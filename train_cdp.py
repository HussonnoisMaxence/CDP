
import torch, os
import time
from omegaconf import OmegaConf
import tqdm
from agent.sac import SACAgent
from utils.networks import eval_mode
from agent.rewardmodel import RewardModel
from agent.VQVAE import VQVAEDiscriminator
import agent.replaybuffer as buffer
from agent.smm import SMMAgent
import utils.utils as utils
import utils.plot as plotter
import utils.oracles as oracles
import numpy as np
from collections import deque

import hydra

import wandb
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
#os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"
class Workspace(object):
    def __init__(self, cfg):  
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device,  torch.cuda.is_available())
        #Get configs
        cfg = cfg.cfg
        
        self.cfg = utils.experiments_settings(cfg)

        ## Set Seeds
        utils.set_seed_everywhere(self.cfg.seed)
        ## Build environment
        self.env = utils.get_env(self.cfg.environment_cfg)
        self.eval_env = utils.get_env(self.cfg.environment_cfg,eval=True)
        self.observation_shape = self.env.observation_space.shape[0]
        self.action_shape = self.env.action_space.shape[0]
        self.action_range = [
                float(self.env.action_space.low.min()),
                float(self.env.action_space.high.max())
            ]
        
        self.env.seed(self.cfg.seed)
        self.env.action_space.seed(self.cfg.seed)
        self.env.observation_space.seed(self.cfg.seed)
        self.eval_env.seed(self.cfg.seed)
        self.eval_env.action_space.seed(self.cfg.seed)
        self.eval_env.observation_space.seed(self.cfg.seed)


        ## Instantiating Logger
        wandb.init(
            mode=self.cfg.wmode,
            project=self.cfg.wproject,
            name=f"{self.cfg.job_id}/{cfg.seed}",
            group=f"{self.cfg.wgname}",
            job_type=self.cfg.wjobtypes,
            notes=self.cfg.wnotes,
            tags=self.cfg.wtags
            )
        wandb.config = cfg

        # dumps to file:
        with open(f"{cfg.dir}/config.yaml", "w") as f:
            OmegaConf.save(cfg, f)

        pref_space = (self.get_shape(self.cfg.pref_embedding),)
        div_space = (self.get_shape(self.cfg.discovery_div_embedding),)
        self.ds = div_space

        self.rbf_smm = buffer.ReplayBufferZ(
            self.env.observation_space.shape,
            self.env.action_space.shape, 
            (self.cfg.discriminator_codebook_size,),
            pref_space, div_space, 
            self.cfg.sac_capacity, 
            self.device)

        self.smm = SMMAgent(
                    obs_dim=self.observation_shape, 
                    action_dim=self.action_shape, 
                    action_range=self.action_range, 
                    device=self.device, 
                    hidden_dim = self.cfg.smm_sac_hidden_dim, 
                    hidden_depth = self.cfg.smm_sac_hidden_depth, 
                    log_std_bounds =  self.cfg.smm_sac_log_std_bound,
                    discount=self.cfg.smm_sac_gamma, 
                    init_temperature=self.cfg.smm_sac_alpha, 
                    alpha_lr=self.cfg.smm_sac_learning_rate, 
                    actor_lr=self.cfg.smm_sac_learning_rate, 
                    actor_update_frequency=self.cfg.smm_sac_actor_train_freq , 
                    critic_lr=self.cfg.smm_sac_learning_rate,
                    critic_tau=self.cfg.smm_sac_polyak_factor, 
                    critic_target_update_frequency=self.cfg.smm_sac_critic_train_freq,
                    batch_size=self.cfg.smm_sac_batch_size, 
                    train_freq=self.cfg.smm_sac_train_freq, 
                    learnable_temperature=self.cfg.smm_sac_alpha_auto,
                    skill_dim=self.cfg.discriminator_codebook_size,
                    hidden_dim_smm=self.cfg.smm_hidden_dim,
                    code_dim_smm=self.cfg.smm_code_dim,
                    vae_beta=self.cfg.smm_vae_beta,
                    sp_lr=self.cfg.smm_sp_lr,
                    vae_lr=self.cfg.smm_vae_lr,
                    prior_dim = div_space[0],
                    scale_factors=self.cfg.exploration_scale_factors,
                    beta_smm=self.cfg.beta_smm
                    )

        
        ## Instantiating Discriminator
        self.discriminator = VQVAEDiscriminator(
            state_size=div_space[0], 
            hidden_size=self.cfg.discriminator_hidden_size,
            codebook_size=self.cfg.discriminator_codebook_size,
            code_size=self.cfg.discriminator_code_size,
            beta=self.cfg.discriminator_beta,
            num_layers=self.cfg.discriminator_num_layers,
            normalize_inputs=self.cfg.discriminator_normalize_inputs
            ).to(self.device)

        self.div_distrib = None
        self.pref_distrib = None

        ## Instantiating Reward Model
        self.reward_model = RewardModel(
                    pref_space[0],
                    ensemble_size=self.cfg.rm_ensemble_size,
                    size_segment=self.cfg.rm_segment_size,
                    lr=self.cfg.rm_learning_rate,
                    mb_size=self.cfg.rm_query_number, #int(self.cfg.max_feedback/10),
                    sampling_mode=self.cfg.rm_sampling_mode,
                    reward_mode=self.cfg.rm_reward_mode,
                    hidden_size=self.cfg.rm_hidden_size,
                    num_layers=self.cfg.rm_num_layers,
                    last_activation=self.cfg.rm_last_activation,
                    last_size=self.cfg.rm_last_size,
                    #Teachers
                    teacher_beta=self.cfg.teacher_beta,
                    teacher_gamma=self.cfg.teacher_gamma,
                    teacher_eps_mistake=self.cfg.teacher_eps_mistake,
                    teacher_eps_skip=self.cfg.teacher_eps_skip,
                    teacher_eps_equal=self.cfg.teacher_eps_equal,
                    device=self.device 
                    )

        self.total_feedback = 0
        self.labeled_queries = 0

        ## Instantiating Agent
        self.rbf = buffer.ReplayBuffer(np.array(self.env.observation_space.shape)+ self.cfg.discriminator_codebook_size,  
                                self.env.action_space.shape, self.cfg.sac_capacity, self.device, window=1)
        
        
        self.agent = SACAgent (
                    obs_dim=self.observation_shape+self.cfg.discriminator_codebook_size, 
                    action_dim=self.action_shape, 
                    action_range=self.action_range, 
                    device=self.device, 
                    hidden_dim = self.cfg.sac_hidden_dim, 
                    hidden_depth = self.cfg.sac_hidden_depth, 
                    log_std_bounds =  self.cfg.sac_log_std_bound,
                    discount=self.cfg.sac_gamma, 
                    init_temperature=self.cfg.sac_alpha, 
                    alpha_lr=self.cfg.sac_learning_rate, 
                    actor_lr=self.cfg.sac_learning_rate, 
                    actor_update_frequency=self.cfg.sac_actor_train_freq , 
                    critic_lr=self.cfg.sac_learning_rate,
                    critic_tau=self.cfg.sac_polyak_factor, 
                    critic_target_update_frequency=self.cfg.sac_critic_train_freq,
                    batch_size=self.cfg.sac_batch_size, 
                    learnable_temperature=self.cfg.sac_alpha_auto,
                    actor_betas=self.cfg.sac_betas,
                    critic_betas=self.cfg.sac_betas,
                    alpha_betas=self.cfg.sac_betas,
                    normalize_state_entropy=True)
        

        self.k_obses = deque(maxlen=self.cfg.k)           

        
        self.settings_info = {
            'env': self.cfg.environment_cfg.name,
            'dir': self.cfg.dir,
            'timestep': 0,
            'pdf': True,
            'get_legends': True,
            'area': self.cfg.environment_cfg.config.goal,
            'plot_area': True,
            'color_mapping': None,
            'phase': 'exploration',
            'get_velocities': False
        }
        if self.cfg.objective == 'smm' or self.cfg.objective == 'smm_prior':
            self.discriminator=None

        self.smm_info = {
            'objective': self.cfg.objective,
            'use_reward': self.cfg.use_reward,
        }
        self.update_after_reset = True
        self.eval_itr = self.cfg.exploration_evaluation_freq
        self.eval_expl = None

    def reset_discr(self):
        self.discriminator = VQVAEDiscriminator(
            state_size=self.ds[0], 
            hidden_size=self.cfg.discriminator_hidden_size,
            codebook_size=self.cfg.discriminator_codebook_size,
            code_size=self.cfg.discriminator_code_size,
            beta=self.cfg.discriminator_beta,
            num_layers=self.cfg.discriminator_num_layers,
            normalize_inputs=self.cfg.discriminator_normalize_inputs
            ).to(self.device)

    def get_shape(self, embedding):
        if embedding == 'state' or embedding == 'next_state':
            shape = self.observation_shape
        elif embedding == 'prior' :
            shape = self.env.prior_space.shape[0]
        elif embedding == 'reward_feature':
            shape = self.cfg.rm_last_size #[-1]
        return shape

    def get_obs(self, T, embedding):
        if embedding == 'state':
            obs = T['obs'][-2]
        elif embedding == 'prior':
            obs = T['prior'][-2]
        elif embedding == 'next_prior':
            obs = T['prior'][-1]
        elif embedding == 'next_state':
            obs = T['obs'][-1]
        elif embedding == 'reward_feature':
            obs = self.reward_model.f_hat(T['obs'][-1])
        return obs 
               

# Phases
    def run_exploration(self):
        self.timestep = 0
        data = []
        self.episode = 0
        metrics = {}

        bar = tqdm.tqdm(total=self.cfg.exploration_total_timesteps, desc='Exploration Phase', smoothing=0)
        while self.timestep < self.cfg.exploration_total_timesteps:

            self.k_obses = deque(maxlen=self.cfg.k)
            self.k_priors = deque(maxlen=self.cfg.k)

            #Initiate the episode
            if self.cfg.prior:
                obs, prior = self.env.reset()
            else:
                obs = self.env.reset()
                prior = obs

            done = False
            done_no_max = False
            episode_step = 0
            episode_reward = 0
            episode_reward_hat = 0

            # Sample skill
            z = torch.randint(low=0, high=self.smm.skill_dim, size=(1,)).to(self.device)
            z_vector = np.zeros(self.smm.skill_dim)
            z_vector[z] = 1


            #next step
            self.k_obses.append(obs)
            self.k_priors.append(prior)

            while not(done):
                
                # sample action for data collection
                if self.timestep < self.cfg.exploration_random_sample:
                    action = self.env.action_space.sample()
                else:
                    with eval_mode(self.smm):
                        action = self.smm.act(np.concatenate([obs, z_vector], axis=-1), sample=True)
                    
                next_obs, reward, done, info = self.env.step(action)
                #
                done = True if episode_step == self.cfg.step_max-1 else False
                done_no_max = done

                #Compute reward
                self.k_obses.append(next_obs)
                self.k_priors.append(info['prior'])
                reward_obs = self.get_obs(T={'obs': self.k_obses,'z':z_vector,'action': action,'prior': self.k_priors}, embedding=self.cfg.pref_embedding)
                if self.timestep >= self.cfg.exploration_start_guide:
                    with torch.no_grad():
                        reward_hat = self.reward_model.r_hat(reward_obs) #
                else:
                    reward_hat = -1

                ## Update reward_model
                metrics = self.update_reward_model(metrics)

                ## Update guide
                metrics = self.update_guide(metrics)

                start_time_init = time.time()
                #Off Policy learning
                metrics = self.update_policy(metrics)
                #
                metrics.update(metrics)

                        
                div_obs = self.get_obs(T={'obs': self.k_obses,'z':z_vector,'action': action,'prior':self.k_priors}, embedding=self.cfg.discovery_div_embedding)

                #Add to replay buffer
                self.rbf_smm.add(obs, action, z_vector, reward_hat, next_obs, done, done_no_max, reward_obs, div_obs)
                if self.total_feedback < self.cfg.max_feedback:
                    self.reward_model.add_data(reward_obs, reward, done)

                obs = next_obs
                prior = info['prior']

                episode_reward += reward
                episode_reward_hat += reward_hat
                self.timestep +=1
                episode_step +=1   

            bar.update(episode_step)
            self.episode += 1
            data.append([info['traj'],z.item()])

            if self.timestep> self.eval_itr:
                ## Evaluate exploration
                metrics_eval = self.evaluate_exploration(self.smm)
                metrics.update(metrics_eval)
                self.eval_itr += self.cfg.exploration_evaluation_freq

            if self.episode%4 == 0:
                metrics.update({
                        'Exploration/return': episode_reward,
                        'Exploration/return_pref': episode_reward_hat,
                        'Exploration/episodes': self.episode,
                        'Exploration/timesteps': self.timestep
                })
                wandb.log(metrics)
                metrics = {}
                

            


        if self.cfg.save_state_visited:
            data = np.array(data)
            plotter.save_data(data, self.settings_info)
            plotter.plot_state_visited(data, self.settings_info)

    def focus(self):
         
        metrics_focus = {}
        values = self.reward_model.r_hat_batch(self.pref_distrib)
        ma = np.max(values)
        mi = np.min(values)
        self.ma = ma
        self.mi = mi

        norm_values = np.array([(v-mi)/(ma-mi) for v in values])

        if self.cfg.beta == 'oracle_beta':
            beta= self.select_beta(ma, mi)
            #e = e*(ma-mi) + mi
            idxs = np.where(norm_values >= beta)[0]
            if len(idxs) == 0 :
                idxs = np.where(norm_values >= np.mean(norm_values))[0]

        elif self.cfg.beta == 'schedule_beta':
            beta = np.mean(values) +  np.std(values)
            idxs = np.where(norm_values >= beta)[0]
            if len(idxs) == 0 :
                idxs = np.where(norm_values >= np.mean(values))[0]
        else:
            beta = self.cfg.beta_pref
            idxs = np.where(norm_values >= beta)[0]

        idxs_t = torch.from_numpy(idxs).long().to(self.device)
        self.div_distrib = self.div_distrib[idxs_t]
        self.pref_distrib = self.pref_distrib[idxs]

        metrics_focus.update({
        })

        return metrics_focus

    def discover(self):
        metrics_discover = {}

        # Normalize dataset
        div_distrib = self.div_distrib 
        self.discriminator.update_normalizer(dataset=div_distrib)
        # Create optimizer
        optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.cfg.discovery_learning_rate)
        
        #Training loop
        indices = list(range(self.div_distrib.size(0)))
        loss_list = []

        
        self.discriminator.train()

        bar = tqdm.tqdm(total=self.cfg.discovery_epoch, desc='Discovery Phase', smoothing=0)
        for iter_idx in range(self.cfg.discovery_epoch):
            # Make batch
            batch_indices = np.random.choice(indices, size=self.cfg.discovery_batch_size)
            batch = dict(next_state=self.div_distrib[batch_indices])
            # Forward + backward pass
            optimizer.zero_grad()
            
            loss = self.discriminator(batch) 
    
            loss.backward()
            optimizer.step()

            # Log progress
            loss_list.append(loss.item())

            bar.update(1)

        metrics_discover.update({
            'Discovery/discrim_loss': np.mean(loss_list)
            })
        self.discriminator.eval()
        return metrics_discover
    
    def learning (self, eval_info):
        metrics = {}
        self.episode = 0
        self.timestep = 0
        bar = tqdm.tqdm(total=self.cfg.learning_total_timesteps, smoothing=0)
        while self.timestep < self.cfg.learning_total_timesteps:
            
            self.k_obses = deque(maxlen=self.cfg.k)
            self.k_priors = deque(maxlen=self.cfg.k)
            #Initiate the episode
            if self.cfg.prior:
                obs, prior = self.env.reset()
            else:
                obs = self.env.reset()
                prior = obs

            done = False
            done_no_max = False
            ep_gt_r = 0
            ep_div_r = 0
            episode_step = 0
            
            #sample skill
            z = torch.randint(low=0, high=self.discriminator.codebook_size, size=(1,)).to(self.device)
            z_vector = np.zeros(self.discriminator.codebook_size)
            z_vector[z] = 1
            self.k_obses.append(obs)
            self.k_priors.append(prior)
            while not(done):
                # sample action for data collection
                if self.timestep <  self.cfg.learning_learning_start:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        with eval_mode(self.agent):
                            action = self.agent.act(np.concatenate([obs, z_vector], axis=-1), sample=True)
                
                next_obs, gt_reward, done, info = self.env.step(action)

                done = True if episode_step == self.cfg.step_max-1 else done
                done_no_max = done

                self.k_obses.append(next_obs)
                self.k_priors.append(info['prior'])
                div_obs = self.get_obs(T={'obs': self.k_obses,'z':z_vector,'action': action,'prior': self.k_priors}, embedding=self.cfg.discovery_div_embedding)
                
                # Get reward from discriminator
                with torch.no_grad():
                    r_div = self.discriminator.compute_logprob_under_latent(
                        dict(
                        next_state=torch.tensor(div_obs,dtype=torch.float).to(self.device).unsqueeze(0), 
                        skill=z)
                        ).cpu().numpy()
                
                #Add to replay buffer
                self.rbf.add(
                    np.concatenate([obs, z_vector], axis=-1), action, r_div, 
                    np.concatenate([next_obs, z_vector], axis=-1), done, done_no_max)

                #Off Policy learning
                if self.timestep >  self.cfg.learning_learning_start:
                    train_metrics = self.agent.update(self.rbf, self.timestep)
                    metrics.update(train_metrics)

                obs = next_obs
                prior = info['prior']
                ep_gt_r += gt_reward
                ep_div_r += r_div
                self.timestep +=1
                episode_step +=1

            metrics.update({
                'Learning/return': ep_gt_r,
                'Learning/return_diveristy': ep_div_r,
                'Learning/timesteps': self.timestep })
            
            wandb.log(metrics)
            if not(self.timestep%self.cfg.learning_evaluation_freq):
                self.evaluate_learning(self.agent, eval_info)
            bar.update(episode_step)


## Updates
    def update_reward_model(self, metrics):
        ## Update reward_model
        time_start_guide = self.timestep >= self.cfg.exploration_start_guide
        time_update_guide =(self.timestep)%(self.cfg.exploration_update_guide) == 0
        time_stop_reward = self.total_feedback < self.cfg.max_feedback

        if  time_start_guide and time_update_guide and time_stop_reward:
            metric_rewards = self.learn_reward(self.timestep)
            self.rbf_smm.relabel_with_predictor(self.reward_model)
            if self.cfg.discovery_div_embedding == 'reward_feature':
                self.rbf_smm.relabel_pref_with_predictor(self.reward_model)
                self.reset_discr()

            metrics.update(metric_rewards)

            if self.update_after_reset:
                self.smm.reset_critic()
                #self.smm.reset_smm() 
            
                self.smm_info = {'objective': self.cfg.objective, 'use_reward': self.cfg.use_reward, 'prefix': 'SMM_reset'}
                metrics_smm = {}
                metrics_smm = self.smm.update_after_reset(
                                    self.rbf_smm, 
                                    self.smm_info,
                                    vqvae=self.discriminator,
                                    gradient_update=self.cfg.exploration_update_after_reset)
                self.update_after_reset = False        
                            #metrics.update(metrics)
        return metrics

    def update_guide(self, metrics):
        ## Update guiding
        time_start_guide = self.timestep >= self.cfg.exploration_start_guide
        time_update_guide = (self.timestep)%(self.cfg.exploration_update_guide) == 0
        if time_start_guide and time_update_guide and self.cfg.use_vqvae:
            metrics_guide = {}

            ## Run Focus phases
            self.div_distrib = self.rbf_smm.get_full_divs(1)
            self.pref_distrib = self.rbf_smm.get_full_prefs(1, tensor=False)
            #self.old_pref_distrib = self.pref_distrib
            if self.cfg.use_reset_critics:
                self.smm.reset_critic()

            if self.cfg.use_preferred_region:
                ## Focus phase
                metrics_guide = self.focus()  
                metrics.update(metrics_guide)

            #if self.cfg.discovery_div_embedding == 'reward_feature':
            #    self.reset_discr()
            ## Discovery phase
            metrics_guide = self.discover()
            metrics.update(metrics_guide)


        return metrics

    def update_policy(self, metrics):
        metrics_policies={}
        if self.timestep > self.cfg.exploration_random_sample:
            if self.timestep < self.cfg.exploration_start_guide:
                self.smm_info = {'objective': 'smm', 'use_reward': False, 'prefix': 'SMM_pt' }
                metrics_policies = self.smm.update(self.timestep, self.rbf_smm, self.smm_info )

            else:        
                self.smm_info = {'objective': self.cfg.objective, 'use_reward': self.cfg.use_reward, 'prefix': 'SMM'}
                metrics_policies = self.smm.update(
                                        self.timestep,
                                        self.rbf_smm, 
                                        self.smm_info,
                                        vqvae=self.discriminator)
            
        metrics.update(metrics_policies)
        return metrics

## other
    def learn_reward(self, timestep):
        metrics_reward = {}
        labeled_queries = 0
        if self.total_feedback==0:
            labeled_queries = self.reward_model.uniform_sampling()
        else:
            labeled_queries = self.reward_model.sampling()
        self.total_feedback += self.reward_model.mb_size
        self.labeled_queries+= labeled_queries 
        if self.labeled_queries>0: 
            for epoch in range(self.cfg.reward_update):
                if self.cfg.teacher_eps_equal > 0:
                    train_acc, loss = self.reward_model.train_soft_reward()
                else:
                    train_acc, loss = self.reward_model.train_reward()

                total_acc = np.mean(train_acc)

                if total_acc > 0.97:
                    break;
                    
            metrics_reward.update({
                        'Reward_model/reward_acc': total_acc,
                        'Reward_model/mean_loss': np.mean(loss),
                        'Reward_model/std_loss': np.std(loss),
                        'Reward_model/max_loss': np.max(loss),
                        'Reward_model/min_loss': np.min(loss),
                        'Reward_model/total_feedback': self.total_feedback,
                        'Reward_model/num_update': epoch,
                        })
        metrics_reward.update({
                        'Reward_model/labeled': self.labeled_queries,
                        })   
        return metrics_reward
    
    def select_beta(self, ma, mi):  
        #Sample target region
        obses = utils.get_pairs(-1, 1, 0.01)
        oracle_pref_region = utils.get_preferred_regions(obses, self.cfg.environment_cfg.config.goal) 
        # compute values
        values = self.reward_model.r_hat_batch(oracle_pref_region)
        norm_values = np.array([(v-mi)/(ma-mi) for v in values])

        ma = np.max(norm_values)
        mi = np.min(norm_values)

        return mi


# Launchers
    def run_oracle(self):
        color_mapping, c = plotter.get_color_mapping(self.cfg.use_vqvae, self.cfg.discovery_div_embedding, 
                                                     self.device, self.reward_model, self.discriminator, self.settings_info)

        self.settings_info.update({
            'color_mapping': color_mapping,
        })
        ## Get exploration
        obses = utils.get_pairs(-1, 1, 0.01) 
        self.settings_info['prefix']='total' 
        plotter.plot_state([obses],  area=None, info=self.settings_info)


        ## Focus phase with oracle reward
        if self.cfg.cdp:
            values = np.array([oracles.reward_area(obs, self.cfg.environment_cfg.config.goal) for obs in obses])
            norm_values = np.array([(v-np.min(values))/(np.max(values)-np.min(values)) for v in values])
            if self.cfg.beta =='default':
                beta = self.cfg.beta_pref
            else:
                beta = np.max(norm_values)

            obses = np.array(obses)[np.where(norm_values >= beta)[0]]
            self.settings_info['prefix']='preferred' 
            plotter.plot_state([obses], area=self.cfg.environment_cfg.config.goal, info=self.settings_info)

        self.div_distrib = torch.from_numpy(obses).float().to(self.device)
        ## Last discovery phase
        self.discover()
        color_mapping, c = plotter.get_color_mapping(self.cfg.use_vqvae, self.cfg.discovery_div_embedding, 
                                                     self.device, self.reward_model, self.discriminator, self.settings_info)

        self.settings_info.update({
            'centroids': c,
            'color_mapping': color_mapping,
            'phase': 'learning'
        })
        self.timestep = 0

        ## Run Skill Learning phase
        self.evaluate_learning(self.agent, self.settings_info)
        self.settings_info['get_legends']= False 
        self.learning(self.settings_info)
        self.evaluate_learning(self.agent, self.settings_info)

    def run_nav2d(self):
        color_mapping, c = plotter.get_color_mapping(self.cfg.use_vqvae, self.cfg.discovery_div_embedding, 
                                                     self.device, self.reward_model, self.discriminator, self.settings_info)

        self.settings_info.update({
            'color_mapping': color_mapping,
        })
        ## Run exploration phase
        self.run_exploration() 
        ## Save Reward Model
        dir = f"{self.settings_info['dir']}/models"
        utils.create_directories(dir)
        self.reward_model.save(dir,-1)

        ## Last Focus phase
        self.div_distrib = self.rbf_smm.get_full_divs(1)
        self.pref_distrib = self.rbf_smm.get_full_prefs(1, tensor=False)
        self.focus()  

        ## Last discovery phase
        self.discover()
        color_mapping, c = plotter.get_color_mapping(self.cfg.use_vqvae, self.cfg.discovery_div_embedding, 
                                                     self.device, self.reward_model, self.discriminator, self.settings_info)

        self.settings_info.update({
            'centroids': c,
            'color_mapping': color_mapping,
            'phase': 'learning'
        })
        self.timestep = 0

        ## Run Skill Learning phase
        self.evaluate_learning(self.agent, self.settings_info)
        self.settings_info['get_legends']= False 
        self.learning(self.settings_info)
        self.evaluate_learning(self.agent, self.settings_info)

    def run_mujoco(self):
        
        self.run_exploration()
        dir = f"{self.settings_info['dir']}/models"
        utils.create_directories(dir)
        self.reward_model.save(dir,-1)
        self.div_distrib = self.rbf_smm.get_full_divs(1)
        self.pref_distrib = self.rbf_smm.get_full_prefs(1, tensor=False)
        self.old_pref_distrib = self.pref_distrib
        self.focus()  

        self.discover()
        self.cfg.record_video=False #True
        self.cfg.plot_pos= True
        metrics_eval = self.evaluate_exploration(self.smm)
        
        self.settings_info.update({
            'phase': 'learning'
        })  
        #plotter.run_test_videos(self.smm, self.path, self.cfg, self.timestep, info='exploration')
        self.timestep = 0
        self.evaluate_learning(self.agent, self.settings_info)
        self.settings_info['get_legends']= False 
        self.learning(self.settings_info)
        #plotter.run_test_videos(self.agent, self.path, self.cfg, self.timestep, info='learning')
        self.settings_info['get_legends']= True 
        self.settings_info['get_velocities'] = True
        self.evaluate_learning(self.agent, self.settings_info)
        self.agent.save(dir,-1)
        
# Evaluations
    def evaluate_exploration(self, agent):
        metrics_eval = {}
        self.settings_info['timestep']= self.timestep
        if self.cfg.exploration_metric or self.cfg.plot_regions or self.cfg.coverage_metric:
            exploration_distrib = self.rbf_smm.get_full_prefs(1, tensor=False)
            if self.cfg.prior:
                if self.cfg.environment_cfg.name =='HC':
                    exploration_distrib = exploration_distrib[:,8]

        if self.cfg.plot_skill_exploration:
            self.settings_info.update({'traj': [], 'centroids': [], 'frames':[]})

            if self.cfg.plot_pos:
                env = self.eval_env
            else:
                env = self.env
            #run skills
            for i in range (self.cfg.discriminator_codebook_size):
                ## plot variable
                #return_skill = 0
                #Initiate the episode
                if self.cfg.prior:
                    obs, prior = env.reset()
                else:
                    obs = env.reset()
                    prior = obs
                done, episode_step = False, 0

                if self.cfg.record_video or self.cfg.plot_pos:
                    env.record = True
             
                #sample skills
                z_vector = np.zeros(self.cfg.discriminator_codebook_size)
                z_vector[i] = 1
                while not(done):
                    with torch.no_grad():
                        with eval_mode(agent):
                            action = agent.act(np.concatenate([obs, z_vector], axis=-1), sample=True)
                    obs, reward, done, info = env.step(action)
                    done = True if episode_step +1 == self.cfg.step_max else False
                    episode_step +=1

                with torch.no_grad():
                    centroid = self.discriminator.get_centroids(dict(skill=torch.tensor(i).to(self.device)))[0].detach().cpu().numpy()   

                self.settings_info['traj'].append(info['traj'])
                if self.cfg.record_video or self.cfg.plot_pos:
                    self.settings_info['frames'].append(info['frames'])
                self.settings_info['centroids'].append(centroid)

            
            if self.cfg.plot_pos:
                plotter.plot_pos(self.settings_info)
                env.record = False
                
            if self.cfg.environment_cfg.type =='2d':
                plotter.plot_traj(self.settings_info)
            
            if self.cfg.environment_cfg.name =='HC':
                plotter.plot_traj_hc(self.settings_info)
            
            if self.cfg.record_video:
                plotter.save_frames(self.settings_info)
                env.record = False
            

        if self.cfg.exploration_metric:
            if self.cfg.environment_cfg.type =='2d':
                coverage,_,_ = utils.count_state_coverage(exploration_distrib, list(self.cfg.environment_cfg.config.goal), precision=2)
            if self.cfg.environment_cfg.type =='mujoco':
                coverage,_,_ = utils.count_state_coverage_hc(exploration_distrib, list(self.cfg.environment_cfg.config.goal))

            metrics_eval.update({
                    'Evaluate_explorations/coverage_target_region': coverage,
                })
            
        if self.cfg.coverage_metric:
            if self.cfg.environment_cfg.type =='2d':
                f1_score , precision, recall = utils.get_f1_bins(self.pref_distrib, list(self.cfg.environment_cfg.config.goal), precision_bins=2)
            if self.cfg.environment_cfg.type =='mujoco':
                f1_score , precision, recall = utils.get_f1_bins_hc(self.pref_distrib[:,8], list(self.cfg.environment_cfg.config.goal), precision_bins=2)
            #e_norm , mean, std= utils.evaluate_beta(self.reward_model, list(self.cfg.environment_cfg.config.goal))
            metrics_eval.update({
                    'Evaluate_rm/f1_score':f1_score , 
                    'Evaluate_rm/precision': precision,
                    'Evaluate_rm/recall': recall  
                })
        #plot regions
        if self.cfg.plot_regions:

                if self.cfg.environment_cfg.name =='HC':
                    if self.cfg.use_preferred_region and self.timestep >= self.cfg.exploration_start_guide :
                        self.settings_info['prefix']='preferred' 
                        pref_distrib = self.pref_distrib[:,8]
                        plotter.plot_state_hc([pref_distrib], info=self.settings_info)

                    self.settings_info['prefix']='total' 
                    plotter.plot_state_hc([exploration_distrib], info=self.settings_info)
                    
                if self.cfg.environment_cfg.type =='2d':    
                    if self.cfg.use_preferred_region and self.timestep >= self.cfg.exploration_start_guide :
                        
                        self.settings_info['prefix']='preferred' 
                        plotter.plot_state([self.pref_distrib], info=self.settings_info)
                    self.settings_info['prefix']='total' 
                    plotter.plot_state([exploration_distrib], info=self.settings_info)

       

        if self.cfg.plot_reward:
            obs = utils.get_pairs(-1, 1, 0.01)
            with torch.no_grad():
                rewards = self.reward_model.r_hat_batch(obs)
            plotter.plot_reward_nc(rewards, obs,   info=self.settings_info)

        if self.cfg.plot_disc:
            if self.cfg.discovery_div_embedding == 'reward_feature':
                c, color_mapping = plotter.plot_discr_region_pref(self.device, self.reward_model, self.discriminator, self.settings_info)
            else:
                c, color_mapping = plotter.plot_discr_region(self.device, self.discriminator, self.settings_info)
        

        # Log values
        #metrics_guide['Evaluate_exploration/step'] = self.timestep
        return metrics_eval #wandb.log(metrics_guide)
    
    def evaluate_learning(self, agent, eval_info):
        reward_mean = 0
        eval_info.update({
            'traj': [],
            'timestep': self.timestep,
            'frames': []
        })
        if eval_info['get_velocities']:   
            eval_info.update({
                'velocities': [[] for i in range(self.cfg.discriminator_codebook_size)]})
        #run skills
        if self.cfg.plot_pos:
            env = self.eval_env
        else:
            env = self.env
        for i in range (self.cfg.discriminator_codebook_size):

            ## plot variable
            return_skill = 0
            #Initiate the episode
            if self.cfg.prior:
                obs, prior = env.reset()
            else:
                obs = env.reset()
            done, episode_step = False, 0
            if self.cfg.record_video or self.cfg.plot_pos:
                env.record = True
            #sample skills
            z_vector = np.zeros(self.cfg.discriminator_codebook_size)
            z_vector[i] = 1
            while not(done):
                with eval_mode(agent):
                    action = agent.act(np.concatenate([obs, z_vector], axis=-1), sample=True)
                obs, reward, done, info = env.step(action)
                done = True if episode_step +1 == self.cfg.step_max else False
                episode_step +=1
                return_skill += reward

                if eval_info['get_velocities']:   
                    eval_info['velocities'][i].append(info['prior'])


            eval_info['traj'].append(info['traj'])
            
            if self.cfg.record_video or self.cfg.plot_pos:
                eval_info['frames'].append(info['frames'])

            reward_mean += return_skill



        if self.cfg.plot_pos:
                plotter.plot_pos(eval_info)
        if self.cfg.environment_cfg.type =='2d': 
            plotter.plot_traj(eval_info)
            
        if self.cfg.environment_cfg.name =='HC':
            plotter.plot_traj_hc(eval_info)

        if self.cfg.record_video:
            plotter.save_frames(eval_info)
        if eval_info['get_velocities']:  
            plotter.plot_velocity(eval_info)
            data = np.array(eval_info['velocities'])       
            plotter.save_data(data, self.settings_info)
            for i in range(100):
                v = data[:,i,:]
                wandb.log({
                    'Eval_velocity/var_velocity':np.var(v)
                })
                for index, s in enumerate(v):
                    wandb.log({
                        f"Eval_velocity/velocity_{index}": s
                    })



@hydra.main(version_base=None, config_path="./configs", config_name="expt.yaml")
def main(cfg):
    workspace = Workspace(cfg)
    if workspace.cfg.environment_cfg.type =='2d':
        if workspace.cfg.exp_setting == 'oracle':
            workspace.run_oracle()
        else:
            workspace.run_nav2d()

    if workspace.cfg.environment_cfg.type =='mujoco':
        workspace.eval_env = utils.get_env(workspace.cfg.environment_cfg,eval=True)
        workspace.run_mujoco()


if __name__ == '__main__':
    main()