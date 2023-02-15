from builtins import int, list, print, range, str
import argparse, torch, os
from statistics import mean
from time import time
from utils.oracles import plot_skill_speed, plot_skill_pos
from agent.sac import SACAgent
from utils.networks import eval_mode
from agent.rewardmodel import RewardModel
from agent.VQVAE import VQVAEDiscriminator
from agent.replaybuffer import ReplayBuffer, ReplayBufferZ
from agent.smm import SMMAgent
from utils.oracles import compute_target_rewards, get_distribution, get_kl_div,  get_preferred_region, get_target_state, sample_target_region
from utils.plot import run_test_videos
from utils.utils import get_env, save_config, set_seed_everywhere, read_config
import numpy as np
from collections import deque
from utils.oracles import plot_state, plot_discriminator, plot_skill

from utils.logger import Logger

torch.backends.cudnn.benchmark = False
from PIL import Image

class Workspace(object):
    def __init__(self, cfg):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #Get configs
        self.cfg = cfg
        
        self.training_cfg = self.cfg['Training']
        self.env_cfg = self.cfg['Environment']
        self.exploration = self.training_cfg['Exploration']
        self.focus_cfg = self.training_cfg['Focus']
        self.discovery_cfg = self.training_cfg['Discovery']
        self.learning_cfg = self.training_cfg['Learning']
        
        self.sac_cfg = self.cfg['SAC']
        self.sac_smm_cfg = self.cfg['SAC_SMM']
        self.discriminator_cfg = self.cfg['Discriminator']
        self.smm_cfg = self.cfg['SMM']
        self.rm_config = self.cfg['Reward_Model']
        self.logger_cfg = self.cfg['Logger']
       

        ## Build environment
        self.env = get_env(self.env_cfg)
        self.observation_shape = self.env.observation_space.shape[0]
        self.action_shape = self.env.action_space.shape[0]
        self.action_range = [
                float(self.env.action_space.low.min()),
                float(self.env.action_space.high.max())
            ]
        self.step_max = self.training_cfg['step_max']
        self.k_obses = deque(maxlen=self.training_cfg['k'])
        ## Set Seeds
        seed = self.training_cfg['seed']
        set_seed_everywhere(seed)
        self.env.seed(seed)
        self.env.action_space.seed(seed)

        ## Instantiating Logger
        self.logger = Logger(
            log_dir=self.logger_cfg['log_dir'],
            file_name=self.logger_cfg['file_name'],
            agent=self.logger_cfg['agent'],
            save_tb=self.logger_cfg['save_tb'],
            log_frequency=self.logger_cfg['log_frequency']
            #,config=self.cfg
        )
        pref_space = (self.get_shape(self.focus_cfg['pref_embedding']),)
        div_space = (self.get_shape(self.discovery_cfg['div_embedding']),)

        self.rbf_smm = ReplayBufferZ(self.env.observation_space.shape, self.env.action_space.shape, 
                                    (self.discriminator_cfg['codebook_size'],), pref_space, div_space , self.sac_cfg['capacity'], self.device, window=1)

        self.smm = SMMAgent(
                    obs_dim=self.observation_shape, 
                    action_dim=self.action_shape, 
                    action_range=self.action_range, 
                    device=self.device, 
                    hidden_dim = self.sac_smm_cfg['policy_kwargs']['hidden_dim'], 
                    hidden_depth = self.sac_smm_cfg['policy_kwargs']['hidden_depth'], 
                    log_std_bounds = self.sac_smm_cfg['policy_kwargs']['log_std_bound'],
                    discount=self.sac_smm_cfg['gamma'], 
                    init_temperature=self.sac_smm_cfg['alpha'], 
                    alpha_lr=self.sac_smm_cfg['learning_rate'], 
                    actor_lr=self.sac_smm_cfg['learning_rate'], 
                    actor_update_frequency=self.sac_smm_cfg['actor_train_freq'], 
                    critic_lr=self.sac_smm_cfg['learning_rate'],
                    critic_tau=self.sac_smm_cfg['polyak_factor'], 
                    critic_target_update_frequency=self.sac_smm_cfg['critic_train_freq'],
                    batch_size=self.sac_smm_cfg['batch_size'], 
                    train_freq=self.sac_smm_cfg['train_freq'], 
                    learnable_temperature=self.sac_smm_cfg['alpha_auto'],
                    skill_dim=self.discriminator_cfg['codebook_size'],
                    hidden_dim_smm=self.smm_cfg['hidden_dim'],
                    vae_beta=self.smm_cfg['vae_beta'],
                    sp_lr=self.smm_cfg['sp_lr'],
                    vae_lr=self.smm_cfg['vae_lr'],
                    prior_dim = div_space[0],
                    scale_factors=self.exploration['scale_factors']
                    )

        
        ## Instantiating Discriminator
        self.discriminator = VQVAEDiscriminator(
            state_size=div_space[0] , 
            hidden_size=self.discriminator_cfg['hidden_size'],
            codebook_size=self.discriminator_cfg['codebook_size'],
            code_size=self.discriminator_cfg['code_size'],
            beta=self.discriminator_cfg['beta'],
            num_layers=self.discriminator_cfg['num_layers'],
            normalize_inputs=self.discriminator_cfg['normalize_inputs']
            ).to(self.device)

        self.div_distrib = None
        self.pref_distrib = None


        self.total_feedback = 0
        ## Instantiating Agent
        self.rbf = ReplayBuffer(np.array(self.env.observation_space.shape)+ self.discriminator_cfg['codebook_size'],  
                                self.env.action_space.shape, self.sac_cfg['capacity'], self.device, window=1)
        
        self.agent = SACAgent (
                    obs_dim=self.observation_shape+self.discriminator_cfg['codebook_size'], 
                    action_dim=self.action_shape, 
                    action_range=self.action_range, 
                    device=self.device, 
                    hidden_dim = self.sac_cfg['policy_kwargs']['hidden_dim'], 
                    hidden_depth = self.sac_cfg['policy_kwargs']['hidden_depth'], 
                    log_std_bounds = self.sac_cfg['policy_kwargs']['log_std_bound'],
                    discount=self.sac_cfg['gamma'], 
                    init_temperature=self.sac_cfg['alpha'], 
                    alpha_lr=self.sac_cfg['learning_rate'], 
                    alpha_betas=self.sac_cfg['betas'],
                    actor_lr=self.sac_cfg['learning_rate'], 
                    actor_betas=self.sac_cfg['betas'], 
                    actor_update_frequency=self.sac_cfg['actor_train_freq'], 
                    critic_lr=self.sac_cfg['learning_rate'],
                    critic_betas=self.sac_cfg['betas'], 
                    critic_tau=self.sac_cfg['polyak_factor'], 
                    critic_target_update_frequency=self.sac_cfg['critic_train_freq'],
                    batch_size=self.sac_cfg['batch_size'], 
                    learnable_temperature=self.sac_cfg['alpha_auto'],
                    normalize_state_entropy=True)
        

        self.mean = None
        self.smm.commit = self.focus_cfg['rts']
        self.density = []
        self.timestep = []

        obs = self.env.reset()

        save_config(self.logger_cfg['log_dir']+'config.yaml', self.cfg)

    def get_shape(self, embedding):
        if embedding == 'state' or embedding == 'state_diff' or embedding == 'next_state':
            shape = self.observation_shape
        elif embedding == 'prior' or embedding == 'prior_diff' or embedding == 'init_prior_diff':
            shape = self.env.prior_space.shape[0]
        elif embedding == 'state_action':
            shape = self.observation_shape+self.action_shape
        elif embedding == 'state_action_skill':
            shape = self.observation_shape+self.action_shape+self.discriminator_cfg['codebook_size']
        elif embedding == 'state_skill':
            shape = self.observation_shape+self.discriminator_cfg['codebook_size']
        elif embedding == 'prior_action':
            shape = self.env.prior_space.shape+self.action_shape
        elif embedding == 'reward_feature':
            shape = self.rm_config['hidden_size'][-1]
        return shape

    def get_obs(self, T, embedding):
        if embedding == 'state':
            obs = T['obs'][-2]
        elif embedding == 'state_action':
            obs = np.concatenate([ T['obs'][-2], T['action']], axis=-1)
        elif embedding == 'state_action_skill':
            obs = np.concatenate([ T['obs'][-2],  T['action'],  T['z']], axis=-1)
        elif embedding == 'state_skill':
            obs = np.concatenate([ T['obs'][-2],   T['action']], axis=-1)
        elif embedding == 'prior_action':
            obs = np.concatenate([T['prior'][-2], T['action']], axis=-1)
        elif embedding == 'state_diff':
            obs = T['obs'][-1] - T['obs'][-2]
        elif embedding == 'prior_diff':
            obs = T['prior'][-1] - T['prior'][0]
        elif embedding == 'prior':
            obs = T['prior'][-2]
        elif embedding == 'next_prior':
            obs = T['prior'][-1]
        elif embedding == 'next_state':
            obs = T['obs'][-1]
        elif embedding == 'state_diff':
            obs = T['obs'][-1] - T['obs'][0]
        elif embedding == 'reward_feature':
            obs = self.reward_model.f_hat(T['obs'][-1])
        return obs 

    def run_exploration(self):
        i,j = 0,0
        self.timestep = 0
        data = []
        self.episode = 0
        while self.timestep < self.exploration['total_timesteps']:
            t = []
            self.k_obses = deque(maxlen=self.training_cfg['k'])
            self.k_priors = deque(maxlen=self.training_cfg['k'])
            #Initiate the episode
            if self.training_cfg['prior']:
                obs, prior = self.env.reset()
                init_prior = prior
            else:
                obs = self.env.reset()
                prior = obs
            init_state = obs
            done = False
            done_no_max = False
            episode_step = 0
            episode_reward = 0
            episode_reward_hat = 0
            # Sample skill
            z = torch.randint(low=0, high=self.smm.skill_dim, size=(1,)).to(self.device)
            z_vector = np.zeros(self.smm.skill_dim)
            z_vector[z] = 1
            t.append(obs)

            #next step
            self.k_obses.append(obs)
            self.k_priors.append(prior)

            while not(done):
                #next step

                # sample action for data collection
                if self.timestep < self.exploration['random_sample']:
                    action = self.env.action_space.sample()
                else:
                    with eval_mode(self.smm):
                        action = self.smm.act(np.concatenate([obs, z_vector], axis=-1), sample=True)
                
                next_obs, reward, done, next_prior = self.env.step(action)

                done = True if episode_step == self.step_max-1 else done
                done_no_max = done

                #Compute reward
                self.k_obses.append(next_obs)
                self.k_priors.append(next_prior)
                reward_obs = self.get_obs(T={'obs': self.k_obses,'z':z_vector,'action': action,'prior': self.k_priors}, embedding=self.focus_cfg['pref_embedding'])

                div_obs = self.get_obs(T={'obs': self.k_obses,'z':z_vector,'action': action,'prior':self.k_priors}, embedding=self.discovery_cfg['div_embedding'])
                self.rbf_smm.add(obs, action, z_vector, reward, next_obs, done, done_no_max, reward_obs, div_obs)

                #init_state = self.k_obses[0]
                #init_prior = self.k_priors[0]
                obs = next_obs
                prior = next_prior
                episode_reward += reward
                self.timestep +=1
                episode_step +=1   
                self.logger.log('Exploration/x', obs[0], self.timestep)
                self.logger.log('Exploration/y', obs[1], self.timestep)
            self.logger.log('Exploration/episode_reward', episode_reward,self.timestep)
            
            self.episode += 1
            print('Exploration:', self.timestep)






    def discover(self, timestep):
        # Normalize dataset
        self.discriminator.update_normalizer(dataset=self.div_distrib)
        # Create optimizer
        optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.discovery_cfg['learning_rate'])
        
        #Training loop
        indices = list(range(self.div_distrib.size(0)))
        loss_list = []
        self.discriminator.train()
        #probs = self.SIR() 
        for iter_idx in range(self.discovery_cfg['epoch']):
            # Make batch
            #batch_indices = np.random.choice(a=indices, size=self.discovery['batch_size'], p=probs, replace=False) #np.random.choice(indices, size=self.discovery['batch_size'])
            batch_indices = np.random.choice(indices, size=self.discovery_cfg['batch_size'])
            batch = dict(next_state=self.div_distrib[batch_indices])
            # Forward + backward pass
            optimizer.zero_grad()
            
            loss = self.discriminator(batch) #, weights=weights)
    
            loss.backward()
            optimizer.step()

            # Log progress
            loss_list.append(loss.item())



    def learning (self):
        
        self.timestep = 0
        while self.timestep < self.learning_cfg['total_timesteps']:
            self.k_obses = deque(maxlen=self.training_cfg['k'])
            self.k_priors = deque(maxlen=self.training_cfg['k'])
            #Initiate the episode
            if self.training_cfg['prior']:
                obs, prior = self.env.reset()
                init_prior = prior
                init_state = obs
            else:
                obs = self.env.reset()
                init_state = obs
                prior = obs
            done = False
            done_no_max = False
            episode_reward = 0
            episode_treward = 0
            episode_step = 0
            
            #sample skill
            z = torch.randint(low=0, high=self.discriminator.codebook_size, size=(1,)).to(self.device)
            z_vector = np.zeros(self.discriminator.codebook_size)
            z_vector[z] = 1
            self.k_obses.append(obs)
            self.k_priors.append(prior)
            while not(done):
                # sample action for data collection
                if self.timestep <  self.learning_cfg['learning_start']:
                    action = self.env.action_space.sample()
                else:
                    with eval_mode(self.agent):
                        action = self.agent.act(np.concatenate([obs, z_vector], axis=-1), sample=True)
                
                next_obs, treward, done, next_prior = self.env.step(action)

                done = True if episode_step == self.step_max-1 else done
                done_no_max = done
                self.k_obses.append(next_obs)
                self.k_priors.append(next_prior)
                div_obs = self.get_obs(T={'obs': self.k_obses,'z':z_vector,'action': action,'prior': self.k_priors}, embedding=self.discovery_cfg['div_embedding'])
                
                # Get reward from discriminator
                reward = self.discriminator.compute_logprob_under_latent(dict(
                    next_state=torch.tensor(div_obs,dtype=torch.float).to(self.device).unsqueeze(0), 
                    skill=z)).item()
                
                #Add to replay buffer
                self.rbf.add(
                    np.concatenate([obs, z_vector], axis=-1), action, reward, 
                    np.concatenate([next_obs, z_vector], axis=-1), done, done_no_max)

                #Off Policy learning
                if self.timestep >  self.learning_cfg['learning_start']:
                    self.agent.update(self.rbf, self.timestep, self.logger)


                obs = next_obs
                prior = next_prior
                episode_reward += reward
                episode_treward += treward
                self.timestep +=1
                episode_step +=1
            self.logger.log('Learning/episode_reward', episode_reward,self.timestep)
            self.logger.log('Learning/episode_reward_div', episode_treward, self.timestep)
            print('Learning:', self.timestep)

    def run(self):

        self.run_exploration()
        if self.env_cfg['type'] == 'MJ':
            run_test_videos(self.smm, self.cfg['Environment'], self.logger, self.discriminator.codebook_size, self.training_cfg, info="smm")
            
        if self.env_cfg['type'] != 'MJ':
            div_distrib = self.rbf_smm.get_full_divs(1)
            pref_distrib = self.rbf_smm.get_full_prefs(1)
            vmax_qs, cm = None, None
            #vmax_qs, cm = plot_discriminator(self.device, self.discriminator, self.logger,  step=0) #plotting stuf
        #plot_state([div_distrib], self.logger, info='Exploration-',goals=[[200/255.0 ,200/255.0]], step=0)
        #plot_state([self.div_distrib], self.logger, info='Focus-',goals=[[200/255.0 ,200/255.0]], step=0) #plotting stuff
        self.div_distrib = self.rbf_smm.get_full_divs(1)
        self.discover(-1)
        self.learning()
        isExist = os.path.exists(self.logger_cfg['log_dir']+'model/')
        if not isExist:
                os.makedirs(self.logger_cfg['log_dir']+'model/')
        self.agent.save(self.logger_cfg['log_dir']+'model',-1)

        if self.env_cfg['type'] == 'MJ':
            #plot_skill_speed( self.agent, self.env, self.logger, self.discriminator.codebook_size,  self.training_cfg['Learning'], self.discriminator, self.device, timestep=-100)
            run_test_videos(self.agent, self.cfg['Environment'], self.logger, self.discriminator.codebook_size, self.training_cfg, info="skill")
        


    def evaluate(self):
        self.agent.load(self.logger_cfg['log_dir']+'model',-1)
        #run_test_videos(self.agent, self.cfg['Environment'], self.logger, self.discriminator.codebook_size, self.training_cfg, info="skillT")
        self.env = get_env(self.env_cfg, eval=True)
        plot_skill_pos( self.agent, self.env, self.logger, self.discriminator.codebook_size,  self.training_cfg['Learning'], self.discriminator, self.device, timestep=-100,infop='pos')
