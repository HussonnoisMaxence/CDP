import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math



from agent.critic import DoubleQCritic
from agent.actor import DiagGaussianActor
from agent.sac import compute_state_entropy
from utils.networks import mlp,  weight_init, TorchRunningMeanStd, to_np, soft_update_params

class VAE(nn.Module):
    def __init__(self, obs_dim, z_dim, code_dim, vae_beta, device):
        super().__init__()
        self.z_dim = z_dim
        self.code_dim = code_dim

        self.make_networks(obs_dim, z_dim, code_dim)
        self.beta = vae_beta

        self.apply(weight_init)
        self.device = device

    def make_networks(self, obs_dim, z_dim, code_dim):
        self.enc = nn.Sequential(nn.Linear(obs_dim + z_dim, 150), nn.ReLU(),
                                 nn.Linear(150, 150), nn.ReLU())
        self.enc_mu = nn.Linear(150, code_dim)
        self.enc_logvar = nn.Linear(150, code_dim)
        self.dec = nn.Sequential(nn.Linear(code_dim, 150), nn.ReLU(),
                                 nn.Linear(150, 150), nn.ReLU(),
                                 nn.Linear(150, obs_dim + z_dim))

    def encode(self, obs_z):
        enc_features = self.enc(obs_z)
        mu = self.enc_mu(enc_features)
        logvar = self.enc_logvar(enc_features)
        stds = (0.5 * logvar).exp()
        return mu, logvar, stds

    def forward(self, obs_z, epsilon):
        mu, logvar, stds = self.encode(obs_z)
        code = epsilon * stds + mu
        obs_distr_params = self.dec(code)
        return obs_distr_params, (mu, logvar, stds)

    def loss(self, obs_z):
        epsilon = torch.randn([obs_z.shape[0], self.code_dim]).to(self.device)
        obs_distr_params, (mu, logvar, stds) = self(obs_z, epsilon)
        kle = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),
                               dim=1).mean()
        log_prob = F.mse_loss(obs_z, obs_distr_params, reduction='none')

        loss = self.beta * kle + log_prob.mean()
        return loss, log_prob.sum(list(range(1, len(log_prob.shape)))).view(
            log_prob.shape[0], 1)

class SMM(nn.Module):
    def __init__(self, obs_dim, z_dim, hidden_dim, vae_beta, device):
        super().__init__()
        self.z_dim = z_dim
        self.z_pred_net = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, z_dim))
        self.vae = VAE(obs_dim=obs_dim,
                       z_dim=z_dim,
                       code_dim=128,
                       vae_beta=vae_beta,
                       device=device)
        self.apply(weight_init)

    def predict_logits(self, obs):
        z_pred_logits = self.z_pred_net(obs)
        return z_pred_logits

    def loss(self, logits, z):
        z_labels = torch.argmax(z, 1)
        return nn.CrossEntropyLoss(reduction='none')(logits, z_labels)


class SMMAgent():
    """SAC algorithm."""
    def __init__(self, 
                obs_dim, 
                action_dim, 
                action_range, 
                device, 
                hidden_dim,
                hidden_depth,
                log_std_bounds,
                discount, 
                init_temperature, 
                alpha_lr, 
                actor_lr, 
                actor_update_frequency, 
                critic_lr,
                critic_tau, 
                critic_target_update_frequency,
                batch_size, 
                train_freq,
                learnable_temperature,
                skill_dim,
                hidden_dim_smm,
                vae_beta,
                sp_lr,
                vae_lr,
                prior_dim=None,
                scale_factors=[1,1,1]
):
        super().__init__()
        self.obs_dim = obs_dim
        self.skill_dim = skill_dim
        self.prior_dim = prior_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.hidden_depth = hidden_depth
        self.exploration=True
        self.ps_scale = scale_factors[0]
        self.exp_scale = scale_factors[1]
        self.div_scale = scale_factors[2]
        self.action_range = action_range
        self.log_std_bounds=log_std_bounds
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature
        self.critic_lr = critic_lr
        self.init_temperature = init_temperature
        self.alpha_lr = alpha_lr
        self.s_ent_stats = TorchRunningMeanStd(shape=[1], device=device)
        #Initiate Critics
        self.critic = DoubleQCritic(obs_dim+self.skill_dim, action_dim, hidden_dim, hidden_depth).to(self.device)
        self.critic_target = DoubleQCritic(obs_dim+self.skill_dim, action_dim, hidden_dim, hidden_depth).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        #Initiate Actors
        self.actor = DiagGaussianActor(obs_dim+self.skill_dim, action_dim, hidden_dim, hidden_depth,
                 log_std_bounds).to(self.device)

        self.smm_dim = self.prior_dim
        ## Initiate the SMM
        self.hidden_dim_smm = hidden_dim_smm
        self.sp_lr = sp_lr
        self.vae_lr=vae_lr
        self.vae_beta= vae_beta
        self.smm = SMM(
                    self.smm_dim,
                    skill_dim,
                    hidden_dim=hidden_dim_smm,
                    vae_beta=vae_beta,
                    device=device).to(device)
        self.pred_optimizer = torch.optim.Adam(
            self.smm.z_pred_net.parameters(), lr=sp_lr)
        self.vae_optimizer = torch.optim.Adam(self.smm.vae.parameters(),
                                              lr=vae_lr)
        self.latent_cond_ent_coef = ...#latent_cond_ent_coef
        self.train_freq = train_freq
        #Alpha
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr)
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=alpha_lr)

        # change mode
        self.train()
        self.critic_target.train()

        self.smm.train()

        # fine tuning SMM agent
        self.ft_returns = np.zeros(skill_dim, dtype=np.float32)
        self.ft_not_finished = [True for z in range(skill_dim)]

        self.min_pref = None
        self.max_pref = None
        self.commit = None
        self.pref_distrib = None
        
    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def reset_critic(self):
        self.critic = DoubleQCritic(self.obs_dim+self.skill_dim, self.action_dim, self.hidden_dim, self.hidden_depth).to(self.device)
        self.critic_target = DoubleQCritic(self.obs_dim+self.skill_dim, self.action_dim, self.hidden_dim, self.hidden_depth).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr)
    
    def reset_actor(self):
        # reset log_alpha
        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=self.alpha_lr)

    def reset_smm(self):
        self.smm = SMM(
                    self.smm_dim,
                    self.skill_dim,
                    hidden_dim=self.hidden_dim_smm,
                    vae_beta=self.vae_beta,
                    device=self.device).to(self.device)
        self.pred_optimizer = torch.optim.Adam(
            self.smm.z_pred_net.parameters(), lr=self.sp_lr)
        self.vae_optimizer = torch.optim.Adam(self.smm.vae.parameters(),
                                              lr=self.vae_lr)
    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic_target.state_dict(), '%s/critic_target_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.discriminator.state_dict(), '%s/discriminator_%s.pt' % (model_dir, step)
        )
        
    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        self.critic_target.load_state_dict(
            torch.load('%s/critic_target_%s.pt' % (model_dir, step))
        )
    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return to_np(action[0])


    def update_critic(self, obs, action, reward, next_obs, 
                      not_done, logger, timestep, print_flag=True):
        
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        


        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        logger.log('smm/critic_loss', critic_loss, timestep)

    
    def update_actor_and_alpha(self, obs, logger, timestep, print_flag=False):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()


        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        logger.log('smm/actor_loss', actor_loss, timestep)


    def update_vae(self, obs_z):
        metrics = dict()
        loss, h_s_z = self.smm.vae.loss(obs_z)
        self.vae_optimizer.zero_grad()

        loss.backward()
        self.vae_optimizer.step()


        #metrics['loss_vae'] = loss.cpu().item()

        return metrics, h_s_z

    def update_pred(self, obs, z):
        metrics = dict()
        logits = self.smm.predict_logits(obs)
        h_z_s = self.smm.loss(logits, z).unsqueeze(-1)
        loss = h_z_s.mean()
        self.pred_optimizer.zero_grad()
        loss.backward()
        self.pred_optimizer.step()

        #metrics['loss_pred'] = loss.cpu().item()

        return metrics, h_z_s
    def norm_reward(self, values):
        ma = torch.max(values)
        mi =  torch.min(values)
        return torch.tensor([(v-mi)/(ma-mi) for v in values])

    def update(self, replay_buffer, logger, timestep, target_reward=False, vqvae=None, mi='reverse', gradient_update=1):
        if timestep % self.train_freq == 0:
            for index in range(gradient_update):
                batch = replay_buffer.sample(self.batch_size)
                obs_z = torch.cat([batch['obs'],batch['skill']], dim=1)
                next_obs_z = torch.cat([batch['next_obs'],batch['skill']], dim=1)

               
                h_z = np.log(self.skill_dim)  # One-hot z encoding            
                h_z *= torch.ones_like(batch['reward']).to(self.device)

                next_obs = batch['divs']
                obs_z_vae =  torch.cat([batch['divs'], batch['skill']], dim=1)

                _, h_s_z = self.update_vae(obs_z_vae)

                h_s_z = h_s_z.detach()

                if vqvae==None:
                    _, h_z_s = self.update_pred(next_obs, batch['skill'])
                    reward =  h_s_z + h_z_s.detach() + h_z
                    if target_reward:
                        reward = reward + batch['reward']

                elif vqvae != None:   
                    z_hat = torch.argmax(batch['skill'], dim=1)

                    if mi=='forward':
                        h_z_s = vqvae.log_approx_posterior(dict(
                                            next_state=next_obs,
                                            skill=z_hat)).view(-1,1).detach()
                        h_z_s = h_z_s+ h_z

                    if mi=='reverse':
                        h_z_s = vqvae.compute_logprob_under_latent(dict(
                                        next_state=next_obs,
                                        skill=z_hat)).view(-1,1).detach()

                    reward = h_s_z + h_z_s
                    if target_reward:
                        reward = reward + batch['reward']


                    
                    #logger.log('smm/reward',torch.mean(reward), timestep)
                    #logger.log('smm/reward_pref',torch.mean(p_s), timestep)
                    #logger.log('smm/p',torch.mean(h_s_z), timestep)
                    #logger.log('smm/h_z_s',torch.mean(h_z_s), timestep)
                
                self.update_critic(obs_z, batch['action'], reward, next_obs_z, batch['not_done_no_max'],
                                                logger, timestep)

                if timestep % self.actor_update_frequency == 0:
                    self.update_actor_and_alpha(obs_z, logger, timestep)

            if timestep % self.critic_target_update_frequency == 0:
                soft_update_params(self.critic, self.critic_target,
                                        self.critic_tau)

    def update_after_reset(self, replay_buffer, logger, timestep, target_reward=False,vqvae=None, mi='reverse', gradient_update=1, policy_update=True):
        for index in range(gradient_update):
            batch = replay_buffer.sample(self.batch_size)
            
            obs_z = torch.cat([batch['obs'],batch['skill']], dim=1)
            next_obs_z = torch.cat([batch['next_obs'],batch['skill']], dim=1)


            h_z = np.log(self.skill_dim)  # One-hot z encoding            
            h_z *= torch.ones_like(batch['reward']).to(self.device)


            next_obs = batch['divs']
            obs_z_vae =  torch.cat([batch['divs'], batch['skill']], dim=1)
            _, h_s_z = self.update_vae(obs_z_vae)
            h_s_z = h_s_z.detach()


            if vqvae==None:
                _, h_z_s = self.update_pred(next_obs, batch['skill'])
                reward =  h_s_z.detach() + h_z_s.detach() + h_z
                if target_reward:
                    reward = reward + batch['reward']

            else:   
                z_hat = torch.argmax(batch['skill'], dim=1)
                if mi=='forward':
                    h_z_s = vqvae.log_approx_posterior(dict(
                                        next_state=next_obs,
                                        skill=z_hat)).view(-1,1).detach()
                    h_z_s = h_z_s+ h_z
                    
                if mi=='reverse':
                    h_z_s = vqvae.compute_logprob_under_latent(dict(
                                    next_state=next_obs,
                                    skill=z_hat)).view(-1,1).detach()
                    h_z_s = h_z_s #+ h_z
                reward = h_s_z + h_z_s
                if target_reward:
                    reward = reward + batch['reward']


                
            self.update_critic(obs_z , batch['action'], reward, next_obs_z , batch['not_done_no_max'], logger,
                               timestep)

            if index % self.actor_update_frequency == 0 and policy_update:
                self.update_actor_and_alpha(obs_z, logger, timestep)

            if index % self.critic_target_update_frequency == 0:
                soft_update_params(self.critic, self.critic_target,
                                         self.critic_tau)

