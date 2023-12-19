import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from agent.critic import DoubleQCritic
from agent.actor import DiagGaussianActor
from utils.networks import  weight_init, to_np, soft_update_params
'''
SMM implementation adapted from from https://github.com/rll-research/BPref
'''

class VAE(nn.Module):
    def __init__(self, obs_dim, z_dim, hidden_dim, code_dim, vae_beta, device):
        super().__init__()
        self.z_dim = z_dim
        self.code_dim = code_dim

        self.make_networks(obs_dim, z_dim, hidden_dim, code_dim)
        self.beta = vae_beta

        self.apply(weight_init)
        self.device = device

    def make_networks(self, obs_dim, z_dim,hidden_dim,  code_dim):
        self.enc = nn.Sequential(nn.Linear(obs_dim + z_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.enc_mu = nn.Linear(hidden_dim, code_dim)

        self.enc_logvar = nn.Linear(hidden_dim, code_dim)

        self.dec = nn.Sequential(nn.Linear(code_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, obs_dim + z_dim))

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
    def __init__(self, obs_dim, z_dim, hidden_dim, code_dim,  vae_beta, device):
        super().__init__()
        self.z_dim = z_dim
        self.z_pred_net = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, z_dim))
        

        self.vae = VAE(obs_dim=obs_dim,
                       z_dim=z_dim,
                       hidden_dim=hidden_dim,
                       code_dim=code_dim,
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
                code_dim_smm,
                vae_beta,
                sp_lr,
                vae_lr,
                prior_dim=None,
                scale_factors=[1,1,1],
                beta_smm=0.5,
):
        super().__init__()
        self.obs_dim = obs_dim
        self.skill_dim = skill_dim
        self.prior_dim = prior_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.hidden_depth = hidden_depth
        self.scale_factors = scale_factors
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
        #Initiate Critics
        self.critic = DoubleQCritic(obs_dim+self.skill_dim, action_dim, hidden_dim, hidden_depth).to(self.device)
        self.critic_target = DoubleQCritic(obs_dim+self.skill_dim, action_dim, hidden_dim, hidden_depth).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        #Initiate Actors
        self.actor = DiagGaussianActor(obs_dim+self.skill_dim, action_dim, hidden_dim, hidden_depth,
                 log_std_bounds).to(self.device)

        self.smm_dim = self.prior_dim
        ## Initiate the SMM
        self.sp_lr = sp_lr
        self.vae_lr = vae_lr
        self.vae_beta = vae_beta
        self.hidden_dim_smm=hidden_dim_smm
        self.beta_smm = beta_smm
        self.smm = SMM(
                    self.smm_dim,
                    skill_dim,
                    code_dim=code_dim_smm,
                    hidden_dim=hidden_dim_smm,
                    vae_beta=vae_beta,
                    device=device).to(device)
        
        self.pred_optimizer = torch.optim.Adam( self.smm.z_pred_net.parameters(), lr=sp_lr)
        self.vae_optimizer = torch.optim.Adam(self.smm.vae.parameters(), lr=vae_lr)
       
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
                      not_done, metrics, prefix):
        
        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()

            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1,
                                target_Q2) - self.alpha * log_prob
            target_Q = reward + (not_done * self.discount * target_V)
            target_Q = target_Q

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        


        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_value_(self.critic.parameters(), 100)
        self.critic_optimizer.step()
        metrics.update({f"{prefix}/critic_loss": critic_loss})
        return metrics                    
    
    def update_actor_and_alpha(self, obs, metrics, prefix):
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

            metrics.update({f"{prefix}/alpha_loss": alpha_loss })

        metrics.update({
            f"{prefix}/actor_loss": actor_loss , 
            f"{prefix}/alpha":self.alpha, 
        })
        return metrics
    
    def update_vae(self, obs_z, metrics, prefix):
        loss, h_s_z = self.smm.vae.loss(obs_z)
        self.vae_optimizer.zero_grad()
        loss.backward()
        self.vae_optimizer.step()
        metrics.update({f"{prefix}/loss_vae":loss})
        return metrics, h_s_z.detach()

    def update_pred(self, obs, z, metrics, prefix):
        logits = self.smm.predict_logits(obs)
        h_z_s = self.smm.loss(logits, z).unsqueeze(-1)
        loss = h_z_s.mean()
        self.pred_optimizer.zero_grad()
        loss.backward()
        self.pred_optimizer.step()
        metrics.update({f"{prefix}/loss_predict":loss})

        return metrics, h_z_s.detach()
    


    def update(self, timestep, replay_buffer, info, vqvae=None, gradient_update=1):
        metrics_update = {}
        start_time_init0 = time.time()
        
        if timestep % self.train_freq == 0:
            for index in range(gradient_update):
                start_time_init = time.time()
                batch = replay_buffer.sample(self.batch_size)
                obs_z = torch.cat([batch['obs'], batch['skill']], dim=1)
                next_obs_z = torch.cat([batch['next_obs'], batch['skill']], dim=1)
                
                obs_z_vae =  torch.cat([batch['divs'], batch['skill']], dim=1)
                next_obs = batch['divs']

                start_time_init = time.time()
                metrics_update, h_s_z = self.update_vae(obs_z_vae, metrics_update, prefix=info['prefix'])
               
                #SMM BASIC
                if info['objective']=='smm': #SMM BASIC
                    #compute reward exploration
                    scaling =  2
                    #h_s_z = norm(x=h_s_z, min=-0.75, max=4)
                    reward_exploration = h_s_z

                    #compute reward diversity
                    h_z = np.log(self.skill_dim)  # One-hot z encoding            
                    h_z *= torch.ones_like(batch['reward']).to(self.device)
                    metrics_update, h_z_s = self.update_pred(next_obs, batch['skill'], metrics_update, prefix=info['prefix'])
                    
                    reward_diversity = h_z + h_z_s
                    #reward_diversity = norm(x=reward_diversity, min=2, max=6)
                    reward = reward_diversity +reward_exploration
                    reward = reward/scaling

                if info['objective']=='smm_prior': #SMM BASIC
                    #compute reward exploration
                    scaling =  sum(self.scale_factors)
                    reward_exploration = self.scale_factors[0]*h_s_z + self.scale_factors[2]*batch['reward']

                    #compute reward diversity
                    h_z = np.log(self.skill_dim)  # One-hot z encoding            
                    h_z *= torch.ones_like(batch['reward']).to(self.device)

                    metrics_update, h_z_s = self.update_pred(next_obs, batch['skill'], metrics_update, info['prefix'])

                    reward_diversity = h_z + h_z_s
                    
                    reward = self.scale_factors[1]*reward_diversity + reward_exploration           
                    reward = reward/scaling    
                    

                    metrics_update.update({
                        f"{info['prefix']}/reward_pref": batch['reward'].mean(),
                    })

                if info['objective']=='cdp': 
                    #compute reward exploration
                    scaling = sum(self.scale_factors)
                    #h_s_z = norm(x=h_s_z, min=-0.75, max=4)
                    reward_exploration = self.scale_factors[0]*h_s_z
                    if info['use_reward']:
                        reward_exploration = reward_exploration + self.scale_factors[2]*batch['reward']

                    #compute reward diversity
                    z_hat = torch.argmax(batch['skill'], dim=1)
                    h_z_s = vqvae.compute_logprob_under_latent(dict(
                                        next_state=next_obs,
                                        skill=z_hat)).view(-1,1).detach()
                    #h_z_s = norm(x=h_z_s, min=-6, max=0)
                    reward_diversity = self.scale_factors[1]*h_z_s
                    reward = reward_diversity + reward_exploration  
                    reward = reward/scaling
                   
                    metrics_update.update({
                        f"{info['prefix']}/reward_pref": batch['reward'].mean(),
                        f"{info['prefix']}/reward_pref_max": batch['reward'].max(),
                        f"{info['prefix']}/reward_pref_min": batch['reward'].min(),
                    })
                
                metrics_update.update({
                        f"{info['prefix']}/reward": reward.mean(),
                        f"{info['prefix']}/reward_min": reward.min(),
                        f"{info['prefix']}/reward_max": reward.max(),
                        f"{info['prefix']}/r_exploration": h_s_z.mean(),
                        f"{info['prefix']}/r_exploration_min": h_s_z.min(),
                        f"{info['prefix']}/r_exploration_max": h_s_z.max(),
                        f"{info['prefix']}/r_div": reward_diversity.mean(),
                        f"{info['prefix']}/r_div_min": reward_diversity.min(),
                        f"{info['prefix']}/r_div_max": reward_diversity.max()
                    })
            
              
                metrics_update = self.update_critic(
                    obs_z, batch['action'], reward, next_obs_z, batch['not_done_no_max'], metrics_update, prefix=info['prefix'])

                if timestep % self.actor_update_frequency == 0:
                    metrics_update = self.update_actor_and_alpha(obs_z, metrics_update, prefix=info['prefix'])

            if timestep % self.critic_target_update_frequency == 0:
                soft_update_params(self.critic, self.critic_target,
                                        self.critic_tau)

            return metrics_update

    def update_after_reset(self, replay_buffer, info, vqvae=None, gradient_update=1, policy_update=True):
        metrics_update = {}
        for index in range(gradient_update):

                batch = replay_buffer.sample(self.batch_size)

                obs_z = torch.cat([batch['obs'], batch['skill']], dim=1)
                next_obs_z = torch.cat([batch['next_obs'], batch['skill']], dim=1)
                
                obs_z_vae =  torch.cat([batch['divs'], batch['skill']], dim=1)
                next_obs = batch['divs']
                metrics_update, h_s_z = self.update_vae(obs_z_vae, metrics_update, info['prefix'])
                #SMM BASIC
                if info['objective']=='smm': #SMM BASIC
                    #compute reward exploration
                    reward_exploration = h_s_z

                    #compute reward diversity
                    h_z = np.log(self.skill_dim)  # One-hot z encoding            
                    h_z *= torch.ones_like(batch['reward']).to(self.device)

                    metrics_update, h_z_s = self.update_pred(next_obs, batch['skill'], metrics_update, info['prefix'])

                    reward_diversity = h_z + h_z_s
                    
                    reward = reward_diversity + reward_exploration

                    metrics_update.update({
                        'SMM/reward': reward.mean(),
                        'SMM/r_exploration': h_s_z.mean(),
                        'SMM/r_div': reward_diversity.mean()
                    })

                if info['objective']=='smm_prior': #SMM BASIC
                    #compute reward exploration
                    scaling =  sum(self.scale_factors)
                    reward_exploration = self.scale_factors[0]*h_s_z + self.scale_factors[2]*batch['reward']

                    #compute reward diversity
                    h_z = np.log(self.skill_dim)  # One-hot z encoding            
                    h_z *= torch.ones_like(batch['reward']).to(self.device)

                    metrics_update, h_z_s = self.update_pred(next_obs, batch['skill'], metrics_update, info['prefix'])

                    reward_diversity = h_z + h_z_s
                    
                    reward = self.scale_factors[1]*reward_diversity + reward_exploration           
                    reward = reward/scaling
                    metrics_update.update({
                        'SMM/reward': reward.mean(),
                        'SMM/reward_pref': batch['reward'].mean(),
                        'SMM/r_exploration': h_s_z.mean(),
                        'SMM/r_div': reward_diversity.mean()
                    })

                if info['objective']=='cdp': 
                    scaling =  sum(self.scale_factors)
                    #compute reward exploration
                    #h_s_z = norm(x=h_s_z, min=-0.75, max=4)
                    reward_exploration =  self.scale_factors[0]*h_s_z
                    if info['use_reward']:
                        reward_exploration = reward_exploration + self.scale_factors[2]*batch['reward'].detach()

                    #compute reward diversity
                    z_hat = torch.argmax(batch['skill'], dim=1)
                    h_z_s = vqvae.compute_logprob_under_latent(dict(
                                        next_state=next_obs,
                                        skill=z_hat)).view(-1,1).detach()
                    
                    reward_diversity = self.scale_factors[1]*h_z_s
                    #h_z_s = norm(x=h_z_s, min=-6, max=0)
                    reward = reward_exploration + reward_diversity 
                    reward = reward/scaling

                metrics_update.update({
                        'SMM/reward': reward.mean(),
                        'SMM/r_exploration': h_s_z.mean(),
                        'SMM/r_div': reward_diversity.mean()
                    })

                
                metrics_update = self.update_critic(obs_z, batch['action'], reward, next_obs_z, batch['not_done_no_max'],
                                            metrics_update, info['prefix'])

                if index % self.actor_update_frequency == 0 and policy_update:
                    metrics_update = self.update_actor_and_alpha(obs_z, metrics_update, info['prefix'])

                if index % self.critic_target_update_frequency == 0:
                    soft_update_params(self.critic, self.critic_target,
                                                    self.critic_tau)
                
        return metrics_update

def norm(x, min, max):
    return (x-min)/(max-min)