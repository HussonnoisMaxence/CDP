from builtins import int, list, print, range, str
import argparse, torch, os
from statistics import mean
from time import time
from utils.oracles import plot_agent
from agent.sac import SACAgent
from utils.networks import eval_mode
from agent.rewardmodel import RewardModel
from agent.VQVAE import VQVAEDiscriminator
from agent.replaybuffer import ReplayBuffer, ReplayBufferZ
from agent.smm import SMMAgent
from utils.oracles import compute_target_rewards, get_distribution, get_kl_div,  get_preferred_region, get_target_state, sample_target_region
from utils.plot import run_test_videos
from utils.utils import get_env, set_seed_everywhere, read_config
import numpy as np
from collections import deque
from utils.oracles import plot_state, plot_discriminator, plot_skill

from utils.logger import Logger

torch.backends.cudnn.benchmark = False

class Workspace(object):
    def __init__(self, cfg):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #Get configs
        self.cfg = cfg
        
        self.training_cfg = self.cfg['Training']
        self.env_cfg = self.cfg['Environment']
        self.learning_cfg = self.training_cfg['Learning']
        
        self.sac_cfg = self.cfg['SAC']
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
        ## Set Seeds
        seed = self.training_cfg['seed']
        set_seed_everywhere(seed)
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        print(self.env)
        print(a)
        ## Instantiating Logger
        self.logger = Logger(
            log_dir=self.logger_cfg['log_dir'],
            file_name=self.logger_cfg['file_name'],
            agent=self.logger_cfg['agent'],
            save_tb=self.logger_cfg['save_tb'],
            log_frequency=self.logger_cfg['log_frequency']
            #,config=self.cfg
        )
       
        ## Instantiating Agent
        self.rbf = ReplayBuffer(self.env.observation_space.shape,  
                                self.env.action_space.shape, self.sac_cfg['capacity'], self.device, window=1)
        
        self.agent = SACAgent (
                    obs_dim=self.observation_shape, 
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

    def learning (self):
        
        self.timestep = 0
        while self.timestep < self.learning_cfg['total_timesteps']:
            #Initiate the episode
            if self.training_cfg['prior']:
                obs, prior = self.env.reset()

            else:
                obs = self.env.reset()
                prior = obs
            done = False
            done_no_max = False
            episode_reward = 0
            episode_step = 0
            
            while not(done):
                # sample action for data collection
                if self.timestep <  self.learning_cfg['learning_start']:
                    action = self.env.action_space.sample()
                else:
                    with eval_mode(self.agent):
                        action = self.agent.act(obs, sample=True)
                
                next_obs, reward, done, next_prior = self.env.step(action)

                done = True if episode_step == self.step_max-1 else done
                done_no_max = done
                
                
                #Add to replay buffer
                self.rbf.add(
                    obs, action, reward, 
                   next_obs, done, done_no_max)

                #Off Policy learning
                if self.timestep >  self.learning_cfg['learning_start']:
                    self.agent.update(self.rbf, self.timestep, self.logger)


                obs = next_obs
                prior = next_prior
                episode_reward += reward
                self.timestep +=1
                episode_step +=1

            print('Learning:', self.timestep, 'Reward:', episode_reward)

    def run(self):

        self.learning()
        plot_agent( self.agent, self.env, self.logger,  self.training_cfg['Learning'],timestep=-100)