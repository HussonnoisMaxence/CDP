from importlib.resources import path

import numpy as np
import random, torch
from agent.replaybuffer import  ReplayBuffer


from utils.logger import Logger
from agent.RMS import RewardModelState
from utils.utils import  read_config, set_seed_everywhere, get_env
from utils.networks import eval_mode
from utils.plot_maze import plot_reward, plot_run
from agent.sac import SACAgent

def train_sac(config):
    ## Initialise technical tool
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## Get config
    sac_config = config['SAC']
    reward_model_config = config['Reward_Model']
    training_config = config['Training']
    logger_config = config['Logger']
    ## Initialise Environment
    env = get_env(config['Environment'])

    observation_shape = env.observation_space.shape[0]
    action_shape = env.action_space.shape[0]
    action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max())
        ]

    ## Set Seeds
    seed = training_config['seed']
    env.seed(seed)
    env.action_space.seed(seed)
    set_seed_everywhere(seed)


    ## Initialise Replay Buffer
    replay_buffer = ReplayBuffer(env.observation_space.shape, env.action_space.shape, sac_config['capacity'], device, window=1)

    ## Initialise Agent
    agent = SACAgent (
                obs_dim=observation_shape, 
                action_dim=action_shape, 
                action_range=action_range, 
                device=device, 
                hidden_dim = sac_config['policy_kwargs']['hidden_dim'], 
                hidden_depth = sac_config['policy_kwargs']['hidden_depth'], 
                log_std_bounds = sac_config['policy_kwargs']['log_std_bound'],
                discount=sac_config['gamma'], 
                init_temperature=sac_config['alpha'], 
                alpha_lr=sac_config['learning_rate'], 
                alpha_betas=sac_config['betas'],
                actor_lr=sac_config['learning_rate'], 
                actor_betas=sac_config['betas'], 
                actor_update_frequency=sac_config['actor_train_freq'], 
                critic_lr=sac_config['learning_rate'],
                critic_betas=sac_config['betas'], 
                critic_tau=sac_config['polyak_factor'], 
                critic_target_update_frequency=sac_config['critic_train_freq'],
                batch_size=sac_config['batch_size'], 
                learnable_temperature=sac_config['alpha_auto'],
                normalize_state_entropy=True)
    
    
    ## Initialise Reward Model
    reward_model = RewardModelState(
        observation_shape,
        ensemble_size=reward_model_config['ensemble_size'],
        size_segment=reward_model_config['segment_size'],
        lr=reward_model_config['learning_rate'],
        mb_size=reward_model_config['batch_size'],
        sampling_mode=reward_model_config['sampling_mode'],
        device=device 
        )

    ## Initialise Logger
    logger = Logger(
        log_dir=logger_config['log_dir'],
        file_name=logger_config['file_name'],
        tracking=logger_config['writer'],
        tracking_interval=logger_config['tracking_interval'],
        comments=logger_config['comments'],
        config=config
    )
    ## Training
    training(env, agent, reward_model, logger, training_config, replay_buffer)
    plot_run(agent, env, logger, training_config, timestep=-1)
    plot_reward(reward_model, logger,env, -1)
    #plot_reward(reward_model, logger, -1)


def training(env, agent, reward_model,  logger, training_config, replay_buffer):
    total_feedback = 0
    labeled_feedback = 0
    ## PRE-TRAINING
    timestep = 0
    step_max = training_config['step_max']
    while timestep < training_config['pt_total_timesteps']:
        #Initiate the episode
        obs = env.reset()
        done = False
        done_no_max = False
        episode_reward_hat = 0
        episode_reward = 0
        episode_step = 0

        while not(done):
            # sample action for data collection
            if timestep < training_config['learning_start']:
                action = env.action_space.sample()
            else:
                with eval_mode(agent):
                    action = agent.act(obs, sample=True)
            
            next_obs, reward, done, _ = env.step(action)
            done = True if episode_step + 1 == step_max else False
            #done_no_max = 0 if episode_step + 1 == env._max_episode_steps else done
            done_no_max = done
            # update policy
            if timestep > training_config['learning_start']:
                agent.update_state_ent(replay_buffer, timestep, logger)

            reward_hat = reward_model.r_hat(obs)    
            reward_model.add_data(obs, reward, done)
            #Add to replay buffer
            replay_buffer.add(obs, action, reward_hat, next_obs, done, done_no_max)


            obs = next_obs
            episode_reward_hat += reward_hat
            episode_reward += reward
            timestep +=1
            episode_step +=1
        print('Timestep:', timestep, 'Cumulative Reward:', episode_reward,'len:', episode_step)


        


    labeled_queries = reward_model.uniform_sampling()
    total_feedback += reward_model.mb_size
    labeled_feedback += labeled_queries

    for epoch in range(training_config['reward_update']):
        train_acc = reward_model.train_reward(logger, timestep)

        total_acc = np.mean(train_acc)
        if total_acc > 0.97:
            break;

    replay_buffer.relabel_with_predictor(reward_model)    


    agent.reset_critic()
        
    # update agent
    agent.update_after_reset(replay_buffer, logger, timestep, 
                    gradient_update=training_config['update_after_reset'], 
                    policy_update=True)

    ## TRAINING
    while timestep < training_config['total_timesteps']:
        #Initiate the episode
        obs = env.reset()
        done = False
        done_no_max = False
        episode_reward_hat = 0
        episode_reward = 0
        episode_step = 0

        while not(done):
            # sample action for data collection
            if timestep < training_config['learning_start']:
                action = env.action_space.sample()
            else:
                with eval_mode(agent):
                    action = agent.act(obs, sample=True)
            
            next_obs, reward, done, _ = env.step(action)
            done = True if episode_step + 1 == step_max else False
            #done_no_max = 0 if episode_step + 1 == env._max_episode_steps else done
            done_no_max = done
            #Compute reward from reward model
            reward_hat = reward_model.r_hat(obs)    
            #Add to Reward replay buffer
            reward_model.add_data(obs, reward, done)
            #Add to replay buffer
            replay_buffer.add(obs, action, reward_hat, next_obs, done, done_no_max)

            # Update Reward Model
            if timestep%(training_config['reward_update_timesteps']-1) == 0 and total_feedback < training_config['max_feedback']:
                labeled_queries = reward_model.sampling()
                total_feedback += reward_model.mb_size
                labeled_feedback += labeled_queries

                for epoch in range(training_config['reward_update']):
                    train_acc = reward_model.train_reward(logger, timestep)
                    total_acc = np.mean(train_acc)
                    if total_acc > 0.97:
                        break;
                print("Reward function is updated!! ACC: " + str(total_acc))

                replay_buffer.relabel_with_predictor(reward_model) 
            # update policy
            if timestep > training_config['learning_start']:
                agent.update(replay_buffer, timestep, logger)
                   
            obs = next_obs
            episode_reward_hat += reward_hat
            episode_reward += reward
            timestep +=1
            episode_step +=1
                       

        print('Timestep:', timestep, 'Cumulative Reward:', episode_reward,'len:', episode_step)



import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('-config')
    args = parser.parse_args()


    if not(args.config):
        print("***config not provided***")

    config = read_config(path='./Projects/Project1/config/maze/square/' + args.config) #'./Projects/pebclone/configs/HC/diayn.yaml')

    train_sac(config)


        