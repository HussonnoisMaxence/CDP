import warnings
import gym
import torch
import numpy as np
import random
import yaml
from envs.mujoco.gym_mujoco.humanoid import HumanoidEnv


def save_config(path: str, config: dict):
    with open(path, 'w') as file:
        documents = yaml.dump(config, file)

def read_config(path: str):
    """read yaml file"""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    f.close()
    return data

from envs.mujoco.gym_mujoco.half_cheetah import HalfCheetahEnv
from envs.mujoco.gym_mujoco.ant import AntEnv
def get_env(env_cfg, eval=False):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="^.*Box bound precision lowered.*$"
        )
        if env_cfg['name'] == "Nav2d":
            cfg = env_cfg['config']
            env = gym.make('gym_nav2d:nav2dGoal-v0')
            env.set_goal(cfg['goal'])
            return env
        if env_cfg['name'] == "Nav2dArea":
            cfg = env_cfg['config']
            env = gym.make('gym_nav2d:nav2dArea-v0')
            env.set_goal(cfg['goal'])
            return env
        if env_cfg['name'] == 'Nav2d2OG':
            cfg = env_cfg['config']
            env = gym.make('gym_nav2d:nav2d2g-v0')
            #env.set_goal(cfg['goal'])
            return env
        if env_cfg['name'] == 'Nav2dSDF':
            cfg = env_cfg['config']
            env = gym.make('gym_nav2d:nav2dsdf-v0')
            #env.set_goal(cfg['goal'])
            return env
        if env_cfg['name'] == "Maze":
            cfg = env_cfg['config']
            return Maze(n=cfg['n'], maze_type=cfg['maze_type'], goal=cfg['goal'], done_on_success=cfg['done_on_success'])
        if env_cfg['name'] == "MazeTree":
            cfg = env_cfg['config']
            return Maze2(n=cfg['n'], maze_type=cfg['maze_type'], goal=cfg['goal'], done_on_success=cfg['done_on_success'])
        if env_cfg['name'] == "HC":
            if eval:
                return HalfCheetahEnv(task=env_cfg['config']['Task'], model_path=env_cfg['config']['eval_path'])
            else:
                return HalfCheetahEnv(task=env_cfg['config']['Task'], model_path=env_cfg['config']['model_path'])

        if env_cfg['name'] == "Ant":
            if eval:
                return AntEnv(task=env_cfg['config']['Task'], model_path=env_cfg['config']['eval_path'])
            else:
                return AntEnv(task=env_cfg['config']['Task'], model_path=env_cfg['config']['model_path'])
        if env_cfg['name'] =='Humanoid':
            return HumanoidEnv(task=env_cfg['config']['Task'])
def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

