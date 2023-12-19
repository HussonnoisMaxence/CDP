import warnings
import gym
import torch
import numpy as np
import random
import yaml


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
        if env_cfg['name'] == "Nav2dArea":
            cfg = env_cfg['config']
            env = gym.make('gym_nav2d:nav2dArea-v0')
            env.set_goal(cfg['goal'])
            return env

        if env_cfg['name'] == "HC":
            cfg = env_cfg['config']
            if eval:
                return HalfCheetahEnv(task= cfg['task'], area=cfg['goal'], model_path= cfg['eval_path'])
            else:
                env = HalfCheetahEnv(task= cfg['task'], area=cfg['goal'], model_path= cfg['model_path'])
                return env

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def my_floor(a, precision=0):
    return np.round(a - (0.5 * 10**(-precision)), precision)

def in_area(a, area):
    if len(area)==4:
        if a[0] >= area[0] and a[0] <= area[2] and a[1] >= area[1] and a[1] <= area[3]:
            return True
        return False
    elif len(area)==2:
        if a >= area[1] and a <= area[0]:
            return True
        return False        


def compute_res_bins(coverage_preferred, coverage, states, states_preferred, area_x,  area_x2,  area_y,  area_y2, precision):
   

    res = ((len(coverage) - len(coverage_preferred))/len(coverage))*100
    res2 = ((len(states) - len(states_preferred))/len(states))*100
    #nbr_bins = (2) * np.power(10,precision)*(2) * np.power(10,precision)
    nbr_bins_preferred = (area_x2 - area_x) * np.power(10,precision)*(area_y2 - area_y) * np.power(10,precision)
    #(len(coverage)/nbr_bins)*100, 
    return (len(coverage_preferred)/nbr_bins_preferred)*100, res, res2

def get_preferred_regions(obses, target_area):
    return np.array([el for el in obses if in_area(el, area=target_area)])

def count_state_coverage(states, target_area, precision=2):
    y = (target_area[3] - target_area[1])
    x = (target_area[2] - target_area[0])
    nbr_bins_preferred= x* np.power(10,precision)*y* np.power(10,precision) 


    states = np.array(states)
    states_preferred = np.array([el for el in states if in_area(el, area=target_area) ])
    
    states_bins_preferred = np.array(my_floor(states_preferred, precision=precision))
    unique_states_bins_preferred, counts = np.unique(states_bins_preferred, return_counts=True, axis=0)

    per_preferred_coverage = (len(unique_states_bins_preferred)/nbr_bins_preferred)*100

    states_bins = my_floor(states, precision=precision)
    unique_states_bins, counts = np.unique(states_bins, return_counts=True, axis=0)

    per_residual_bins  =  (1 - (len(unique_states_bins_preferred)/len(unique_states_bins)))*100


    states_preferred = np.array([el for el in states if in_area(el, area=target_area) ])
    per_residual_dense =  (1 - (len(states_preferred)/len(states)))*100
    '''
    states_floor_preferred = np.array(my_floor(states_preferred, precision=precision))
    states_bins = my_floor(states, precision=precision)
    unique_states_bins, counts = np.unique(states_bins, return_counts=True,axis=0)
    unique_states_bins_preferred = np.array([el for el in unique_states_bins if in_area(el, area=target_area)])

    coverage_preferred, counts = np.unique(states_floor_preferred, return_counts=True, axis=0)
    
    nbr_bins_preferred=(target_area[2] - target_area[0]) * np.power(10,precision)*(target_area[3] - target_area[1]) * np.power(10,precision)
    per_preferred_coverage = (len(coverage_preferred)/nbr_bins_preferred)*100


    res = ((len(coverage) - len(coverage_preferred))/len(coverage))*100
    res2 = ((len(states) - len(states_preferred))/len(states))*100
    #nbr_bins = (2) * np.power(10,precision)*(2) * np.power(10,precision)
    
    #(len(coverage)/nbr_bins)*100, 

    '''
    return per_preferred_coverage , per_residual_bins, per_residual_dense

def evaluate_beta(rm, target_area, precision=2):

    all_states = utils.get_pairs(-1, 1, 0.01) 
    oracle_pref_region = np.array([el for el in all_states if in_area(el, area=target_area) ])


    values = rm.r_hat_batch(oracle_pref_region)
    ma_v = (np.max(values)+1)/2
    mi_v = (np.min(values)+1)/2
    e_norm = ma_v-mi_v
    mean = np.mean(values)
    std = np.std(values)
    
    return e_norm , mean, std


def get_f1_bins(states, target_area, precision_bins=2):

    states = np.array(states)
    states_bins = np.unique(np.array(my_floor(states, precision=precision_bins)),axis=0)

    states_preferred = np.array([el for el in states if in_area(el, area=target_area) ])
    states_preferred_bins = np.unique(np.array(my_floor(states_preferred, precision=precision_bins)),axis=0)
    
    precision = len(states_preferred)/ len(states)
    
    all_states = utils.get_pairs(-1, 1, 10**(-precision_bins)) 
    oracle_pref_region = np.array([el for el in all_states if in_area(el, area=target_area) ])
    recall =  len(states_preferred_bins)/ len(oracle_pref_region)
    if precision == 0:
        f1_score = 0
    else: 
        f1_score = 2*(precision*recall)/(precision+recall)
    
    return f1_score , precision, recall

def get_f1_bins_hc(states, target_area, precision_bins=2):

    states = np.array(states)
    states_bins = np.unique(np.array(my_floor(states, precision=precision_bins)),axis=0)

    states_preferred = np.array([el for el in states if in_area(el, area=target_area) ])
    states_preferred_bins = np.unique(np.array(my_floor(states_preferred, precision=precision_bins)), axis=0)
    
    precision = len(states_preferred_bins)/ len(states_bins)
    all_states = np.arange(-8, -2 + 10**(-precision_bins) , 10**(-precision_bins))
    oracle_pref_region = np.array([el for el in all_states if in_area(el, area=target_area) ])
    recall =  len(states_preferred_bins)/ len(oracle_pref_region)
    if precision == 0:
        f1_score = 0
    else: 
        f1_score = 2*(precision*recall)/(precision+recall)
    
    return f1_score , precision, recall

def get_f1(states, target_area, precision=2):

    states = np.array(states)
    states_preferred = np.array([el for el in states if in_area(el, area=target_area) ])
    
    precision = len(states_preferred)/ len(states)
    
    all_states = utils.get_pairs(-1, 1, 0.01) 
    oracle_pref_region = np.array([el for el in all_states if in_area(el, area=target_area) ])
    recall =  len(states_preferred)/ len(oracle_pref_region)
    if precision == 0:
        f1_score = 0
    else: 
        f1_score = 2*(precision*recall)/(precision+recall)
    

    print(f1_score , precision, recall)

    return f1_score , precision, recall


def count_state_coverage_hc(states, target_area, precision=2):
    states = np.array(states)
    states_floor = my_floor(states, precision=precision)
    coverage, counts = np.unique(states_floor, return_counts=True,axis=0)
    
    states_preferred = np.array([el for el in states if in_area(el, area=target_area) ])
    states_floor_preferred = np.array(my_floor(states_preferred, precision=precision))

    coverage_preferred, counts = np.unique(states_floor_preferred, return_counts=True, axis=0)
    #(len(coverage)/nbr_bins)*100, 
    #(len(coverage)/nbr_bins)*100, 
    res = ((len(coverage) - len(coverage_preferred))/len(coverage))*100
    res2 = ((len(states) - len(states_preferred))/len(states))*100
    #nbr_bins = (2) * np.power(10,precision)*(2) * np.power(10,precision)
    nbr_bins_preferred=-(target_area[1] - target_area[0]) * np.power(10,precision) #*(area_y2 - area_y) * np.power(10,precision)
    #(len(coverage)/nbr_bins)*100, 
    return (len(coverage_preferred)/nbr_bins_preferred)*100, res, res2



def count_visited_bins(positions, area, step=0.01):
    x_min, y_min, x_max, y_max = area[0], area[1], area[2], area[3]
    x_bins = int(np.ceil((x_max - x_min) / step)) + 1
    y_bins = int(np.ceil((y_max - y_min) / step)) + 1
    total_bins = x_bins * y_bins

    positions = np.array(positions)
    x_indices = ((positions[:, 0] - x_min) / step).astype(int)
    y_indices = ((positions[:, 1] - y_min) / step).astype(int)

    valid_indices = np.where((0 <= positions[:, 0]) & (positions[:, 0] <= 1) & (0 <= positions[:, 1]) & (positions[:, 1] <= 1))
    visited_bins = np.unique((x_indices[valid_indices], y_indices[valid_indices]), axis=1).shape[1]

    percentage = (visited_bins / total_bins) * 100

    return percentage



def get_pairs(range_start, range_end, step):
    x_values = np.arange(range_start, range_end +step , step)
    pairs = np.transpose([np.tile(x_values, len(x_values)), np.repeat(x_values, len(x_values))])
    return pairs


import os
from datetime import datetime
def create_run_name(cfg):
    date = datetime.now().strftime("%d:%m")
    date2 = datetime.now().strftime("%H:%M")
    return f"{cfg.experiment_path}/{date}/{cfg.jobid}/{cfg.seed}"


def create_exp_dir(cfg):
    dir_name=cfg.dir
    if cfg.job_id is None:
        jobid=str(os.system("squeue -u $USER | tail -1| awk '{print $1}'"))
        dt = datetime.now().strftime('%Y-%m-%d') 
        dt2 = datetime.now().strftime('%H:%M:%S')                              
        jobid+= f'/{dt2}'
        dir_name += f'/{dt}/{jobid}'
    else:
        dt = datetime.now().strftime('%Y-%m-%d')
        dir_name += f'/{dt}/{cfg.job_id}'
    dir_name+=f'/{cfg.seed}'
    return dir_name

def create_exp_wgname(cfg):
    dir_name=cfg.wname
    if cfg.job_id is None:
        jobid=str(os.system("squeue -u $USER | tail -1| awk '{print $1}'"))
        dt = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        jobid+= f'_{dt}'
        dir_name += f'_{jobid}'
        print(dir_name)
    else:
        dir_name += f'_{cfg.job_id}'
    return dir_name


def create_dir(path):
    isdir = os.path.isdir(path)
    if isdir:
        ...
        #print(f"{path} allready exist")
    else:
        os.makedirs(path)


import utils.utils as utils
def experiments_settings(cfg):
    if cfg.wmode =='disabled':
        cfg.dir = utils.create_exp_dir(cfg)
        utils.create_directories(cfg.dir)

    else:
        cfg.dir = utils.create_exp_dir(cfg)
        utils.create_directories(cfg.dir)

    cfg.wgname = utils.create_exp_wgname(cfg)
    return cfg

def create_directories(path):
    create_dir(path)


def undense(states, precision=2):
    states_bins, index = np.unique(np.array(my_floor(states, precision=precision)),return_index=True,  axis=0)
    return index
