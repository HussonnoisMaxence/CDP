from builtins import max, object, print, zip
import numpy as np
import torch
import utils
'''
Replay-buffer adapted from from https://github.com/rll-research/BPref
'''

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device, window=1):
        self.capacity = capacity
        self.device = device
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        self.obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        self.data_name = ['obs', 'action', 'reward', 'next_obs', 'not_done', 'not_done_no_max' ]
        self.obses = np.empty((capacity, *self.obs_shape), dtype=self.obs_dtype)
        self.next_obses = np.empty((capacity, *self.obs_shape), dtype=self.obs_dtype)
        self.actions = np.empty((capacity, *self.action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.window = window

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
    
    def add_batch(self, obs, action, reward, next_obs, done, done_no_max):
        
        next_index = self.idx + self.window
        if next_index >= self.capacity:
            self.full = True
            maximum_index = self.capacity - self.idx
            np.copyto(self.obses[self.idx:self.capacity], obs[:maximum_index])
            np.copyto(self.actions[self.idx:self.capacity], action[:maximum_index])
            np.copyto(self.rewards[self.idx:self.capacity], reward[:maximum_index])
            np.copyto(self.next_obses[self.idx:self.capacity], next_obs[:maximum_index])
            np.copyto(self.not_dones[self.idx:self.capacity], done[:maximum_index] <= 0)
            np.copyto(self.not_dones_no_max[self.idx:self.capacity], done_no_max[:maximum_index] <= 0)
            remain = self.window - (maximum_index)
            if remain > 0:
                np.copyto(self.obses[0:remain], obs[maximum_index:])
                np.copyto(self.actions[0:remain], action[maximum_index:])
                np.copyto(self.rewards[0:remain], reward[maximum_index:])
                np.copyto(self.next_obses[0:remain], next_obs[maximum_index:])
                np.copyto(self.not_dones[0:remain], done[maximum_index:] <= 0)
                np.copyto(self.not_dones_no_max[0:remain], done_no_max[maximum_index:] <= 0)
            self.idx = remain
        else:
            np.copyto(self.obses[self.idx:next_index], obs)
            np.copyto(self.actions[self.idx:next_index], action)
            np.copyto(self.rewards[self.idx:next_index], reward)
            np.copyto(self.next_obses[self.idx:next_index], next_obs)
            np.copyto(self.not_dones[self.idx:next_index], done <= 0)
            np.copyto(self.not_dones_no_max[self.idx:next_index], done_no_max <= 0)
            self.idx = next_index
        
    def relabel_with_predictor(self, predictor):
        batch_size = 200
        total_iter = int(self.idx/batch_size)
        
        if self.idx > batch_size*total_iter:
            total_iter += 1
            
        for index in range(total_iter):
            last_index = (index+1)*batch_size
            if (index+1)*batch_size > self.idx:
                last_index = self.idx
                
            obses = self.obses[index*batch_size:last_index]
            inputs = np.concatenate([obses], axis=-1)
            
            pred_reward = predictor.r_hat_batch(inputs)
            self.rewards[index*batch_size:last_index] = pred_reward
            
    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        return dict(zip( self.data_name, [obses, actions, rewards, next_obses, not_dones, not_dones_no_max]))
    
    def sample_state_ent(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)
        
        if self.full:
            full_obs = self.obses
        else:
            full_obs = self.obses[: self.idx]
        full_obs = torch.as_tensor(full_obs, device=self.device)
        
        return obses, full_obs, actions, rewards, next_obses, not_dones, not_dones_no_max

    def get_full_obs(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        
        if self.full:
            full_obs = self.next_obses
        else:
            full_obs = self.next_obses[: self.idx]
        
        full_obs = torch.as_tensor(full_obs, device=self.device)
        full_skills = torch.as_tensor(full_skills, device=self.device)
        #full_obs = torch.cat([full_obs,full_skills], dim=1)
        return full_obs

    def reset(self):
        self.obses = np.empty((self.capacity, *self.obs_shape), dtype=self.obs_dtype)
        self.next_obses = np.empty((self.capacity, *self.obs_shape), dtype=self.obs_dtype)
        self.actions = np.empty((self.capacity, *self.action_shape), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((self.capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((self.capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False




class ReplayBufferZ(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, skill_shape, pref_shape, div_shape, capacity, device, window=1):
        self.capacity = capacity
        self.device = device

        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.skill_shape = skill_shape
        self.pref_shape = pref_shape
        self.div_shape = div_shape
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        self.obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        self.data_name = ['obs', 'action', 'skill', 'reward', 'next_obs', 'not_done', 'not_done_no_max', 'prefs', 'divs']

        self.obses = np.empty((capacity, *obs_shape), dtype=self.obs_dtype )
        self.next_obses = np.empty((capacity, *obs_shape), dtype=self.obs_dtype )
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.skills= np.empty((capacity, *skill_shape), dtype=np.float32)
        self.prefs = np.empty((capacity, *pref_shape), dtype=np.float32)
        self.divs= np.empty((capacity, *div_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        self.window = window

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, skill, reward, next_obs, done, done_no_max, pref, div):

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.skills[self.idx], skill)
        np.copyto(self.prefs[self.idx],pref)
        np.copyto(self.divs[self.idx], div)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
    
  
            
    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)
        #print(self.idx)
        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()

        prefs = torch.as_tensor(self.prefs[idxs], device=self.device).float()
        divs = torch.as_tensor(self.divs[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        skills = torch.as_tensor(self.skills[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)
        return dict(zip( self.data_name, [obses, actions, skills, rewards, next_obses, not_dones, not_dones_no_max, prefs, divs]))
    
    def relabel_with_predictor(self, predictor):
        batch_size = 200
        total_iter = int(self.idx/batch_size)
        
        if self.idx > batch_size*total_iter:
            total_iter += 1
            
        for index in range(total_iter):
            last_index = (index+1)*batch_size
            if (index+1)*batch_size > self.idx:
                last_index = self.idx
            inputs = self.prefs[index*batch_size:last_index]
            pred_reward = np.sum([p.r_hat_batch(inputs) for p in predictor])

            self.rewards[index*batch_size:last_index] = pred_reward


    def get_full_prefs(self, batch):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch)
        if self.full:
            full_prefs = self.prefs

        else:
            full_prefs = self.prefs[: self.idx]

        full_prefs = torch.as_tensor(full_prefs, device=self.device)
        return full_prefs


    def get_full_divs(self, batch):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch)
        if self.full:
            full_divs = self.divs

        else:
            full_divs = self.divs[: self.idx]

        full_divs = torch.as_tensor(full_divs, device=self.device)
        return full_divs



