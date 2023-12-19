# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from gym import utils
import numpy as np
from gym.envs.mujoco import mujoco_env
import gym
import math

class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):

  def __init__(self,
               expose_all_qpos=False,
               task='forward',
               area=None,
               target_velocity=None,
               model_path='half_cheetah.xml'):
    # Settings from
    # https://github.com/openai/gym/blob/master/gym/envs/__init__.py
    self._expose_all_qpos = expose_all_qpos
    self._task = task
    self._target_velocity = target_velocity
    self.target_area = None
    self.positions = []
    self.frames = []
    if self._task == 'area_velocity':
      self.set_target_area(area)

    self.record = False
    self.prior_low_state = np.array([-1]) # x_agent,y_agent, x_goal, y_goal, distance
    self.prior_high_state = np.array([1])
    self.prior_space = gym.spaces.Box(self.prior_low_state, self.prior_high_state, dtype=np.float32)
    xml_path =  "envs/mujoco/assets/"
    model_path = os.path.abspath(os.path.join(xml_path, model_path))
    self._max_episode_steps = 1000
    mujoco_env.MujocoEnv.__init__(
        self,
        model_path,
        5)
    utils.EzPickle.__init__(self)

  def set_target_area(self, area):
    self.target_area = area
    self.target_velocity = (area[0] + area[1]) / 2

  def step(self, action):
    xposbefore = self.sim.data.qpos[0]
    zposbefore = self.sim.data.qpos[1]
    self.do_simulation(action, self.frame_skip)
    xposafter = self.sim.data.qpos[0]
    zposafter = self.sim.data.qpos[1]
    xvelafter = self.sim.data.qvel[0]
    ob = self._get_obs()
    reward_ctrl = -0.1 * np.square(action).sum()

    if self._task == 'default':
      reward_vel = 0.
      reward_run = (xposafter - xposbefore) / self.dt
      reward = reward_ctrl + reward_run
    elif self._task == 'jump':
      reward_vel = 0.
      reward_run = (zposafter - zposbefore) / self.dt
      reward = reward_ctrl + reward_run      
    elif self._task == 'area_velocity':
      reward_vel = 0.0
      if xvelafter >= self.target_area[0] and xvelafter <= self.target_area[1]:
        reward_vel = 1.0
      else:
        reward_vel = - math.sqrt(pow((self.target_velocity-xvelafter), 2)) #/500
      reward = reward_ctrl + reward_vel #(reward_vel+1)/2

    elif self._task == 'forward':
      reward_f =  0.1 if (xposafter - xposbefore) > 0 else 0
      reward = reward_f + reward_ctrl

    elif self._task == 'backward':
      reward_f =  0.1 if (xposafter - xposbefore) < -0.5 else 0
      reward = reward_f + reward_ctrl  

    elif self._task == 'target_velocity':
      reward_vel = -(self._target_velocity - xvelafter)**2
      reward = reward_ctrl + reward_vel

    elif self._task == 'run_back':
      reward_vel = 0.0
      reward_run = (xposbefore - xposafter) / self.dt
      reward = reward_ctrl + reward_run

    done = False
    self.positions.append(xposafter)
    if self.record:
      frame = self.render(mode='rgb_array')
      self.frames.append(frame)



    info = {
      'x_pos': xposafter ,
      'y_pos': 0,
      'prior':np.array([xvelafter]) ,
      'traj': self.positions,
      'frames': np.array(self.frames)
    }
    return ob, reward, done, info  #xvelafter]) #self.norms(xposafter)
    
  def norms(self, x):
    return np.array([x])/10.0
  
  def _get_obs(self):
    if self._expose_all_qpos:
      return np.concatenate(
          [self.sim.data.qpos.flat, self.sim.data.qvel.flat])
    return np.concatenate([
        self.sim.data.qpos.flat[1:],
        self.sim.data.qvel.flat,
    ])

  def reset_model(self):
    qpos = self.init_qpos + self.np_random.uniform(
        low=-.1, high=.1, size=self.sim.model.nq)
    qvel = self.init_qvel + self.np_random.randn(self.sim.model.nv) * .1
    self.set_state(qpos, qvel)
    xvelafter = self.sim.data.qvel[0]
    self.positions = [self.sim.data.qpos[0]]
    self.frames = []
    return self._get_obs(), np.array([xvelafter]) #np.array([xvelafter])

  def viewer_setup(self):
    self.viewer.cam.distance = self.model.stat.extent * 0.5
