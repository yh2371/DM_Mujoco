#!/usr/bin/env python3
import numpy as np
import math
import random
from os import getcwd
import sys
import os

# Add the parent directory of 'code' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mujoco.mocap_v2 import MocapDM
from mujoco.mujoco_interface import MujocoInterface
from mujoco.mocap_util import JOINT_WEIGHT
from mujoco_py import load_model_from_xml, MjSim, MjViewer

from gym.envs.mujoco import mujoco_env
from gym import utils

from config import Config
from pyquaternion import Quaternion

from transformations import quaternion_from_euler

BODY_JOINTS = ["chest", "neck", "right_shoulder", "right_elbow", 
            "left_shoulder", "left_elbow", "right_hip", "right_knee", 
            "right_ankle", "left_hip", "left_knee", "left_ankle"]

DOF_DEF = {"root": 3, "chest": 3, "neck": 3, "right_shoulder": 3, 
           "right_elbow": 1, "right_wrist": 0, "left_shoulder": 3, "left_elbow": 1, 
           "left_wrist": 0, "right_hip": 3, "right_knee": 1, "right_ankle": 3, 
           "left_hip": 3, "left_knee": 1, "left_ankle": 3}

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

def com_velocity(sim):
    # center of mass velocity
    mass = np.expand_dims(sim.model.body_mass, 1)
    vel = sim.data.cvel
    return (np.sum(mass * vel, 0) / np.sum(mass))

_exp_weighted_averages = {}

def smooth(label, new_value, smoothing_factor=0.9):
    """
    Implements an exponential running smoothing filter.
    Several inputs can be filtered parallely by providing different labels.
    :param label: give your filtered data a name.
                  Will be used as a dict key to save current filtered value.
    :return: current filtered value for the provided label
    """
    global _exp_weighted_averages

    if label not in _exp_weighted_averages:
        _exp_weighted_averages[label] = new_value
        return new_value

    new_average = smoothing_factor * new_value + (1 - smoothing_factor) * _exp_weighted_averages[label]
    _exp_weighted_averages[label] = new_average

    return new_average

class DPEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, mocap_path, xml_path):
        xml_file_path = xml_path

        self.mocap = MocapDM()
        self.interface = MujocoInterface()
        self.load_mocap(mocap_path)

        self.weight_pose = 0.5
        self.weight_vel = 0.05
        self.weight_root = 0.2
        self.weight_end_eff = 0.15
        self.weight_com = 0.1

        self.scale_pose = 2.0
        self.scale_vel = 0.1
        self.scale_end_eff = 40.0
        self.scale_root = 5.0
        self.scale_com = 10.0
        self.scale_err = 1.0

        self.reference_state_init()
        self.idx_curr = -1
        self.idx_tmp_count = -1

        self.pos_rew, self.vel_rew, self.com_rew = 0,0,0
        # track episode duration
        self.ep_dur = 0
        # track mean episode return for ET-reward calculation
        self.ep_rews = []
        self.mean_epret_smoothed = 0

        mujoco_env.MujocoEnv.__init__(self, xml_file_path, 6)
        utils.EzPickle.__init__(self)

    def _quat2euler(self, quat):
        tmp_quat = np.array([quat[1], quat[2], quat[3], quat[0]])
        euler = euler_from_quaternion(tmp_quat, axes='rxyz')
        return euler

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()[7:] # ignore root joint
        velocity = self.sim.data.qvel.flat.copy()[6:] # ignore root joint
        return np.concatenate((position, velocity))

    def reference_state_init(self):
        self.idx_init = random.randint(0, self.mocap_data_len-1)
        # self.idx_init = 0
        self.idx_curr = self.idx_init
        self.idx_tmp_count = 0

    def early_termination(self):
        target_config = self.mocap.data_config[self.idx_curr][7:] # to exclude root joint
        curr_config = self.get_joint_configs()
        err_configs = self.calc_config_errs(curr_config, target_config)
        if err_configs >= 15.0:
            return True
        return False

    def get_joint_configs(self):
        data = self.sim.data
        return data.qpos[7:] # to exclude root joint

    def get_root_configs(self):
        data = self.sim.data
        return data.qpos[3:7] # to exclude x coord

    def load_mocap(self, filepath):
        self.mocap.load_mocap(filepath)
        self.mocap_dt = self.mocap.dt
        self.mocap_data_len = len(self.mocap.data)

    def calc_config_errs(self, env_config, mocap_config):
        assert len(env_config) == len(mocap_config)
        return np.sum(np.abs(env_config - mocap_config))

    def calc_root_reward(self): # including root joint
        curr_root = self.mocap.data_config[self.idx_curr][3:7]
        target_root = self.get_root_configs()
        assert len(curr_root) == len(target_root)
        assert len(curr_root) == 4

        q_0 = Quaternion(curr_root[0], curr_root[1], curr_root[2], curr_root[3])
        q_1 = Quaternion(target_root[0], target_root[1], target_root[2], target_root[3])

        q_diff =  q_0.conjugate * q_1
        tmp_diff = q_diff.angle

        err_root = abs(tmp_diff)
        reward_root = math.exp(-err_root)
        return reward_root
    
    def goal_reward(self, target_direction, desired_speed):
        """
        Compute the reward based on the target heading objective.
        """
        # Compute the speed along the target direction
        current_velocity = com_velocity(self.sim)[:3] #linear only
        speed_along_target = np.dot(current_velocity, target_direction)

        # Compute the squared difference between desired speed and speed along target
        speed_difference_squared = (desired_speed - speed_along_target) 
        # Compute the reward
        reward = np.exp(-25 * max(0, speed_difference_squared)**2)
        return reward

    def calc_config_reward(self):
        assert len(self.mocap.data) != 0
        err_configs = 0.0

        target_config = self.mocap.data_config[self.idx_curr][7:] # to exclude root joint
        self.curr_frame = target_config
        curr_config = self.get_joint_configs()

        err_configs = self.calc_config_errs(curr_config, target_config)
        # reward_config = math.exp(-self.scale_err * self.scale_pose * err_configs)
        reward_config = math.exp(-err_configs)

        return reward_config, err_configs

    def step(self, action):
        self.step_len = 1
        step_times = 1
        #print(action)
        self.do_simulation(action, step_times)

        reward_config,  err_config = self.calc_config_reward()
        # reward_root = 10 * self.calc_root_reward()
        # reward = reward_config + reward_root     

        info = dict()

        self.idx_curr += 1
        self.idx_curr = self.idx_curr % self.mocap_data_len

        observation = self._get_obs()
        done = bool(self.is_done() or err_config >= 10.0)
        reward = self.get_reward(done)
        if not done:
            self.ep_rews.append(reward) #track rewards
            self.ep_dur += 1

        return observation, reward, done, info
    
    def _get_ET_reward(self):
        """ Punish falling hard and reward reaching episode's end a lot. """

        # calculate a running mean of the ep_return
        self.mean_epret_smoothed = smooth('mimic_env_epret', np.sum(self.ep_rews), 0.5)
        self.ep_rews = []

        # reward reaching the end of the episode without falling
        # reward = expected cumulative future reward
        max_eplen_reached = self.ep_dur >= 200
        if max_eplen_reached:
            # estimate future cumulative reward expecting getting the mean reward per step
            mean_step_rew = self.mean_epret_smoothed / self.ep_dur
            act_ret_est = np.sum(mean_step_rew * np.power(0.99, np.arange(self.ep_dur)))
            reward = act_ret_est
        # punish for ending the episode early
        else:
            reward = -1 * self.mean_epret_smoothed

        return reward

    def get_reward(self, done: bool):
        """ Returns the reward of the current state.
            :param done: is True, when episode finishes and else False"""
        return self._get_ET_reward() if done \
            else self.get_imitation_reward() + 0.2 #alive bonus

    def get_imitation_reward(self):
        """ DeepMimic imitation reward function """

        # get rew weights
        weights =  [0.8, 0.2, 0, 0]

        w_pos, w_vel, w_com, w_pow = weights
        pos_rew = self.get_pose_reward()
        vel_rew = self.get_vel_reward()
        com_rew = self.get_com_reward() #root joint
        
        self.pos_rew, self.vel_rew, self.com_rew = pos_rew, vel_rew, com_rew

        # target heading reward
        direction = np.array([-1,1,0])
        direction = direction/np.linalg.norm(direction)
        imit_rew = w_pos * pos_rew + w_vel * vel_rew + w_com * com_rew + self.goal_reward(direction,2)

        return imit_rew

    def get_pose_reward(self):
        qpos = self.sim.data.qpos.copy()[7:]
        ref_pos = self.mocap.data_config[self.idx_curr].copy()[7:]

        dif = qpos - ref_pos
        dif_sqrd = np.square(dif)
        sumv = np.sum(dif_sqrd)
        pose_rew = np.exp(-3 * sumv)
        return pose_rew

    def get_vel_reward(self):
        qvel = self.sim.data.qvel.copy()[6:]
        ref_vel = self.mocap.data_vel[self.idx_curr].copy()[6:]

        difs = qvel - ref_vel
        dif_sqrd = np.square(difs)
        dif_sum = np.sum(dif_sqrd)
        vel_rew = np.exp(-0.05 * dif_sum)
        return vel_rew

    def get_com_reward(self):
        curr_root = self.mocap.data_config[self.idx_curr][3:7].copy()
        target_root = self.get_root_configs().copy()
        dif = curr_root - target_root
        dif_sqrd = np.square(dif)
        sumv = np.sum(dif_sqrd)
        com_rew = np.exp(-16 * sumv)
        return com_rew

    def is_done(self):
        mass = np.expand_dims(self.model.body_mass, 1)
        xpos = self.sim.data.xipos
        z_com = (np.sum(mass * xpos, 0) / np.sum(mass))[2]
        done = bool((z_com < 0.7) or (z_com > 1.2))
        return done

    def goto(self, pos):
        self.sim.data.qpos[:] = pos[:]
        self.sim.forward()

    def get_time(self):
        return self.sim.data.time

    def reset_model(self):
        self.reference_state_init()
        qpos = self.mocap.data_config[self.idx_init]
        qvel = self.mocap.data_vel[self.idx_init]
        self.set_state(qpos, qvel)
        observation = self._get_obs()
        self.idx_tmp_count = -self.step_len
        return observation

    def reset_model_init(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        pass

if __name__ == "__main__":
    env = DPEnv()
    env.reset_model()

    import cv2
    from VideoSaver import VideoSaver
    width = 640
    height = 480

    action_size = env.action_space.shape[0]
    ac = np.zeros(action_size)
    while True:
        qpos = env.mocap.data_config[env.idx_curr]
        qvel = np.zeros_like(env.mocap.data_vel[env.idx_curr])
        env.set_state(qpos, qvel)
        env.sim.step()
        env.idx_curr += 1
        if env.idx_curr == env.mocap_data_len:
            env.idx_curr = env.idx_curr % env.mocap_data_len
        env.render()