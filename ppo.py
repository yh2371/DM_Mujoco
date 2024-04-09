import sys
import os
# Add the parent directory of 'code' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
import torch.nn.functional as F
# import gymasium
from dp_env_3 import DPEnv
from policy import MlpPolicy
from VideoSaver import VideoSaver
import numpy as np
import time
from torch.distributions.normal import Normal
import mujoco
from mujoco import viewer
# import mediapy as media

class PPO:
    def __init__(self, ob_dim=56, ac_dim=28, hid_size=100, num_hid_layers=2, lr=1e-5, gamma=0.99, penalty_coef=0.1, env=None):
        self.policy = MlpPolicy(ob_dim, ac_dim, hid_size, num_hid_layers)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.penalty_coef = penalty_coef
        self.kl_target = 0.01
        self.env = env
        self.timesteps_per_batch = 600
        self.max_timesteps_per_episode = 200
        self.verbose = True
        self.clip_ratio = 0.2
        ### Feel free to add additional terms ###

    def calculate_advantages(self, rtgs, vpred):
        ### TODO ###
        A_k = rtgs - vpred.detach()
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
        return A_k

    def update_penalty(self, kl_loss):
        ### TODO ###
        if kl_loss < self.kl_target / 1.5:
            self.penalty_coef *= 1.5
        elif kl_loss > self.kl_target * 1.5:
            self.penalty_coef /= 1.5

    def compute_log_prob(self, actions, mean, logstd):
        ### TODO ###
        std = logstd.exp()
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(axis=-1)
        return log_probs#, entropy

        # std = logstd.exp()
        # var = std.pow(2)
        # log_probs = -0.5 * ((actions - mean) ** 2 / var + 2 * logstd + np.log(2 * np.pi))
        # log_probs = log_probs.sum(dim=-1, keepdim=True)
        # entropy = 0.5 * (logstd + np.log(2 * np.pi * np.e)).sum(dim=-1).mean()
        # return log_probs

    def compute_loss(self, log_probs, old_log_probs, vpred, advantages, rtgs):
        ### TODO ###
        ratios = torch.exp(log_probs - old_log_probs)
        clip_adv = torch.clamp(ratios, 1-self.clip_ratio, 1+self.clip_ratio) * advantages
        actor_loss = -(torch.min(ratios * advantages, clip_adv)).mean()
        critic_loss = F.smooth_l1_loss(vpred.view(-1,1), rtgs.view(-1,1))

        kl_loss = (old_log_probs - log_probs).mean()

        return actor_loss, critic_loss, kl_loss

    def update(self, obs, old_log_probs, advantages, rtgs):  

        action, vpred, mean, logstd = ppo_agent.policy.act(obs)
        #print(action, vpred, mean, logstd)
        curr_log_probs = self.compute_log_prob(action, mean, logstd).view((-1,1))
        
        actor_loss, critic_loss, kl_loss = self.compute_loss(curr_log_probs, old_log_probs, vpred, advantages, rtgs)
        total_loss = actor_loss + critic_loss #- kl_loss * self.penalty_coef 
        if self.verbose:
            print(f"Actor loss: {actor_loss}, Critic loss: {critic_loss}, KL loss: {kl_loss}")

        # Backprop
        ppo_agent.optimizer.zero_grad()
        total_loss.backward()
        ppo_agent.optimizer.step()

        # Adjust penalty coefficient
        with torch.no_grad():
            self.update_penalty(kl_loss)

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []
        batch_vpred = []

        t = 0 # Keeps track of how many timesteps we've run so far this batch

        while t < self.timesteps_per_batch:
            ep_rews = [] # rewards collected per episode

            # Reset the environment per episode
            obs = self.env.reset_model()
            done = False

            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.max_timesteps_per_episode):

                t += 1

                action, vpred, mean, logstd = self.policy.act(torch.tensor(obs, dtype=torch.float32).view(1,56))
                log_probs = self.compute_log_prob(action, mean, logstd)

                action = action.detach().numpy()
                log_probs = log_probs.detach().numpy()
                obs, rew, done, _ = self.env.step(action)

                batch_obs.append(obs)
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_probs)
                batch_vpred.append(vpred)

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        # Reshape data as tensors
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_vpred = torch.tensor(batch_vpred, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)                                                             # ALG STEP 4

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_vpred

    def compute_rtgs(self, batch_rews):
		# The rewards-to-go (rtg) per episode per batch to return.

        batch_rtgs = []
        # Iterate through each episode
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

            
# Initialize environment
env = DPEnv("../mujoco_file/motions/humanoid3d_walk.txt", "../mujoco_file/humanoid_deepmimic/envs/asset/dp_env_v3.xml")
env.reset_model()
renderer = mujoco.Renderer(env.m)
data = mujoco.MjData(env.m)
    
# Perform behavior cloning

# Initialize PPO Agent
ppo_agent = PPO(env=env)

# Load BC weights
ppo_agent.policy.load_state_dict(torch.load("bc_weights.pth"),)

MAX_ITER = 200
MAX_STEPS = 200
N_UPDATES = 10

for _ in range(MAX_ITER):

    # Rollouts
    batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_vpred = ppo_agent.rollout()                     # ALG STEP 3
    print(f"ITER {_}: Average Estimated Return: {batch_vpred.mean()}, Average Reward per Episode: {batch_rtgs.mean()}")

    action, V, mean, logstd = ppo_agent.policy.act(batch_obs)
    advantages = ppo_agent.calculate_advantages(batch_rtgs, V)
    print("#"*20)
    for _ in range(20):                                             
        ppo_agent.update(batch_obs, batch_log_probs, advantages, batch_rtgs)
    print("#"*20)

    # Initialize video saver
    width = 320
    height = 240
    # vid_save = VideoSaver(width=width, height=height, fps = 30)
    # import cv2
    # video = cv2.VideoWriter("test.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 20.0, (width,height))
    
    # One evaluation loop for visualization
    ppo_agent.policy.eval()
    env.reset_model()
    steps = 0
    frames = []
    with viewer.launch_passive(env.m, env.md) as v:
        while True:
            # Get observation from environment
            obs = torch.tensor(env._get_obs().reshape((1,-1)),dtype=torch.float32)

            # Use policy network to predict action
            action = ppo_agent.policy.act(obs)[0].detach().numpy()

            # Step the environment with the predicted action
            
            next_obs, reward, done, _ = env.step(action)
            mujoco.mj_step(env.m, env.md)

            v.sync()
            time.sleep(0.1)
            # Render the environment
            #env.render()
            # mujoco.mj_forward(env.m, data)
            # renderer.update_scene(data)
            # frame = renderer.render()
            #print(frame.shape)
            # frames.append(frame)

            #media.show_image(renderer.render())

            # Optionally save video, comment out to view simulation
            # vid_save.addFrame(seg)
            # video.write(frame)
            steps +=1 
            if steps >= MAX_STEPS:
                print("Done")
                break

        # Check if episode is done
        if done:
            print("Done")
            break
            
        ### STUDENT CODE START ###
        # Add reward plotting
        ### STUDENT CODE END ###
    # cv2.destroyAllWindows()
    # video.release()
    break
    # Close video saver
    # vid_save.close()
