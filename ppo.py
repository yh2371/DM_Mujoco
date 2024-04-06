import torch
import torch.optim as optim
import torch.nn.functional as F
import gym
from dp_env import DPEnv
from policy import MlpPolicy
from VideoSaver import VideoSaver
import numpy as np
import time
import imageio

class PPO:
    def __init__(self, ob_space, ac_space, hid_size=100, num_hid_layers=2, lr=5e-5, gamma=0.95, penalty_coef=0.1):
        self.policy = MlpPolicy(ob_space, ac_space, hid_size, num_hid_layers)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.penalty_coef = penalty_coef
        self.kl_target = 0.01
        self.clip = 0.2
        ### Feel free to add additional terms ###

    def calculate_targets_advantages(self, rewards, next_states, dones, vpred):
        ### TODO ###
        _, next_value = self.policy.forward(next_states) 
        td_targets = rewards + self.gamma * next_value * (1 - dones)
        advantages = td_targets - vpred.detach()
        return td_targets, advantages

    def update_penalty(self, kl_loss):
        ### TODO ###
        if kl_loss < self.kl_target / 1.5:
            self.penalty_coef *= 1.5
        elif kl_loss > self.kl_target * 1.5:
            self.penalty_coef /= 1.5

    def compute_log_prob(self, actions, mean, logstd):
        ### TODO ###
        std = logstd.exp()
        var = std.pow(2)
        log_probs = -0.5 * ((actions - mean) ** 2 / var + 2 * logstd + np.log(2 * np.pi))
        log_probs = log_probs.sum(dim=-1, keepdim=True)
        entropy = 0.5 * (logstd + np.log(2 * np.pi * np.e)).sum(dim=-1).mean()

        return log_probs, entropy

    def compute_loss(self, log_probs, old_log_probs, vpred, advantages, td_targets):
        ### TODO ###
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages

        policy_loss = (-torch.min(surr1, surr2)).mean()
        value_loss = F.smooth_l1_loss(vpred.squeeze(), td_targets.detach())

        kl_loss = (old_log_probs - log_probs).mean()
        kl_penalty = kl_loss * self.penalty_coef

        return policy_loss, value_loss, kl_loss, kl_penalty

    def update(self, states, actions, rewards, next_states, dones, old_log_probs, adv_targets):        
        # Fetch mean, std of gaussian
        pdparams, vpred = self.policy.forward(states)
        mean, logstd = pdparams[:, :self.policy.pdtype.param_shape()[0] // 2], pdparams[:, self.policy.pdtype.param_shape()[0] // 2:]

        # Compute log probabilities, assuming diagonal gaussian distribution
        log_probs, entropy = self.compute_log_prob(actions, mean, logstd)

        # Compute targets, advantages 
        td_targets, advantages = self.calculate_targets_advantages(rewards, next_states, dones, vpred)

        # Compute losses
        policy_loss, value_loss, kl_loss, kl_penalty = self.compute_loss(log_probs, old_log_probs, vpred, advantages, td_targets)
        total_loss = policy_loss - kl_penalty + value_loss - 0.01 * entropy

        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Adjust penalty coefficient
        with torch.no_grad():
            self.update_penalty(kl_loss)
            
# Initialize environment
env = DPEnv()
env2 = DPEnv()
env.reset()
env2.reset()
    
# Perform behavior cloning
ob_space = env.observation_space
ac_space = env.action_space

# Initialize PPO Agent
ppo_agent = PPO(ob_space, ac_space)

# Load BC weights
ppo_agent.policy.load_state_dict(torch.load("bc_weights.pth"))

# Initialize video saver
width = 500
height = 500
vid_save = VideoSaver(width=width, height=height, fps = 30)

max_episodes = 1000
max_steps = 400

for episode in range(max_episodes):
    if episode == 200:
        ppo_agent.policy.unfreeze_all_parameters()
    env.reset()
    state = torch.tensor(env._get_obs().reshape((1,-1)),dtype=torch.float32)
    total_reward = 0
    for step in range(max_steps):
        action, not_useful_value = ppo_agent.policy.act(torch.FloatTensor(state).view((1,56)))
        next_state, reward, done, _ = env.step(action.detach().numpy())
        mean, logstd = pdparams[:, :self.policy.pdtype.param_shape()[0] // 2], pdparams[:, self.policy.pdtype.param_shape()[0] // 2:]
        log_probs, _ = self.compute_log_prob(actions, mean, logstd)
        # if step == 0:
        # imageio.imwrite(f"render/env_{episode}_{step}.png", env.render(mode='rgb_array'))
        # qpos = env2.mocap.data_config[env.idx_curr]
        # qvel = env2.mocap.data_vel[env.idx_curr]
        # env2.set_state(qpos, qvel)
        # imageio.imwrite(f"render/env2_{episode}_{step}.png", env2.render(mode='rgb_array'))
        total_reward += reward
        ppo_agent.update(torch.tensor(state, dtype=torch.float32).view(1,56), action, reward, torch.tensor(next_state, dtype=torch.float32).view(1,56), done, log_probs, adv_targets=None)
        state = next_state
        if done or step == max_steps - 1:
            print("Episode:", episode, "Total Steps:", step, "Total Reward:", total_reward)
            break


# One evaluation loop
ppo_agent.policy.eval()
env.reset_model()
for i in range(300):
    # Get observation from environment
    obs = torch.tensor(env._get_obs().reshape((1,-1)),dtype=torch.float32)

    # Use policy network to predict action
    action = ppo_agent.policy.act(obs)[0].detach().numpy()

    # Step the environment with the predicted action
    # time.sleep(.002)
    next_obs, reward, done, _ = env.step(action)

    env.calc_config_reward()

    # Render the environment
    #env.render()

    # Optionally save video, comment out to view simulation
    vid_save.addFrame(env.render(mode='rgb_array'))

    # Check if episode is done
    if done:
        print("Done")
        break
        
    ### STUDENT CODE START ###
    # Add reward plotting
    ### STUDENT CODE END ###

# Close video saver
vid_save.close()
