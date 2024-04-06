"""
	This file will run Stable Baselines PPO2 or PPO for Beginners code
	with the input seed and environment.
"""

# import gym
import os
import argparse
from network import FeedForwardNN
from policy import MlpPolicy
from dp_env import DPEnv
import torch


def train_ppo_for_beginners(args):
	"""
		Trains with PPO for Beginners on specified environment.

		Parameters:
			args - the arguments defined in main.

		Return:
			None
	"""
	# Import ppo for beginners
	from ppo_optimized import PPO

	# from ppo_for_beginners.network import FeedForwardNN

	total_timesteps = 1405000000 # 1.4 million
	#humanoid hyperparameters
	hyperparameters = {'n_steps': 2048, 'batch_size': 128, 'gae_lambda': 0.95, 'gamma': 0.99, 'n_epochs': 10,
							'ent_coef': 0.001, 'learning_rate': 2.5e-4, 'clip_range': 0.2, 'verbose': 1, 'seed': args.seed}

	# Make the environment and model, and train
	env = DPEnv()
	model = PPO(FeedForwardNN, env, **hyperparameters)
	# model.actor.load_state_dict(torch.load("bc_actor_weights.pth"))
	model.learn(total_timesteps)



# Load BC weights
	# model.policy.load_state_dict(torch.load("bc_weights.pth"))

	# # Initialize video saver
	# width = 500
	# height = 500
	# vid_save = VideoSaver(width=width, height=height, fps = 30)



	# # One evaluation loop
	# ppo_agent.policy.eval()
	# env.reset_model()
	# for i in range(300):
	# 	# Get observation from environment
	# 	obs = torch.tensor(env._get_obs().reshape((1,-1)),dtype=torch.float32)

	# 	# Use policy network to predict action
	# 	action = ppo_agent.policy.act(obs)[0].detach().numpy()

	# 	# Step the environment with the predicted action
	# 	# time.sleep(.002)
	# 	next_obs, reward, done, _ = env.step(action)

	# 	env.calc_config_reward()

	# 	# Render the environment
	# 	#env.render()

	# 	# Optionally save video, comment out to view simulation
	# 	vid_save.addFrame(env.render(mode='rgb_array'))

	# 	# Check if episode is done
	# 	if done:
	# 		print("Done")
	# 		break
			
	# 	### STUDENT CODE START ###
	# 	# Add reward plotting
	# 	### STUDENT CODE END ###

	# # Close video saver
	# vid_save.close()

def main(args):
	"""
		An intermediate function that will call either PPO2 learn or PPO for Beginners learn.

		Parameters:
			args - the arguments defined below

		Return:
			None
	"""

	train_ppo_for_beginners(args)

if __name__ == '__main__':
	# Parse arguments
	parser = argparse.ArgumentParser()

	parser.add_argument('--t', dest='optimized', type=bool, default=True)         # A flag for optimization
	parser.add_argument('--code', dest='code', type=str, default='')               # Can be 'stable_baselines_ppo' or 'ppo_for_beginners' / 'ppo_for_beginners/trick'
	parser.add_argument('--seed', dest='seed', type=int, default=None)             # An int for our seed
	parser.add_argument('--env', dest='env', type=str, default='')                 # Formal name of environment

	args = parser.parse_args()

	# Collect data
	main(args)