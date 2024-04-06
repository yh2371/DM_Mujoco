import sys
import os

# Add the parent directory of 'code' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from VideoSaver import VideoSaver
from dp_env import DPEnv
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# from policy import MlpPolicy, build_policy_network
from network import FeedForwardNN

# Behavior cloning
def behavior_cloning(obs, actions, policy_network, num_epochs=10):

    ### STUDENT CODE START###
    # Define your loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(policy_network.parameters(), lr=0.0001)
    
    # Convert your data to PyTorch tensors and create datasets
    X_train_tensor = torch.tensor(obs, dtype=torch.float32)
    y_train_tensor = torch.tensor(actions, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Train the model
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            # Forward pass
            outputs = policy_network.forward(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Print loss after each epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    # Save the trained model weights
    torch.save(policy_network.state_dict(), 'bc_actor_weights.pth')

    ### STUDENT CODE END ###
    return policy_network


# Main function
def main():
    # Prepare expert data
    expert_data = np.load("./expert.npz", allow_pickle=True)
    obs = expert_data['obs']
    actions = expert_data['acs']

    # Initialize environment
    env = DPEnv()
    env.reset_model()
    
    # Perform behavior cloning
    ob_space = env.observation_space
    ac_space = env.action_space
    print(ob_space.shape)
    print(ac_space.shape)
    policy_net = FeedForwardNN(ob_space.shape[0], ac_space.shape[0])
    # policy_net = behavior_cloning(obs, actions, policy_net, 130)  
    policy_net.load_state_dict(torch.load('bc_actor_weights.pth'))
    policy_net.eval()
    

    # Initialize video saver
    width = 500
    height = 500
    vid_save = VideoSaver(width=width, height=height, fps = 30)

    # One evaluation loop
    policy_net.eval()
    num_steps = 300

    for i in range(num_steps):
        print(f"Step {i}/{num_steps}")
        # Get observation from environment
        obs = torch.tensor(env._get_obs().reshape((1,-1)),dtype=torch.float32)

        # Use policy network to predict action
        action = policy_net.forward(obs).detach().numpy()

        # Step the environment with the predicted action
        next_obs, reward, done, _ = env.step(action)

        env.calc_config_reward()

        # Render the environment
        #env.render()

        # Optionally save video, comment out to view simulation
        vid_save.addFrame(env.render(mode='rgb_array'))

        # Check if episode is done
        if done:
            break
            
        ### STUDENT CODE START ###
        # Add reward plotting
        ### STUDENT CODE END ###

    # Close video saver
    vid_save.close()


if __name__ == "__main__":
    main()
