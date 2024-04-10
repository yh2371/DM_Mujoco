import sys
import os

# Add the parent directory of 'code' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
# from VideoSaver import VideoSaver
from dp_env_3 import DPEnv
from hw_utils import viz_policy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from policy import MlpPolicy, build_policy_network

# Behavior cloning
def behavior_cloning(obs, actions, policy_network, num_epochs=200):

    ### STUDENT CODE START###
    # Define your loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(policy_network.parameters())
    
    # Convert your data to PyTorch tensors and create datasets
    X_train_tensor = torch.tensor(obs, dtype=torch.float32)
    y_train_tensor = torch.tensor(actions, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Train the model
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            # Forward pass
            outputs = policy_network.act(inputs)[0]
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Print loss after each epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    # Save the trained model weights
    torch.save(policy_network.state_dict(), 'bc_weights.pth')

    ### STUDENT CODE END ###
    return policy_network


# Main function
def main():
    # Prepare expert data
    expert_data = np.load("./expert.npz", allow_pickle=True)
    obs = expert_data['obs']
    actions = expert_data['acs']

    # Initialize environment
    
    # Perform behavior cloning
    ob_dim=56
    ac_dim=28
    policy_net = build_policy_network(ob_dim, ac_dim)
    # policy_net = behavior_cloning(obs, actions, policy_net)  
    policy_net.load_state_dict(torch.load('bc_weights.pth'))

    # Initialize video saver
    # width = 500
    # height = 500
    # vid_save = VideoSaver(width=width, height=height, fps = 30)

    # One evaluation loop
    env = DPEnv("../mujoco_file/motions/humanoid3d_walk.txt", "../mujoco_file/humanoid_deepmimic/envs/asset/dp_env_v3.xml")
    env.reset_model()
    policy_net.eval()
    viz_policy(env, policy_net)


if __name__ == "__main__":
    main()
