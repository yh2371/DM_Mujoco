import sys
import os

# Add the parent directory of 'code' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from mujoco.mocap_v2 import MocapDM
import cv2
import numpy as np
from VideoSaver import VideoSaver
from dp_env import DPEnv


# Define a simple neural network policy
def build_policy_network(input_dim, output_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(output_dim, activation='tanh')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Behavior cloning using supervised learning
def behavior_cloning_with_supervised_learning(expert_data, test_size=0.2, num_epochs=1000000):
    # Prepare expert data
    state = np.vstack(expert_data.data_config[:])[:,7:]  # States
    print(state.shape)
    vel = np.vstack(expert_data.data_vel[:])[:,6:]     # Actions
    obs = np.hstack([state[:-1], vel[:-1]])
    actions = np.vstack(expert_data.data_config[1:])[:,7:]  
    print(actions.shape)

    # Split data into train and test sets
    print(len(obs))
    print(len(actions))
    X_train, X_test, y_train, y_test = train_test_split(obs, actions, test_size=test_size, random_state=42)

    # Build policy network
    policy_network = build_policy_network(X_train.shape[1], y_train.shape[1])
    
    # Compile the model
    policy_network.compile(optimizer='adam', loss='mse')

    # Train the model
    policy_network.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epochs, batch_size=32)

    # After training your model
    policy_network.save_weights('bc_weights.h5')

    return policy_network


# Main function
def main():
    # Initialize MocapDM to load expert data
    expert_data = MocapDM()
    expert_data.load_mocap("../mujoco/motions/humanoid3d_walk.txt")

    # Perform behavior cloning using supervised learning
    policy_net = behavior_cloning_with_supervised_learning(expert_data)

    # Initialize environment
    env = DPEnv("../mujoco/motions/humanoid3d_walk.txt", "/DeepMimic_mujoco/src/mujoco/humanoid_deepmimic/envs/asset/dp_env_v3.xml")
    env.reset_model()

    # Initialize video saver
    width = 640
    height = 480
    #vid_save = VideoSaver(width=width, height=height)

    # Load motion data (if needed)
    # env.load_mocap("/home/mingfei/Documents/DeepMimic/mujoco/motions/humanoid3d_crawl.txt")

    # Get action size from environment
    action_size = env.action_space.shape
    print(action_size)
    actions = np.vstack(expert_data.data_config[1:])[:,7:]  
    idx = 0

    # Main loop
    while True:
        # Get observation from environment
        obs = env._get_obs().reshape((1,-1))

        # Use policy network to predict action
        action = policy_net.predict(obs)

        # Step the environment with the predicted action
        next_obs, reward, done, _ = env.step(action)

        env.calc_config_reward()

        # Render the environment
        env.render()

        # Optionally save video
        #vid_save.write(env.render(mode='rgb_array'))

        # Check if episode is done
        if done:
            continue
            break

    # # Close video saver
    # vid_save.close()


if __name__ == "__main__":
    main()
