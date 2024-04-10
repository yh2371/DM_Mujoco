import mujoco
from mujoco import viewer
import time
import torch
MAX_STEPS = 200
def viz_policy(env, policy_net):
    for _ in range(10):
        env.reset_model()
        steps = 0
        with viewer.launch_passive(env.m, env.md) as v:
            #change the camera angle
            v.cam.distance = 10
            done = False
            while not done:
                # Get observation from environment
                obs = torch.tensor(env._get_obs().reshape((1,-1)),dtype=torch.float32)

                # Use policy network to predict action
                action = policy_net.forward(obs)[0].detach().numpy()

                # Step the environment with the predicted action
                
                next_obs, reward, done, _ = env.step(action)
                mujoco.mj_step(env.m, env.md)

                v.sync()
                time.sleep(0.04)
                steps +=1 
                if steps >= MAX_STEPS:
                    print("Done")
                    break

        # Check if episode is done   