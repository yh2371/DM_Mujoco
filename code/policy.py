import sys
import os
# Add the parent directory of 'code' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import make_pdtype
import gym
from torch.distributions.normal import Normal


class MlpPolicy(nn.Module):
    def __init__(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        super(MlpPolicy, self).__init__()

        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob_shape = (sequence_length,) + tuple(ob_space.shape)

	    ### TODO ###
        self.actor = nn.ModuleList([nn.Linear(np.prod(ob_space.shape), hid_size)] + [nn.Linear(hid_size, hid_size) for _ in range(num_hid_layers)])  
        self.critic = nn.Linear(np.prod(ob_space.shape), hid_size)
        self.vpred = nn.Linear(hid_size, 1)

        self.mean_pred = nn.Linear(hid_size, pdtype.param_shape()[0]//2)
        self.logstd = nn.Parameter(torch.zeros(1, pdtype.param_shape()[0]//2))

    def forward(self, ob):
    	### TODO ###
        vpred = self.vpred(F.relu(self.critic(ob))) 
        
        obz = ob
        for fc in self.actor:
            obz = F.relu(fc(obz))
        mean = self.mean_pred(obz)     
        logstd = self.logstd#_pred(obz)  
 
        return mean, logstd, vpred

    def act(self, ob):
    	### TODO ###
        mean, logstd, vpred = self.forward(ob)  

        dist = Normal(mean, logstd.exp())
        ac = dist.sample()
        #ac = mean + torch.randn_like(mean) * logstd.exp()

        return ac, vpred, mean, logstd

    def get_trainable_variables(self):
        return filter(lambda p: p.requires_grad, self.parameters())
        
def build_policy_network(ob_space, ac_space, hid_size = 100, num_layers = 2):
    model = MlpPolicy(ob_space, ac_space, hid_size, num_layers)
    return model
