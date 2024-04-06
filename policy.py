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

class MlpPolicy(nn.Module):
    def __init__(self, ob_space, ac_space, hid_size = 100, num_hid_layers = 2, gaussian_fixed_var=True):
        super(MlpPolicy, self).__init__()

        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob_shape = (sequence_length,) + tuple(ob_space.shape)

	    ### TODO ###
        self.obfilter = nn.Linear(np.prod(ob_space.shape), hid_size)
        
        self.vpred = nn.Linear(hid_size, 1)

        self.polfc = nn.ModuleList([nn.Linear(hid_size, hid_size) for _ in range(num_hid_layers)])
        self.polfinal = nn.Linear(hid_size, pdtype.param_shape()[0]//2)

        self.logstd = nn.Parameter(torch.zeros(1, pdtype.param_shape()[0]//2))
        self.freeze_parameters_except_vpred()
    def unfreeze_all_parameters(self):
        # Unfreeze all parameters in the model
        for param in self.parameters():
            param.requires_grad = True
    def freeze_parameters_except_vpred(self): # so that value can be trained with arbitary reward function
    # so value function can be somewhat accurate
        # Freeze all parameters in the model
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze the parameters of vpred
        for param in self.vpred.parameters():
            param.requires_grad = True

    def forward(self, ob):
    	### TODO ###
        obz = F.relu(self.obfilter(ob))
        
        vpred = self.vpred(obz)
        
        for fc in self.polfc:
            obz = F.relu(fc(obz))

        mean = self.polfinal(obz)       
        pdparam = torch.cat([mean, mean * 0.0 + self.logstd.exp()], dim=1)
 
        return pdparam, vpred

    def act(self, ob):
    	### TODO ###
        pdparam, vpred = self.forward(ob)        
        ac = pdparam[:, :self.pdtype.param_shape()[0]//2] + torch.randn_like(pdparam[:, :self.pdtype.param_shape()[0]//2]) * pdparam[:, self.pdtype.param_shape()[0]//2:]

        return ac, vpred

    def get_trainable_variables(self):
        return filter(lambda p: p.requires_grad, self.parameters())
        
def build_policy_network(ob_space, ac_space, hid_size = 100, num_layers = 2):
    model = MlpPolicy(ob_space, ac_space, hid_size, num_layers)
    return model
