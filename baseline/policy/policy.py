import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from abc import ABC, abstractmethod
from typing import Tuple

import math

toTensor = lambda x : torch.tensor(x, dtype=torch.float32)


class BasePolicy(nn.Module, ABC) :
    def __init__(self) :
        super(BasePolicy, self).__init__()
    
    @abstractmethod
    def forward(self, state:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor] :
        pass
    
    @abstractmethod
    def sample(self, action_pdf:torch.Tensor, deterministic:bool=False) :
        pass
    
    @abstractmethod
    def log_prob(self, action_pdf:torch.Tensor, action:torch.Tensor) :
        pass
    
    @abstractmethod
    def entropy(self, action_pdf:torch.Tensor) :
        pass


class ContinuMlpPolicy(BasePolicy) :
    def __init__(
            self, state_dim:int, action_dim:int, hidden_layers:list=[64,64],
            policy_active=torch.tanh, value_active=torch.relu,
            sd_init:float=0.5, sd_rng:Tuple[float, float]=(0.1, 0.6)
            ) :
        
        super(ContinuMlpPolicy, self).__init__()
        
        # activation function
        self.policy_active = policy_active
        self.value_active = value_active

        # layers
        tmp = [state_dim] + hidden_layers
        self.policy_hidden_layer = nn.ModuleList([nn.Linear(tmp[k], tmp[k + 1]) for k in range(len(tmp) - 1)])
        self.policy_last_layer = nn.Sequential(nn.Linear(hidden_layers[-1], action_dim), nn.Tanh())

        tmp = [state_dim] + hidden_layers
        self.value_hidden_layer = nn.ModuleList([nn.Linear(tmp[k], tmp[k + 1]) for k in range(len(tmp) - 1)])
        self.value_last_layer = nn.Sequential(nn.Linear(hidden_layers[-1], 1))

        # action standard deviation parameter
        self.sd_log_rng = (math.log(sd_rng[0]), math.log(sd_rng[1]))

        self.sd_param = nn.Parameter(torch.full((1, action_dim), math.log(sd_init), dtype=torch.float32))
    

    def forward(self, state:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor] :
        
        # policy-mu
        mu = state
        for layer in self.policy_hidden_layer :
            mu = self.policy_active(layer(mu))
        mu = self.policy_last_layer(mu)

        # policy-sd
        self.sd_param.data.clip_(*self.sd_log_rng)
        sd = self.sd_param.exp().expand(*mu.shape)
        
        # value
        value = state
        for layer in self.value_hidden_layer :
            value = self.policy_active(layer(value))
        value = self.value_last_layer(value)
        
        return torch.stack((mu, sd), dim=2), value
    

    def sample(self, action_pdf:torch.Tensor, deterministic:bool=False) :
        if deterministic :
            return action_pdf[:,:,0]
        
        dist = Normal(action_pdf[:,:,0], action_pdf[:,:,1])
        return dist.sample().clip(-1.0, 1.0)
    

    def log_prob(self, action_pdf:torch.Tensor, action:torch.Tensor) :
        dist = Normal(action_pdf[:,:,0], action_pdf[:,:,1])
        return dist.log_prob(action)
    
    
    def entropy(self, action_pdf:torch.Tensor) :
        dist = Normal(action_pdf[:,:,0], action_pdf[:,:,1])
        return dist.entropy()