# -*- coding: utf-8 -*-
import time
import torch
from torch import nn
from torch.distributions import Normal



class Actor_Model(nn.Module):
    def __init__(self, state_dim):
        super(Actor_Model, self).__init__()
        self.Dense1 = nn.Linear(state_dim, 256)
        self.Dense1act = nn.ReLU(inplace=True)
        self.Dense2 = nn.Linear(256, 128)
        self.Dense2act = nn.ReLU(inplace=True)
        self.mean_out = nn.Linear(128, 1)
        self.meanact = nn.Tanh()
        torch.nn.init.uniform_(self.mean_out.weight, a=-3e-3, b=3e-3)
        self.sigma_out = nn.Linear(128, 1)
        torch.nn.init.uniform_(self.sigma_out.weight, a=-3e-3, b=3e-3)
        self.sigmaact = nn.Softplus()

    def forward(self, state_vector):
        common_vector = self.Dense1(state_vector)
        common_vector = self.Dense1act(common_vector)
        common_vector = self.Dense2(common_vector)
        common_vector = self.Dense2act(common_vector)
        mean_out = self.mean_out(common_vector)
        mean_out = self.meanact(mean_out)

        sigma_out = self.sigma_out(common_vector)
        sigma_out = self.sigmaact(sigma_out)
        return mean_out, sigma_out


class Critic_Model(nn.Module):
    def __init__(self, state_dim):
        super(Critic_Model, self).__init__()
        self.Dense1 = nn.Linear(state_dim, 256)
        self.Dense1act = nn.ReLU(inplace=True)
        self.Dense2 = nn.Linear(256, 128)
        self.Dense2act = nn.ReLU(inplace=True)
        self.Dense3 = nn.Linear(128, 1)
        self.Dense3act = nn.ReLU(inplace=True)

    def forward(self, state_vector):
        value_vector = self.Dense1(state_vector)
        value_vector = self.Dense1act(value_vector)
        value_vector = self.Dense2(value_vector)
        value_vector = self.Dense2act(value_vector)
        value_vector = self.Dense3(value_vector)
        value_out = self.Dense3act(value_vector)
        return value_out
