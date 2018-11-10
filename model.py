import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=[1024, 512]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (list(int)): Number of nodes in hidden_layers
        """
        super(Actor, self).__init__()
        self.action_size = action_size
        self.state_size = state_size
        self.seed = torch.manual_seed(seed)
        self.bn_input = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc_units[0])
        self.layers = nn.ModuleList([nn.Linear(fc_units[i], fc_units[i+1]) for i in range(len(fc_units) - 1)])
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(fc_units[i]) for i in range(len(fc_units) - 1)])
        if action_size == 2:
            self.movement = nn.Linear(fc_units[-1], 1)
            self.jump = nn.Linear(fc_units[-1], 1)
        elif action_size == 4:
            self.m1 = nn.Linear(fc_units[-1], 1)
            self.j1 = nn.Linear(fc_units[-1], 1)
            self.m2 = nn.Linear(fc_units[-1], 1)
            self.j2 = nn.Linear(fc_units[-1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        for l in self.layers:
            l.weight.data.uniform_(*hidden_init(l))
        if self.action_size == 2:
            self.movement.weight.data.uniform_(-3e-3, 3e-3)
            self.jump.weight.data.uniform_(-3e-3, 3e-3)
        elif self.action_size == 4:
            self.m1.weight.data.uniform_(-3e-3, 3e-3)
            self.j1.weight.data.uniform_(-3e-3, 3e-3)
            self.m2.weight.data.uniform_(-3e-3, 3e-3)
            self.j2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        state = state.reshape(-1, self.state_size)
        x = self.bn_input(state)
        x = F.tanh(self.fc1(x))
        for i, l in enumerate(self.layers):
            x = self.bn_layers[i](x)
            x = F.tanh(l(x))
        if self.action_size == 2:
            m = F.tanh(self.movement(x))
            j = F.sigmoid(self.jump(x))
            return torch.cat((m, j)).reshape(-1, 2)
        elif self.action_size == 4:
            m1 = F.tanh(self.m1(x))
            j1 = F.sigmoid(self.j1(x))
            m2 = F.tanh(self.m2(x))
            j2 = F.sigmoid(self.j2(x))
            return torch.cat((m1, j1, m2, j2)).reshape(-1, 4)

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc_units= [1024, 512], action_cat_layer=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (list(int)): Number of nodes in hidden_layers
            action_cat_layer (int): Index of hidden layers to concatenate actions
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_cat_layer = action_cat_layer
        self.bn_input = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc_units[0])
        self.layers = nn.ModuleList([nn.Linear(fc_units[i] + (action_size if i == action_cat_layer else 0), fc_units[i+1]) for i in range(len(fc_units) - 1)])
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(fc_units[i] + (action_size if i == action_cat_layer else 0) )for i in range(len(fc_units) - 1)])
        self.output = nn.Linear(fc_units[-1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        for l in self.layers:
            l.weight.data.uniform_(*hidden_init(l))
        self.output.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        state = state.reshape(-1, self.state_size)
        x = self.bn_input(state)
        x = F.relu(self.fc1(x))
        for i, l in enumerate(self.layers):
            if i == self.action_cat_layer:
                x = torch.cat((x, action), dim=1)
            x = self.bn_layers[i](x)
            x = F.relu(l(x))
        return self.output(x)
