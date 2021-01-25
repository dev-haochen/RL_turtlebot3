#!/usr/bin/env python

import random
import numpy as np
from collections import namedtuple, deque 

import torch
from torch import device
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

DEVICE = device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_unit=64, fc2_unit=64):
        super(QNetwork, self).__init__()
        self._seed = torch.manual_seed(seed)
        self._fc1 = nn.Linear(state_size, fc1_unit)
        self._fc2 = nn.Linear(fc1_unit, fc2_unit)
        self._fc3 = nn.Linear(fc2_unit, action_size)

    def forward(self, state):
        x = F.relu(self._fc1(state))
        x = F.relu(self._fc2(x))
        return self._fc3(x)

class ReplayBuffer():
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self._action_size = action_size
        self._batch_size = batch_size
        self._buffer = deque(buffer_size)
        self._seed = random.seed(seed)
        self._experience = namedtuple("Experience", "state action reward next_state done")

    def __len__(self):
        return len(self._buffer)

    def __repr__(self):
        return

    def add(self, state, action, reward, next_state, done):
        exp = self._experience(state, action, reward, next_state, done)
        self._buffer.append(exp)

    def sample(self):
        random_experiences = random.sample(self._buffer, k=self._batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for exp in random_experiences:
            states.append(exp.state)
            actions.append(exp.action)
            rewards.append(exp.reward)
            next_states.append(exp.next_state)
            dones.append(exp.done)

        states = torch.from_numpy(np.vstack(states)).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack(actions)).float().to(DEVICE)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack(dones)).float().to(DEVICE)

        return(states, actions, rewards, next_states, dones)

    def save(self):
        pass

    def load(self):
        pass

class QLearningAgent():
    
    def __init__(self, state_size, action_size, epsilon, lr, gamma, buffer_size=10000, batch_size=16, seed=0):
        
        self._state_size = state_size
        self.action_size = action_size

        self.qnet_local = QNetwork(state_size, action_size, seed).to(DEVICE)
        self.qnet_target = QNetwork(state_size, action_size, seed).to(DEVICE)

        self.optimizer = optim
        
        self._memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)

    def learn(self):
        pass

    def choose_action(self):
        pass

    def load(self):
        pass

    def save(self):
        pass


if __name__ == '__main__':
    pass