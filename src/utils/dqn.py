#!/usr/bin/env python

import random
from gym.core import ActionWrapper
import numpy as np
from collections import namedtuple, deque 

import torch
from torch import device
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

DEVICE = device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_unit=256, fc2_unit=64):
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
        self._buffer = deque(maxlen=buffer_size)
        self._seed = random.seed(seed)
        self._experience = namedtuple("Experience", "state action reward next_state")

    def __len__(self):
        return len(self._buffer)

    def __repr__(self):
        return

    def add(self, state, action, reward, next_state):
        exp = self._experience(state, action, reward, next_state)
        self._buffer.append(exp)

    def sample(self):
        random_experiences = random.sample(self._buffer, k=self._batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []

        for exp in random_experiences:
            states.append(exp.state)
            actions.append(exp.action)
            rewards.append(exp.reward)
            next_states.append(exp.next_state)

        states = torch.tensor(states, dtype=torch.float, device=DEVICE)
        actions = torch.tensor(actions, dtype=torch.int64, device=DEVICE).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.int, device=DEVICE).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float, device=DEVICE)

        return(states, actions, rewards, next_states)

class QRunningAgent():
    def __init__(self, path):

        try:
            checkpoint = torch.load(path)
            self._action_size = checkpoint['action_size']
            self._state_size = checkpoint['state_size']
            self._state_dict = checkpoint['local_model']
        except:
            print('Something went wrong during loading the models.')
            raise

        self._qnet_local = QNetwork(self._state_size, self._action_size, 0).to(DEVICE)
        self._qnet_local.load_state_dict(checkpoint['local_model'])

    def choose_action(self, state):

        state_tens = torch.tensor(state, device=DEVICE, dtype=torch.float)
        self._qnet_local.eval()
        with torch.no_grad():
            pred_q_values = self._qnet_local(state_tens.unsqueeze(0))
        self._qnet_local.train()
        return torch.argmax(pred_q_values).item()

class QLearningAgent():
    
    def __init__(self, state_size, action_size, epsilon, lr, gamma, buffer_size=100000, batch_size=64, seed=0):
        
        self._state_size = state_size
        self._action_size = action_size

        self._gamma = gamma
        self._eqsilon = epsilon
        self._lr = lr

        self._seed = seed
        self._buffer_size = buffer_size
        self._batch_size = batch_size

        # create deep q networks
        self._qnet_local = QNetwork(state_size, action_size, seed).to(DEVICE)
        self._qnet_target = QNetwork(state_size, action_size, seed).to(DEVICE)

        # create optimizer
        self._optimizer = optim.Adam(self._qnet_local.parameters(), lr=lr)
        
        # create replay buffer 
        self._memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)

    @property
    def epsilon(self):
        
        return self._eqsilon

    @epsilon.setter
    def epsilon(self, eps):
        
        if isinstance(eps, float):
            if not (0.0 < eps < 1.0):
                raise ValueError('Epsilon value must be in range 0.0 and 1.0.')
        else:
            raise TypeError('The provided argument must be of type float.')

        self._eqsilon = eps

    @property
    def gamma(self):

        return self._gamma

    @property
    def lr(self):

        return self._lr

    def _do_learning(self, experiences):
        
        states, actions, rewards, next_states = experiences

        self._qnet_local.train()
        self._qnet_target.eval()

        q_values = self._qnet_local(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self._qnet_target(next_states).max(1)[0].unsqueeze(1)
            
        expected_q_values = rewards + self._gamma * next_q_values

        # calculate MSE loss
        loss = F.mse_loss(q_values, expected_q_values).to(DEVICE)
        
        self._optimizer.zero_grad() # clear param in optimizer
        loss.backward() # compute new param
        self._optimizer.step() # update param in _qnet_local
        self._soft_update() # update param in _qnet_target with soft update strategy

    def _soft_update(self, tau=1e-3):

        for target_param, local_param in zip(self._qnet_target.parameters(), self._qnet_local.parameters()):
            target_param.data.copy_(tau*local_param + (1-tau)*target_param.data)

    def learn(self, state, action, reward, next_state, threshold=64):
        
        self._memory.add(state, action, reward, next_state)

        if len(self._memory) > threshold:
            exp = self._memory.sample()
            self._do_learning(exp)

    def choose_action(self, state):

        if random.random() >  self._eqsilon:
            state_tens = torch.tensor(state, device=DEVICE, dtype=torch.float)
            self._qnet_local.eval()
            with torch.no_grad():
                pred_q_values = self._qnet_local(state_tens.unsqueeze(0))
            self._qnet_local.train()
            ret = (torch.argmax(pred_q_values).item(), 'predicted')
        else:
            ret = (random.choice(range(self._action_size)), 'random choice')

        return ret

    def load(self, path):
        
        try:
            checkpoint = torch.load(path)
            self._gamma = checkpoint['gamma']
            self._eqsilon = checkpoint['epsilon']
            self._lr = checkpoint['learning_rate']
            self._action_size = checkpoint['action_size']
            self._state_size = checkpoint['state_size']
            self._seed = checkpoint['seed']
        except:
            print('Something went wrong during loading the models.')
            raise

        self._qnet_local = QNetwork(self._state_size, self._action_size, self._seed).to(DEVICE)
        self._qnet_local.load_state_dict(checkpoint['local_model'])
        self._qnet_target = QNetwork(self._state_size, self._action_size, self._seed).to(DEVICE)
        self._qnet_local.load_state_dict(checkpoint['target_model'])

        self._optimizer = optim.Adam(self._qnet_local.parameters(), lr=self._lr)
        self._optimizer.load_state_dict(checkpoint['optimizer'])

    def save(self, path):
        
        try:
            torch.save({
                'local_model': self._qnet_local.state_dict(),
                'target_model': self._qnet_target.state_dict(),
                'optimizer': self._optimizer.state_dict(),
                'epsilon': self._eqsilon,
                'gamma': self._gamma,
                'learning_rate': self._lr,
                'action_size' : self._action_size,
                'state_size' : self._state_size,
                'seed' : self._seed,
                'buffer_size': self._buffer_size,
                'batch_size': self._batch_size
            }, path)
        except:
            print('Something went wrong during saving the models.')
            raise

if __name__ == '__main__':
    pass