'''
Defines base class for contextual bandits
'''

import torch
import numpy as np

import sys
sys.path.append('../')
from config import DEVICE

class Bandit():
    def __init__(self, label, bandit_params, x, y):
        self.n_samples = bandit_params['n_samples']
        self.buffer_size = bandit_params['buffer_size']
        self.batch_size = bandit_params['batch_size']
        self.num_batches = bandit_params['num_batches']
        self.lr = bandit_params['lr']
        self.epsilon = bandit_params['epsilon']
        self.hidden_units = bandit_params['hidden_units']
        self.mode = bandit_params['mode']
        self.cumulative_regrets = [0]
        self.buffer_x, self.buffer_y = [], []
        self.x, self.y = x, y
        self.label = label
        self.init_net()

    def get_agent_reward(self, eaten, edible):
        if not eaten:
            return 0
        if eaten and edible:
            return 5
        elif eaten and not edible:
            return 5 if np.random.rand() > 0.5 else -35

    def get_oracle_reward(self, edible):
        return 5*edible 

    def take_action(self, mushroom):
        context, edible = self.x[mushroom], self.y[mushroom]
        eat_tuple = torch.FloatTensor(np.concatenate((context, [1, 0]))).unsqueeze(0).to(DEVICE)
        reject_tuple = torch.FloatTensor(np.concatenate((context, [0, 1]))).unsqueeze(0).to(DEVICE)

        # evaluate reward for actions
        with torch.no_grad():
            self.net.eval()
            reward_eat = sum([self.net(eat_tuple) for _ in range(self.n_samples)]).item()
            reward_reject = sum([self.net(reject_tuple) for _ in range(self.n_samples)]).item()

        eat = reward_eat > reward_reject
        # epsilon-greedy agent
        if np.random.rand() < self.epsilon:
            eat = (np.random.rand() < 0.5)
        agent_reward = self.get_agent_reward(eat, edible)

        # record context, action, reward
        action = torch.Tensor([1, 0] if eat else [0, 1])
        self.buffer_x.append(np.concatenate((context, action)))
        self.buffer_y.append(agent_reward)

        # calculate regret
        regret = self.get_oracle_reward(edible) - agent_reward
        self.cumulative_regrets.append(self.cumulative_regrets[-1]+regret)

    def update(self, mushroom):
        self.take_action(mushroom)
        l = len(self.buffer_x)

        if l <= self.batch_size:
            idx_pool = int(self.batch_size//l + 1)*list(range(l))
            idx_pool = np.random.permutation(idx_pool[-self.batch_size:])
        elif l > self.batch_size and l < self.buffer_size:
            idx_pool = int(l//self.batch_size)*self.batch_size
            idx_pool = np.random.permutation(list(range(l))[-idx_pool:])
        else:
            idx_pool = np.random.permutation(list(range(l))[-self.buffer_size:])

        context_pool = torch.Tensor([self.buffer_x[i] for i in idx_pool]).to(DEVICE)
        value_pool = torch.Tensor([self.buffer_y[i] for i in idx_pool]).to(DEVICE)
        
        for i in range(0, len(idx_pool), self.batch_size):
            self.loss_info = self.loss_step(context_pool[i:i+self.batch_size], value_pool[i:i+self.batch_size], i//self.batch_size)

    def init_net(self):
        raise NotImplementedError

    def loss_step(self, x, y, batch_id):
        raise NotImplementedError

    def log_progress(self, step):
        raise NotImplementedError