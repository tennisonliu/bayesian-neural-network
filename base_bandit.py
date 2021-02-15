import torch
import numpy as np
from config import *

class MushroomBandit():
    def __init__(self, bandit_params, x, y):
        print("Bandit Parameters: ")
        print(bandit_params)
        self.n_samples = bandit_params['n_samples']
        self.buffer_size = bandit_params['buffer_size']
        self.batch_size = bandit_params['batch_size']
        self.num_batches = bandit_params['num_batches']
        self.lr = bandit_params['lr']
        self.epsilon = 0
        self.cumulative_regrets = [0]
        self.buffer_x, self.buffer_y = [], []
        self.x, self.y = x, y
        self.init_net()
        self.init_buffer()
    
    def init_net(self):
        raise NotImplementedError

    def init_buffer(self):
        for i in np.random.choice(range(len(self.x)), self.buffer_size):
            eat = np.random.rand() > 0.5
            action = [1, 0] if eat else [0, 1]
            self.buffer_x.append(np.concatenate((self.x[i], action)))
            self.buffer_y.append(self.get_agent_reward(eat, self.y[i]))
        print(f'Size of buffer after initialisation: {len(self.buffer_x)}, {len(self.buffer_y)}')

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
        # eat_tuple = torch.FloatTensor(torch.cat((context, torch.FloatTensor([1, 0])))).unsqueeze(0).to(device)
        # reject_tuple = torch.FloatTensor(torch.cat((context, torch.FloatTensor([0, 1])))).unsqueeze(0).to(device)
        eat_tuple = torch.FloatTensor(np.concatenate((context, [1, 0]))).unsqueeze(0).to(device)
        reject_tuple = torch.FloatTensor(np.concatenate((context, [0, 1]))).unsqueeze(0).to(device)

        # evaluate reward for actions
        with torch.no_grad():
            self.net.eval()
            reward_eat = sum([self.net(eat_tuple) for _ in range(self.n_samples)]).item()
            reward_reject = sum([self.net(reject_tuple) for _ in range(self.n_samples)]).item()
            # reward_eat = self.net(eat_tuple).item()
            # reward_reject = self.net(reject_tuple).item()

        eat = reward_eat > reward_reject
        # epsilon-greedy agent
        if np.random.rand() < self.epsilon:
            eat = (np.random.rand() < 0.5)
        agent_reward = self.get_agent_reward(eat, edible)

        # record context, action, reward
        action = torch.Tensor([1, 0] if eat else [0, 1])
        self.buffer_x.append(np.concatenate((context, action)))
        self.buffer_y.append(agent_reward)
        # self.buffer_x = torch.cat((self.buffer_x, torch.cat((context, action)).unsqueeze(0)))
        # self.buffer_y = torch.cat((self.buffer_y, torch.FloatTensor([agent_reward]).unsqueeze(0)))

        # calculate regret
        regret = self.get_oracle_reward(edible) - agent_reward
        self.cumulative_regrets.append(self.cumulative_regrets[-1]+regret)

    def update(self, mushroom):
        self.take_action(mushroom)
        l = len(self.buffer_x)
        idx_pool = range(l) if l >= self.buffer_size else ((int(self.buffer_size//l) + 1)*list(range(l)))
        idx_pool = np.random.permutation(idx_pool[-self.buffer_size:])
        context_pool = torch.Tensor([self.buffer_x[i] for i in idx_pool]).to(device)
        value_pool = torch.Tensor([self.buffer_y[i] for i in idx_pool]).to(device)
        for i in range(0, self.buffer_size, self.batch_size):
            loss_info = self.loss_step(context_pool[i:i+self.batch_size], value_pool[i:i+self.batch_size], i//self.batch_size)
    
        return loss_info, self.cumulative_regrets[-1]

    def loss_step(self, x, y, batch_id):
        raise NotImplementedError