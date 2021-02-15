import torch
import numpy as np
from net import BayesianNetwork
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
        self.epsilon = bandit_params['epsilon'] if 'epsilon' in bandit_params else 0
        self.cumulative_regrets = [0]
        self.buffer_x, self.buffer_y = [], []
        self.x, self.y = torch.FloatTensor(x), torch.FloatTensor(y)   # pass by pointer
        self.init_net()
    
    def init_net(self):
        ####
        # raise NotIMplementedError
        ####
        model_params = {
            'input_shape': self.x.shape[1]+2,
            'classes': 1 if len(self.y.shape)==1 else self.y.shape[1],
            'num_batches': self.num_batches,
            'batch_size': self.batch_size
        }
        print("BNN Parameters: ")
        print(model_params)
        self.net = BayesianNetwork(model_params).to(device)
        self.optimiser = torch.optim.Adam(self.net.parameters(), lr=self.lr)

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
        eat_tuple = torch.FloatTensor(torch.cat((context, torch.FloatTensor([1, 0])))).unsqueeze(0).to(device)
        reject_tuple = torch.FloatTensor(torch.cat((context, torch.FloatTensor([0, 1])))).unsqueeze(0).to(device)

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
        ####
        # raise not implemented error
        ####
        beta = 2 ** (64 - (batch_id + 1)) / (2 ** self.num_batches - 1) 
        self.net.train()
        self.net.zero_grad()
        loss_info = self.net.sample_elbo(x, y, beta, self.n_samples)
        if batch_id == 63:
            print(loss_info)
        net_loss = loss_info[0]
        net_loss.backward()
        self.optimiser.step()
        return loss_info