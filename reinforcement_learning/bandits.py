'''
Two Contextual bandits 
1) BNN and Thompson Sampling -> BNN_Bandit
2) MLP and epsilon-greedy poligy -> Greedy_Bandit
Both derived from base_bandit.py class
'''
import torch
import numpy as np
from .base_bandit import Bandit
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('../')
from logger_utils import *
from networks import BayesianNetwork, MLP
from config import DEVICE

class BNN_Bandit(Bandit):
    def __init__(self, label, *args):
        super().__init__(label, *args)
        self.writer = SummaryWriter(comment=f"_{label}_training")
    
    def init_net(self, parameters):
        model_params = {
            'input_shape': self.x.shape[1]+2,
            'classes': 1 if len(self.y.shape)==1 else self.y.shape[1],
            'batch_size': self.batch_size,
            'hidden_units': parameters['hidden_units'],
            'mode': parameters['mode'],
            'mu_init': parameters['mu_init'],
            'rho_init': parameters['rho_init'],
            'prior_init': parameters['prior_init']
        }
        self.net = BayesianNetwork(model_params).to(DEVICE)
        self.optimiser = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimiser, step_size=5000, gamma=0.5)
        print(f'Bandit {self.label} Parameters: ')
        print(f'buffer_size: {self.buffer_size}, batch size: {self.batch_size}, number of samples: {self.n_samples}, epsilon: {self.epsilon}')
        print("BNN Parameters: ")
        print(f'input shape: {model_params["input_shape"]}, hidden units: {model_params["hidden_units"]}, output shape: {model_params["classes"]}, lr: {self.lr}')

    def loss_step(self, x, y, batch_id):
        beta = 2 ** (self.num_batches - (batch_id + 1)) / (2 ** self.num_batches - 1) 
        self.net.train()
        self.net.zero_grad()
        loss_info = self.net.sample_elbo(x, y, beta, self.n_samples)
        net_loss = loss_info[0]
        net_loss.backward()
        self.optimiser.step()
        return loss_info

    def log_progress(self, step):
        write_weight_histograms(self.writer, self.net, step)
        write_loss_scalars(self.writer, self.loss_info, step)
        self.writer.add_scalar('logs/cumulative_regret', self.cumulative_regrets[-1], step)

class Greedy_Bandit(Bandit):
    def __init__(self, label, *args):
        super().__init__(label, *args)
        self.writer = SummaryWriter(comment=f"_{label}_training"),
    
    def init_net(self):
        model_params = {
            'input_shape': self.x.shape[1]+2,
            'classes': 1 if len(self.y.shape)==1 else self.y.shape[1],
            'batch_size': self.batch_size,
            'hidden_units': self.hidden_units,
            'mode': self.mode
        }
        self.net = MLP(model_params).to(DEVICE)
        self.optimiser = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimiser, step_size=5000, gamma=0.5)
        print(f'Bandit {self.label} Parameters: ')
        print(f'buffer_size: {self.buffer_size}, batch size: {self.batch_size}, number of samples: {self.n_samples}, epsilon: {self.epsilon}')
        print("MLP Parameters: ")
        print(f'input shape: {model_params["input_shape"]}, hidden units: {model_params["hidden_units"]}, output shape: {model_params["classes"]}, lr: {self.lr}')

    def loss_step(self, x, y, batch_id):
        self.net.train()
        self.net.zero_grad()
        net_loss = torch.nn.functional.mse_loss(self.net(x).squeeze(), y, reduction='sum')
        net_loss.backward()
        self.optimiser.step()
        return net_loss

    def log_progress(self, step):
        write_loss(self.writer[0], self.loss_info, step)
        self.writer[0].add_scalar('logs/cumulative_regret', self.cumulative_regrets[-1], step)