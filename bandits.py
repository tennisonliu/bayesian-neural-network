import torch
import numpy as np
from networks import BayesianNetwork, MLP
from base_bandit import MushroomBandit
from torch.utils.tensorboard import SummaryWriter
from config import *
from utils import *

class BNN_Bandit(MushroomBandit):
    def __init__(self, label, *args):
        super().__init__(*args)
        self.writer = SummaryWriter(comment=f"_{label}_agent_training")
    
    def init_net(self):
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

    def loss_step(self, x, y, batch_id):
        beta = 2 ** (64 - (batch_id + 1)) / (2 ** self.num_batches - 1) 
        self.net.train()
        self.net.zero_grad()
        loss_info = self.net.sample_elbo(x, y, beta, self.n_samples)
        # if batch_id == 0:
        #     print(loss_info)
        net_loss = loss_info[0]
        net_loss.backward()
        self.optimiser.step()
        return loss_info

    def log_progress(self, step):
        write_weight_histograms(self.writer, self.net, step)
        write_loss_scalars(self.writer, self.loss_info, self.cumulative_regrets[-1], step)

class Greedy_Bandit(MushroomBandit):
    def __init__(self, epsilon, label, *args):
        super().__init__(*args)
        self.epsilon = epsilon
        self.writer = SummaryWriter(comment=f"_{label}_agent_training"),
    
    def init_net(self):
        model_params = {
            'input_shape': self.x.shape[1]+2,
            'classes': 1 if len(self.y.shape)==1 else self.y.shape[1],
            'num_batches': self.num_batches,
            'batch_size': self.batch_size
        }
        print("MLP Parameters: ")
        print(model_params)
        self.net = MLP(model_params).to(device)
        self.optimiser = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def loss_step(self, x, y, batch_id):
        self.net.train()
        self.net.zero_grad()
        net_loss = torch.nn.functional.mse_loss(self.net(x).squeeze(), y, reduction='sum')
        net_loss.backward()
        self.optimiser.step()
        return net_loss

    def log_progress(self, step):
        write_loss(self.writer[0], self.loss_info, self.cumulative_regrets[-1], step)