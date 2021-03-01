
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys
sys.path.append('../')
from utils import *
from networks import BayesianNetwork, MLP
from config import DEVICE

class BNN_Classification():
    def __init__(self, label, parameters):
        super().__init__()
        self.writer = SummaryWriter(comment=f"_{label}_training")
        self.label = label
        self.lr = parameters['lr']
        self.hidden_units = parameters['hidden_units']
        self.mode = parameters['mode']
        self.batch_size = parameters['batch_size']
        self.num_batches = parameters['num_batches']
        self.n_samples = parameters['train_samples']
        self.x_shape = parameters['x_shape']
        self.classes = parameters['classes']
        self.init_net()
    
    def init_net(self):
        model_params = {
            'input_shape': self.x_shape,
            'classes': self.classes,
            'batch_size': self.batch_size,
            'hidden_units': self.hidden_units,
            'mode': self.mode
        }
        self.net = BayesianNetwork(model_params).to(DEVICE)
        self.optimiser = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimiser, step_size=5000, gamma=0.5)
        print(f'Regression Task {self.label} Parameters: ')
        print(f'number of samples: {self.n_samples}')
        print("BNN Parameters: ")
        print(f'batch size: {self.batch_size}, input shape: {model_params["input_shape"]}, hidden units: {model_params["hidden_units"]}, output shape: {model_params["classes"]}, lr: {self.lr}')

    def train_step(self, train_data, test_data):
        self.net.train()
        for idx, (x, y) in enumerate(tqdm(train_data)):
            beta = 2 ** (self.num_batches - (idx + 1)) / (2 ** self.num_batches - 1) 
            x, y = x.to(DEVICE), y.to(DEVICE)
            self.net.zero_grad()
            self.loss_info = self.net.sample_elbo(x, y, beta, self.n_samples)
            net_loss = self.loss_info[0]
            net_loss.backward()
            self.optimiser.step()

    def log_progress(self, step):
        write_weight_histograms(self.writer, self.net, step)
        write_loss_scalars(self.writer, self.loss_info, step)
