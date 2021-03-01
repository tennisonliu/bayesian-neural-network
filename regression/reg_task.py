'''
Two Regression Models
1) BNN -> BNN_Regression
2) MLP -> MLP_Regression
'''
import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('../')
from logger_utils import *
from networks import BayesianNetwork, MLP
from config import DEVICE
from plotting import create_regression_plot

class BNN_Regression():
    def __init__(self, label, parameters):
        super().__init__()
        self.writer = SummaryWriter(comment=f"_{label}_training")
        self.label = label
        self.batch_size = parameters['batch_size']
        self.num_batches = parameters['num_batches']
        self.n_samples = parameters['num_training_samples']
        self.x_shape = parameters['x_shape']
        self.y_shape = parameters['y_shape']
        self.noise_tol = parameters['noise_tolerance']
        self.lr = parameters['lr']
        self.save_model_path = f'{parameters["save_dir"]}/{label}_model.pt'
        self.best_loss = np.inf
        self.init_net(parameters)
    
    def init_net(self, parameters):
        if not os.path.exists(parameters["save_dir"]):
            os.makedirs(parameters["save_dir"])

        model_params = {
            'input_shape': self.x_shape,
            'classes': self.y_shape,
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
        print(f'Regression Task {self.label} Parameters: ')
        print(f'number of samples: {self.n_samples}, noise tolerance: {self.noise_tol}')
        print("BNN Parameters: ")
        print(f'batch size: {self.batch_size}, input shape: {model_params["input_shape"]}, hidden units: {model_params["hidden_units"]}, output shape: {model_params["classes"]}, lr: {self.lr}')

    def train_step(self, train_data):
        self.net.train()
        for idx, (x, y) in enumerate(train_data):
            beta = 2 ** (self.num_batches - (idx + 1)) / (2 ** self.num_batches - 1) 
            x, y = x.to(DEVICE), y.to(DEVICE)
            self.net.zero_grad()
            self.loss_info = self.net.sample_elbo(x, y, beta, self.n_samples, sigma=self.noise_tol)
            net_loss = self.loss_info[0]
            net_loss.backward()
            self.optimiser.step()
        self.epoch_loss = net_loss.item()

    def evaluate(self, x_test, n_samples):
        self.net.eval()
        with torch.no_grad():
            y_test = np.zeros((n_samples, x_test.shape[0]))
            for s in range(n_samples):
                tmp = self.net(x_test.to(DEVICE)).detach().cpu().numpy()
                y_test[s,:] = tmp.reshape(-1)
            return y_test

    def log_progress(self, step):
        write_weight_histograms(self.writer, self.net, step)
        write_loss_scalars(self.writer, self.loss_info, step)

    def evaluate(self, X_test, samples, train_ds):
        self.net.eval()
        y_test = np.zeros((samples, X_test.shape[0]))
        for s in range(samples):
            tmp = self.net(X_test.to(DEVICE)).detach().cpu().numpy()
            y_test[s,:] = tmp.reshape(-1)
        create_regression_plot(X_test.cpu().numpy(), y_test, train_ds, 'bnn')

class MLP_Regression():
    def __init__(self, label, parameters):
        super().__init__()
        self.writer = SummaryWriter(comment=f"_{label}_training")
        self.label = label
        self.lr = parameters['lr']
        self.hidden_units = parameters['hidden_units']
        self.mode = parameters['mode']
        self.batch_size = parameters['batch_size']
        self.num_batches = parameters['num_batches']
        self.x_shape = parameters['x_shape']
        self.y_shape = parameters['y_shape']
        self.save_model_path = f'{parameters["save_dir"]}/{label}_model.pt'
        self.best_loss = np.inf
        self.init_net(parameters)
    
    def init_net(self, parameters):
        if not os.path.exists(parameters["save_dir"]):
            os.makedirs(parameters["save_dir"])

        model_params = {
            'input_shape': self.x_shape,
            'classes': self.y_shape,
            'batch_size': self.batch_size,
            'hidden_units': self.hidden_units,
            'mode': self.mode
        }
        self.net = MLP(model_params).to(DEVICE)
        self.optimiser = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimiser, step_size=5000, gamma=0.5)
        print("MLP Parameters: ")
        print(f'batch size: {self.batch_size}, input shape: {model_params["input_shape"]}, hidden units: {model_params["hidden_units"]}, output shape: {model_params["classes"]}, lr: {self.lr}')

    def train_step(self, train_data):
        self.net.train()
        for _, (x, y) in enumerate(train_data):
            x, y = x.to(DEVICE), y.to(DEVICE)
            self.net.zero_grad()
            self.loss_info = torch.nn.functional.mse_loss(self.net(x), y, reduction='sum')
            self.loss_info.backward()
            self.optimiser.step()

        self.epoch_loss = self.loss_info.item()

    def evaluate(self, x_test, n_samples=0):
        self.net.eval()
        with torch.no_grad():
            y_test = self.net(x_test.to(DEVICE)).detach().cpu().numpy()
            return y_test

    def log_progress(self, step):
        write_loss(self.writer, self.loss_info, step)

    def evaluate(self, X_test, samples, train_ds):
        self.net.eval()
        y_test = self.net(X_test.to(DEVICE)).detach().cpu().numpy()
        create_regression_plot(X_test.cpu().numpy(), y_test.reshape(1, -1), train_ds, 'mlp')