'''
Classification Task problems for three BNNs:
1) BBB -> BNN_Classification
2) MLP -> MLP_Classification
3) MC-Dropout -> MCDropout_Classification
'''

import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys
sys.path.append('../')
from utils.logger_utils import *
from networks import BayesianNetwork, MLP, MLP_Dropout
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
        self.test_samples = parameters['test_samples']
        self.x_shape = parameters['x_shape']
        self.classes = parameters['classes']
        self.mu_init = parameters['mu_init']
        self.rho_init = parameters['rho_init']
        self.prior_init = parameters['prior_init']
        self.mixture_prior = parameters['mixture_prior']
        self.save_model_path = f'{parameters["save_dir"]}/{label}_model.pt'
        self.local_reparam = parameters['local_reparam']
        self.best_acc = 0.
        self.init_net(parameters)
    
    def init_net(self, parameters):
        if not os.path.exists(parameters["save_dir"]):
            os.makedirs(parameters["save_dir"])

        model_params = {
            'input_shape': self.x_shape,
            'classes': self.classes,
            'batch_size': self.batch_size,
            'hidden_units': self.hidden_units,
            'mode': self.mode,
            'mu_init': self.mu_init,
            'rho_init': self.rho_init,
            'prior_init': self.prior_init,
            'mixture_prior': self.mixture_prior,
            'local_reparam': self.local_reparam
        }
        self.net = BayesianNetwork(model_params).to(DEVICE)
        self.optimiser = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimiser, step_size=100, gamma=0.5)
        print(f'Classification Task {self.label} Parameters: ')
        print(f'number of samples: {self.n_samples}')
        print("BNN Parameters: ")
        print(f'batch size: {self.batch_size}, input shape: {model_params["input_shape"]}, hidden units: {model_params["hidden_units"]}, output shape: {model_params["classes"]}, lr: {self.lr}')

    def train_step(self, train_data):
        self.net.train()
        for idx, (x, y) in enumerate(tqdm(train_data)):
            beta = 2 ** (self.num_batches - (idx + 1)) / (2 ** self.num_batches - 1) 
            x, y = x.to(DEVICE), y.to(DEVICE)
            self.net.zero_grad()
            if self.local_reparam:
                self.loss_info = self.net.sample_elbo_lr(x, y, beta, self.n_samples)
            else:
                self.loss_info = self.net.sample_elbo(x, y, beta, self.n_samples)            
            net_loss = self.loss_info[0]
            net_loss.backward()
            self.optimiser.step()

    def predict(self, X):
        probs = torch.zeros(size=[self.batch_size, self.classes]).to(DEVICE)
        for _ in torch.arange(self.test_samples):
            out = torch.nn.Softmax(dim=1)(self.net(X, sample=True))
            probs = probs + out / self.test_samples
        preds = torch.argmax(probs, dim=1)
        return preds, probs

    def evaluate(self, test_loader):
        self.net.eval()
        print('Evaluating on validation data')
        correct = 0
        total = 0

        with torch.no_grad():
            for data in tqdm(test_loader):
                X, y = data
                X, y = X.to(DEVICE), y.to(DEVICE)
                preds, _ = self.predict(X)
                total += self.batch_size
                correct += (preds == y).sum().item()
        self.acc = correct / total
        print(f'{self.label} validation accuracy: {self.acc}')    

    def log_progress(self, step):
        write_weight_histograms(self.writer, self.net, step)
        write_loss_scalars(self.writer, self.loss_info, step)
        write_acc(self.writer, self.acc, step)


class MLP_Classification():
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
        self.classes = parameters['classes']
        self.save_model_path = f'{parameters["save_dir"]}/{label}_model.pt'
        self.best_acc = 0.
        self.dropout = parameters['dropout']
        self.init_net(parameters)
    
    def init_net(self, parameters):
        if not os.path.exists(parameters["save_dir"]):
            os.makedirs(parameters["save_dir"])

        model_params = {
            'input_shape': self.x_shape,
            'classes': self.classes,
            'batch_size': self.batch_size,
            'hidden_units': self.hidden_units,
            'mode': self.mode,
            'dropout': self.dropout,
        }
        if self.dropout:
            self.net = MLP_Dropout(model_params).to(DEVICE)
            print('MLP Dropout Parameters: ')
            print(f'batch size: {self.batch_size}, input shape: {model_params["input_shape"]}, hidden units: {model_params["hidden_units"]}, output shape: {model_params["classes"]}, lr: {self.lr}')
        else:
            self.net = MLP(model_params).to(DEVICE)
            print('MLP Parameters: ')
            print(f'batch size: {self.batch_size}, input shape: {model_params["input_shape"]}, hidden units: {model_params["hidden_units"]}, output shape: {model_params["classes"]}, lr: {self.lr}')
        self.optimiser = torch.optim.SGD(self.net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimiser, step_size=100, gamma=0.5)

    def train_step(self, train_data):
        self.net.train()
        for _, (x, y) in enumerate(tqdm(train_data)):
            x, y = x.to(DEVICE), y.to(DEVICE)
            self.net.zero_grad()
            self.loss_info = torch.nn.functional.cross_entropy(self.net(x), y, reduction='sum')
            self.loss_info.backward()
            self.optimiser.step()

    def predict(self, X):
        probs = torch.nn.Softmax(dim=1)(self.net(X))
        preds = torch.argmax(probs, dim=1)
        return preds, probs

    def evaluate(self, test_loader):
        self.net.eval()
        print('Evaluating on validation data')
        correct = 0
        total = 0

        with torch.no_grad():
            for data in tqdm(test_loader):
                X, y = data
                X, y = X.to(DEVICE), y.to(DEVICE)
                preds, _ = self.predict(X)
                total += self.batch_size
                correct += (preds == y).sum().item()
        self.acc = correct / total
        print(f'{self.label} validation accuracy: {self.acc}')    

    def log_progress(self, step):
        write_loss(self.writer, self.loss_info, step)
        write_acc(self.writer, self.acc, step)

class MCDropout_Classification():
    def __init__(self, label, parameters):
        super().__init__()
        self.writer = SummaryWriter(comment=f"_{label}_training")
        self.label = label
        self.lr = parameters['lr']
        self.hidden_units = parameters['hidden_units']
        self.mode = parameters['mode']
        self.batch_size = parameters['batch_size']
        self.num_batches = parameters['num_batches']
        self.test_samples = parameters['test_samples']
        self.x_shape = parameters['x_shape']
        self.classes = parameters['classes']
        self.save_model_path = f'{parameters["save_dir"]}/{label}_model.pt'
        self.best_acc = 0.
        self.dropout = parameters['dropout']
        self.init_net(parameters)
    
    def init_net(self, parameters):
        if not os.path.exists(parameters["save_dir"]):
            os.makedirs(parameters["save_dir"])

        model_params = {
            'input_shape': self.x_shape,
            'classes': self.classes,
            'batch_size': self.batch_size,
            'hidden_units': self.hidden_units,
            'mode': self.mode,
            'dropout': self.dropout
        }
        self.net = MLP_Dropout(model_params).to(DEVICE)
        self.optimiser = torch.optim.SGD(self.net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimiser, step_size=100, gamma=0.5)
        print('MCDropout Network Parameters: ')
        print(f'batch size: {self.batch_size}, input shape: {model_params["input_shape"]}, hidden units: {model_params["hidden_units"]}, output shape: {model_params["classes"]}, lr: {self.lr}')

    def train_step(self, train_data):
        self.net.train()
        for _, (x, y) in enumerate(tqdm(train_data)):
            x, y = x.to(DEVICE), y.to(DEVICE)
            self.net.zero_grad()
            self.loss_info = torch.nn.functional.cross_entropy(self.net(x), y, reduction='sum')
            self.loss_info.backward()
            self.optimiser.step()

    def predict(self, X):
        probs = torch.zeros(size=[self.batch_size, self.classes]).to(DEVICE)
        for _ in torch.arange(self.test_samples):
            out = torch.nn.Softmax(dim=1)(self.net(X))
            probs = probs + out / self.test_samples
        preds = torch.argmax(probs, dim=1)
        return preds, probs

    def evaluate(self, test_loader):
        self.net.eval()
        self.net.enable_dropout() # Set dropout layers to train mode
        print('Evaluating on validation data')
        correct = 0
        total = 0

        with torch.no_grad():
            for data in tqdm(test_loader):
                X, y = data
                X, y = X.to(DEVICE), y.to(DEVICE)
                preds, _ = self.predict(X)
                total += self.batch_size
                correct += (preds == y).sum().item()
        self.acc = correct / total
        print(f'{self.label} validation accuracy: {self.acc}')      

    def log_progress(self, step):
        write_loss(self.writer, self.loss_info, step)
        write_acc(self.writer, self.acc, step)