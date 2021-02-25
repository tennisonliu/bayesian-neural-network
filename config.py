'''
Configuration file, defines parameters for RL (RLConfig), regression (RegConfig) and classification (ClassConfig)
'''
import torch

global DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RegConfig:
    train_size = 2048
    batch_size = 128
    lr = 1e-4
    epochs = 1000
    hidden_units = 400
    mode = 'regression'
    train_samples = 2 
    test_smaples = 100

class RLConfig:
    data_dir = 'data/agaricus-lepiota.data'
    batch_size = 64
    num_batches = 64
    buffer_size = batch_size * num_batches
    lr = 1e-4
    training_steps = 50000
    hidden_units = 100
    mode = 'regression'
