import torch

class Config:
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = 'data/agaricus-lepiota.data'
    batch_size = 64
    num_batches = 64
    n_samples = 2
    buffer_size = batch_size * num_batches
    lr = 1e-4
    NB_STEPS = 50000