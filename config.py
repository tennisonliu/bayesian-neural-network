'''
Configuration file, defines parameters for RL (RLConfig), regression (RegConfig) and classification (ClassConfig)
'''
import torch

global DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RegConfig:
    save_dir = './saved_models'
    train_size = 1024
    batch_size = 128
    lr = 1e-3
    epochs = 1000
    train_samples = 5                   # number of train samples for MC gradients
    test_samples = 10                   # number of test samples for MC averaging
    num_test_points = 400               # number of test points
    mode = 'regression'
    mixture_prior = False               # whether to use mixture prior, unconnect prior_init line to use mixture prior
    hidden_units = 400                  # number of hidden units
    noise_tolerance = .1                # log likelihood sigma
    mu_init = [-0.2, 0.2]               # range for mu 
    rho_init = [-5, -4]                 # range for rho
    # prior_init = [0.5, -0, -6]        # mixture weight, log(sigma1), log(sigma2)
    prior_init = [1]
    regression_clusters = False         # generate synthetic data with gap between data clusters

class RLConfig:
    data_dir = 'data/agaricus-lepiota.data'
    batch_size = 64
    num_batches = 64
    buffer_size = batch_size * num_batches  # buffer to track latest batch of mushrooms
    lr = 1e-4
    training_steps = 50000
    mode = 'regression'
    hidden_units = 100                      # number of hidden units
    mixture_prior = True
    mu_init = [-0.2, 0.2]                   # range for mu 
    rho_init = [-5, -4]                     # range for rho
    prior_init = [0.5, -0, -6]              # mixture weight, log(sigma1), log(sigma2)

class ClassConfig:
    batch_size = 128
    lr = 1e-4
    epochs = 300
    hidden_units = 1200
    mode = 'classification'
    train_samples = 2
    test_samples = 10
    x_shape = 28 * 28                       # input shape
    classes = 10                            # number of output classes
    mu_init = [-0.2, 0.2]                   # range for mu 
    rho_init = [-5, -4]                     # range for rho
    #prior_init = [0.5, -0, -8]             # mixture weight, log(sigma1), log(sigma2)
    prior_init = [1.]
    mixture_prior = False
    save_dir = './saved_models'
    local_reparam = True