import torch
from config import *
from networks import BayesianNetwork, MLP, MLP_Dropout

def load_bnn_class_model(saved_model):
    '''Load model weights from path saved_model.'''
    config = ClassConfig

    model_params = {
        'input_shape': config.x_shape,
        'classes': config.classes,
        'batch_size': config.batch_size,
        'hidden_units': config.hidden_units,
        'mode': config.mode,
        'mu_init': config.mu_init,
        'rho_init': config.rho_init,
        'prior_init': config.prior_init,
        'mixture_prior': config.mixture_prior
    }
    model = BayesianNetwork(model_params)
    model.load_state_dict(torch.load(saved_model))
    model.eval()
    return model

bnn_model = load_bnn_class_model('./saved_models/bnn_classification_model.pt')


