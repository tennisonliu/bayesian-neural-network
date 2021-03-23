'''
Helper function to load trained models.
'''
import torch
from torch import nn
from config import *
from networks import BayesianNetwork, MLP, MLP_Dropout

def load_bnn_class_model(saved_model, local_reparam):
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
        'mixture_prior': config.mixture_prior,
        'local_reparam': local_reparam
    }
    model = BayesianNetwork(model_params)
    model.load_state_dict(torch.load(saved_model))

    return model.eval()

def load_mlp_class_model(saved_model):
    '''Load model weights from path saved_model.'''
    config = ClassConfig
    model_params = {
        'input_shape': config.x_shape,
        'classes': config.classes,
        'batch_size': config.batch_size,
        'hidden_units': config.hidden_units,
        'mode': config.mode,
        'dropout': False
    }
    model = MLP(model_params)
    model.load_state_dict(torch.load(saved_model))

    return model.eval()

def load_dropout_class_model(saved_model):
    '''Load model weights from path saved_model.'''
    config = ClassConfig
    model_params = {
        'input_shape': config.x_shape,
        'classes': config.classes,
        'batch_size': config.batch_size,
        'hidden_units': config.hidden_units,
        'mode': config.mode,
        'dropout': True
    }
    model = MLP_Dropout(model_params)
    model.load_state_dict(torch.load(saved_model))

    return model.eval()