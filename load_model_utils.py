import torch
from torch import nn
from config import *
from networks import BayesianLinear, MLP, MLP_Dropout

global DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BayesianNetwork(nn.Module):
    ''' Bayesian Neural Network '''
    def __init__(self, model_params):
        super().__init__()
        self.input_shape = model_params['input_shape']
        self.classes = model_params['classes']
        self.batch_size = model_params['batch_size']
        self.hidden_units = model_params['hidden_units']
        self.mode = model_params['mode']
        self.mu_init = model_params['mu_init']
        self.rho_init = model_params['rho_init']
        self.prior_init = model_params['prior_init']
        self.mixture_prior = model_params['mixture_prior']
        layer = BayesianLinear

        self.l1 = layer(self.input_shape, self.hidden_units, self.mu_init, self.rho_init, self.prior_init, self.mixture_prior)
        self.l1_act = nn.ReLU()
        self.l2 = layer(self.hidden_units, self.hidden_units, self.mu_init, self.rho_init, self.prior_init, self.mixture_prior)
        self.l2_act = nn.ReLU()
        self.l3 = layer(self.hidden_units, self.classes, self.mu_init, self.rho_init, self.prior_init, self.mixture_prior)
    
    def forward(self, x, sample=False):
        if self.mode == 'classification':
            x = x.view(-1, self.input_shape) # Flatten images
        x = self.l1_act(self.l1(x, sample))
        x = self.l2_act(self.l2(x, sample))
        x = self.l3(x, sample)
        return x

    def log_prior(self):
        return self.l1.log_prior + self.l2.log_prior + self.l3.log_prior
    
    def log_variational_posterior(self):
        return self.l1.log_variational_posterior + self.l2.log_variational_posterior + self.l3.log_variational_posterior

    def kl_cost(self):
        return self.l1.kl_cost + self.l2.kl_cost + self.l3.kl_cost

    def get_nll(self, outputs, target, sigma=1.):
        if self.mode == 'regression':
            nll = -torch.distributions.Normal(outputs, sigma).log_prob(target).sum()
        elif self.mode == 'classification':
            nll = nn.CrossEntropyLoss(reduction='sum')(outputs, target)
        else:
            raise Exception("Training mode must be either 'regression' or 'classification'")
        return nll

    def sample_elbo(self, input, target, beta, samples, sigma=1.):
        ''' Sample ELBO for BNN w/o Local Reparameterisation '''
        assert self.local_reparam==False, 'sample_elbo() method returns loss for BNNs without local reparameterisation, alternatively use sample_elbo_lr()'
        log_priors = torch.zeros(samples).to(DEVICE)
        log_variational_posteriors = torch.zeros(samples).to(DEVICE)
        negative_log_likelihood = torch.zeros(1).to(DEVICE)

        for i in range(samples):
            output = self.forward(input, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
            negative_log_likelihood += self.get_nll(output, target, sigma)

        log_prior = beta*log_priors.mean()
        log_variational_posterior = beta*log_variational_posteriors.mean()
        negative_log_likelihood = negative_log_likelihood / samples
        loss = log_variational_posterior - log_prior + negative_log_likelihood
        return loss, log_priors.mean(), log_variational_posteriors.mean(), negative_log_likelihood

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