'''
Components required to define Bayesian Neural Network
-> Gaussian mixture prior + variational Gaussian posterior
-> Bayesian fully connected layer
-> Bayesian Neural Network
-> Non-Bayesian MLP
-> MLP with MC Dropout
'''
import torch
import math
from torch import nn
from config import *

class ScaleMixtureGaussian:
    ''' Scale Mxiture Prior '''
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0,sigma1)
        self.gaussian2 = torch.distributions.Normal(0,sigma2)

    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1-self.pi) * prob2)).sum()

class GaussianNode:
    ''' Stochastic Weight Nodes '''
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0,1)
    
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(DEVICE)
        return self.mu + self.sigma * epsilon
    
    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi)) - torch.log(self.sigma) - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()

class BayesianLinear(nn.Module):
    ''' FC Layer with Bayesian Weights '''
    def __init__(self, in_features, out_features, mu_init, rho_init, prior_init, mixture_prior=True):
        super().__init__()

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(*mu_init))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(*rho_init))
        self.weight = GaussianNode(self.weight_mu, self.weight_rho)

        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(*mu_init))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(*rho_init))
        self.bias = GaussianNode(self.bias_mu, self.bias_rho)

        if mixture_prior:
            assert len(prior_init)==3, "Scale Mixture Prior requires three values in prior initialisation"
            self.weight_prior = ScaleMixtureGaussian(prior_init[0], math.exp(prior_init[1]), math.exp(prior_init[2]))
            self.bias_prior = ScaleMixtureGaussian(prior_init[0], math.exp(prior_init[1]), math.exp(prior_init[2]))
        else:
            assert len(prior_init)==1, "Gaussian Prior requires one value in prior initialisation"
            self.weight_prior = torch.distributions.Normal(0, prior_init[0])
            self.bias_prior = torch.distributions.Normal(0, prior_init[0])

        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.training or sample: 
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu

        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight).sum() + self.bias_prior.log_prob(bias).sum()
            self.log_variational_posterior = self.weight.log_prob(weight).sum() + self.bias.log_prob(bias).sum()

        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return nn.functional.linear(input, weight, bias)

class BayesianLinearLR(nn.Module):
    ''' FC Layer with Bayesian Weights and Local Reparameterisation '''
    def __init__(self, in_features, out_features, mu_init, rho_init, prior_init, mixture_prior=False):
        super().__init__()

        self.weight_mu = nn.Parameter(torch.Tensor(in_features, out_features).uniform_(*mu_init))
        self.weight_rho = nn.Parameter(torch.Tensor(in_features, out_features).uniform_(*rho_init))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(*mu_init))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(*rho_init))
        self.normal = torch.distributions.Normal(0,1)

        assert len(prior_init)==1, "Gaussian Prior requires one value in prior initialisation"
        self.weight_prior = [0, prior_init[0]]
        self.bias_prior = [0, prior_init[0]]
        self.weight_kl_cost = 0
        self.bias_kl_cost = 0
        self.kl_cost = 0

    def compute_kl_cost(self, p_params, q_params):
        ''' Compute closed-form KL Divergence between two Gaussians '''
        [p_mu, p_sigma] = p_params
        [q_mu, q_sigma] = q_params
        kl_cost = 0.5 * (2*torch.log(p_sigma/q_sigma) - 1 + (q_sigma/p_sigma).pow(2) + ((p_mu - q_mu)/p_sigma).pow(2)).sum()
        return kl_cost

    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.training or sample: 
            w_sigma = torch.log1p(torch.exp(self.weight_rho))
            b_sigma = torch.log1p(torch.exp(self.bias_rho))
            activation_mu = torch.mm(input, self.weight_mu)
            activation_sigma = torch.sqrt(torch.mm(input.pow(2), w_sigma.pow(2)))

            w_epsilon = self.normal.sample(activation_sigma.size()).to(DEVICE)
            b_epsilon = self.normal.sample(b_sigma.size()).to(DEVICE)
            activation_w = activation_mu + activation_sigma * w_epsilon
            activation_b = self.bias_mu + b_sigma * b_epsilon

            activation =  activation_w + activation_b.unsqueeze(0).expand(input.shape[0], -1)

        else:
            activation = torch.mm(input, self.weight_mu) + self.b_mu

        if self.training or calculate_log_probs:
            self.weight_kl_cost = self.compute_kl_cost(self.weight_prior, [self.weight_mu, w_sigma]).sum()
            self.bias_kl_cost = self.compute_kl_cost(self.bias_prior, [self.bias_mu, b_sigma]).sum()
            self.kl_cost = self.weight_kl_cost + self.bias_kl_cost

        return activation

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
        self.local_reparam = model_params['local_reparam']

        if self.local_reparam:
            layer = BayesianLinearLR
        else:
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

    def sample_elbo_lr(self, input, target, beta, samples, sigma=1.):
        ''' Sample ELBO for BNN w/ Local Reparameterisation '''
        assert self.local_reparam==True, 'sample_elbo_lr() method returns loss for BNNs with local reparameterisation, alternatively use sample_elbo()'
        kl_costs = torch.zeros(samples).to(DEVICE)
        negative_log_likelihood = torch.zeros(1).to(DEVICE)

        for i in range(samples):
            output = self.forward(input, sample=True)
            kl_costs[i] = self.kl_cost()
            negative_log_likelihood += self.get_nll(output, target, sigma)

        kl_cost = beta*kl_costs.mean()
        negative_log_likelihood = negative_log_likelihood / samples
        loss = kl_cost + negative_log_likelihood
        return loss, kl_costs.mean(), negative_log_likelihood

class MLP(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.input_shape = model_params['input_shape']
        self.classes = model_params['classes']
        self.batch_size = model_params['batch_size']
        self.hidden_units = model_params['hidden_units']
        self.mode = model_params['mode']

        self.net = nn.Sequential(
            nn.Linear(self.input_shape, self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units, self.classes))
    
    def forward(self, x):
        if self.mode == 'classification':
            assert len(x.shape) == 4, "Input dimensions incorrect, expected shape = (batch_size, sample, x_dim[0], x_dim[1])"
            x = x.view(-1, self.input_shape) # Flatten images
        else:    
            assert len(x.shape) == 2, "Input dimensions incorrect, expected shape = (batch_size, sample,...)"
        
        x = self.net(x)
        return x

class MLP_Dropout(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.input_shape = model_params['input_shape']
        self.classes = model_params['classes']
        self.batch_size = model_params['batch_size']
        self.hidden_units = model_params['hidden_units']
        self.mode = model_params['mode']

        self.net = nn.Sequential(
            nn.Linear(self.input_shape, self.hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_units, self.classes))
    
    def forward(self, x):
        if self.mode == 'classification':
            assert len(x.shape) == 4, "Input dimensions incorrect, expected shape = (batch_size, sample, x_dim[0], x_dim[1])"
            x = x.view(-1, self.input_shape) # Flatten images
        else:    
            assert len(x.shape) == 2, "Input dimensions incorrect, expected shape = (batch_size, sample,...)"
        
        x = self.net(x)
        return x

    def enable_dropout(self):
        ''' Enable the dropout layers during test-time '''
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

