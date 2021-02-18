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
        epsilon = self.normal.sample(self.rho.size()).to(device)
        return self.mu + self.sigma * epsilon
    
    def log_prob(self, input):
        ## :~
        return (-math.log(math.sqrt(2 * math.pi)) - torch.log(self.sigma) - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()

class BayesianLinear(nn.Module):
    ''' FC Layer with Bayesian Weights '''
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))  # variance initialised between (-0.2, 0.2)
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))    # mean initialised between (-5, -4)
        self.weight = GaussianNode(self.weight_mu, self.weight_rho)

        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))
        self.bias = GaussianNode(self.bias_mu, self.bias_rho)

        self.weight_prior = ScaleMixtureGaussian(0.5, math.exp(-0), math.exp(-6))   # prior mean 0.5, sigma1 = exp(-0), sigma2 = exp(-6)
        self.bias_prior = ScaleMixtureGaussian(0.5, math.exp(-0), math.exp(-6))
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
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return nn.functional.linear(input, weight, bias)

class BayesianNetwork(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.input_shape = model_params['input_shape']
        self.classes = model_params['classes']
        self.batch_size = model_params['batch_size']
        self.num_batches = model_params['num_batches']

        self.l1 = BayesianLinear(self.input_shape, 100)
        self.l1_act = nn.ReLU()
        self.l2 = BayesianLinear(100, 100)
        self.l2_act = nn.ReLU()
        self.l3 = BayesianLinear(100, self.classes)
        # self.l3_act = nn.LogSoftmax(dim=1)   # Log Softmax to work with NLLLoss ## dim=1? ## do not need softmax since we are trying to estimate reward
    
    def forward(self, x, sample=False):
        assert len(x.shape) == 2, "Input dimensions incorrect, expected shape = (batch_size, sample,...)"
        x = self.l1_act(self.l1(x, sample))
        x = self.l2_act(self.l2(x, sample))
        # x = self.l3_act(self.l3(x, sample))
        x = self.l3(x, sample)
        return x

    def log_prior(self):
        return self.l1.log_prior + self.l2.log_prior + self.l3.log_prior
    
    def log_variational_posterior(self):
        return self.l1.log_variational_posterior + self.l2.log_variational_posterior + self.l3.log_variational_posterior

    def get_nll(self, outputs, target, sigma=1.):
        ll = torch.distributions.Normal(outputs, sigma).log_prob(target).sum()
        return -ll
        # nll = nn.functional.mse_loss(outputs, target, reduction='sum')
        # return torch.div(nll, (2*sigma**2)) + torch.log(torch.tensor([sigma])).to(device)
        # return nll

    def sample_elbo(self, input, target, beta, samples):
        outputs = torch.zeros(samples, self.batch_size, self.classes).to(device)
        log_priors = torch.zeros(samples).to(device)
        log_variational_posteriors = torch.zeros(samples).to(device)
        for i in range(samples):
            outputs[i] = self.forward(input, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        log_prior = beta*log_priors.mean()
        log_variational_posterior = beta*log_variational_posteriors.mean()
        negative_log_likelihood = self.get_nll(outputs.mean(0).squeeze(), target)
        loss = log_variational_posterior - log_prior + negative_log_likelihood
        return loss, log_priors.mean(), log_variational_posteriors.mean(), negative_log_likelihood

class MLP(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.input_shape = model_params['input_shape']
        self.classes = model_params['classes']
        self.batch_size = model_params['batch_size']
        self.num_batches = model_params['num_batches']

        self.net = nn.Sequential(
            nn.Linear(self.input_shape, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, self.classes))
    
    def forward(self, x):
        assert len(x.shape) == 2, "Input dimensions incorrect, expected shape = (batch_size, sample,...)"
        x = self.net(x)
        return x

