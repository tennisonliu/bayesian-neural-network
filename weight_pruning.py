'''
Weight pruning experiments.
''' 
import torch
import numpy as np
import matplotlib.pyplot as plt
from load_model_utils import *
import seaborn as sns

def collect_weights(model, bnn=False):
    '''Collect all weights from model in a list'''
    mus = []
    rhos = []
    weights = []
    for name, param in model.named_parameters():
        if 'mu' in name:
            mus.append(param.flatten().tolist())
        elif 'rho' in name:
            rhos.append(param.flatten().tolist())
        else:
            weights.append(param.flatten().tolist())
    
    # flatten nested lists
    mus = [item for sublist in mus for item in sublist]
    rhos = [item for sublist in rhos for item in sublist]
    weights = [item for sublist in weights for item in sublist]

    if bnn:
        sigmas = [rho_to_sigma(rho) for rho in rhos]
        weights = [mus, sigmas]

    return weights

def rho_to_sigma(rho): 
    return np.log(1 + np.exp(rho))

def sample_bnn_weights(mu, sigma):
    return np.random.normal(mu, sigma)

def plot_histogram(weights_list, labels):
    fig = plt.gcf()   
    fig.set_size_inches(5, 3.5) 
    for weights, label in zip(weights_list, labels):
        plt.hist(weights, alpha=0.5, density=True, label=label, bins=100)
    plt.legend()
    plt.savefig('./graphs/weights_histogram.png')
    plt.close()

def snr_plots(snr):
    # density plot
    fig = plt.gcf()  
    fig.set_size_inches(5, 4) 
    sns.kdeplot(data=snr, alpha=0.5, fill=True)
    plt.xlabel('Signal-to-noise ratio')
    plt.savefig('./graphs/snr_density.png')
    plt.close()

    # cdf plot
    fig = plt.gcf()   
    fig.set_size_inches(5, 4) 
    sns.kdeplot(data=snr, cumulative=True, fill=False)
    plt.xlabel('Signal-to-noise ratio')
    plt.savefig('./graphs/snr_cdf.png')
    plt.close()

def compute_snr(mu, sigma):
    '''Compute signal-to-noise ratio in decibels'''
    return 10*np.log10(abs(mu)/sigma)

def prune_weights(model, weights, snr, drop_percentage=0.5):
    '''Remove weights with the lowest SNR'''
    # convert percentage to snr threshold
    snr_threshold = np.percentile(weights, 100*drop_percentage)

    # TODO: complete weight pruning
    #for module in model.modules():
    #    with torch.no_grad():
    #        model.module.weight[34].zero_()
    pass

def main():
    # load models
    bnn_model = load_bnn_class_model('./saved_models/bnn_classification_model.pt')
    #mlp_model = load_mlp_class_model('./saved_models/mlp_classification_model.pt')
    #dropout_model = load_dropout_class_model('./saved_models/dropout_classification_model.pt')
       
    # collect weights
    bnn_mus, bnn_sigmas = collect_weights(bnn_model, bnn=True)
    bnn_weights = [sample_bnn_weights(mu, sigma) for mu, sigma in zip(bnn_mus, bnn_sigmas)]
    #mlp_weights = collect_weights(mlp_model)
    #dropout_weights = collect_weights(dropout_model)
    
    # create histogram
    plot_histogram(
        [bnn_weights], #[bnn_weights, mlp_weights, dropout_weights], 
        ['BNN']) #['BNN', 'Vanilla SGD', 'Dropout'])

    # plot snr densities
    snr = [compute_snr(mu, sigma) for mu, sigma in zip(bnn_mus, bnn_sigmas)]
    snr_plots(snr)

if __name__=='__main__':
    main()



