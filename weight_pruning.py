'''
Weight pruning experiments.
''' 
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.load_model_utils import *
import seaborn as sns
from networks import BayesianLinear
from utils.data_utils import create_data_class
from tqdm import tqdm
from config import DEVICE
import copy
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
    plt.style.use('seaborn-colorblind')
    fig = plt.figure(figsize=(9, 6))
    for weights, label in zip(weights_list, labels):
        sns.kdeplot(weights, label=label, fill=True, clip=[-0.3, 0.3])
    plt.xlim(-0.3, 0.3)
    plt.ylabel('Probability Density', fontsize=20)
    plt.xlabel('Weight', fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(loc=2, prop={'size': 18})
    plt.savefig('./graphs/weights_histogram.pdf', dpi=5000, format='pdf', bbox_inches ='tight', pad_inches=0.1)

def snr_plots(snr):
    # TODO : UPDATE THIS TO NEW PLOTTING FORMAT
    # density plot
    plt.style.use('seaborn-colorblind')
    fig = plt.figure(figsize=(9, 6))
    sns.kdeplot(data=snr, alpha=0.5, fill=True)
    plt.xlabel('SNR', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.savefig('./graphs/snr_density.pdf', dpi=5000, format='pdf', bbox_inches='tight', pad_inches=0.1)

    # cdf plot
    fig = plt.gcf()   
    fig.set_size_inches(5, 4) 
    sns.kdeplot(data=snr, cumulative=True, fill=False)
    plt.xlabel('SNR', fontsize=20)
    plt.ylabel('CDF', fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.savefig('./graphs/snr_cdf.pdf', dpi=5000, format='pdf', bbox_inches='tight', pad_inches=0.1)

def compute_snr(mu, sigma):
    '''Compute signal-to-noise ratio in decibels'''
    return 10*np.log10(abs(mu)/sigma)

def prune_weights(model, snrs, drop_percentage=0.5):
    '''Remove weights with the lowest SNR'''
    # convert percentage to snr threshold
    snr_threshold = np.percentile(snrs, 100*drop_percentage)

    with torch.no_grad():
        # loop through weights
        for layer in model.children():
            if isinstance(layer, BayesianLinear):
                weight_mus = layer.weight_mu.data
                weight_rhos = layer.weight_rho.data
                bias_mus = layer.bias_mu.data
                bias_rhos = layer.bias_rho.data
                weight_sigmas = torch.log1p(torch.exp(weight_rhos))
                bias_sigmas = torch.log1p(torch.exp(bias_rhos))

                # weights
                snrs = 10*torch.log10(torch.abs(weight_mus)/weight_sigmas)
                assert snrs.shape[0]==weight_mus.shape[0] and snrs.shape[1]==weight_mus.shape[1], 'SNR not same shape as mus'      
                mask = snrs > snr_threshold
                mask = mask.long()
                layer.weight_mu.data = weight_mus*mask
                layer.weight_rho.data = weight_rhos*mask
                
                # biases
                snrs = 10*torch.log10(torch.abs(bias_mus)/bias_sigmas)
                assert snrs.shape[0]==bias_mus.shape[0], 'SNR not same shape as mus'      
                mask = snrs > snr_threshold
                mask = mask.long()
                layer.bias_mu.data = bias_mus*mask
                layer.bias_rho.data = bias_rhos*mask
    
def predict(model, X):
    probs = torch.nn.Softmax(dim=1)(model(X))
    preds = torch.argmax(probs, dim=1)
    return preds

def evaluate(model, test_loader):
    print('Evaluating on validation data')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(test_loader):
            X, y = data
            X, y = X.to(DEVICE), y.to(DEVICE)
            preds = predict(model, X)
            total += model.batch_size
            correct += (preds == y).sum().item()
    acc = correct / total
    print(f'Validation accuracy: {acc}') 


def main():
    # load models
    bnn_model = load_bnn_class_model('./saved_models/bnn_classification_model.pt', False)
    bnn_model.to(DEVICE)
    mlp_model = load_mlp_class_model('./saved_models/mlp_classification_model.pt')
    dropout_model = load_dropout_class_model('./saved_models/dropout_classification_model.pt')
       
    # collect weights
    bnn_mus, bnn_sigmas = collect_weights(bnn_model, bnn=True)
    bnn_weights = [sample_bnn_weights(mu, sigma) for mu, sigma in zip(bnn_mus, bnn_sigmas)]
    mlp_weights = collect_weights(mlp_model)
    dropout_weights = collect_weights(dropout_model)
    
    # create weights histogram
    plot_histogram(
        [bnn_weights, mlp_weights, dropout_weights], 
        ['BBB', 'Vanilla SGD', 'Dropout']
    )
    
    # plot snr densities
    snr = [compute_snr(mu, sigma) for mu, sigma in zip(bnn_mus, bnn_sigmas)]
    # snr_plots(snr)

    # perform pruning
    pruned_bnn_model = copy.deepcopy(bnn_model)
    print('-------- before pruning --------')
    print(pruned_bnn_model.state_dict())
    
    print('-------- after pruning --------')
    prune_weights(pruned_bnn_model, snr, drop_percentage=.8)
    print(pruned_bnn_model.state_dict())

    # evaluate pruned model
    with torch.no_grad():
        bnn_model.eval()
        pruned_bnn_model.eval()
        test_ds = create_data_class(train=False, batch_size=128, shuffle=False)
        evaluate(pruned_bnn_model.to(DEVICE), test_ds)
        evaluate(bnn_model.to(DEVICE), test_ds)
    
    
if __name__=='__main__':
    main()



