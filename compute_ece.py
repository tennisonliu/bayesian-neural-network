'''
Computes ECE and plots Reliability Diagram for all classification models
'''
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torch import nn
from tqdm import tqdm
from utils.data_utils import create_data_class
from classification.class_task import BNN_Classification, MLP_Classification, MCDropout_Classification
from config import *

class ECELoss(nn.Module):
    ''' compute expected calibration error '''
    def __init__(self, bin_step=0.1, num_classes=10):
        ''' n_bins: number of confidence interval bins '''
        super(ECELoss, self).__init__()
        self.bin_step = bin_step
        self.num_classes = num_classes

    def forward(self, probs, labels):
        pred_class = np.argmax(probs, axis=1)

        # make confidence, preds, labels one-hot
        expanded_preds = np.reshape(probs, -1)
        pred_class_OH = np.reshape(get_one_hot(pred_class, self.num_classes), -1)
        target_class_OH = np.reshape(get_one_hot(labels, self.num_classes), -1)
        correct_vec = (target_class_OH*(pred_class_OH == target_class_OH)).astype(int)

        # generate bins
        bins = np.arange(0, 1.1, self.bin_step)
        bin_idxs = np.digitize(expanded_preds, bins, right=True)
        bin_idxs = bin_idxs - 1
        
        bin_centers = bins[1:] - self.bin_step/2
        bin_counts = np.ones(len(bin_centers))
        bin_corrects = np.zeros(len(bin_centers))
        bin_confidence = np.zeros(len(bin_centers))

        min_idx = self.num_classes
        if min(bin_idxs) < min_idx:
            min_idx = min(bin_idxs)
        
        for nbin in range(len(bin_centers)):
            bin_counts[nbin] = np.sum((bin_idxs==nbin).astype(int))
            bin_corrects[nbin] = np.sum(correct_vec[bin_idxs==nbin])
            bin_confidence[nbin] = np.mean(expanded_preds[bin_idxs==nbin])

        have_data = bin_counts > 0  
        bin_acc = bin_corrects[have_data] / bin_counts[have_data]

        ece = 0
        for i in range(len(bin_confidence)):
            ece += np.absolute(bin_confidence[i]-bin_acc[i]) * bin_counts[i]/np.sum(bin_counts)

        return ece, bin_centers[have_data], bin_acc

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def evaluate_ece(model, test_loader):
    ece_criterion = ECELoss(bin_step=0.1).to(DEVICE)
    probs_list, labels_list = [], []
    with torch.no_grad():
        for data in tqdm(test_loader):
            X, y = data
            X, y = X.to(DEVICE), y.to(DEVICE)
            _, probs = model.predict(X)
            probs_list.append(probs)
            labels_list.append(y)
        probs = torch.cat(probs_list)
        labels = torch.cat(labels_list)

        probs = probs.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
    
    return ece_criterion(probs, labels)

def main():
    np.random.seed(0)
    config = ClassConfig
    test_ds = create_data_class(train=False, batch_size=config.batch_size, shuffle=False)

    params = {
        'lr': config.lr,
        'hidden_units': config.hidden_units,
        'mode': config.mode,
        'batch_size': config.batch_size,
        'epochs': config.epochs,
        'x_shape': config.x_shape,
        'classes': config.classes,
        'num_batches': 0,
        'train_samples': config.train_samples,
        'test_samples': 5,
        'mu_init': config.mu_init,
        'rho_init': config.rho_init,
        'prior_init': config.prior_init,
        'mixture_prior':config.mixture_prior,
        'save_dir': config.save_dir,
    }

    models = {
        'BBB': BNN_Classification('bnn_classification', {**params, 'local_reparam': False, 'dropout': False}),
        # 'BBB-LR': BNN_Classification('bnn_classification_lr', {**params, 'local_reparam': True, 'dropout': False}),
        'MLP': MLP_Classification('mlp_classification', {**params, 'dropout': False}),
        'MC-Dropout': MCDropout_Classification('mcdropout_classification', {**params, 'dropout': True}),
        }

    for _, model in models.items():
        model.net.load_state_dict(torch.load(model.save_model_path))

    plt.style.use('seaborn-colorblind')
    fig = plt.figure(figsize=(9, 6))
    for m_name, model in models.items():
        print(f'Computing calibraiton error and plotting reliability diagram for {m_name}')
        model.net.to(DEVICE)
        with torch.no_grad():
            model.net.eval()
            if m_name == 'MC_Dropout':
                model.net.enable_dropout()
            ece, confidence_list, accuracy_list = evaluate_ece(model, test_ds)
            print(f'Expected Calibration Error: {ece:.4f}')
            plt.plot(confidence_list, accuracy_list, marker='o', linewidth=2, label=f'{m_name}')
    plt.plot([0.05, 0.95], [0.05, 0.95], '--', linewidth=2)
    plt.legend(loc=2, prop={'size': 18})
    plt.xlabel('Confidence', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(np.arange(0.05, 1.0, 0.1), fontsize=20)
    plt.savefig('graphs/reliability_diagram.pdf', dpi=5000, format='pdf', bbox_inches='tight', pad_inches=0.1)

if __name__ == "__main__":
    main()