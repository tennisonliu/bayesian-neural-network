import torch
from config import *
from data_utils import create_data_class
from networks import BayesianNetwork
import matplotlib.pyplot as plt
import numpy as np
import os
import foolbox


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
        'mixture_prior': config.mixture_prior,
        'local_reparam': False
    }
    model = BayesianNetwork(model_params)
    model.load_state_dict(torch.load(saved_model))
    model.eval()
    return model

def sample_predict(model, X, samp):
        probs = torch.zeros(size=[1, 10])
        for _ in torch.arange(samp):
            out = torch.nn.Softmax(dim=1)(model(X, sample=True))
            probs = probs + out / samp
        pred = torch.argmax(probs, dim=1)
        return pred

NUM_IMS = 5000
SAMPS = 20

ds = create_data_class(0,NUM_IMS,0)
bnn_model = load_bnn_class_model('./saved_models/bnn_classification_model.pt')

for i, (X, y) in enumerate(ds):
    if i==0:
        break

image_dir = 'adv_images'
im_path = os.path.join(image_dir, 'im.png')
plt.imsave(im_path, 1-X[0,0], cmap='gray')

pred = bnn_model(X).argmax(dim=1)
print('MAP accuracy on unperturbed images ', ((pred==y).sum()/float(len(y))).numpy())

pred_sam = sample_predict(bnn_model, X, SAMPS)
print('Posterior accuracy on unperturbed images ', ((pred_sam==y).sum()/float(len(y))).numpy())

fmodel_bnn = foolbox.models.PyTorchModel(bnn_model, bounds=(0, 1), device='cpu')

print('Generating adversarial examples')
attack = foolbox.attacks.FGSM()

epsilons=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
_, adversarial, success = attack(fmodel_bnn, X.squeeze(), y, epsilons=epsilons)
print('Done. Testing...')

for i in range(len(epsilons)):
    print('epsilon= ', epsilons[i])
    adv = adversarial[i]
    plt.imsave(os.path.join(image_dir, f'adv_eps_{epsilons[i]}.png'), 1-adv[0], cmap='gray')
    diff = X[0,0]-adv[0]
    diff = (diff+1)/2
    diff[0,0]=0
    diff[0,1]=1
    plt.imsave(os.path.join(image_dir, f'diff_eps_{epsilons[i]}.png'), diff, cmap='gray')


    pred_adv = bnn_model(adv).argmax(dim=1)
    print('MAP accuracy on adversarial images ', ((pred_adv==y).sum()/float(len(y))).numpy())

    pred_adv_sam=sample_predict(bnn_model, adv, SAMPS)
    print('Posterior accuracy on adversarial images ', ((pred_adv_sam==y).sum()/float(len(y))).numpy())
