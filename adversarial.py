import torch
from config import *
from data_utils import create_data_class
from networks import BayesianNetwork, MLP, MLP_Dropout
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
    model.eval()
    return model

def sample_predict(model, X, samp):
        probs = torch.zeros(size=[1, 10])
        for _ in torch.arange(samp):
            out = torch.nn.Softmax(dim=1)(model(X, sample=True))
            probs = probs + out / samp
        pred = torch.argmax(probs)
        return pred

ds = create_data_class(0,1,0)
bnn_model = load_bnn_class_model('./saved_models/bnn_classification_model.pt')
#mlp_model = load_mlp_class_model('./saved_models/mlp_classification_model.pt')

image_dir = 'adv_images'
im_path = os.path.join(image_dir, 'im.png')

for i, (X, y) in enumerate(ds):
    if i==2:
        break

plt.imsave(im_path, 1-X[0,0], cmap='gray')

#print(mlp_model(X).argmax())
print(bnn_model(X).argmax())

fmodel_bnn = foolbox.models.PyTorchModel(bnn_model, bounds=(0, 1), device='cpu')

attack = foolbox.attacks.FGSM()
_, adversarial, success = attack(fmodel_bnn, X[0], y, epsilons=[0.0, 0.001, 0.01, 0.03, 0.1])

adv = adversarial[4]
plt.imsave(os.path.join(image_dir, 'adv.png'), 1-adv[0] ,cmap='gray')

print(bnn_model(adv).argmax())

print(sample_predict(bnn_model, adv, 20))
