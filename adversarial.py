import torch
from config import *
from data_utils import create_data_class
from networks import BayesianNetwork, MLP, MLP_Dropout
import matplotlib.pyplot as plt
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


ds = create_data_class(0,1,0)
bnn_model = load_bnn_class_model('./saved_models/bnn_classification_model.pt')
#mlp_model = load_mlp_class_model('./saved_models/mlp_classification_model.pt')

image_dir = 'adv_images'
im_path = os.path.join(image_dir, 'im.png')
X, y = next(iter(ds))

plt.imsave(im_path, X[0,0], cmap='gray')

#print(mlp_model(X).argmax())
print(bnn_model(X).argmax())

fmodel_bnn = foolbox.models.PyTorchModel(bnn_model, bounds=(0, 255), num_classes=10)

attack = foolbox.attacks.FGSM(fmodel_bnn)
adversarial = attack(X[0], y)
print('adversarial class', np.argmax(fmodel_bnn.forward_one(adversarial)))
