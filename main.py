import numpy as np
from reinforcement_learning.bandits import BNN_Bandit, Greedy_Bandit
from regression.reg_task import BNN_Regression, MLP_Regression
from classification.class_task import BNN_Classification
from tqdm import tqdm
from config import *
from data_utils import *

def reg_trainer():
    ''' Regression Task Trainer '''
    config = RegConfig
    X, Y = create_data_reg(train_size=config.train_size)
    train_ds = PrepareData(X, Y)
    train_ds = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    params = {
        'lr': config.lr,
        'hidden_units': config.hidden_units,
        'mode': config.mode,
        'batch_size': config.batch_size,
        'num_batches': len(train_ds),
        'x_shape': X.shape[1],
        'y_shape': Y.shape[1],
        'num_training_samples': config.train_samples
    }

    models = {
        'bnn': BNN_Regression('bnn_regression', params),
        'mlp': MLP_Regression('mlp_regression', params),
    }

    epochs = config.epochs
    print(f"Initialising training on {DEVICE}...")
    for epoch in tqdm(range(epochs)):
        for _, model in models.items():
            model.train_step(train_ds)
            model.log_progress(epoch)
            model.scheduler.step()

def rl_trainer():
    ''' RL Bandit Task Trainer'''
    config = RLConfig
    X, Y = read_data_rl(config.data_dir)

    params = {
        'buffer_size': config.buffer_size,
        'batch_size': config.batch_size,
        'num_batches': config.num_batches,
        'lr': config.lr,
        'hidden_units': config.hidden_units,
        'mode': config.mode
    }

    bandits = {
        'bnn': BNN_Bandit('bnn_bandit', {**params, 'n_samples':2, 'epsilon':0}, X, Y),
        'greedy': Greedy_Bandit('greedy_bandit', {**params, 'n_samples':1, 'epsilon':0}, X, Y),
        '0.01-greedy': Greedy_Bandit('0.01_greedy_bandit', {**params, 'n_samples':1, 'epsilon':0.01}, X, Y),
        '0.05-greedy': Greedy_Bandit('0.05_greedy_bandit', {**params, 'n_samples':1, 'epsilon':0.05}, X, Y)
    }

    training_steps = config.training_steps
    print(f"Initialising training on {DEVICE}...")
    training_data_len = len(X)
    for step in tqdm(range(training_steps)):
        mushroom = np.random.randint(training_data_len)
        for _, bandit in bandits.items():
            bandit.update(mushroom)
            bandit.scheduler.step()
            if (step+1)%100 == 0:
                bandit.log_progress(step)


def class_trainer():
    ''' MNIST classification Task Trainer'''
    config = ClassConfig

    train_ds = DataLoader(
        datasets.MNIST(
        './mnist', train=True, download=True,
        transform=transforms.ToTensor()),
        batch_size=config.batch_size, shuffle=True, drop_last=True)

        
    test_ds = DataLoader(
        datasets.MNIST(
        './mnist', train=False, download=True,
        transform=transforms.ToTensor()),
        batch_size=config.batch_size, shuffle=False, drop_last=True)

    params = {
        'lr': config.lr,
        'hidden_units': config.hidden_units,
        'mode': config.mode,
        'batch_size': config.batch_size,
        'epochs': config.epochs,
        'x_shape': config.x_shape,
        'classes': config.classes,
        'num_batches': len(train_ds),
        'train_samples': config.train_samples,
        'test_samples': config.test_smaples
    }

    models = {'bnn': BNN_Classification('bnn_classification', params)}
    
    epochs = config.epochs
    print(f"Initialising training on {DEVICE}...")
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        for _, model in models.items():
            model.train_step(train_ds, test_ds)
            model.log_progress(epoch)
            model.scheduler.step()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', metavar='m', choices=['regression', 'classification', 'rl'], type=str)
    args = parser.parse_args()

    if args.model == 'regression':
        reg_trainer()
    elif args.model == 'classification':
        class_trainer()
    else:
        rl_trainer()