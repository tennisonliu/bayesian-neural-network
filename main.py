'''
Main script with trainers
'''
import numpy as np
from reinforcement_learning.bandits import BNN_Bandit, Greedy_Bandit
from regression.reg_task import BNN_Regression, MLP_Regression
from tqdm import tqdm
from config import *
from data_utils import *
from plot_utils import *

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
        'num_training_samples': config.train_samples,
        'noise_tolerance': config.noise_tolerance,
        'mu_init': config.mu_init,
        'rho_init': config.rho_init,
        'prior_init': config.prior_init,
        'save_dir': config.save_dir
    }

    models = {
        'bnn_reg': BNN_Regression('bnn_regression', params),
        'mlp_reg': MLP_Regression('mlp_regression', params),
    }

    epochs = config.epochs
    print(f"Initialising training on {DEVICE}...")

    # training loop
    for epoch in tqdm(range(epochs)):
        for m_name, model in models.items():
            model.train_step(train_ds)
            model.log_progress(epoch)
            model.scheduler.step()
            # save best model
            if model.epoch_loss < model.best_loss:
                model.best_loss = model.epoch_loss
                torch.save(model.net.state_dict(), model.save_model_path)

    # evaluate
    print("Evaluating and generating plots...")
    x_test = torch.linspace(-2., 2, config.test_samples).reshape(-1, 1)
    for m_name, model in models.items():
        model.net.load_state_dict(torch.load(model.save_model_path, map_location=torch.device(DEVICE)))
        y_test = model.evaluate(x_test, config.test_samples)
        if m_name == 'bnn_reg':
            create_regression_plot(x_test.cpu().numpy(), y_test, train_ds, m_name)
        else:
            create_regression_plot(x_test.cpu().numpy(), y_test.reshape(1, -1), train_ds, m_name)


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
        'mode': config.mode,
        'mu_init': config.mu_init,
        'rho_init': config.rho_init,
        'prior_init': config.prior_init
    }

    bandits = {
        'bnn_bandit': BNN_Bandit('bnn_bandit', {**params, 'n_samples':2, 'epsilon':0}, X, Y),
        'greedy_bandit': Greedy_Bandit('greedy_bandit', {**params, 'n_samples':1, 'epsilon':0}, X, Y),
        '0.01_greedy_bandit': Greedy_Bandit('0.01_greedy_bandit', {**params, 'n_samples':1, 'epsilon':0.01}, X, Y),
        '0.05_greedy_bandit': Greedy_Bandit('0.05_greedy_bandit', {**params, 'n_samples':1, 'epsilon':0.05}, X, Y)
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