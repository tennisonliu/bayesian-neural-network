import numpy as np
from config import *
from utils import read_data
from bandits import Greedy_Bandit, BNN_Bandit
from tqdm import tqdm

def main():
    X, Y = read_data()

    params = {
        'buffer_size': Config.buffer_size,
        'batch_size': Config.batch_size,
        'num_batches': Config.num_batches,
        'lr': Config.lr
    }

    bandits = {
        'bnn': BNN_Bandit('bnn', {**params, 'n_samples':Config.n_samples}, X, Y),
        'greedy': Greedy_Bandit(0.0, 'greedy', {**params, 'n_samples':1}, X, Y),
        '0.01-greedy': Greedy_Bandit(0.01, '0.01_greedy', {**params, 'n_samples':1}, X, Y),
        '0.05-greedy': Greedy_Bandit(0.05, '0.05_greedy', {**params, 'n_samples':1}, X, Y)
    }

    NB_STEPS = Config.NB_STEPS
    print(f"Initialising training on {device}...")
    num_training_data = len(X)
    for step in tqdm(range(NB_STEPS)):
        mushroom = np.random.randint(num_training_data)
        for _, bandit in bandits.items():
            bandit.update(mushroom)
            if (step+1)%100 == 0:
                bandit.log_progress(step)

if __name__ == "__main__":
    main()