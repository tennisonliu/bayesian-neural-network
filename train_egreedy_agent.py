import numpy as np
from config import *
from utils import *
from bandits import Greedy_Bandit
from tqdm import tqdm

def main():
    X, Y = read_data()

    bandit_params = {
        'n_samples': 1,
        'buffer_size': Config.buffer_size,
        'batch_size': Config.batch_size,
        'num_batches': Config.num_batches,
        'lr': Config.lr
    }

    greedy_bandits = {
        'greedy': Greedy_Bandit(epsilon=0.0, bandit_params, X, Y),
        '0.01-greedy': Greedy_Bandit(epsilon=0.01, bandit_params, X, Y),
        '0.05-greedy': Greedy_Bandit(epsilon=0.05, bandit_params, X, Y)
    }
    writers = {
        'greedy': SummaryWriter(comment=f"_0.00_greedy_agent_training"),
        '0.01-greedy': SummaryWriter(comment=f"_0.01_greedy_agent_training"),
        '0.05-greedy': SummaryWriter(comment=f"_0.05_greedy_agent_training")
    }
    NB_STEPS = Config.NB_STEPS
    print(f"Initialising training on {device}...")
    num_samples = len(X)
    for step in tqdm(range(NB_STEPS)):
        mushroom = np.random.randint(num_samples)
        for name, bandit in greedy_bandits.items():
            loss_info, regret = bandit.update(mushroom)
            if (step+1)%100 == 0:
                write_loss(writers[name], step, loss_info, regret)

    # ## Train 0.05-greedy agent
    # epsilon = 0.00
    # greedy_bandit = Greedy_Bandit(epsilon, bandit_params, X, Y)
    # NB_STEPS = Config.NB_STEPS
    # writer = SummaryWriter(comment=f"_{epsilon}_greedy_agent_training")
    # print(f"Initialising training on {device}...")
    # num_samples = len(X)
    # for step in tqdm(range(NB_STEPS)):
    #     mushroom = np.random.randint(num_samples)
    #     loss_info, regret = greedy_bandit.update(mushroom)
    #     if (step+1)%100 == 0:
    #         write_loss(writer, step, loss_info, regret)

if __name__ == "__main__":
    main()