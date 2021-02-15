import numpy as np
from config import *
from utils import *
from bandits import Greedy_Bandit
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def main():
    X, Y = read_data()

    bandit_params = {
        'n_samples': 1,
        'buffer_size': Config.buffer_size,
        'batch_size': Config.batch_size,
        'num_batches': Config.num_batches,
        'lr': Config.lr
    }
    ## Train 0.05-greedy agent
    epsilon = 0.05
    greedy_bandit = Greedy_Bandit(epsilon, bandit_params, X, Y)
    NB_STEPS = Config.NB_STEPS
    writer = SummaryWriter(comment=f"_{epsilon}_greedy_agent_training")
    print(f"Initialising training on {device}...")
    num_samples = len(X)
    for step in tqdm(range(NB_STEPS)):
        mushroom = np.random.randint(num_samples)
        loss_info, regret = greedy_bandit.update(mushroom)
        if (step+1)%100 == 0:
            write_loss(writer, step, loss_info, regret)

if __name__ == "__main__":
    main()