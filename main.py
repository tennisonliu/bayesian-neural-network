import numpy as np
from config import *
from utils import *
from bandit import MushroomBandit
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def main():
    X, Y = read_data()

    bandit_params = {
        'n_samples': Config.n_samples,
        'buffer_size': Config.buffer_size,
        'batch_size': Config.batch_size,
        'num_batches': Config.num_batches,
        'lr': Config.lr
    }
    mushroom_bandit = MushroomBandit(bandit_params, X, Y)
    NB_STEPS = Config.NB_STEPS
    writer = SummaryWriter(comment=f"_BNN_training")
    print(f"Initialising training on {device}...")
    num_samples = len(X)
    for step in tqdm(range(NB_STEPS)):
        mushroom = np.random.randint(num_samples)
        loss_info, regret = mushroom_bandit.update(mushroom)
        if (step+1)%100 == 0:
            write_weight_histograms(writer, mushroom_bandit.net, step)
            write_loss_scalars(writer, step, loss_info, regret)

if __name__ == "__main__":
    main()