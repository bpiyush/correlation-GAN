import argparse
import tqdm
import numpy as np
from termcolor import colored
import wandb
import random
import torch
import os
from torch.optim import Adam

from config import Config
from data.synthetic_dataset import SyntheticDataset
from networks.generator import Generator
from networks.discriminator import Discriminator
from models.wgan_gp import WGAN_GP

import psutil
from utils.logger import Logger
from data.dataloader import create_data_loader
logger = Logger()


def set_cpu_limit(start, end):
    process = psutil.Process()
    curr_cpus = process.cpu_affinity()
    new_cpus = range(start, end)

    print(colored("=> You were using {} CPUs. Setting {} CPUs based on your input.".format(len(curr_cpus), len(new_cpus)), 'yellow'))
    process.cpu_affinity(list(new_cpus))


def seed_everything(seed=0, harsh=False):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    if harsh:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    seed_everything()

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', type=str, required=True, help='version of the config')
    parser.add_argument('-cpu', '--cpu_range', nargs='*', type=int, required=True, help='start and end index of CPUs to be used: Example: 1 10')
    args = parser.parse_args()

    start, end = args.cpu_range
    set_cpu_limit(start, end)

    config = Config(args.version)
    dataloader = create_data_loader(config)

    run_name = 'correlation-GAN_{}'.format(config.version)
    wandb.init(name=run_name, dir=config.checkpoint_dir, notes=config.description)
    wandb.config.update(config.__dict__)

    logger.log("Assembling {} model ...".format(colored('WGAN-GP', 'red')))
    wgan_gp = WGAN_GP(config)

    logger.log("Starting training for {} epochs ...".format(config.num_epochs))
    wgan_gp.train(dataloader, config.num_epochs)

    import ipdb; ipdb.set_trace()


