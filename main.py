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
from models.dcgan import DCGAN

import psutil
from utils.logger import Logger
from data.dataloader import create_data_loader
logger = Logger()


def set_cpu_limit(start, end):
    process = psutil.Process()
    curr_cpus = process.cpu_affinity()
    new_cpus = range(start, end)

    output = "=> You were using {} CPUs. Setting {} CPUs based on your input."
    print(colored(output.format(len(curr_cpus), len(new_cpus)), 'yellow'))
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


def setup_wandb_credentials(username, project, run_dir):
    os.environ['WANDB_ENTITY'] = username
    os.environ['WANDB_PROJECT'] = project
    os.environ['WANDB_DIR'] = run_dir


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', type=str, required=True,
                        help='version of the config')
    parser.add_argument('-a', '--arch', type=str, choices=['dcgan', 'wgan_gp'],
                        required=True, help='model architecture to be used')
    parser.add_argument('-cpu', '--cpu_range', nargs='*', type=int, required=True,
                        help='start and end index of CPUs to be used: Example: 1 10')
    args = parser.parse_args()

    # Setup CPU limit
    start, end = args.cpu_range
    set_cpu_limit(start, end)

    # Setup config
    config = Config(args.version, args.arch)

    # set seed for eveything
    seed_everything(config.system['seed'])

    # create data loader
    dataloader = create_data_loader(config)

    # Setup W&B credentials
    run_name = 'correlation-GAN_{}_{}'.format(config.arch, config.version)
    username, project, run_dir = "bpiyush", "correlation-GAN", config.checkpoint_dir
    setup_wandb_credentials(username, project, run_dir)

    wandb.init(name=run_name, dir=run_dir, notes=config.description, entity=username)
    wandb.config.update(config.__dict__)

    architecture = eval(config.arch.upper())
    logger.log("Assembling {} model ...".format(colored(config.arch.upper(), 'red')))
    model = architecture(config)

    logger.log("Starting training for {} epochs ...".format(config.training['num_epochs']))
    model.train(dataloader, config.training['num_epochs'])


