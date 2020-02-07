import imageio
import numpy as np
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import autograd
from torch.optim import Adam
from termcolor import colored

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, '/home/users/piyushb/projects/correlation-GAN')
from utils.logger import Logger
from data.dataloader import create_data_loader
from utils.visualize import plot_original_vs_generated
from networks.generator import DeconvGenerator
from networks.discriminator import ConvDiscriminator
logger = Logger()

MAX_SAMPLES = 10000
WANDB_LOG_GAP = 10

class DCGAN(object):
    """docstring for DCGAN"""
    def __init__(self, config):
        super(DCGAN, self).__init__()
        self.config = config

        _data_related_attrs = ['dimension', 'size', 'rho']
        self.read_attributes_from_config('data', _data_related_attrs)

        _model_related_attrs = ['noise_dim', 'num_channels_prefinal', 'n_critic', 'num_channels_first', 'num_layers']
        self.read_attributes_from_config('model', _model_related_attrs)

        _hparam_related_attrs = ['g_lr', 'g_betas', 'd_lr', 'd_betas', 'g_weight_decay', 'd_weight_decay']
        self.read_attributes_from_config('hyperparams', _hparam_related_attrs)

        self.input_image_dimension = self.get_input_image_dimension(self.dimension)
        self.out_h, self.out_w = self.input_image_dimension, self.input_image_dimension

        self.use_cuda = self.config.system['use_cuda']
        self.device = torch.device('cuda')

        self._build_model()
        self._setup_optimizers()
        self._define_losses()

        if self.use_cuda:
            self.D = self.D.to(self.device)
            self.G = self.G.to(self.device)

        self.max_samples = MAX_SAMPLES
        if self.max_samples > config.data['size']:
            self.max_samples = config.data['size']

        self.fixed_Z = torch.randn(self.max_samples, self.noise_dim)
        if self.use_cuda:
            self.fixed_Z = self.fixed_Z.to(self.device)

        self.wandb_steps = 0

    def get_input_image_dimension(self, data_dimension):
        return int(np.ceil(np.sqrt(data_dimension)))

    def read_attributes_from_config(self, section_name, attributes):
        section = getattr(self.config, section_name)
        for attr in attributes:
            setattr(self, attr, section[attr])

    def _build_model(self):

        logger.log("Loading {} network ...".format(colored('deconvolutional generator', 'red')))
        self.G = DeconvGenerator(self.out_h, self.out_w, self.noise_dim, self.num_channels_prefinal, num_layers=self.num_layers)

        logger.log("Loading {} network ...".format(colored('convolutional discriminator', 'red')))
        self.D = ConvDiscriminator(num_channels_first=self.num_channels_first, num_layers=self.num_layers)

        wandb.watch([self.G, self.D])

    def _setup_optimizers(self):

        self.G_opt = Adam(self.G.parameters(), lr=self.g_lr, betas=tuple(self.g_betas), weight_decay=self.g_weight_decay)
        self.D_opt = Adam(self.D.parameters(), lr=self.d_lr, betas=tuple(self.d_betas), weight_decay=self.d_weight_decay)

    def _define_losses(self):
        self.criterion = nn.BCEWithLogitsLoss()

    def train(self, data_loader, n_epochs):

        self.global_steps = 0

        for epoch in range(1, n_epochs + 1):
            logger.log('Starting epoch {}...'.format(colored(epoch, 'yellow')))
            self._train_epoch(data_loader, epoch)

    def _train_epoch(self, data_loader, epoch):
        self.d_steps, self.g_steps = 0, 0
        epoch_d_loss, epoch_g_loss = 0.0, 0.0
        iterator = tqdm(data_loader)

        for data_dict in iterator:
            self.d_steps += 1

            X = data_dict['image']
            X = Variable(X)
            self.batch_size = X.shape[0]

            Z = torch.randn(self.batch_size, self.noise_dim)
            Z = Variable(Z)

            if self.use_cuda:
                X = X.to(self.device)
                X = X.to(self.device)

            loss_dict = self._train_step(X, Z)

            d_loss = loss_dict['d_loss']; epoch_d_loss += d_loss;
            if 'g_loss' in loss_dict:
                self.g_steps += 1
                g_loss = loss_dict['g_loss']; epoch_g_loss += g_loss;

                self.global_steps += 1
                wandb.log(loss_dict, step=self.global_steps)

                iterator.set_description('Epoch: {} | {} | V {} | G: {:.3f} | D: {:.3f}'.format(epoch, self.config.arch, self.config.version, g_loss, d_loss))
                iterator.refresh()

            if self.wandb_steps % WANDB_LOG_GAP == 0:
                self.wandb_steps += 1
                self._update_wandb(data_loader)

        epoch_d_loss /= self.d_steps; epoch_g_loss /= self.g_steps;
        logger.log('Epoch completed => G: {:.3f} D: {:.3f}'.format(epoch_g_loss, epoch_d_loss))

    def _train_step(self, X, Z):

        loss_dict = {}

        ################# D training step ###################
        real_label, fake_label = 1, 0

        self.D.zero_grad()

        D_X, D_X_logits = self.D(X, return_logits=True)
        real_labels = torch.full((self.batch_size,), real_label)
        if self.use_cuda:
            real_labels = real_labels.to(self.device)
        D_loss_real = self.criterion(D_X_logits.view(-1), real_labels)
        # D_loss_real.backward()

        G_Z = self.G(Z)
        D_G_Z, D_G_Z_logits = self.D(G_Z, return_logits=True)
        fake_labels = torch.full((self.batch_size,), fake_label)
        if self.use_cuda:
            fake_labels = fake_labels.to(self.device)
        D_loss_fake = self.criterion(D_G_Z_logits.view(-1), fake_labels)
        # D_loss_fake.backward()

        D_loss = D_loss_real + D_loss_fake
        D_loss.backward(retain_graph=True)

        self.D_opt.step()

        loss_dict.update({'d_loss': D_loss.item(), 'd_out_real': D_X.mean().item(), 'd_out_fake_1': D_G_Z.mean().item()})
        ################# XXXXXXXXXXXXXXXX ###################

        if self.d_steps % self.n_critic == 0:

            ################# G training step ###################
            self.G.zero_grad()

            # Since D has been updated, we compute D(G(Z)) again
            D_G_Z_, D_G_Z_logits_ = self.D(G_Z, return_logits=True)

            real_looking_fake_labels = torch.full((self.batch_size,), 1)
            if self.use_cuda:
                real_looking_fake_labels = real_looking_fake_labels.to(self.device)

            G_loss = self.criterion(D_G_Z_logits_.view(-1), real_looking_fake_labels)
            G_loss.backward()

            self.G_opt.step()

            loss_dict.update({'g_loss': G_loss.item(), 'd_out_fake_2': D_G_Z_.mean().item()})
            ################# XXXXXXXXXXXXXXXX ###################

        return loss_dict

    def flatten_image(self, image):
        if len(image.shape) == 3:
            tensor = image.view(-1)
            return tensor[:self.dimension]
        elif len(image.shape) == 4:
            tensor = image.reshape((image.shape[0], -1))
            return tensor[:, :self.dimension]

    def _update_wandb(self, data_loader, samples_to_visualize=500, seed=0):

        dataset = data_loader.dataset

        # Fix the seed since we need the same original data across epochs
        np.random.seed(seed)
        indices = np.random.choice(len(dataset), size=samples_to_visualize, replace=False)

        X = np.array([self.flatten_image(dataset[i]['image']).cpu().numpy() for i in indices])

        Z = self.fixed_Z[indices]
        G_Z = self.flatten_image(self.G(Z)).detach().cpu().numpy()

        figure = plot_original_vs_generated(X, G_Z)
        wandb.log({"Original vs Generated: Scatter plot": wandb.Image(figure)}, step=self.wandb_steps)


if __name__ == '__main__':
    from config import Config
    from data.synthetic_dataset import SyntheticDataset

    config = Config('default.yml', 'dcgan')
    dataloader = create_data_loader(config)

    run_name = 'correlation-GAN_{}_{}'.format(config.arch, config.version)
    wandb.init(name=run_name, dir=config.checkpoint_dir, notes=config.description)
    wandb.config.update(config.__dict__)

    dcgan = DCGAN(config)
    dcgan.train(dataloader, 10)
        
        