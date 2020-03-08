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
from networks.generator import DCGenerator
from networks.discriminator import DCDiscriminator
logger = Logger()

MAX_SAMPLES = 10000
WANDB_LOG_GAP = 1

class DCGAN(object):
    """docstring for DCGAN"""
    def __init__(self, config):
        super(DCGAN, self).__init__()
        self.config = config

        self.preprocess = self.config.data['preprocess']
        _data_related_attrs = ['dimension', 'size']
        self.read_attributes_from_config('data_config', _data_related_attrs)

        _model_related_attrs = ['noise_dim', 'num_channels_prefinal', 'n_critic', 'num_channels_first', 'num_layers', 'use_batch_norm', 'use_init', 'last_layer_activation']
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
        if self.max_samples > config.data_config['size']:
            self.max_samples = config.data_config['size']

        self.fixed_Z = torch.randn(self.max_samples, self.noise_dim)
        if self.use_cuda:
            self.fixed_Z = self.fixed_Z.to(self.device)

    def get_input_image_dimension(self, data_dimension):
        return int(np.ceil(np.sqrt(data_dimension)))

    def read_attributes_from_config(self, section_name, attributes):
        section = getattr(self.config, section_name)
        for attr in attributes:
            setattr(self, attr, section[attr])

    # custom weights initialization called on G and D
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)

    def _build_model(self):

        logger.log("Loading {} network ...".format(colored('deconvolutional generator', 'red')))
        act = 'tanh' if self.preprocess else None
        self.G = DCGenerator(self.out_h, self.out_w, self.noise_dim, self.num_channels_prefinal,
                             self.use_batch_norm, num_layers=self.num_layers, last_layer_activation=act)

        if self.use_init:
            # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
            self.G.apply(self.weights_init)

        self.G = self.G.train()

        logger.log("Loading {} network ...".format(colored('convolutional discriminator', 'red')))
        self.D = DCDiscriminator(self.input_image_dimension, num_channels_first=self.num_channels_first,
                                 use_batch_norm=self.use_batch_norm, num_layers=self.num_layers)

        if self.use_init:
            # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
            self.D.apply(self.weights_init)

        self.D = self.D.train()

        wandb.watch([self.G, self.D])

    def _setup_optimizers(self):

        self.G_opt = Adam(self.G.parameters(), lr=self.g_lr, betas=tuple(self.g_betas), weight_decay=self.g_weight_decay)
        self.D_opt = Adam(self.D.parameters(), lr=self.d_lr, betas=tuple(self.d_betas), weight_decay=self.d_weight_decay)

    def _define_losses(self):
        self.criterion = nn.BCEWithLogitsLoss()

    def train(self, data_loader, n_epochs):

        self.global_num_iters = 0

        for epoch_num in range(1, n_epochs + 1):
            logger.log('Starting epoch {}...'.format(colored(epoch_num, 'yellow')))
            self._train_epoch(data_loader, epoch_num)

    def _train_epoch(self, data_loader, epoch):
        self.d_steps, self.g_steps = 0, 0
        epoch_d_loss, epoch_g_loss = 0.0, 0.0
        iterator = tqdm(data_loader)
        num_iteratrions_per_epoch = len(data_loader.dataset)

        for data_dict in iterator:

            self.global_num_iters += 1
            wandb_logs = {}

            self.d_steps += 1

            X = data_dict['image']
            X = Variable(X)
            self.batch_size = X.shape[0]

            Z = torch.randn(self.batch_size, self.noise_dim)
            Z = Variable(Z)

            if self.use_cuda:
                X = X.to(self.device)
                Z = Z.to(self.device)

            loss_dict = self._train_step(X, Z)

            d_loss = loss_dict['d_loss']; epoch_d_loss += d_loss;
            if 'g_loss' in loss_dict:
                self.g_steps += 1
                g_loss = loss_dict['g_loss']; epoch_g_loss += g_loss;

                iterator.set_description('Epoch: {} | {} | V {} | G: {:.3f} | D: {:.3f}'.format(epoch, self.config.arch, self.config.version, g_loss, d_loss))
                iterator.refresh()

            wandb_logs.update(loss_dict)

            figure = self._get_scatter_plot(data_loader)
            wandb_logs.update({"Original vs Generated: Scatter plot": wandb.Image(figure)})
            wandb.log(wandb_logs, step=self.global_num_iters)

        epoch_d_loss /= self.d_steps; epoch_g_loss /= self.g_steps;
        logger.log(colored('Epoch completed => G: {:.3f} D: {:.3f}'.format(epoch_g_loss, epoch_d_loss), 'blue'))

    def _train_step(self, X, Z):

        loss_dict = {}

        ################# D training step ###################
        real_label, fake_label = np.random.uniform(0.9, 1.0), np.random.uniform(0.0, 0.1)
        if np.random.uniform(0, 1) > 0.8:
            real_label, fake_label = fake_label, real_label

        self.D.zero_grad()

        D_X, D_X_logits = self.D(X, return_logits=True)
        real_labels = torch.full((self.batch_size,), real_label)
        if self.use_cuda:
            real_labels = real_labels.to(self.device)
        D_loss_real = self.criterion(D_X_logits.view(-1), real_labels)

        G_Z = self.G(Z)
        D_G_Z, D_G_Z_logits = self.D(G_Z, return_logits=True)
        fake_labels = torch.full((self.batch_size,), fake_label)
        if self.use_cuda:
            fake_labels = fake_labels.to(self.device)
        D_loss_fake = self.criterion(D_G_Z_logits.view(-1), fake_labels)


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


    def postprocess(self, data_loader, X):
        dataset = data_loader.dataset
        return dataset.preprocessor.inverse_transform(X)
        

    def _get_scatter_plot(self, data_loader, samples_to_visualize=200, seed=0):

        # setting up models for evaluation
        self.G = self.G.eval()

        dataset = data_loader.dataset

        # Fix the seed since we need the same original data across epochs
        np.random.seed(seed)
        indices = np.random.choice(self.size, size=samples_to_visualize, replace=False)

        X = np.array([self.flatten_image(dataset[i]['image']).cpu().numpy() for i in indices])

        Z = self.fixed_Z[indices]
        G_Z = self.flatten_image(self.G(Z)).detach().cpu().numpy()

        if self.config.data['preprocess']:
            X = self.postprocess(data_loader, X)
            G_Z = self.postprocess(data_loader, G_Z)

        figure = plot_original_vs_generated(X, G_Z)
        return figure


if __name__ == '__main__':
    from config import Config
    from data.synthetic_dataset import SyntheticDataset

    config = Config('default.yml', 'dcgan')
    dataloader = create_data_loader(config)

    run_name = 'correlation-GAN_{}_{}'.format(config.arch, config.version)
    os.environ['WANDB_ENTITY'] = "wadhwani"
    os.environ['WANDB_PROJECT'] = "correlation-GAN"
    run_dir = config.paths['CKPT_DIR']
    os.environ['WANDB_DIR'] = run_dir
    wandb.init(name=run_name, dir=run_dir, notes=config.description)
    wandb.config.update(config.__dict__)

    dcgan = DCGAN(config)
    dcgan.train(dataloader, 10)
        
        