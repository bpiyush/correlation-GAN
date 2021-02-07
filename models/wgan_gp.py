from os.path import join
import imageio
import numpy as np
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
# from torchvision.utils import make_grid
from torch.autograd import Variable
from torch import autograd
from torch.optim import Adam
from termcolor import colored
# from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, '/home/users/piyushb/projects/correlation-GAN')
from utils.logger import Logger
from utils.metrics import kl_divergence, js_divergence, reconstruction_error
from utils.io import save_model, save_pkl
from data.dataloader import create_data_loader
from utils.visualize import plot_original_vs_generated, plot_correlation
from networks.generator import Generator
from networks.discriminator import Discriminator
logger = Logger()

MAX_SAMPLES = 10000
CKPT_SAVE_INTERVAL = 20

class WGAN_GP():
    def __init__(self, config):

        self.config = config
        self.data_config = self.config.data_config
        self.latent_dim = config.model['latent_dim']
        self.n_critic = config.model['n_critic']
        self.gamma = config.hyperparams['gamma']
        self.use_cuda = config.system['use_cuda']

        self.G, self.D, self.G_opt, self.D_opt = self._build_model()

        self.steps = 0
        self.max_samples = MAX_SAMPLES
        if self.max_samples > config.data_config['size']:
            self.max_samples = config.data_config['size']
        self._fixed_z = torch.randn(self.max_samples, self.latent_dim)
        self.hist = []

        self.seed = config.system['seed']

        if self.use_cuda:
            self._fixed_z = self._fixed_z.cuda()
            self.G.cuda()
            self.D.cuda()

    def _build_model(self):
        device = torch.device('cuda')

        data_dimension = self.config.data_config['dimension']
        generator_hidden_layers = self.config.model['generator_hidden_layers']
        use_dropout = self.config.model['use_dropout']
        drop_prob = self.config.model['drop_prob']
        use_ac_func = self.config.model['use_ac_func']
        activation = self.config.model['activation']
        disc_hidden_layers = self.config.model['disc_hidden_layers']
        last_layer_act = self.config.model['last_layer_activation']

        logger.log("Loading {} network ...".format(colored('generator', 'red')))
        gen_fc_layers = [self.latent_dim, *generator_hidden_layers, data_dimension]
        generator = Generator(gen_fc_layers, use_dropout, drop_prob, use_ac_func, activation, last_layer_act).to(device)
        # generator = generator.train()

        logger.log("Loading {} network ...".format(colored('discriminator', 'red')))
        disc_fc_layers = [data_dimension, *disc_hidden_layers, 1]
        discriminator = Discriminator(disc_fc_layers, use_dropout, drop_prob, use_ac_func, activation).to(device)
        # discriminator = discriminator.train()

        wandb.watch([generator, discriminator])

        g_optimizer, d_optimizer = self._setup_optimizers(generator, discriminator)

        return generator, discriminator, g_optimizer, d_optimizer

    def _setup_optimizers(self, generator, discriminator):
        hparam_attrs_from_config = ['g_lr', 'g_betas', 'd_lr', 'd_betas']
        for attr in hparam_attrs_from_config:
            setattr(self, attr, self.config.hyperparams[attr])

        g_optimizer = Adam(generator.parameters(), lr=self.g_lr, betas=tuple(self.g_betas), weight_decay=self.config.hyperparams['weight_decay'])
        d_optimizer = Adam(discriminator.parameters(), lr=self.d_lr, betas=tuple(self.d_betas))

        return g_optimizer, d_optimizer

    def train(self, data_loader, n_epochs):

        best_metric_so_far = np.inf

        for epoch in range(1, n_epochs + 1):
            logger.log('Starting epoch {}...'.format(colored(epoch, 'yellow')))

            # run the train epoch
            self._train_epoch(data_loader)

            # compute metrics and visualizations post the epoch
            original_data, generated_data = self._get_original_and_generated_data(data_loader, self.seed)
            comparison = self._compare_original_and_generated_data(original_data, generated_data)

            # save checkpoint based on certain metrics
            reconstruction_error = comparison['Reconstruction error']
            save_mode = 'no-save'
            if reconstruction_error < best_metric_so_far:
                save_mode = 'best'
                best_metric_so_far = reconstruction_error

                logger.log("[{}] Saving training (O/G) data".format(colored(save_mode, 'red')))
                self._save_training_data(original_data, generated_data)

            if epoch % CKPT_SAVE_INTERVAL == 1 and save_mode != 'best':
                save_mode = 'regular'

            if save_mode != 'no-save':
                metric_value = 'reconstruction error: {}'.format(np.round(reconstruction_error, 2))
                metric_value = colored(metric_value, 'blue')
                logger.log("[{}] Saving model checkpoint based on {}".format(colored(save_mode, 'red'), metric_value))
                self._save_model(epoch, save_mode, comparison)

            # update metrics and visualisations on W&B
            self._update_wandb(comparison, epoch)

    def _train_epoch(self, data_loader):
        iterator = tqdm(data_loader)
        for data_dict in iterator:
            self.steps += 1

            data = data_dict['point']

            data = Variable(data)
            if self.use_cuda:
                data = data.cuda()

            d_loss, grad_penalty = self._discriminator_train_step(data)
            self.hist.append({'d_loss': d_loss, 'grad_penalty': grad_penalty})

            if self.steps % self.n_critic == 0:
                g_loss = self._generator_train_step(data.size(0))
                self.hist[-1]['g_loss'] = g_loss

                iterator.set_description('Itn: {} | V {} | G: {:.3f} | D: {:.3f} | GP: {:.3f}'.format(self.steps, self.config.version, g_loss, d_loss, grad_penalty))
                iterator.refresh()

        status = colored('Epoch completed', 'yellow')
        logger.log('{} => G: {:.3f} D: {:.3f} GP: {:.3f}'.format(status, g_loss, d_loss, grad_penalty))

    def _discriminator_train_step(self, data):
        batch_size = data.size(0)
        generated_data = self._sample(batch_size)
        grad_penalty = self._gradient_penalty(data, generated_data)
        d_loss = self.D(generated_data).mean() - self.D(data).mean() + grad_penalty
        self.D_opt.zero_grad()
        d_loss.backward()
        self.D_opt.step()
        return d_loss.item(), grad_penalty.item()

    def _generator_train_step(self, batch_size):
        self.G_opt.zero_grad()
        generated_data = self._sample(batch_size)
        g_loss = -self.D(generated_data).mean()
        g_loss.backward()
        self.G_opt.step()
        return g_loss.item()

    def _gradient_penalty(self, data, generated_data, gamma=10):
        batch_size = data.size(0)
        epsilon = torch.rand(batch_size, 1)
        epsilon = epsilon.expand_as(data)

        if self.use_cuda:
            epsilon = epsilon.cuda()

        interpolation = epsilon * data.data + (1 - epsilon) * generated_data.data
        interpolation = Variable(interpolation, requires_grad=True)

        if self.use_cuda:
            interpolation = interpolation.cuda()

        interpolation_logits = self.D(interpolation)
        grad_outputs = torch.ones(interpolation_logits.size())

        if self.use_cuda:
            grad_outputs = grad_outputs.cuda()

        gradients = autograd.grad(outputs=interpolation_logits,
                                  inputs=interpolation,
                                  grad_outputs=grad_outputs,
                                  create_graph=True,
                                  retain_graph=True)[0]

        gradients = gradients.view(batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        return self.gamma * ((gradients_norm - 1) ** 2).mean()

    def _sample(self, n_samples):
        z = Variable(torch.randn(n_samples, self.latent_dim))
        if self.use_cuda:
            z = z.cuda()
        return self.G(z)

    def _get_original_and_generated_data(self, data_loader, seed):
        dataset = data_loader.dataset

        # Fix the seed since we need the same original data across epochs
        np.random.seed(seed)
        indices = np.random.choice(len(dataset), size=min(len(dataset), self.max_samples), replace=False)

        original_data = np.array([dataset[i]['point'].cpu().numpy() for i in indices])

        # eval_generator = self.G.eval()
        generated_data = self.G(self._fixed_z).detach().cpu().numpy()

        if self.config.data['preprocess']:
            # if original data is preprocessed, generated needs to be post processed
            preprocessor = dataset.preprocessor
            original_data = preprocessor.inverse_transform(original_data)
            generated_data = preprocessor.inverse_transform(generated_data)

        return original_data, generated_data

    def _compute_metrics(self, original_data, generated_data):

        assert original_data.shape[0] == generated_data.shape[0]

        kld = kl_divergence(original_data, generated_data)
        jsd = js_divergence(original_data, generated_data)
        recs_error = reconstruction_error(original_data, generated_data)

        metrics = {'kld': kld, 'jsd': jsd, 'reconstruction_error': recs_error}

        return metrics

    def _compare_original_and_generated_data(self, original_data, generated_data):

        scatter_plot = plot_original_vs_generated(original_data, generated_data)
        correlation_plot = plot_correlation(original_data, generated_data, self.data_config['cleaned_columns'])

        metrics = self._compute_metrics(original_data, generated_data)

        comparison = {
            "Original vs Generated: Scatter plot": wandb.Image(scatter_plot),
            "Original vs Generated: Correlation": wandb.Image(correlation_plot),
            "KL Divergence": metrics['kld'],
            "JS Divergence": metrics['jsd'],
            "Reconstruction error": metrics['reconstruction_error']
        }

        return comparison

    def _save_model(self, epoch, save_mode, comparison_dict):
        ckpt_parent_folder = self.config.paths['CKPT_DIR']

        # saving generator
        save_model(self.G, 'generator', self.G_opt, epoch, comparison_dict, ckpt_parent_folder, save_mode)
        # saving discriminator
        save_model(self.D, 'discriminator', self.D_opt, epoch, comparison_dict, ckpt_parent_folder, save_mode)

    def _save_training_data(self, original_data, generated_data):
        ckpt_parent_folder = self.config.paths['CKPT_DIR']
        train_data = {'config': self.config, 'original': original_data, 'generated': generated_data}

        fpath = join(ckpt_parent_folder, 'train_data_best.pkl')
        save_pkl(train_data, fpath)

    def _update_wandb(self, comparison_dict, epoch_number):
        wandb.log(self.hist[epoch_number], step=epoch_number)
        wandb.log(comparison_dict, step=epoch_number)


if __name__ == '__main__':
    from config import Config
    from data.synthetic_dataset import SyntheticDataset
    from networks.generator import Generator
    from networks.discriminator import Discriminator

    config = Config('default.yml')
    dataloader = create_data_loader(config)

    run_name = 'correlation-GAN_{}'.format(config.version)
    wandb.init(name=run_name, dir=config.paths['CKPT_DIR'], notes=config.description)
    wandb.config.update(config.__dict__)

    device = torch.device('cuda')

    use_dropout = [True, True, False]
    drop_prob = [0.5, 0.5, 0.5]
    use_ac_func = [True, True, False]
    activation = 'relu'
    latent_dim = 10

    gen_fc_layers = [latent_dim, 16, 32, 2]
    generator = Generator(gen_fc_layers, use_dropout, drop_prob, use_ac_func, activation).to(device)

    disc_fc_layers = [2, 32, 16, 1]
    discriminator = Discriminator(disc_fc_layers, use_dropout, drop_prob, use_ac_func, activation).to(device)

    wandb.watch([generator, discriminator])

    g_optimizer = Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    d_optimizer = Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))

    wgan_gp = WGAN_GP(config, generator, discriminator, g_optimizer, d_optimizer, latent_shape)
    wgan_gp.train(dataloader, 200)


