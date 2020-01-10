import imageio
import numpy as np
import torch
import torch.nn as nn
# from torchvision.utils import make_grid
from torch.autograd import Variable
from torch import autograd
from torch.optim import Adam
# from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, '/home/users/piyushb/projects/correlation-GAN')
from utils.logger import Logger
from data.dataloader import create_data_loader
logger = Logger()

class WGAN_GP():
    def __init__(self, generator, discriminator, g_optmizer, d_optimizer,
                 latent_shape, n_critic=2, gamma=10,
                 save_every=20, use_cuda=True, logdir=None):

        self.G = generator
        self.D = discriminator
        self.G_opt = g_optmizer
        self.D_opt = d_optimizer
        attributes = ['latent_shape', 'n_critic', 'gamma', 'save_every', 'use_cuda']
        for attr in attributes:
            setattr(self, attr, eval(attr))
        # self.writer = SummaryWriter(logdir)
        self.steps = 0
        self._fixed_z = torch.randn(64, latent_shape)
        self.hist = []
        # self.images = []

        if self.use_cuda:
            self._fixed_z = self._fixed_z.cuda()
            self.G.cuda()
            self.D.cuda()

    def train(self, data_loader, n_epochs):
        # self._save_gif()
        for epoch in range(1, n_epochs + 1):
            logger.log('Starting epoch {}...'.format(epoch))
            self._train_epoch(data_loader)

            # if epoch % self.save_every == 0 or epoch == n_epochs:
            #     torch.save(self.G.state_dict(), self.dataset_name + '_gen_{}.pt'.format(epoch))
            #     torch.save(self.D.state_dict(), self.dataset_name + '_disc_{}.pt'.format(epoch))

    def _train_epoch(self, data_loader):
        for i, data_dict in enumerate(data_loader):
            self.steps += 1

            data = data_dict['point']
            data = Variable(data)
            if self.use_cuda:
                data = data.cuda()

            d_loss, grad_penalty = self._discriminator_train_step(data)
            # self.writer.add_scalars('losses', {'d_loss': d_loss, 'grad_penalty': grad_penalty}, self.steps)
            self.hist.append({'d_loss': d_loss, 'grad_penalty': grad_penalty})

            # if i % 200 == 0:
            #     img_grid = make_grid(self.G(self._fixed_z).cpu().data, normalize=True)
            #     self.writer.add_image('images', img_grid, self.steps)

            if self.steps % self.n_critic == 0:
                g_loss = self._generator_train_step(data.size(0))
                # self.writer.add_scalars('losses', {'g_loss': g_loss}, self.steps)
                self.hist[-1]['g_loss'] = g_loss

        logger.log('g_loss: {:.3f} d_loss: {:.3f} grad_penalty: {:.3f}'.format(g_loss, d_loss, grad_penalty))

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
        z = Variable(torch.randn(n_samples, self.latent_shape))
        if self.use_cuda:
            z = z.cuda()
        return self.G(z)

    # def _save_gif(self):
    #     grid = make_grid(self.G(self._fixed_z).cpu().data, normalize=True)
    #     grid = np.transpose(grid.numpy(), (1, 2, 0))
    #     self.images.append(grid)
    #     imageio.mimsave('{}.gif'.format(self.dataset_name), self.images)


if __name__ == '__main__':
    from config import Config
    from data.synthetic_dataset import SyntheticDataset
    from networks.generator import Generator
    from networks.discriminator import Discriminator

    config = Config('default.yml')
    dataloader = create_data_loader(config)

    device = torch.device('cuda')

    use_dropout = [True, True, False]
    drop_prob = [0.5, 0.5, 0.5]
    use_ac_func = [True, True, False]
    activation = 'relu'
    latent_shape = 10

    gen_fc_layers = [latent_shape, 16, 32, 2]
    generator = Generator(gen_fc_layers, use_dropout, drop_prob, use_ac_func, activation).to(device)

    disc_fc_layers = [2, 32, 16, 1]
    discriminator = Discriminator(disc_fc_layers, use_dropout, drop_prob, use_ac_func, activation).to(device)

    g_optimizer = Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    d_optimizer = Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))

    wgan_gp = WGAN_GP(generator, discriminator, g_optimizer, d_optimizer, latent_shape)
    wgan_gp.train(dataloader, 100)
    import ipdb; ipdb.set_trace()


