import math
import argparse
import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.insert(0, '/home/users/piyushb/projects/correlation-GAN')
from networks.generic import LinearModel


class Generator(LinearModel):
    """
    Class for definition of linear generator for structured data
    """
    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func, activation):
        super(Generator, self).__init__(fc_layers, use_dropout, drop_prob, use_ac_func, activation)
        
    def forward(self, inputs):
        return self.fc_blocks(inputs)


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class DeconvGenerator(nn.Module):
    """
    Class definition of DCGAN generator
    """
    def __init__(self, out_h, out_w, noise_dim, num_channels_prefinal, num_layers=4):
        super(DeconvGenerator, self).__init__()

        self.num_layers = num_layers
        self.noise_dim = noise_dim
        self.size_dict = [{'height': out_h, 'width': out_w, 'channels': 1}]

        for _ in range(num_layers - 1):
            next_layer_ht, next_layer_wd = self.size_dict[0]['height'], self.size_dict[0]['width']
            curr_layer_info = {'height': conv_out_size_same(next_layer_ht, 2), 'width': conv_out_size_same(next_layer_wd, 2)}
            curr_layer_info.update({'channels': int(math.pow(2, _)) * num_channels_prefinal})
            self.size_dict.insert(0, curr_layer_info)

        out_features = 4 * num_channels_prefinal * self.size_dict[0]['height'] * self.size_dict[0]['width']

        self.projector = nn.Linear(in_features=noise_dim, out_features=out_features)
        self.projector_batch_norm = self.batch_norm(self.size_dict[0]['channels'])
        self.relu = nn.ReLU(True)
        self.lrelu = nn.LeakyReLU(True)
        self.prelu = nn.PReLU()
        self.tanh = nn.Tanh()

        self.deconv_net = nn.Sequential()
        for i in range(num_layers - 1):
            in_channels = self.size_dict[i]['channels']
            out_channels = self.size_dict[i + 1]['channels']
            # padding and output_padding are chosen to make sure we get output sizes as per self.size_dict
            deconv_layer = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=2, padding=2, output_padding=1)
            self.deconv_net.add_module(name="deconv_layer_{}".format(i + 1), module=deconv_layer)

            batch_norm_layer = self.batch_norm(out_channels)
            self.deconv_net.add_module(name='batch_norm_layer_{}'.format(i + 1), module=batch_norm_layer)

            if i < num_layers - 2:
                self.deconv_net.add_module(name='activation_{}'.format(i + 1), module=self.lrelu)

        self.deconv_net.add_module(name='activation_{}'.format(num_layers - 1), module=self.tanh)


    def batch_norm(self, output_dim, eps=1e-5, momentum=0.9):
        return nn.BatchNorm2d(output_dim, eps=eps, momentum=momentum)


    def forward(self, z):
        assert z.shape[-1] == self.noise_dim

        x = self.projector(z)
        x = x.reshape((-1, self.size_dict[0]['channels'], self.size_dict[0]['height'], self.size_dict[0]['width']))
        x = self.projector_batch_norm(x)
        x = self.relu(x)

        out = self.deconv_net(x)

        return out


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True, choices=['linear-generator', 'deconv-generator'])
    args = parser.parse_args()

    if args.model == 'linear-generator':
        out_dim = 10
        fc_layers = [10, 16, 32, out_dim]
        use_dropout = [True, True, False]
        drop_prob = [0.5, 0.5, 0.5]
        use_ac_func = [True, True, False]
        activation = 'relu'

        device = torch.device('cuda')
        generator = Generator(fc_layers, use_dropout, drop_prob, use_ac_func, activation).to(device)

        noise = torch.rand((1, 10)).to(device)
        generated = generator(noise)

        assert generated.shape == (1, out_dim)
    else:
        out_h, out_w = 16, 16
        noise_dim = 100
        generator = DeconvGenerator(out_h=out_h, out_w=out_w, noise_dim=noise_dim, num_channels_prefinal=64)
        z = torch.rand((1, noise_dim))
        image = generator(z)

    import ipdb; ipdb.set_trace()
        
