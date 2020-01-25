import sys
import argparse
import math
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/home/users/piyushb/projects/correlation-GAN')
from networks.generic import LinearModel

class Discriminator(LinearModel):
    """
    Class for definition of linear discriminator for structured data
    """
    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func, activation):
        super(Discriminator, self).__init__(fc_layers, use_dropout, drop_prob, use_ac_func, activation)
        
    def forward(self, inputs):
        return self.fc_blocks(inputs)

class ConvDiscriminator(nn.Module):
    """docstring for ConvDiscriminator"""
    def __init__(self, num_channels_first=64, num_layers=4, input_image_channels=1):
        super(ConvDiscriminator, self).__init__()

        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        self.channel_sizes = [input_image_channels] + [num_channels_first * int(math.pow(2, i)) for i in range(num_layers)]

        self.conv_net = nn.Sequential()

        for i in range(1, num_layers):
            in_channels = self.channel_sizes[i - 1]
            out_channels = self.channel_sizes[i]
            conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=2, padding=2)
            batch_norm_layer = self.batch_norm(out_channels)
            activation = self.lrelu if i < num_layers - 1 else self.sigmoid

            name = '%s_{}'.format(i)
            self.conv_net.add_module(name=name % 'conv_layer', module=conv_layer)
            self.conv_net.add_module(name=name % 'batch_norm_layer', module=batch_norm_layer)
            self.conv_net.add_module(name=name % 'activation', module=activation)

    def forward(self, image):
        conv_output = self.conv_net(image)
        prod_of_other_dims = np.prod(np.array([*conv_output.shape[1:]]))
        conv_output = conv_output.reshape((-1, prod_of_other_dims))
        self.linear = nn.Linear(in_features=prod_of_other_dims, out_features=1)
        y = self.sigmoid(self.linear(conv_output))

        return y


    def batch_norm(self, output_dim, eps=1e-5, momentum=0.9):
        return nn.BatchNorm2d(output_dim, eps=eps, momentum=momentum)

        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True, choices=['linear-discriminator', 'conv-discriminator'])
    args = parser.parse_args()

    if args.model == 'linear-discriminator':
        in_dim = 10
        out_dim = 1
        fc_layers = [in_dim, 32, 16, out_dim]
        use_dropout = [True, True, False]
        drop_prob = [0.5, 0.5, 0.5]
        use_ac_func = [True, True, False]
        activation = 'relu'

        device = torch.device('cuda')
        discriminator = Discriminator(fc_layers, use_dropout, drop_prob, use_ac_func, activation).to(device)

        input_ = torch.rand((1, in_dim)).to(device)
        disc_output = discriminator(input_)

        assert disc_output.shape == (1, out_dim)

    else:
        discriminator = ConvDiscriminator()

        input_image_ht, input_image_wt = 64, 64
        image = torch.rand([1, 1, input_image_ht, input_image_wt])
        y = discriminator(image)

    import ipdb; ipdb.set_trace()

