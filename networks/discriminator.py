import sys
import argparse
import math
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/home/users/piyushb/projects/correlation-GAN')
from networks.generic import LinearModel, Flatten

class Discriminator(LinearModel):
    """
    Class for definition of linear discriminator for structured data
    """
    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func, activation):
        super(Discriminator, self).__init__(fc_layers, use_dropout, drop_prob, use_ac_func, activation)
        
    def forward(self, inputs):
        return self.fc_blocks(inputs)

class DCDiscriminator(nn.Module):
    """docstring for DCDiscriminator"""
    def __init__(self, input_image_side, input_image_channels=1, num_channels_first=64, num_layers=4, use_batch_norm=True):
        super(DCDiscriminator, self).__init__()

        lrelu = nn.LeakyReLU()
        sigmoid = nn.Sigmoid()

        self.channel_sizes = [input_image_channels] + [num_channels_first * int(math.pow(2, i)) for i in range(num_layers)]

        # defining the conv network
        self.conv_net = nn.Sequential()
        for i in range(1, num_layers):
            in_channels = self.channel_sizes[i - 1]
            out_channels = self.channel_sizes[i]
            conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=2, padding=2)
            batch_norm_layer = self.batch_norm(out_channels)
            activation = lrelu if i < num_layers - 1 else sigmoid

            name = '%s_{}'.format(i)
            self.conv_net.add_module(name=name % 'conv_layer', module=conv_layer)
            if use_batch_norm:
                self.conv_net.add_module(name=name % 'batch_norm_layer', module=batch_norm_layer)
            self.conv_net.add_module(name=name % 'activation', module=activation)

        # defining the FCN layer
        input_image_size = (input_image_channels, input_image_side, input_image_side)
        final_flattened_size = self.get_flattened_size(input_image_size)
        self.fc_net = nn.Sequential()
        self.fc_net.add_module(name='flattener', module=Flatten())
        linear_layer = nn.Linear(in_features=final_flattened_size, out_features=1)
        self.fc_net.add_module(name='linear_layer', module=linear_layer)

    def get_flattened_size(self, input_image_size):
        temp_batch_size = 10

        image = torch.zeros((temp_batch_size, *input_image_size))
        conv_output = self.conv_net(image)
        prod_of_other_dims = np.prod(np.array([*conv_output.shape[1:]]))

        return prod_of_other_dims

    def forward(self, image, return_logits=False):
        conv_output = self.conv_net(image)
        logits = self.fc_net(conv_output)

        sigmoid = nn.Sigmoid()
        y = sigmoid(logits)

        if return_logits:
            return y, logits
        else:
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
        discriminator = DCDiscriminator()

        input_image_ht, input_image_wt = 64, 64
        image = torch.rand([1, 1, input_image_ht, input_image_wt])
        y = discriminator(image)

    import ipdb; ipdb.set_trace()

