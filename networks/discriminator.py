import torch

import sys
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

if __name__ == '__main__':
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

    import ipdb; ipdb.set_trace()
