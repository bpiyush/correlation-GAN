import torch

from generic import LinearModel

class Generator(LinearModel):
    """
    Class for definition of linear generator for structured data
    """
    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func, activation):
        super(Generator, self).__init__(fc_layers, use_dropout, drop_prob, use_ac_func, activation)
        
    def forward(self, inputs):
        return self.fc_blocks(inputs)

if __name__ == '__main__':
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

    import ipdb; ipdb.set_trace()
        
