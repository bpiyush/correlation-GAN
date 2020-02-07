import sys
sys.path.insert(0, '/home/users/piyushb/projects/correlation-GAN')
from config import Config
from os.path import join
import torch
import numpy as np
import argparse
from termcolor import colored

from utils.io import load_pkl

THRESHOLD = 0.05

class SyntheticDataset(object):
    """docstring for SyntheticDataset"""
    def __init__(self, config):
        super(SyntheticDataset, self).__init__()
        self.config = config

        self.dataset, self.key = self.load_dataset()

        self.data_dimension = self.config.data['dimension']
        self.input_image_side = int(np.ceil(np.sqrt(self.data_dimension)))

    def load_dataset(self):
        dataset_path = join(self.config.paths['DATA_DIR'], self.config.data['type'], self.config.data['name'], self.config.data['version'] + '.pkl')
        dataset = load_pkl(dataset_path)

        for key in dataset:
            if key == 'metainfo':
                continue
            if self.config.data['rho'] - THRESHOLD < dataset[key]['rho'] <= self.config.data['rho'] + THRESHOLD:
                print(colored("=> Loading 2D synthetic dataset with rho: {}".format(np.round(self.config.data['rho'], 3)), 'yellow'))
                return dataset[key], key

        raise Exception('This dataset does not have any sub-dataset with given rho.')

    def squarify_tensor(self, tensor, out_image_side):
        assert tensor.dtype == torch.float32 and isinstance(out_image_side, int)

        flattened_image = torch.zeros(out_image_side * out_image_side).float()
        flattened_image[:tensor.shape[0]] = tensor
        image = flattened_image.reshape((self.input_image_side, self.input_image_side))

        return image

    def __len__(self):
        """Returns number of examples in dataset object"""
        return self.dataset['data'].shape[0]

    def __getitem__(self, index):

        if self.config.arch == 'wgan_gp':
            return {'point': torch.tensor(self.dataset['data'].values[index]).float()}
        elif self.config.arch == 'dcgan':
            input_ = torch.tensor(self.dataset['data'].values[index]).float()
            image = self.squarify_tensor(input_, self.input_image_side)
            return {'image': image.unsqueeze(0)} # Send out (1, side, side) sized image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', type=str, default='default.yml')
    parser.add_argument('-a', '--arch', type=str, default='wgan_gp', choices=['wgan_gp', 'dcgan'])
    args = parser.parse_args()

    config = Config(args.conf, args.arch)
    dataset = SyntheticDataset(config)

    instance = dataset[0]
    import ipdb; ipdb.set_trace()

