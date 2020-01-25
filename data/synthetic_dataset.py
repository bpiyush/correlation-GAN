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

    def __len__(self):
        """Returns number of examples in dataset object"""
        return self.dataset['data'].shape[0]

    def __getitem__(self, index):
        return {'point': torch.tensor(self.dataset['data'].values[index]).float()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', type=str, default='default.yml')
    args = parser.parse_args()

    config = Config(args.conf)
    dataset = SyntheticDataset(config)
    import ipdb; ipdb.set_trace()


