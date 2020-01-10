import sys
sys.path.insert(0, '/home/users/piyushb/projects/correlation-GAN')
from config import Config
from os.path import join
import torch

from utils.io import load_pkl

THRESHOLD = 0.05

class SyntheticDataset(object):
    """docstring for SyntheticDataset"""
    def __init__(self, config, mode):
        super(SyntheticDataset, self).__init__()
        self.config = config
        self.mode = mode

        self.dataset, self.key = self.load_dataset()

    def load_dataset(self):
        dataset_path = join(self.config.paths['DATA_DIR'], self.config.data_type, self.config.dataset_name, self.config.dataset_version + '.pkl')
        dataset = load_pkl(dataset_path)

        for key in dataset:
            if key == 'metainfo':
                continue
            if self.config.rho - THRESHOLD < dataset[key]['rho'] <= self.config.rho + THRESHOLD:
                return dataset[key], key

        raise Exception('This dataset does not have any sub-dataset with given rho.')

    def __len__(self):
        """Returns number of examples in dataset object"""
        return self.dataset['data'].shape[0]

    def __getitem__(self, index):
        return {'point': torch.tensor(self.dataset['data'].values[index]).float()}


if __name__ == '__main__':
    config = Config('default.yml')
    dataset = SyntheticDataset(config, 'train')
    import ipdb; ipdb.set_trace()


