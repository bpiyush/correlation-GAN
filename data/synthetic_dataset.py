import sys
sys.path.insert(0, '/home/users/piyushb/projects/correlation-GAN')
from config import Config
from os.path import join
import torch
import numpy as np
import argparse
from termcolor import colored
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

from utils.io import load_pkl, read_yml

THRESHOLD = 0.05

class SyntheticDataset(object):
    """docstring for SyntheticDataset"""
    def __init__(self, config):
        super(SyntheticDataset, self).__init__()
        self.config = config

        self.data_config, data_folder = self.load_data_config()

        self.preprocessor = Pipeline([('standard scaler', StandardScaler()), ('minmax scaler', MinMaxScaler(feature_range=(-1.0, 1.0)))])
        self.dataset, self.data_related_params = self.load_dataset(data_folder)

        self.data_dimension = self.data_config['dimension']
        self.input_image_side = int(np.ceil(np.sqrt(self.data_dimension)))

    def load_data_config(self):
        data_type, dataset_name = self.config.data['type'], self.config.data['name']
        data_config_path = join(self.config.paths['HOME'], 'configs/data/{}/{}.yml'.format(data_type, dataset_name))
        data_config = read_yml(data_config_path)

        data_folder = join(self.config.paths['DATA_DIR'], data_type, dataset_name)
        return data_config, data_folder

    def load_dataset(self, data_folder):
        dataset_path = join(data_folder, 'cleaned.pkl')
        global_dataset = load_pkl(dataset_path)

        dim, size = self.data_config['dimension'], self.data_config['size']
        print(colored("=> Loading synthetic {}D dataset with size {}".format(dim, size), 'yellow'))

        data_version = self.config.data['version']
        local_dataset = global_dataset[str(data_version)]

        exclude_keys = ['data']
        data_related_params = {k: local_dataset[k] for k in set(list(local_dataset.keys())) - set(exclude_keys)}
        data = local_dataset['data']

        self.preprocessor.fit(data.values)
        return data, data_related_params


    def squarify_tensor(self, tensor, out_image_side):
        assert tensor.dtype == torch.float32 and isinstance(out_image_side, int)

        flattened_image = torch.zeros(out_image_side * out_image_side).float()
        flattened_image[:tensor.shape[0]] = tensor
        image = flattened_image.reshape((self.input_image_side, self.input_image_side))

        return image

    def __len__(self):
        """Returns number of examples in dataset object"""
        return self.dataset.shape[0]

    def __getitem__(self, index):

        if self.config.data['preprocess']:
            input_ = self.preprocessor.transform(self.dataset.values[index].reshape(1, -1)).reshape(self.data_dimension)
            input_ = torch.tensor(input_).float()
        else:
            input_ = torch.tensor(self.dataset.values[index]).float()

        if self.config.arch == 'wgan_gp':
            return {'point': input_} # send out a (D, ) dimensional vector

        elif self.config.arch == 'dcgan':
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

