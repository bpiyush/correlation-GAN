import argparse
from os import makedirs
from os.path import join, basename, expanduser, splitext

from utils.constants import DATA_DIR
from utils.io import read_yml, save_yml


USER_TO_HOME = {
    'piyushb': '/home/users/{}/projects/correlation-GAN',
}

class Config(object):
    """docstring for Config"""
    def __init__(self, version, arch):
        super(Config, self).__init__()
        self.version = version
        self.arch = arch

        username = expanduser('~').split('/')[-1]
        self.paths = self.define_paths(username)

        config_path = join(self.paths['HOME'], 'configs', arch, version)
        self.__dict__.update(read_yml(config_path))

        self.data_config = self.load_data_config()

        self.data['sample_num'] = self.__dict__.get('data').get('sample_num', 500)
        self.data['sample_run'] = self.__dict__.get('data').get('sample_run', False)
        if self.data['sample_run']:
            self.data_config['size'] = self.data['sample_num']
        self.data['preprocess'] = self.__dict__.get('data').get('preprocess', False)

        # set default seed
        self.system['seed'] = self.__dict__.get('system').get('seed', 0)

        # set defaults for model decisions
        self.model['use_batch_norm'] = self.__dict__.get('model').get('use_batch_norm', True)
        self.model['use_init'] = self.__dict__.get('model').get('use_init', True)
        self.model['last_layer_activation'] = self.__dict__.get('model').get('last_layer_activation', 'tanh')

        save_yml(self.__dict__, join(self.paths['CONFIG_DIR'], 'config.yml'))

    def define_paths(self, username):
        HOME = USER_TO_HOME[username].format(username)
        COMMON_DIR = join(HOME, 'common')
        OUT_DIR = '/scratch/users/{}/correlation-GAN/{}/{}'.format(username, self.arch, splitext(self.version)[0])
        CKPT_DIR = join(OUT_DIR, 'checkpoints')
        LOG_DIR = join(OUT_DIR, 'logs')
        CONFIG_DIR = join(OUT_DIR, 'configs')
        DATA_DIR = '/home/users/piyushb/data/'

        makedirs(CONFIG_DIR, exist_ok=True)
        makedirs(CKPT_DIR, exist_ok=True)
        makedirs(LOG_DIR, exist_ok=True)

        keys = ['HOME', 'CKPT_DIR', 'LOG_DIR', 'CONFIG_DIR', 'COMMON_DIR', 'DATA_DIR']
        dir_dict = {}
        for key in keys:
            dir_dict[key] = eval(key)
        return dir_dict

    def load_data_config(self):
        data_type, dataset_name = self.data['type'], self.data['name']
        data_config_path = join(self.paths['HOME'], 'configs/data/{}/{}.yml'.format(data_type, dataset_name))
        data_config = read_yml(data_config_path)

        return data_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', type=str, default='default.yml')
    parser.add_argument('-a', '--arch', type=str, default='wgan_gp', choices=['wgan_gp', 'dcgan'])
    args = parser.parse_args()

    config = Config(args.conf, args.arch)
    import ipdb; ipdb.set_trace()