from os import makedirs
from os.path import join, basename, expanduser, splitext

from utils.constants import DATA_DIR
from utils.io import read_yml


USER_TO_HOME = {
    'piyushb': '/home/users/{}/projects/correlation-GAN',
}

class Config(object):
    """docstring for Config"""
    def __init__(self, version):
        super(Config, self).__init__()
        self.version = version

        username = expanduser('~').split('/')[-1]
        self.paths = self.define_paths(username)

        config_path = join(self.paths['HOME'], 'configs', version)
        self.__dict__.update(read_yml(config_path))

        self.checkpoint_dir = join(self.paths['CKPT_DIR'], splitext(self.version)[0])
        self.log_dir = join(self.paths['LOG_DIR'], self.version)

        for path in [self.checkpoint_dir, self.log_dir]:
            makedirs(path, exist_ok=True)

    def define_paths(self, username):
        HOME = USER_TO_HOME[username].format(username)
        COMMON_DIR = join(HOME, 'common')
        OUT_DIR = '/scratch/users/{}/correlation-GAN/'.format(username)
        CKPT_DIR = join(OUT_DIR, 'checkpoints')
        LOG_DIR = join(OUT_DIR, 'logs')
        CONFIG_DIR = join(OUT_DIR, 'configs')
        DATA_DIR = '/home/users/piyushb/data/'

        makedirs(CONFIG_DIR, exist_ok=True)
        keys = ['HOME', 'CKPT_DIR', 'LOG_DIR', 'CONFIG_DIR', 'COMMON_DIR', 'DATA_DIR']
        dir_dict = {}
        for key in keys:
            dir_dict[key] = eval(key)
        return dir_dict

if __name__ == '__main__':
    config = Config('default.yml')
    import ipdb; ipdb.set_trace()