import sys
sys.path.insert(0, '/home/users/piyushb/projects/correlation-GAN')
from torch.utils.data import DataLoader

from config import Config
from data.synthetic_dataset import SyntheticDataset


def create_data_loader(config, drop_last=True, shuffle=True):
    if config.data['type'] == 'synthetic':
        dataset = SyntheticDataset(config)
    else:
        NotImplementedError
    dataloader = DataLoader(dataset=dataset, batch_size=config.training['batch_size'], shuffle=shuffle,
                            drop_last=drop_last, pin_memory=True, num_workers=config.system['num_workers'])

    return dataloader

if __name__ == '__main__':
    config = Config('default.yml')
    dataloader = create_data_loader(config)
    import ipdb; ipdb.set_trace()