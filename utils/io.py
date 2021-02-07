import yaml
import pickle
import json
from os import makedirs
from os.path import join
from collections import OrderedDict
import torch

def load_pkl(path, encoding='ascii'):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding=encoding)

    return data


def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


def read_yml(filepath):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    return data


def save_yml(data, path):
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return


def _get_state_dict(model_dict):
    result = OrderedDict()

    for (k, v) in model_dict.items():
        result[k] = v

    return result


def save_model(model, model_name, optimizer, epoch_num, metrics, parent_folder, save_mode):

    makedirs(parent_folder, exist_ok=True)

    save_fname = '{}_'.format(epoch_num) * (save_mode != 'best') + '{}_{}.pth.tar'.format(model_name, save_mode)

    model_save_path = join(parent_folder, save_fname)
    model_state_dict = _get_state_dict(model.state_dict())
    model_state =  {
        'epoch': epoch_num + 1,
        'state_dict': model_state_dict,
        'metrics': metrics,
        'optimizer' : optimizer.state_dict()
    }

    torch.save(model_state, model_save_path)


def copy_state_dict(cur_state_dict, pre_state_dict, prefix=''):
    def _get_params(key):
        key = prefix + key
        try:
            out = pre_state_dict[key]
        except:
            try:
                out = pre_state_dict[key[7:]]
            except:
                try:
                    out = pre_state_dict["module." + key]
                except:
                    try:
                        out = pre_state_dict[key[14:]]
                    except:
                        out = None
        return out

    for k in cur_state_dict.keys():
        # import pdb;pdb.set_trace()
        v = _get_params(k)
        try:
            if v is None:
                print('parameter {} not found'.format(k))
                continue
            cur_state_dict[k].copy_(v)
        except:
            print('copy param {} failed'.format(k))
            continue


def load_model(new_model, saved_model_path):
    generator_cgt_kp = torch.load(saved_model_path)
    copy_state_dict(new_model.state_dict(), generator_cgt_kp['state_dict'], prefix='')

