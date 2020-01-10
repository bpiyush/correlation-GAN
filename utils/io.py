import yaml
import pickle
import json

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