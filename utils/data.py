import sys
import pdb
import json
import yaml
import pickle
import numpy as np
import pandas as pd
from os import makedirs
from os.path import join, exists
import matplotlib.pyplot as plt

from utils.constants import DATA_DIR, PROJECT_HOME
from utils.io import read_yml, load_pkl


def load_data_config(data_type, dataset_name):
    filepath = join(PROJECT_HOME, 'configs/data', data_type, '{}.yml'.format(dataset_name))
    return read_yml(filepath)


def load_dataset(dataset_name, data_type, state='raw'):
    
    filepath = join(DATA_DIR, data_type, dataset_name, '{}_data.csv'.format(state))
    data = pd.read_csv(filepath, index_col=False, low_memory=False)
    
    return data


def load_cleaned_dataset(dataset_name, data_type, load_subset=False):
    
    if data_type == 'real':
        fname = 'clean_data.csv' if not load_subset else 'clean_subset_data.csv'
        filepath = join(DATA_DIR, data_type, dataset_name, fname)
        data = pd.read_csv(filepath, index_col=False, low_memory=False)
    else:
        fname = 'cleaned.pkl'
        filepath = join(DATA_DIR, data_type, dataset_name, fname)
        data = load_pkl(filepath)
    
    return data


def peek_into_dataset(dataset_name, state='raw'):
    data = load_dataset(dataset_name, state)
    print(data.head(3))
    return


def sample_subset_of_dataset(data, num_examples_reqd, seed=0):
    
    np.random.seed(seed)
    idx = np.random.randint(0, data.shape[0], num_examples_reqd)
    
    return data.iloc[idx, :]


def save_subset_of_dataset(dataset_name, data_type, num_examples_reqd):
    
    data = load_dataset(dataset_name, 'clean')
    
    print("=> Sampling a subset of {} examples from dataset ...".format(num_examples_reqd))
    data_subset = sample_subset_of_dataset(data, num_examples_reqd) 
    
    print("=> Saving the new subset dataset ...")
    save_filepath = join(DATA_DIR, data_type, dataset_name, "clean_subset_data.csv")
    data_subset.to_csv(save_filepath, index=False)
    
    return


def check_data_type(column_name, all_columns, num_real):
    if not isinstance(all_columns, list):
        all_columns = list(all_columns)

    if all_columns.index(column_name) < num_real:
        return 'real'
    return 'categorical'


def check_column_type_from_config(data_config, column_name):
    return check_data_type(column_name, data_config['cleaned_columns'], data_config['num_real'])


def close_enough(val1, val2, eps=1e-4):
    if np.abs(val1 - val2) < eps:
        return True
    return False

