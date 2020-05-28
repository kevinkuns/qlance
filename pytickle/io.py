'''
utility functions for loading and saving data
'''

import numpy as np
import h5py


def str2byte(str_list):
    byte_list = [np.string_(val) for val in str_list]
    return byte_list


def byte2str(byte_list):
    str_list = [str(val, 'utf-8') for val in byte_list]
    return str_list


def dict_to_hdf5(dictionary, path, h5file):
    # FIXME: ordered dicts as well?
    for key, val in dictionary.items():
        if isinstance(val, dict):
            dict2hdf5(val, path + '/' + key, h5file)
        else:
            h5file[path + '/' + key] = val


def hdf5_to_dict(h5_group):
    dictionary = {}
    for key, val in h5_group.items():
        if isinstance(val, h5py.Group):
            dictionary[key] = hdf5_to_dict(val)
        else:
            dictionary[key] = val[()]
    return dictionary
