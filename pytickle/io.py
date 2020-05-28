'''
utility functions for loading and saving data
'''

import numpy as np
import h5py


def str2byte(str_list):
    """Convert a list of strings to a byte list for hdf5 saving
    """
    byte_list = [np.string_(val) for val in str_list]
    return byte_list


def byte2str(byte_list):
    """Convert a byte list loaded from hdf5 into a normal string list
    """
    str_list = [str(val, 'utf-8') for val in byte_list]
    return str_list


def dict_to_hdf5(dictionary, path, h5file):
    """Save a dictionary to hdf5

    Inputs:
      dictionary: the dictionary
      path: the full path or key to use
      h5file: the hdf5 file
    """
    # FIXME: ordered dicts as well?
    for key, val in dictionary.items():
        if isinstance(val, dict):
            dict_to_hdf5(val, path + '/' + key, h5file)
        else:
            h5file[path + '/' + key] = val


def hdf5_to_dict(h5_group):
    """Load an hdf5 group as a dict

    Inputs:
      h5_group: the Group

    Returns:
      dictionary: a normal dict
    """
    dictionary = {}
    for key, val in h5_group.items():
        if isinstance(val, h5py.Group):
            dictionary[key] = hdf5_to_dict(val)
        else:
            dictionary[key] = val[()]
    return dictionary


def none_to_hdf5(key, kind, h5file):
    """Save a None variable or empty dictionary to hdf5

    Inputs:
      key: Dataset key
      kind: type of variable:
        * 'scalar': for a scalar, i.e. None
        * 'list': for a list, i.e. []
        * 'dict': for a dict, i.e. {}
      h5file: the hdf5 file
    """
    h5file.create_dataset(key, data=h5py.Empty('f'))
    h5file[key].attrs['type'] = kind


def hdf5_to_none(key, h5file):
    """Load an hdf5 None variable or empty dictionary

    Inputs:
      key: Dataset keys
      h5file: the hdf5 file

    Returns: None (if it's a scalar) or {} (if it's a dict)
    """
    # do not get the the element with h5file[key][()]
    # this doesn't have attrs
    val = h5file[key]
    if isinstance(val[()], h5py.Empty):
        # return None
        kind = val.attrs['type']
        if kind == 'scalar':
            return None
        elif kind == 'list':
            return []
        elif kind == 'dict':
            return {}
        else:
            raise ValueError('Unrecognized None type')
    else:
        raise ValueError('This dataset is not None')


# just hack this for now and make more general later
# this is only going to work for the current data structures with care
def mech_plants_to_hdf5(dictionary, path, h5file):
    for key, val in dictionary.items():
        if isinstance(val, dict):
            mech_plants_to_hdf5(val, path + '/' + key, h5file)
        else:
            if key in (['zs', 'ps']):
                if len(val) == 0:
                    none_to_hdf5(path + '/' + key, 'list', h5file)
                else:
                    h5file[path + '/' + key] = val
            else:
                h5file[path + '/' + key] = val


def hdf5_to_mech_plants(h5_group, h5file):
    dictionary = {}
    for key, val in h5_group.items():
        if isinstance(val, h5py.Group):
            dictionary[key] = hdf5_to_mech_plants(val, h5file)
        else:
            if key in (['zs', 'ps']):
                if isinstance(val[()], h5py.Empty):
                    # path = h5_group.name + '/' + key
                    # dictionary[key] = hdf5_to_none(path, h5file)
                    dictionary[key] = []
                else:
                    dictionary[key] = val[()]
            else:
                dictionary[key] = val[()]
    return dictionary


def hdf5_to_noiseAC(h5_group, h5file):
    # yes, this actually works; super hacky...
    if isinstance(h5_group, h5py.Dataset):
        return {}
    else:
        return hdf5_to_mech_plants(h5_group, h5file)


def possible_none_to_hdf5(val, key, kind, h5file):
    if kind == 'dict':
        if len(val) == 0:
            none_to_hdf5(key, kind, h5file)
        else:
            dict_to_hdf5(val, key, h5file)
    elif kind == 'list':
        if val is None:
            none_to_hdf5(key, kind, h5file)
        else:
            h5file[key] = val


def hdf5_to_possible_none(key, kind, h5file):
    if isinstance(h5file[key][()], h5py.Empty):
        return hdf5_to_none(key, h5file)
    elif kind == 'list':
        return h5file[key][()]
    elif kind == 'dict':
        return hdf5_to_dict(h5file[key])
        # return hdf5_to_possible_none(key, 'dict', h5file)

# # def possible_none_to_hdf5(val, key, h5file):
# #     if isinstance(val, (list, np.ndarray)):
# #         if len(list) == 0:
# #             none_to_hdf5(key, 'list', h5file)
# #         else:
# #             h5file[key] = val
# #     elif isinstance(val, dict):
# #         if len(val) == 0:
# #             none_to_hdf5(key, 'dict', h5file)
# #         else:
# #             dict_to_hdf5(possible_none_to_hdf5(val), key, h5file)
# #     else:
# #         if val is None:
# #             none_to_hdf5(key, 'scalar', h5file)
# #         else:
# #             h5file[key] = val


# # # def hdf5_to_possible_none(key, h5file):
