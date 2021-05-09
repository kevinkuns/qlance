'''
utility functions for loading and saving data
'''

import numpy as np
import h5py
from numbers import Number


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


def possible_none_to_hdf5(val, key, h5file):
    if isinstance(val, dict):
        if len(val) == 0:
            none_to_hdf5(key, 'dict', h5file)
        else:
            dict_to_hdf5(val, key, h5file)
    elif isinstance(val, (np.ndarray, list)):
        if len(val) == 0:
            none_to_hdf5(key, 'list', h5file)
        else:
            h5file[key] = val
    elif isinstance(val, Number):
        h5file[key] = val
    elif val is None:
        none_to_hdf5(key, 'scalar', h5file)
    else:
        raise ValueError('Don\'t know what to do')


def hdf5_to_possible_none(key, h5file):
    if isinstance(h5file[key], h5py.Group):
        return hdf5_to_dict(h5file[key])
    elif isinstance(h5file[key][()], h5py.Empty):
        return hdf5_to_none(key, h5file)
    else:
        return h5file[key][()]


def read_foton_file(filename):
    """Read a foton filter file
    """
    f = open(filename,'r')

    #initialize the dictionary
    p = {}
    p['filename'] = filename
    # loop through file
    s = f.readline()
    while 1:
        if not s: break
        arg = s.split()

        # check type of line
        if len(arg) < 2:
            pass

        elif len(arg) > 2 and arg[0] == '#':
            if arg[1] == 'MODULES':
                # initialization of all filters
                for ii in range(2, len(arg)):
                    p[arg[ii]] = empty_filter_dict()
            elif arg[1] == 'SAMPLING':
                # sampling rate of this rt model
                p['fs'] = float(arg[3])
            elif arg[1] == 'DESIGN':
                fname = arg[2]
                ind = int(arg[3])
                if not fname in p:
                    p[fname] = {}
                if not ind in p[fname]:
                    p[fname][ind] = {}
                p[fname][ind]['design'] = ''.join(arg[4:])

        elif len(arg) == 12:
            # this is an actual filter sos definition
            fname = arg[0]
            index = int(arg[1])

            p[fname][index]['name'] = arg[6]
            p[fname][index]['gain'] = float(arg[7])
            gain = float(arg[7])
            soscoef = np.array(arg[8:12], dtype='float')
            order = int(arg[3])
            for kk in range(0, order - 1):
                s = f.readline()
                arg = s.split()
                temp = np.array(arg[0:4], dtype='float')
                soscoef = np.vstack((soscoef, temp))
            if order == 1:
                soscoef = np.vstack((soscoef, soscoef)) #for indexing convenience

            # reshape the sos coeffs
            coef = np.zeros((order,6))
            for jj in range(order):
                if jj == 0:
                    coef[jj][0] = gain
                    coef[jj][1] = gain*soscoef[jj][2]
                    coef[jj][2] = gain*soscoef[jj][3]
                else:
                    coef[jj][0] = 1.
                    coef[jj][1] = soscoef[jj][2]
                    coef[jj][2] = soscoef[jj][3]
                coef[jj][3] = 1.
                coef[jj][4] = soscoef[jj][0]
                coef[jj][5] = soscoef[jj][1]
            p[fname][index]['sos_coeffs'] = np.squeeze(coef)
        s = f.readline()

    f.close()
    return p


def empty_filter_dict():
    '''
    returns an "empty" filter bank structure for initialization
    '''
    fb = {}
    for ii in range(10):
        fb[ii] = {}
        fb[ii]['name']='<empty>'
        fb[ii]['sos_coeffs'] = np.array([1.,0.,0.,1.,0.,0.])
        fb[ii]['fs'] = 16384
        fb[ii]['design'] = 'none'
    return fb
