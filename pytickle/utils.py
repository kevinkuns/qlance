'''
Some misc. functions.
'''

import numpy as np
import matlab
from numbers import Number
from collections import OrderedDict
from copy import deepcopy


def mat2py(mat_arr):
    """Convert matlab arrays into python arrays.
    """
    return np.squeeze(np.array(mat_arr))


def py2mat(py_arr, is_complex=False):
    """Convert python arrays into matlab arrays.
    """
    if isinstance(py_arr, list):
        mat_arr = py_arr
    elif isinstance(py_arr, np.ndarray):
        mat_arr = py_arr.tolist()
        if isinstance(mat_arr, Number):
            mat_arr = [mat_arr]
    elif isinstance(py_arr, Number):
        mat_arr = [py_arr]
    else:
        raise TypeError('Invalid type ' + str(type(py_arr)) + ' for py_arr')
    return matlab.double(mat_arr, is_complex=is_complex)


def str2mat(string):
    """Ensure that strings are properly formatted for matlab.
    """
    return ''.join("'{0}'".format(string))


def addOpticklePath(eng, path=None):
    """Add the Optickle path to the matlab path

    This must be run once everytime a matlab engine is initialized

    Inputs:
      eng: the matlab engine
      path: If None (default) the path is taken from 'OPTICKLE_PATH'
            if a string, adds that string to the path
    """
    if path is None:
        cmd = "addpath(genpath(getenv('OPTICKLE_PATH')));"
    else:
        cmd = "addpath(genpath({:s}));".format(str2mat(path))
    eng.eval(cmd, nargout=0)


def normalizeDOF(dof):
    """Normalize a degree of freedom for use in transfer functions, etc
    """
    vals = np.array(list(dof.values()))
    norm = np.sum(np.abs(vals))
    dof_normalized = OrderedDict({k: v / norm for k, v in dof.items()})
    return dof_normalized


def remove_conjugates(arr):
    """Removes the complex conjugates from an array of complex numbers

    Raises an error if not all of the complex numbers have conjugates.
    Purely real numbers are unchanged.

    Useful for defining Finesse transfer functions because Finesse
    automatically adds the conjugate poles or zeros and so must be defined
    with only one of the conjugates.

    Inputs:
      arr: the array of complex numbers

    Returns:
      an array with the real and complex numbers with positive imaginary part
    """
    arr = np.array(arr)
    real = arr[np.isreal(arr)]
    pos = np.sort(arr[arr.imag > 0])
    neg = np.sort(arr[arr.imag < 0].conj())

    # make sure all of the complex numbers have conjugates
    if len(pos) != len(neg):
        raise ValueError('Not all of the complex numbers have conjugates')
    if not np.allclose(pos, neg):
        raise ValueError('Some numbers have unequal conjugates')

    return np.hstack((real, pos))


def add_conjugates(arr):
    """Adds the conjugates to an array of complex numbers ignoring real numbers

    This is effectively the inverse of remove_conjugates
    """
    arr = np.array(arr)
    inds = np.iscomplex(arr)
    return np.hstack((arr, arr[inds].conj()))


def mag2db(arr, pow=False):
    """Convert magnidude to decibels
    """
    if pow:
        return 10 * np.log10(arr)
    else:
        return 20 * np.log10(arr)


def siPrefix(num, tex=False):
    """Breaks a number up in SI notation

    Returns:
      pref: the SI prefix, e.g. k for kilo, n for nano, etc.
      num: the base number
      tex: if True the micro prefix is '$\mu$' instead of 'u'

    Example
      siPrefix(1300) = 'k', 1.3
      siPrefix(2e-10) = 'p', 200
    """
    if num == 0:
        exp = 0
    else:
        exp = np.floor(np.log10(np.abs(num)))
    posPrefixes = ['', 'k', 'M', 'G', 'T']
    negPrefixes = ['m', 'u', 'n', 'p']
    try:
        if np.sign(exp) >= 0:
            ind = int(np.abs(exp) // 3)
            pref = posPrefixes[ind]
            num = num / np.power(10, 3*ind)
        else:
            ind = int((np.abs(exp) - 1) // 3)
            pref = negPrefixes[ind]
            num = num * np.power(10, 3*(ind + 1))
    except IndexError:
        pref = ''
    if tex:
        if pref == 'u':
            pref = r'$\mu$'
    return pref, num


def assertType(data, dtype):
    """Convert some data into a specified type

    Converts a single data point to a list, dictionary or OrderedDict
    of a single element

    Inputs:
      data: the data
      dtype: the type of data: list, dict, or OrderedDict

    Returns:
      data: the converted data
    """
    data = deepcopy(data)
    if isinstance(data, str):
        if dtype == list:
            data = [data]
        elif dtype == dict:
            data = {data: 1}
        elif dtype == OrderedDict:
            data = OrderedDict({data: 1})
    return data


def append_str_if_unique(array, elements):
    """Append elements to an array only if that element is unique

    Inputs:
      array: the array to which the elements should be appended
      elements: the elements to append to the array
    """
    if isinstance(elements, str):
        elements = [elements]
    elif isinstance(elements, dict) or isinstance(elements, OrderedDict):
        elements = elements.keys()

    for element in elements:
        if element not in array:
            array.append(element)



def printLine(arr, pad):
    """Helper function for showfDC
    """
    line = ''
    for elem in arr:
        pref, num = siPrefix(np.abs(elem)**2)
        pad1 = pad - len(pref)
        line += '{:{pad1}.1f} {:s}W|'.format(num, pref, pad1=pad1)
    return line


def printHeader(freqs, pad):
    """Helper function for showfDC
    """
    line = ''
    if len(freqs.shape) == 0:
        freqs = [freqs]
    for freq in freqs:
        pref, freq = siPrefix(freq)
        freq = round(freq)
        pad1 = pad - len(pref)
        if freq == 0:
            line += '{:>{pad}s}|'.format('DC', pad=(pad + 3))
        else:
            line += '{:+{pad1}.0f} {:s}Hz|'.format(freq, pref, pad1=pad1)
    return line
