"""
Code for interfacing with matlab from within python
"""


import numpy as np
import matlab
from numbers import Number

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
        cmd1 = "OPTICKLE_PATH__ = getenv('OPTICKLE_PATH');"
    else:
        cmd1 = "OPTICKLE_PATH__ = {:s};".format(str2mat(path))
    cmd2 = "addpath(genpath(OPTICKLE_PATH__));"
    eng.eval(cmd1, nargout=0)
    eng.eval(cmd2, nargout=0)
