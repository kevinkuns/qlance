"""
Convenience functions for checking if values are close
"""

import numpy as np


def _close(val1, val2, func, *args, **kwargs):
    norm = np.median(np.abs(val1))
    if norm == 0:
        return np.all(val2 == np.zeros_like(val2))
    else:
        return func(val1/norm, val2/norm, *args, **kwargs)


def isclose(val1, val2, *args, **kwargs):
    return _close(val1, val2, np.isclose, *args, **kwargs)


def allclose(val1, val2, *args, **kwargs):
    return _close(val1, val2, np.allclose, *args, **kwargs)


def check_dicts(dict1, dict2):
    if len(dict1) != len(dict2):
        raise ValueError('dictionaries are definitely not the same')

    equal = []
    for key, val1 in dict1.items():
        equal.append(allclose(val1, dict2[key]))
    return equal
