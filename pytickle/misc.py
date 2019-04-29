'''
Some misc. functions.
'''

from __future__ import division
import numpy as np


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
