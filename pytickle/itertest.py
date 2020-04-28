import numpy as np
import itertools


m1 = np.ones((2, 2))
m2 = np.ones((2, 2, 8))
m3 = np.ones((2, 2)) * 2
m4 = np.ones((2, 2, 8)) * 3
m5 = np.ones((2, 2)) * 4


def getTestPoints(sigTo, sigFrom):
    tstpnts = itertools.cycle(['err', 'ctrl', 'comp', 'drive', 'sens'])
    start = False
    blap = []
    for tstpnt in tstpnts:
        if tstpnt != sigFrom and not start:
            continue
        start = True
        blap.append(tstpnt)
        if tstpnt == sigTo:
            break
    return blap


def getTF(sigTo, sigFrom):
    tstpnts = itertools.cycle(['err', 'ctrl', 'comp', 'drive', 'sens'])
    mats = itertools.cycle([m1, m2, m3, m4, m5])
    start = False
    tf = 1
    for tstpnt, mat in zip(tstpnts, mats):
        if tstpnt != sigFrom and not start:
            continue
        start = True
        tf = multiplyMat(tf, mat)
        if tstpnt == sigTo:
            break
    return tf


def multiplyMat(mat1, mat2):
    if np.isscalar(mat1) or np.isscalar(mat2):
        return mat1*mat2

    dim1 = len(mat1.shape)
    dim2 = len(mat2.shape)
    str1 = ''.join(['i', 'j', 'f'][:dim1])
    str2 = ''.join(['j', 'k', 'f'][:dim2])
    str3 = ''.join(['i', 'k', 'f'][:max(dim1, dim2)])
    cmd = '{:s},{:s}->{:s}'.format(str1, str2, str3)
    return np.einsum(cmd, mat1, mat2)
