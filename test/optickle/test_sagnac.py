"""
Unit tests for an Optickle polarization Sagnac
"""

import matlab.engine
import numpy as np
import pytickle.optickle as pyt
import pytickle.plant as plant
import optSagnac
import os
import close
import pytest

eng = matlab.engine.start_matlab()
pyt.addOpticklePath(eng)

data = np.load('data/optickle_sagnac_data.npz')

ff = data['ff']
CARM = {'EX': 1, 'EY': 1}
DARM = {'EX': 1, 'EY': -1}

opt = optSagnac.optSagnac(eng, 'opt')
opt00 = optSagnac.optSagnac(eng, 'opt00', 0, 6, 10)
opt45 = optSagnac.optSagnac(eng, 'opt45', 45, 6, 10)

opt.run(ff, noise=False)
opt00.run(ff)
opt45.run(ff)

opt.save('test_sagnac.hdf5')
opt2 = plant.OpticklePlant()
opt2.load('test_sagnac.hdf5')
os.remove('test_sagnac.hdf5')

opt00.save('test_sagnac_00.hdf5')
opt00_2 = plant.OpticklePlant()
opt00_2.load('test_sagnac_00.hdf5')
os.remove('test_sagnac_00.hdf5')

opt45.save('test_sagnac_45.hdf5')
opt45_2 = plant.OpticklePlant()
opt45_2.load('test_sagnac_45.hdf5')
os.remove('test_sagnac_45.hdf5')


tfCARM = opt.getTF('REFL_I', CARM)
tfDARM = opt.getTF('AS_DIFF', DARM)
qn00 = opt00.getQuantumNoise('AS_DIFF')
qn45 = opt45.getQuantumNoise('AS_DIFF')


def test_DARM():
    assert close.allclose(tfDARM, data['tfDARM'])


def test_CARM():
    assert close.allclose(tfCARM, data['tfCARM'])


def test_qn00():
    assert close.allclose(qn00, data['qn00'])


def test_qn45():
    assert close.allclose(qn45, data['qn45'])


def test_reload_DARM():
    assert close.allclose(tfDARM, opt2.getTF('AS_DIFF', DARM))


def test_reload_CARM():
    assert close.allclose(tfCARM, opt2.getTF('REFL_I', CARM))


def test_reload_qn00():
    assert close.allclose(qn00, opt00_2.getQuantumNoise('AS_DIFF'))


def test_reload_qn45():
    assert close.allclose(qn45, opt45_2.getQuantumNoise('AS_DIFF'))
