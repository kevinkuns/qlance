"""
Unit tests for an Optickle polarization Sagnac
"""

import matlab.engine
import numpy as np
import pytickle.optickle as pyt
import optSagnac

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

tfCARM = opt.getTF('REFL_I', CARM)
tfDARM = opt.getTF('AS_DIFF', DARM)
qn00 = opt00.getQuantumNoise('AS_DIFF')
qn45 = opt45.getQuantumNoise('AS_DIFF')


def test_DARM():
    assert np.allclose(tfDARM, data['tfDARM'])


def test_CARM():
    assert np.allclose(tfCARM, data['tfCARM'])


def test_qn00():
    assert np.allclose(qn00, data['qn00'])


def test_qn45():
    assert np.allclose(qn45, data['qn45'])
