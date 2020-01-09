"""
Unit tests for optickle PDH frequency response and sweeps
"""

import matlab.engine
import numpy as np
import pytickle.optickle as pyt
import pytest

eng = matlab.engine.start_matlab()
pyt.addOpticklePath(eng)


fmod = 11e3
gmod = 0.1
vRF = np.array([-fmod, 0, fmod])
Pin = 1
Ti = 0.01
Lcav = 40e3

data = np.load('data/optickle_PDH_data.npz')


def optFP(opt_name):
    opt = pyt.PyTickle(eng, opt_name, vRF)

    opt.addMirror('IX', Thr=Ti)
    opt.addMirror('EX', Thr=0)
    opt.addLink('IX', 'fr', 'EX', 'fr', Lcav)
    opt.addLink('EX', 'fr', 'IX', 'fr', Lcav)
    opt.setCavityBasis('EX', 'IX')

    opt.addSource('Laser', np.sqrt(Pin)*(vRF == 0))
    opt.addRFmodulator('Mod', fmod, gmod*1j)
    opt.addLink('Laser', 'out', 'Mod', 'in', 0)
    opt.addLink('Mod', 'out', 'IX', 'bk', 0)

    opt.addSink('REFL')
    opt.addLink('IX', 'bk', 'REFL', 'in', 0)
    opt.addReadout('REFL', fmod, 0)

    return opt


class TestFreqResp:

    opt = optFP('optFR')
    ff = np.logspace(-2, 4, 1000)
    opt.tickle(ff, noise=False)

    def test_tfI(self):
        tfI = self.opt.getTF('REFL_I', 'EX')
        assert np.allclose(tfI, data['tfI'])

    def test_tfQ(self):
        tfQ = self.opt.getTF('REFL_Q', 'EX')
        assert np.allclose(tfQ, data['tfQ'])


class TestSweep:

    opt = optFP('optSweep')
    ePos = {'EX': 5e-9}
    sPos = {k: -v for k, v in ePos.items()}
    opt.sweepLinear(sPos, ePos, 1000)

    def test_sweepI(self):
        poses, sweepI = self.opt.getSweepSignal('REFL_I', 'EX')
        assert np.allclose(sweepI, data['sweepI'])

    def test_sweepQ(self):
        poses, sweepQ = self.opt.getSweepSignal('REFL_Q', 'EX')
        assert np.allclose(sweepQ, data['sweepQ'])
