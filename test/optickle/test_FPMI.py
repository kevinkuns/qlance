"""
Unit tests for optickle FPMI frequency response and sweeps
"""

import matlab.engine
import numpy as np
import pytickle.optickle as pyt
import close
import pytest

eng = matlab.engine.start_matlab()
pyt.addOpticklePath(eng)


data = np.load('data/optickle_FPMI_data.npz', allow_pickle=True)

fmod = 55e6
gmod = 0.1
vRF = np.array([-fmod, 0, fmod])
Pin = 1
Ti = 0.1
Lcav = 40e3
lm = 3
ls = 1.36
lx = lm + ls/2
ly = lm - ls/2


def optMI(opt_name):
    opt = pyt.PyTickle(eng, opt_name, vRF)

    opt.addMirror('EX', Thr=0)
    opt.addMirror('EY', Thr=0)
    opt.addMirror('IX', Thr=Ti)
    opt.addMirror('IY', Thr=Ti)
    opt.addBeamSplitter('BS', aoi=45, Thr=0.5)

    opt.addSource('Laser', np.sqrt(Pin)*(vRF == 0))
    opt.addRFmodulator('Mod', fmod, gmod*1j)
    opt.addLink('Laser', 'out', 'Mod', 'in', 0)
    opt.addLink('Mod', 'out', 'BS', 'frA', 1)

    opt.addLink('BS', 'bkA', 'IX', 'bk', lx)
    opt.addLink('IX', 'fr', 'EX', 'fr', Lcav)
    opt.addLink('EX', 'fr', 'IX', 'fr', Lcav)
    opt.addLink('IX', 'bk', 'BS', 'bkB', lx)

    opt.addLink('BS', 'frA', 'IY', 'bk', ly)
    opt.addLink('IY', 'fr', 'EY', 'fr', Lcav)
    opt.addLink('EY', 'fr', 'IY', 'fr', Lcav)
    opt.addLink('IY', 'bk', 'BS', 'frB', ly)

    opt.addSink('AS')
    opt.addLink('BS', 'bkB', 'AS', 'in', 1)
    opt.addReadout('AS', fmod, 0)

    return opt


class TestFreqResp:

    opt = optMI('optFR')
    ff = np.logspace(0, 4, 400)
    opt.run(ff, noise=False)

    def test_tfQ_EM(self):
        tfQ_EM = self.opt.getTF('AS_Q', {'EX': 0.5, 'EY': -0.5})
        assert close.allclose(tfQ_EM, data['tfQ_EM'])

    def test_tfI_EM(self):
        tfI_EM = self.opt.getTF('AS_I', {'EX': 0.5, 'EY': -0.5})
        assert close.allclose(tfI_EM, data['tfI_EM'])

    def test_tfQ_IM(self):
        tfQ_IM = self.opt.getTF('AS_Q', {'IX': 0.5, 'IY': -0.5})
        assert close.allclose(tfQ_IM, data['tfQ_IM'])


class TestSchnupp:
    lmean = (lx + ly)/2
    lasy = np.linspace(0, 3, 100)
    lxx = lmean + lasy/2
    lyy = lmean - lasy/2

    optAsy = optMI('optAsy')

    powf0 = np.zeros_like(lasy)
    powfu = np.zeros_like(lasy)
    powfl = np.zeros_like(lasy)

    for ii in range(len(lasy)):
        optAsy.setLinkLength('BS', 'IX', lxx[ii])
        optAsy.setLinkLength('IX', 'BS', lxx[ii])
        optAsy.setLinkLength('BS', 'IY', lyy[ii])
        optAsy.setLinkLength('IY', 'BS', lyy[ii])

        optAsy.run(1, noise=False)
        powf0[ii] = optAsy.getDCpower('BS', 'AS', 0)
        powfu[ii] = optAsy.getDCpower('BS', 'AS', fmod)
        powfl[ii] = optAsy.getDCpower('BS', 'AS', -fmod)

    def test_powf0(self):
        assert close.allclose(self.powf0, data['powf0'])

    def test_powfu(self):
        assert close.allclose(self.powfu, data['powfu'])

    def test_powfl(self):
        assert close.allclose(self.powfl, data['powfl'])


def computeSweep(ePos):
    optSweep = optMI('optSweep')
    # ePos = {drive: 180 * optSweep.lambda0[0]/360}
    sPos = {k: -v for k, v in ePos.items()}
    optSweep.sweepLinear(sPos, ePos, 1001)
    return optSweep


def getSweepPowers(opt, drive):
    locs = {'XARM': ('IX', 'EX'),
            'YARM': ('IY', 'EY'),
            'BSX': ('BS', 'IX'),
            'BSY': ('BS', 'IY'),
            'AS': ('BS', 'AS')}
    # locs = ['XARM', 'YARM', 'BSX', 'BSY', 'AS']
    powers = {loc: {} for loc in locs.keys()}
    poses = {}
    for loc, (sopt, eopt) in locs.items():
        poses[loc], powers[loc]['f0'] = opt.getSweepPower(drive, sopt, eopt, 0)
        _, powers[loc]['fu'] = opt.getSweepPower(drive, sopt, eopt, fmod)
        _, powers[loc]['fl'] = opt.getSweepPower(drive, sopt, eopt, -fmod)

    return poses, powers


def assert_power(optic, ePos, powers_ref):
    # opt = computeSweep({optic: 180 * 1064e-9/360})
    opt = computeSweep(ePos)
    _, powers = getSweepPowers(opt, optic)
    rslt = [close.allclose(powers[key]['f0'], powers_ref[key]['f0'])
            for key in powers.keys()]
    return rslt


def test_powers_XARM():
    ePos = {'EX': 180 * 1064e-9/360}
    assert all(assert_power('EX', ePos, data['powersXARM'][()]))


def test_powers_IX():
    ePos = {'IX': 180 * 1064e-9/360}
    assert all(assert_power('IX', ePos, data['powersIX'][()]))


def test_powers_XARM_DIFF():
    ePos = {'EX': 180 * 1064e-9/360}
    ePos['IX'] = ePos['EX']
    assert all(assert_power('EX', ePos, data['powersXARM_DIFF'][()]))


def test_powers_XARM_COMM():
    ePos = {'EX': 180 * 1064e-9/360}
    ePos['IX'] = -ePos['EX']
    assert all(assert_power('EX', ePos, data['powersXARM_COMM'][()]))
