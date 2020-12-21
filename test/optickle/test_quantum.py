"""
Unit tests for optickle quantum noise: homodyne detectors and squeezers
"""

import matlab.engine
import numpy as np
import pytickle.optickle as pyt
from pytickle.controls import resRoots
import close
import pytest


eng = matlab.engine.start_matlab()
pyt.addOpticklePath(eng)


ref_data = np.load('data/optickle_quantum_data.npz', allow_pickle=True)
ref_data = ref_data['data'][()]

fmin = 3
fmax = 5e3
npts = 100
ff = np.logspace(np.log10(fmin), np.log10(fmax), npts)
DARM = dict(EX=1/2, EY=-1/2)

M = 100
Q = 50
f0 = 1
Pin = 50e3
Ti = 0.14
Larm = 4e3
lm = 3
ls = 0.01
lx = lm + ls/2
ly = lm - ls/2
fmod = 55e6
gmod = 0.1
poles = np.array(resRoots(f0, Q, Hz=False))


def optFPMI(eng, opt_name, sqAng, sqdB, rf=True):
    if rf:
        vRF = np.array([-fmod, 0, fmod])
    else:
        vRF = 0
    opt = pyt.PyTickle(eng, opt_name, vRF=vRF)

    opt.addMirror('EX')
    opt.addMirror('EY')
    opt.addMirror('IX', Thr=Ti)
    opt.addMirror('IY', Thr=Ti)
    opt.addBeamSplitter('BS')

    for optic in ['EX', 'EY', 'IX', 'IY']:
        opt.setMechTF(optic, [], poles, 1/M)

    Tpo = 1 - 1/Pin
    opt.addMirror('LO_PO', Thr=Tpo)

    opt.addSource('Laser', np.sqrt(Pin)*(vRF == 0))
    if rf:
        opt.addRFmodulator('Mod', fmod, gmod*1j)
        opt.addLink('Laser', 'out', 'Mod', 'in', 0)
        opt.addLink('Mod', 'out', 'LO_PO', 'fr', 0)
    else:
        opt.addLink('Laser', 'out', 'LO_PO', 'fr', 0)

    opt.addLink('LO_PO', 'bk', 'BS', 'frA', 0)

    opt.addLink('BS', 'bkA', 'IX', 'bk', lx)
    opt.addLink('IX', 'fr', 'EX', 'fr', Larm)
    opt.addLink('EX', 'fr', 'IX', 'fr', Larm)
    opt.addLink('IX', 'bk', 'BS', 'bkB', lx)

    opt.addLink('BS', 'frA', 'IY', 'bk', ly)
    opt.addLink('IY', 'fr', 'EY', 'fr', Larm)
    opt.addLink('EY', 'fr', 'IY', 'fr', Larm)
    opt.addLink('IY', 'bk', 'BS', 'frB', ly)

    opt.addMirror('AS_PO', aoi=45, Thr=0.5)
    opt.addLink('BS', 'bkB', 'AS_PO', 'fr', 0)

    if rf:
        opt.addSink('AS')
        opt.addLink('AS_PO', 'fr', 'AS', 'in', 0)
        opt.addReadout('AS', fmod, 0)

    opt.addSqueezer('Sqz', sqAng=sqAng, sqdB=sqdB)
    opt.addLink('Sqz', 'out', 'BS', 'bkA', 0)

    return opt


def homodyne_lo(opt, phi):
    opt.addHomodyneReadout('AS', phi, qe=0.9)
    opt.addLink('AS_PO', 'bk', 'AS_BS', 'fr', 0)


def homodyne_po(opt, phi):
    opt.addHomodyneReadout('AS', qe=0.9, LOpower=0)
    opt.addLink('AS_PO', 'bk', 'AS_BS', 'fr', 0)
    opt.addLink('LO_PO', 'fr', 'AS_LOphase', 'fr', 0)
    opt.setPosOffset('AS_LOphase', 0.5*(phi - 90)/360 * 1064e-9)


def get_data(sqAng, sqdB, phi):
    opt_lo = optFPMI(eng, 'opt_lo', sqAng=sqAng, sqdB=sqdB)
    opt_po = optFPMI(eng, 'opt_po', sqAng=sqAng, sqdB=sqdB)
    homodyne_lo(opt_lo, phi)
    homodyne_po(opt_po, phi)

    opt_lo.run(ff)
    opt_po.run(ff)

    data = {}

    for key, opt in zip(['lo', 'po'], [opt_lo, opt_po]):
        data[key] = dict(
            qnoise_DIFF=opt.getQuantumNoise('AS_DIFF'),
              qnoise_AS_I=opt.getQuantumNoise('AS_I'),
              qnoise_AS_Q=opt.getQuantumNoise('AS_Q'),
              qnoise_AS_DC=opt.getQuantumNoise('AS_DC'),
              tf_DIFF=opt.getTF('AS_DIFF', DARM),
              tf_AS_I=opt.getTF('AS_I', DARM),
              tf_AS_Q=opt.getTF('AS_Q', DARM),
              tf_AS_DC=opt.getTF('AS_DC', DARM))

    return data


def get_results(tst_data, chk_data):
    return [close.allclose(chk_data[key], val) for key, val in tst_data.items()]


class Test90_15_0:
    data = get_data(90, 15, 0)
    chk_data = ref_data['d90_15_0']

    def test_lo(self):
        rslts = get_results(self.data['lo'], self.chk_data['lo'])
        assert all(rslts)

    def test_po(self):
        rslts = get_results(self.data['po'], self.chk_data['po'])
        assert all(rslts)


class Test0_15_0:
    data = get_data(0, 15, 0)
    chk_data = ref_data['d0_15_0']

    def test_lo(self):
        rslts = get_results(self.data['lo'], self.chk_data['lo'])
        assert all(rslts)

    def test_po(self):
        rslts = get_results(self.data['po'], self.chk_data['po'])
        assert all(rslts)


class Test145_10_30:
    data = get_data(145, 10, 30)
    chk_data = ref_data['d145_10_30']

    def test_lo(self):
        rslts = get_results(self.data['lo'], self.chk_data['lo'])
        assert all(rslts)

    def test_po(self):
        rslts = get_results(self.data['po'], self.chk_data['po'])
        assert all(rslts)


class Test45_10_160:
    data = get_data(45, 10, 160)
    chk_data = ref_data['d45_10_160']

    def test_lo(self):
        rslts = get_results(self.data['lo'], self.chk_data['lo'])
        assert all(rslts)

    def test_po(self):
        rslts = get_results(self.data['po'], self.chk_data['po'])
        assert all(rslts)


class Test45_10_160_carrier_only:
    opt_lo = optFPMI(eng, 'opt_lo', sqAng=45, sqdB=10, rf=False)
    opt_po = optFPMI(eng, 'opt_po', sqAng=45, sqdB=10, rf=False)
    homodyne_lo(opt_lo, 160)
    homodyne_po(opt_po, 160)
    opt_lo.run(ff)
    opt_po.run(ff)

    chk_data = ref_data['d45_10_160']

    def test_lo(self):
        qnoise = self.opt_lo.getQuantumNoise('AS_DIFF')
        tf = self.opt_lo.getTF('AS_DIFF', DARM)
        rslt1 = close.allclose(
            qnoise, self.chk_data['lo']['qnoise_DIFF'], rtol=1e-2, atol=1e-2)
        rslt2 = close.allclose(
            tf, self.chk_data['lo']['tf_DIFF'], rtol=1e-2, atol=1e-2)
        assert all([rslt1, rslt2])

    def test_po(self):
        qnoise = self.opt_po.getQuantumNoise('AS_DIFF')
        tf = self.opt_po.getTF('AS_DIFF', DARM)
        rslt1 = close.allclose(
            qnoise, self.chk_data['po']['qnoise_DIFF'], rtol=1e-2, atol=1e-2)
        rslt2 = close.allclose(
            tf, self.chk_data['po']['tf_DIFF'], rtol=1e-2, atol=1e-2)
        assert all([rslt1, rslt2])
