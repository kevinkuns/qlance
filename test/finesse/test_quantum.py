"""
Unit tests for finesse quantum noise: homodyne detectors and squeezers
"""

import numpy as np
import qlance.finesse as fin
import pykat
from qlance.controls import resRoots
import close
import pytest


ref_data = np.load('data/finesse_quantum_data.npz', allow_pickle=True)
ref_data = ref_data['data'][()]

fmin = 3
fmax = 5e3
npts = 100
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


def katFPMI(sqAng, sqdB, rf=True):
    kat = pykat.finesse.kat()

    fin.addMirror(kat, 'EX')
    fin.addMirror(kat, 'EY')
    fin.addMirror(kat, 'IX', Thr=Ti)
    fin.addMirror(kat, 'IY', Thr=Ti)
    fin.addBeamSplitter(kat, 'BS')

    kat.BS.phi = np.sqrt(2) * 45

    Tpo = 1 - 1/Pin
    fin.addBeamSplitter(kat, 'LO_PO', Thr=Tpo, aoi=45, comp=True)

    for optic in ['EX', 'EY', 'IX', 'IY']:
        fin.setMechTF(kat, optic, [], poles, 1/M)

    fin.addLaser(kat, 'Laser', Pin)
    if rf:
        fin.addModulator(kat, 'Mod', fmod, gmod, 1, 'pm')
        fin.addSpace(kat, 'Laser_out', 'Mod_in', 0)
        fin.addSpace(kat, 'Mod_out', 'LO_PO_frI', 0)
    else:
        fin.addSpace(kat, 'Laser_out', 'LO_PO_frI', 0)

    fin.addSpace(kat, 'LO_PO_bkT', 'BS_frI', 0)

    fin.addSpace(kat, 'BS_bkT', 'IX_bk', lx)
    fin.addSpace(kat, 'IX_fr', 'EX_fr', Larm)

    fin.addSpace(kat, 'BS_frR', 'IY_bk', ly)
    fin.addSpace(kat, 'IY_fr', 'EY_fr', Larm)

    fin.addFaradayIsolator(kat, 'FI')
    fin.addSpace(kat, 'BS_bkO', 'FI_fr_in', 0)

    fin.addBeamSplitter(kat, 'AS_PO', aoi=45, Thr=0.5)
    fin.addSpace(kat, 'FI_fr_out', 'AS_PO_frI', 0)

    if rf:
        fin.addReadout(kat, 'AS', 'AS_PO_bkT', fmod, 0)

    fin.addSqueezer(kat, 'Sqz', sqAng, sqdB)
    fin.addSpace(kat, 'Sqz_out', 'FI_bk_in', 0)

    return kat


def homodyne_lo(kat, phi):
    fin.addHomodyneReadout(kat, 'AS', phi, qe=0.9)
    fin.addSpace(kat, 'AS_PO_frR', 'AS_BS_frI', 0)
    fin.monitorAllQuantumNoise(kat)


def homodyne_po(kat, phi):
    fin.addHomodyneReadout(kat, 'AS', qe=0.9, LOpower=0)
    fin.addSpace(kat, 'AS_PO_frR', 'AS_BS_frI', 0)
    fin.addSpace(kat, 'LO_PO_frR', 'AS_LOphase_frI', 0)
    kat.AS_LOphase.phi = phi/2
    fin.monitorAllQuantumNoise(kat)


def getKatFR(kat):
    katFR = fin.KatFR(kat, all_drives=False)
    katFR.addDrives(['EX', 'EY'])
    return katFR


def get_data(sqAng, sqdB, phi):
    kat_lo = katFPMI(sqAng, sqdB)
    kat_po = katFPMI(sqAng, sqdB)
    homodyne_lo(kat_lo, phi)
    homodyne_po(kat_po, phi)
    kat_lo = getKatFR(kat_lo)
    kat_po = getKatFR(kat_po)

    kat_lo.run(fmin, fmax, npts, rtype='opt')
    kat_po.run(fmin, fmax, npts, rtype='opt')

    data = {}

    for key, kat in zip(['lo', 'po'], [kat_lo, kat_po]):
        data[key] = dict(
            qnoise_DIFF=kat.getQuantumNoise('AS_DIFF'),
              qnoise_AS_I=kat.getQuantumNoise('AS_I'),
              qnoise_AS_Q=kat.getQuantumNoise('AS_Q'),
              qnoise_AS_DC=kat.getQuantumNoise('AS_DC'),
              tf_DIFF=kat.getTF('AS_DIFF', DARM),
              tf_AS_I=kat.getTF('AS_I', DARM),
              tf_AS_Q=kat.getTF('AS_Q', DARM),
              tf_AS_DC=kat.getTF('AS_DC', DARM))

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
    kat_lo = katFPMI(sqAng=45, sqdB=10, rf=False)
    kat_po = katFPMI(sqAng=45, sqdB=10, rf=False)
    homodyne_lo(kat_lo, 160)
    homodyne_po(kat_po, 160)
    kat_lo = getKatFR(kat_lo)
    kat_po = getKatFR(kat_po)

    kat_lo.run(fmin, fmax, npts, rtype='opt')
    kat_po.run(fmin, fmax, npts, rtype='opt')

    chk_data = ref_data['d45_10_160']

    def test_lo(self):
        qnoise = self.kat_lo.getQuantumNoise('AS_DIFF')
        tf = self.kat_lo.getTF('AS_DIFF', DARM)
        rslt1 = close.allclose(
            qnoise, self.chk_data['lo']['qnoise_DIFF'], rtol=1e-2, atol=1e-2)
        rslt2 = close.allclose(
            tf, self.chk_data['lo']['tf_DIFF'], rtol=1e-2, atol=1e-2)
        assert all([rslt1, rslt2])

    def test_po(self):
        qnoise = self.kat_po.getQuantumNoise('AS_DIFF')
        tf = self.kat_po.getTF('AS_DIFF', DARM)
        rslt1 = close.allclose(
            qnoise, self.chk_data['po']['qnoise_DIFF'], rtol=1e-2, atol=1e-2)
        rslt2 = close.allclose(
            tf, self.chk_data['po']['tf_DIFF'], rtol=1e-2, atol=1e-2)
        assert all([rslt1, rslt2])
