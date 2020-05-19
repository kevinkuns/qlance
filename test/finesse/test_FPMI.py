"""
Unit tests for finesse FPMI frequency response and sweeps
"""

import numpy as np
import pytickle.finesse as fin
import pykat
import pykat.components as kcmp
import pykat.detectors as kdet
import pykat.commands as kcmd
import pytest


data = np.load('data/finesse_FPMI_data.npz', allow_pickle=True)

fmod = 55e6
gmod = 0.1
Pin = 1
Ti = 0.1
Lcav = 40e3
lm = 3
ls = 1.36
lx = lm + ls/2
ly = lm - ls/2
dpx = 90
dpb = 0  # 45 * np.sqrt(2)
aoib = 45
Ri = 34e3
Re = 36e3


def katMI(dpx, dpb, aoib):
    kat = pykat.finesse.kat()

    fin.addMirror(kat, 'EX', Thr=0, phi=dpx, Chr=1/Re)
    fin.addMirror(kat, 'EY', Thr=0, Chr=1/Re)
    fin.addMirror(kat, 'IX', Thr=Ti, phi=dpx, Chr=1/Ri)
    fin.addMirror(kat, 'IY', Thr=Ti, Chr=1/Ri)
    fin.addBeamSplitter(kat, 'BS', aoi=aoib, phi=dpb)

    fin.addLaser(kat, 'Laser', Pin)
    fin.addModulator(kat, 'Mod', fmod, gmod, 5, 'pm')
    fin.addSpace(kat, 'Laser_out', 'Mod_in', 0)
    fin.addSpace(kat, 'Mod_out', 'BS_frI', 1)

    fin.addSpace(kat, 'BS_bkT', 'IX_bk', lx)
    fin.addSpace(kat, 'IX_fr', 'EX_fr', Lcav)
    fin.setCavityBasis(kat, 'IX_fr', 'EX_fr')

    fin.addSpace(kat, 'BS_frR', 'IY_bk', ly)
    fin.addSpace(kat, 'IY_fr', 'EY_fr', Lcav)
    fin.setCavityBasis(kat, 'IY_fr', 'EY_fr')

    fin.addSpace(kat, 'BS_bkO', 'AS_in', 1, new_comp='AS')

    kat.phase = 2

    return kat


class TestFreqResp:

    kat = katMI(90, 0, 45)
    fin.addReadout(kat, 'AS', 'AS_in', fmod, 0)
    katTF = fin.KatFR(kat)
    katTF.run(1, 1e4, 400)

    def test_tfQ_EM(self):
        tfQ_EM = self.katTF.getTF('AS_Q', {'EX': 0.5, 'EY': -0.5})
        assert np.allclose(tfQ_EM, data['tfQ_EM'])

    def test_tfI_EM(self):
        tfI_EM = self.katTF.getTF('AS_I', {'EX': 0.5, 'EY': -0.5})
        assert np.allclose(tfI_EM, data['tfI_EM'])

    def test_tfQ_IM(self):
        tfQ_IM = self.katTF.getTF('AS_Q', {'IX': 0.5, 'IY': -0.5})
        assert np.allclose(tfQ_IM, data['tfQ_IM'])


def runAsy(ls):
    kat = katMI(90, 0, 45)
    kat.verbose = False
    kat.add(kdet.ad('AS_f0', 0, 'AS_in'))
    kat.add(kdet.ad('AS_fu', fmod, 'AS_in'))
    kat.add(kdet.ad('AS_fl', -fmod, 'AS_in'))
    kat.s_BS_IX.L = lm + ls/2
    kat.s_BS_IY.L = lm - ls/2

    kat.noxaxis = True
    return kat.run()


class TestSchnupp:

    lasy = np.linspace(0, 3, 100)
    powf0 = np.zeros_like(lasy)
    powfu = np.zeros_like(lasy)
    powfl = np.zeros_like(lasy)

    for li, ls in enumerate(lasy):
        outAsy = runAsy(ls)
        powf0[li] = outAsy['AS_f0']**2
        powfu[li] = outAsy['AS_fu']**2
        powfl[li] = outAsy['AS_fl']**2

    def test_powf0(self):
        assert np.allclose(self.powf0, data['powf0'])

    def test_powfu(self):
        assert np.allclose(self.powfu, data['powfu'])

    def test_powfl(self):
        assert np.allclose(self.powfl, data['powfl'])


def add_ads(kat, name, node):
    kat.add(kdet.ad(name + '_f0', 0, node))
    kat.add(kdet.ad(name + '_fu', fmod, node))
    kat.add(kdet.ad(name + '_fl', -fmod, node))


def katSweep(comp, dp=0, verbose=False):
    kat = katMI(90, 0, 45)
    kat.verbose = verbose
    add_ads(kat, 'XARM', 'IX_fr')
    add_ads(kat, 'YARM', 'IY_fr')
    add_ads(kat, 'BSX', 'IX_bk')
    add_ads(kat, 'BSY', 'IY_bk')
    add_ads(kat, 'AS', 'AS_in')

    kat.add(kcmd.xaxis('lin', np.array(
        [-180, 180]) - dp, kat.components[comp].phi, 1000))
    kat.parse('yaxis abs')

    return kat


def katSweep2(drives, spos, epos, dp=0, verbose=False):
    kat = katMI(90, 0, 45)
    kat.verbose = verbose
    add_ads(kat, 'XARM', 'IX_fr')
    add_ads(kat, 'YARM', 'IY_fr')
    add_ads(kat, 'BSX', 'IX_bk')
    add_ads(kat, 'BSY', 'IY_bk')
    add_ads(kat, 'AS', 'AS_in')

    kat = fin.KatSweep(kat, drives, relative=False)
    kat.sweep(spos - dp, epos - dp, 1000)
    return kat


def assert_amp(out, sweep, key):
    return np.allclose(out[key], data[sweep][()][key])


def assert_amp2(kat, probe, sweep):
    _, sig = kat.getSweepSignal(probe, 'EX')
    return np.allclose(sig, data[sweep][()][probe])


class TestSweep:

    katXARM = katSweep('EX', -90)
    outXARM = katXARM.run()

    katIX = katSweep('IX', 90)
    outIX = katIX.run()

    keys = ['AS_f0', 'XARM_f0', 'YARM_f0', 'BSX_f0', 'BSY_f0']

    def test_amps_XARM(self):
        rslts = [assert_amp(self.outXARM, 'sweep_XARM', key)
                 for key in self.keys]
        assert all(rslts)

    def test_amps_IX(self):
        rslts = [assert_amp(self.outIX, 'sweep_IX', key) for key in self.keys]
        assert all(rslts)


class TestSweepXARM:

    katXARM_DIFF = katSweep2({'EX': 1, 'IX': 1}, -180, 180)
    katXARM_COMM = katSweep2({'EX': 1, 'IX': -1}, -180, 180)

    keys = list(data['sweep_XARM_DIFF'][()].keys())

    def test_amps_XARM_DIFF(self):
        rslts = [assert_amp2(self.katXARM_DIFF, key, 'sweep_XARM_DIFF')
                 for key in self.keys]
        assert all(rslts)

    def test_amps_XARM_COMM(self):
        rslts = [assert_amp2(self.katXARM_COMM, key, 'sweep_XARM_COMM')
                 for key in self.keys]
        assert all(rslts)
