"""
Unit tests for finesse PDH frequency response and sweeps
"""

import numpy as np
import pytickle.finesse as fin
import pykat
import pykat.components as kcmp
import pykat.commands as kcom
import pytest


fmod = 11e3
gmod = 0.1
Pin = 1
Ti = 0.01
Lcav = 40e3

data = np.load('data/finesse_PDH_data.npz')


def katFP():
    kat = pykat.finesse.kat()

    fin.addMirror(kat, 'EX', Thr=0)
    fin.addMirror(kat, 'IX', Thr=Ti)
    kat.add(kcmp.space('s_IX_EX', 'IX_fr', 'EX_fr', Lcav))

    kat.add(kcmp.laser('Laser', 'Laser_out', P=Pin))
    kat.add(kcmp.modulator('Mod', 'Mod_in', 'Mod_out', fmod, gmod, 5, 'pm'))
    kat.add(kcmp.space('s_Laser_Mod', 'Laser_out', 'Mod_in', 0))
    kat.add(kcmp.space('s_Mod_IX', 'Mod_out', 'IX_bk', 0))

    kat.phase = 2

    return kat


class TestFreqResp:

    kat = katFP()
    fin.addReadout(kat, 'REFL', 'IX_bk', 11e3, 0)
    katFR = fin.KatFR(kat)
    katFR.tickle(1e-2, 1e4, 1000)

    def test_tfI(self):
        tfI = self.katFR.getTF('REFL_I', 'EX')
        assert np.allclose(tfI, data['tfI'])

    def test_tfQ(self):
        tfQ = self.katFR.getTF('REFL_Q', 'EX')
        assert np.allclose(tfQ, data['tfQ'])


class TestSweep:

    kat = katFP()
    fin.addReadout(kat, 'REFL', 'IX_bk', 11e3, 0, freqresp=False)
    ePos = 5e-9 * 360/kat.lambda0
    kat.add(kcom.xaxis('lin', [-ePos, ePos], kat.EX.phi, 1000))
    kat.parse('yaxis abs')
    out = kat.run()

    def test_sweepI(self):
        sweepI = self.out['REFL_I']
        assert np.allclose(sweepI, data['sweepI'])

    def test_sweepQ(self):
        sweepQ = self.out['REFL_Q']
        assert np.allclose(sweepQ, data['sweepQ'])


class TestFreqRespSetProbe:

    kat = katFP()
    fin.addReadout(kat, 'REFL', 'IX_bk', 11e3, 0, freqresp=False)
    for probe in ['REFL_DC', 'REFL_I', 'REFL_Q']:
        fin.set_probe_response(kat, probe, 'fr')
    katFR = fin.KatFR(kat)
    katFR.tickle(1e-2, 1e4, 1000)

    def test_tfI(self):
        tfI = self.katFR.getTF('REFL_I', 'EX')
        assert np.allclose(tfI, data['tfI'])

    def test_tfQ(self):
        tfQ = self.katFR.getTF('REFL_Q', 'EX')
        assert np.allclose(tfQ, data['tfQ'])


class TestSweepSetProbe:

    kat = katFP()
    fin.addReadout(kat, 'REFL', 'IX_bk', 11e3, 0, freqresp=True)
    for probe in ['REFL_DC', 'REFL_I', 'REFL_Q']:
        fin.set_probe_response(kat, probe, 'dc')
    ePos = 5e-9 * 360/kat.lambda0
    kat.add(kcom.xaxis('lin', [-ePos, ePos], kat.EX.phi, 1000))
    kat.parse('yaxis abs')
    out = kat.run()

    def test_sweepI(self):
        sweepI = self.out['REFL_I']
        assert np.allclose(sweepI, data['sweepI'])

    def test_sweepQ(self):
        sweepQ = self.out['REFL_Q']
        assert np.allclose(sweepQ, data['sweepQ'])
