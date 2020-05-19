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
    fin.addSpace(kat, 'IX_fr', 'EX_fr', Lcav)

    fin.addLaser(kat, 'Laser', Pin)
    fin.addModulator(kat, 'Mod', fmod, gmod, 5, 'pm')
    fin.addSpace(kat, 'Laser_out', 'Mod_in', 0)
    fin.addSpace(kat, 'Mod_out', 'IX_bk', 0)

    fin.addReadout(kat, 'REFL', 'IX_bk', fmod, 0)
    for det_name in kat.detectors.keys():
        fin.monitorShotNoise(kat, det_name)

    kat.phase = 2

    return kat


class TestFreqResp:

    kat = katFP()
    # fin.addReadout(kat, 'REFL', 'IX_bk', 11e3, 0)
    katFR = fin.KatFR(kat)
    katFR.run(1e-2, 1e4, 1000)

    def test_tfI(self):
        tfI = self.katFR.getTF('REFL_I', 'EX')
        assert np.allclose(tfI, data['tfI'])

    def test_tfQ(self):
        tfQ = self.katFR.getTF('REFL_Q', 'EX')
        assert np.allclose(tfQ, data['tfQ'])

    def test_qnI(self):
        qnI = self.katFR.getQuantumNoise('REFL_I')
        assert np.allclose(qnI, data['qnQ'])

    def test_qnQ(self):
        qnQ = self.katFR.getQuantumNoise('REFL_Q')
        assert np.allclose(qnQ, data['qnQ'])

    def test_qnDC(self):
        qnDC = self.katFR.getQuantumNoise('REFL_DC')
        assert np.allclose(qnDC, data['qnDC'])


class TestSweep:

    kat = katFP()
    # fin.addReadout(kat, 'REFL', 'IX_bk', 11e3, 0, freqresp=False)
    ePos = 5e-9 * 360/kat.lambda0
    # kat.add(kcom.xaxis('lin', [-ePos, ePos], kat.EX.phi, 1000))
    # kat.parse('yaxis abs')
    # out = kat.run()
    katSweep = fin.KatSweep(kat, 'EX')
    katSweep.sweep(-ePos, ePos, 1000)

    def test_sweepI(self):
        _, sweepI = self.katSweep.getSweepSignal('REFL_I', 'EX')
        assert np.allclose(sweepI, data['sweepI'])

    def test_sweepQ(self):
        _, sweepQ = self.katSweep.getSweepSignal('REFL_Q', 'EX')
        assert np.allclose(sweepQ, data['sweepQ'])


class TestFreqRespSetProbe:

    kat = katFP()
    # fin.addReadout(kat, 'REFL', 'IX_bk', 11e3, 0, freqresp=False)
    # for probe in ['REFL_DC', 'REFL_I', 'REFL_Q']:
    #     fin.set_probe_response(kat, probe, 'fr')
    # now this step is done automatically
    # fin.set_all_probe_response(kat, 'fr')
    katFR = fin.KatFR(kat)
    katFR.run(1e-2, 1e4, 1000)

    def test_tfI(self):
        tfI = self.katFR.getTF('REFL_I', 'EX')
        assert np.allclose(tfI, data['tfI'])

    def test_tfQ(self):
        tfQ = self.katFR.getTF('REFL_Q', 'EX')
        assert np.allclose(tfQ, data['tfQ'])


class TestSweepSetProbe:

    kat = katFP()
    # fin.addReadout(kat, 'REFL', 'IX_bk', 11e3, 0, freqresp=True)
    # kat.add(kcom.xaxis('lin', [-ePos, ePos], kat.EX.phi, 1000))
    # kat.parse('yaxis abs')
    # out = kat.run()
    # for probe in ['REFL_DC', 'REFL_I', 'REFL_Q']:
    #     fin.set_probe_response(kat, probe, 'dc')
    # now this step is done automatically
    # fin.set_all_probe_response(kat, 'dc')
    ePos = 5e-9 * 360/kat.lambda0
    katSweep = fin.KatSweep(kat, 'EX')
    katSweep.sweep(-ePos, ePos, 1000)

    def test_sweepI(self):
        _, sweepI = self.katSweep.getSweepSignal('REFL_I', 'EX')
        assert np.allclose(sweepI, data['sweepI'])

    def test_sweepQ(self):
        _, sweepQ = self.katSweep.getSweepSignal('REFL_Q', 'EX')
        assert np.allclose(sweepQ, data['sweepQ'])
