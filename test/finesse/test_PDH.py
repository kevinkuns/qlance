"""
Unit tests for finesse PDH frequency response and sweeps
"""

import numpy as np
import pytickle.finesse as fin
import pykat
import pykat.components as kcmp
import pykat.commands as kcom
import pytickle.plant as plant
import os
import close
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

    for optic in ['EX', 'IX']:
        fin.setMechTF(kat, optic, [], [0, 0], 1)

    fin.addLaser(kat, 'Laser', Pin)
    fin.addModulator(kat, 'Mod', fmod, gmod, 5, 'pm')
    fin.addSpace(kat, 'Laser_out', 'Mod_in', 0)
    fin.addSpace(kat, 'Mod_out', 'IX_bk', 0)

    fin.addReadout(kat, 'REFL', 'IX_bk', fmod, 0)
    fin.monitorAllQuantumNoise(kat)

    kat.phase = 2

    return kat


class TestFreqResp:

    kat = katFP()
    katFR = fin.KatFR(kat)
    katFR.run(1e-2, 1e4, 1000)
    katFR.run(1e-2, 1e4, 1000, dof='freq')
    katFR.runDC()
    katFR.run(1e-2, 1e4, 1000, dof='amp')
    katFR.save('test_PDH.hdf5')
    katFR2 = plant.FinessePlant()
    katFR2.load('test_PDH.hdf5')
    os.remove('test_PDH.hdf5')

    def test_tfI(self):
        tfI = self.katFR.getTF('REFL_I', 'EX')
        assert close.allclose(tfI, data['tfI'])

    def test_tfQ(self):
        tfQ = self.katFR.getTF('REFL_Q', 'EX')
        assert close.allclose(tfQ, data['tfQ'])

    def test_qnI(self):
        qnI = self.katFR.getQuantumNoise('REFL_I')
        assert close.allclose(qnI, data['qnI'])

    def test_qnQ(self):
        qnQ = self.katFR.getQuantumNoise('REFL_Q')
        assert close.allclose(qnQ, data['qnQ'])

    def test_qnDC(self):
        qnDC = self.katFR.getQuantumNoise('REFL_DC')
        assert close.allclose(qnDC, data['qnDC'])

    def test_freqI(self):
        tf = self.katFR.getTF('REFL_I', 'Laser', dof='freq')
        assert close.allclose(tf, data['tfI_freq'])

    def test_freqQ(self):
        tf = self.katFR.getTF('REFL_Q', 'Laser', dof='freq')
        assert close.allclose(tf, data['tfQ_freq'])

    def test_ampI(self):
        tf = self.katFR.getTF('REFL_I', 'Laser', dof='amp')
        assert close.allclose(tf, data['tfI_amp'])

    def test_ampQ(self):
        tf = self.katFR.getTF('REFL_Q', 'Laser', dof='amp')
        assert close.allclose(tf, data['tfQ_amp'])

    def test_DC_DC(self):
        sig = self.katFR.getSigDC('REFL_DC')
        assert close.isclose(sig, data['dcDC'])

    def test_DC_I(self):
        sig = self.katFR.getSigDC('REFL_I')
        assert close.isclose(sig, data['dcI'])

    def test_DC_Q(self):
        sig = self.katFR.getSigDC('REFL_Q')
        assert close.isclose(sig, data['dcQ'])

    def test_reload_tfI(self):
        tfI = self.katFR2.getTF('REFL_I', 'EX')
        assert close.allclose(tfI, data['tfI'])

    def test_reload_tfQ(self):
        tfQ = self.katFR2.getTF('REFL_Q', 'EX')
        assert close.allclose(tfQ, data['tfQ'])

    def test_reload_qnI(self):
        qnI = self.katFR2.getQuantumNoise('REFL_I')
        assert close.allclose(qnI, data['qnI'])

    def test_reload_qnQ(self):
        qnQ = self.katFR2.getQuantumNoise('REFL_Q')
        assert close.allclose(qnQ, data['qnQ'])

    def test_reload_qnDC(self):
        qnDC = self.katFR2.getQuantumNoise('REFL_DC')
        assert close.allclose(qnDC, data['qnDC'])

    def test_reload_freqI(self):
        tf = self.katFR2.getTF('REFL_I', 'Laser', dof='freq')
        assert close.allclose(tf, data['tfI_freq'])

    def test_reload_freqQ(self):
        tf = self.katFR2.getTF('REFL_Q', 'Laser', dof='freq')
        assert close.allclose(tf, data['tfQ_freq'])

    def test_reload_ampI(self):
        tf = self.katFR2.getTF('REFL_I', 'Laser', dof='amp')
        assert close.allclose(tf, data['tfI_amp'])

    def test_reload_ampQ(self):
        tf = self.katFR2.getTF('REFL_Q', 'Laser', dof='amp')
        assert close.allclose(tf, data['tfQ_amp'])

    def test_reload_DC_DC(self):
        sig = self.katFR2.getSigDC('REFL_DC')
        assert close.isclose(sig, data['dcDC'])

    def test_reload_DC_I(self):
        sig = self.katFR2.getSigDC('REFL_I')
        assert close.isclose(sig, data['dcI'])

    def test_reload_DC_Q(self):
        sig = self.katFR2.getSigDC('REFL_Q')
        assert close.isclose(sig, data['dcQ'])


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
        assert close.allclose(sweepI, data['sweepI'])

    def test_sweepQ(self):
        _, sweepQ = self.katSweep.getSweepSignal('REFL_Q', 'EX')
        assert close.allclose(sweepQ, data['sweepQ'])


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
        assert close.allclose(tfI, data['tfI'])

    def test_tfQ(self):
        tfQ = self.katFR.getTF('REFL_Q', 'EX')
        assert close.allclose(tfQ, data['tfQ'])


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
        assert close.allclose(sweepI, data['sweepI'])

    def test_sweepQ(self):
        _, sweepQ = self.katSweep.getSweepSignal('REFL_Q', 'EX')
        assert close.allclose(sweepQ, data['sweepQ'])
