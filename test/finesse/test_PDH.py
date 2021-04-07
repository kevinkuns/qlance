"""
Unit tests for finesse PDH frequency response and sweeps
"""

import numpy as np
import qlance.finesse as fin
from qlance.controls import DegreeOfFreedom
import pykat
import qlance.plant as plant
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
    fin.monitorMotion(kat, 'EX')

    kat.phase = 2

    return kat


class TestFreqResp:

    kat = katFP()
    katFR = fin.KatFR(kat)
    katFR.run(1e-2, 1e4, 1000)
    katFR.run(1e-2, 1e4, 1000, doftype='freq')
    katFR.runDC()
    katFR.run(1e-2, 1e4, 1000, doftype='amp')
    katFR.save('test_PDH.hdf5')
    katFR2 = plant.FinessePlant()
    katFR2.load('test_PDH.hdf5')
    os.remove('test_PDH.hdf5')

    def test_tfI(self):
        ex = DegreeOfFreedom('EX')
        ex2 = DegreeOfFreedom('EX', probes='REFL_I')
        ex3 = DegreeOfFreedom('EX', probes='REFL_DC')
        tfI1 = self.katFR.getTF('REFL_I', 'EX')
        tfI2 = self.katFR.getTF('REFL_I', ex)
        tfI3 = self.katFR.getTF(ex2)
        tfI4 = self.katFR.getTF('REFL_I', ex3)
        c1 = close.allclose(tfI1, data['tfI'])
        c2 = close.allclose(tfI2, data['tfI'])
        c3 = close.allclose(tfI3, data['tfI'])
        c4 = close.allclose(tfI4, data['tfI'])
        assert np.all([c1, c2, c3, c4])

    def test_tfQ(self):
        ex = DegreeOfFreedom(name='EX', drives='EX', doftype='pos')
        ex2 = DegreeOfFreedom(name='EX', drives='EX', doftype='pos',
                              probes='REFL_Q')
        tfQ1 = self.katFR.getTF('REFL_Q', 'EX')
        tfQ2 = self.katFR.getTF('REFL_Q', ex)
        tfQ3 = self.katFR.getTF(ex2)
        c1 = close.allclose(tfQ1, data['tfQ'])
        c2 = close.allclose(tfQ2, data['tfQ'])
        c3 = close.allclose(tfQ3, data['tfQ'])
        assert np.all([c1, c2, c3])

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
        laser = DegreeOfFreedom('Laser', 'freq')
        laser2 = DegreeOfFreedom('Laser', probes='REFL_I', doftype='freq')
        tf1 = self.katFR.getTF('REFL_I', 'Laser', doftype='freq')
        tf2 = self.katFR.getTF('REFL_I', laser)
        tf3 = self.katFR.getTF(laser2)
        c1 = close.allclose(tf1, data['tfI_freq'])
        c2 = close.allclose(tf2, data['tfI_freq'])
        c3 = close.allclose(tf3, data['tfI_freq'])
        assert np.all([c1, c2])

    def test_freqQ(self):
        tf = self.katFR.getTF('REFL_Q', 'Laser', doftype='freq')
        assert close.allclose(tf, data['tfQ_freq'])

    def test_ampI(self):
        laser = DegreeOfFreedom('Laser', 'amp')
        laser2 = DegreeOfFreedom('Laser', probes='REFL_I', doftype='amp')
        tf1 = self.katFR.getTF('REFL_I', 'Laser', doftype='amp')
        tf2 = self.katFR.getTF('REFL_I', laser)
        tf3 = self.katFR.getTF(laser2)
        c1 = close.allclose(tf1, data['tfI_amp'])
        c2 = close.allclose(tf2, data['tfI_amp'])
        c3 = close.allclose(tf3, data['tfI_amp'])
        assert np.all([c1, c2, c3])

    def test_ampQ(self):
        tf = self.katFR.getTF('REFL_Q', 'Laser', doftype='amp')
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

    def test_amp2pos(self):
        ex = DegreeOfFreedom('EX')
        laser_amp = DegreeOfFreedom('Laser', 'amp')
        amp2pos = self.katFR.getMechTF(ex, laser_amp)
        assert close.allclose(amp2pos, data['amp2pos'])

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
        tf = self.katFR2.getTF('REFL_I', 'Laser', doftype='freq')
        assert close.allclose(tf, data['tfI_freq'])

    def test_reload_freqQ(self):
        tf = self.katFR2.getTF('REFL_Q', 'Laser', doftype='freq')
        assert close.allclose(tf, data['tfQ_freq'])

    def test_reload_ampI(self):
        tf = self.katFR2.getTF('REFL_I', 'Laser', doftype='amp')
        assert close.allclose(tf, data['tfI_amp'])

    def test_reload_ampQ(self):
        tf = self.katFR2.getTF('REFL_Q', 'Laser', doftype='amp')
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
    EX = DegreeOfFreedom('EX')
    katSweep = fin.KatSweep(kat, 'EX')
    katSweep2 = fin.KatSweep(kat, EX)
    katSweep.sweep(-ePos, ePos, 1000)
    katSweep2.sweep(-ePos, ePos, 1000)

    def test_sweepI(self):
        _, sweepI = self.katSweep.getSweepSignal('REFL_I', 'EX')
        assert close.allclose(sweepI, data['sweepI'])

    def test_sweepQ(self):
        _, sweepQ = self.katSweep.getSweepSignal('REFL_Q', 'EX')
        assert close.allclose(sweepQ, data['sweepQ'])

    def test_sweepI2(self):
        _, sweepI = self.katSweep2.getSweepSignal('REFL_I', 'EX')
        assert close.allclose(sweepI, data['sweepI'])

    def test_sweepQ2(self):
        _, sweepQ = self.katSweep2.getSweepSignal('REFL_Q', 'EX')
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
