"""
Unit tests for pytickle controls
"""

import matlab.engine
import numpy as np
import pytickle.optickle as pyt
import pytickle.controls as ctrl
import pytest

data = np.load('data/optickle_lsc_data.npz', allow_pickle=True)

eng = matlab.engine.start_matlab()
pyt.addOpticklePath(eng)
eng.eval("addpath(genpath('data'));", nargout=0)

opt = pyt.PyTickle(eng, 'opt')
eng.eval("par = struct;", nargout=0)
eng.eval("par = paramCE1;", nargout=0)
eng.eval("par.PR.Thr = 0.01;", nargout=0)
eng.eval("opt = optCE_homo(par, 1);", nargout=0)
eng.eval("probesCE_homo(opt, par, 1);", nargout=0)
opt.loadMatModel()

ff = np.logspace(np.log10(3), np.log10(7e3), 300)
opt.run(ff)

DARM = {'EX': 1, 'EY': -1}
MICH = {'BS': 1, 'SR': 1/np.sqrt(2), 'PR': -1/np.sqrt(2)}

filtDARM = ctrl.Filter(
    ctrl.catzp(-2*np.pi*20, -2*np.pi*800),
    ctrl.catzp(0, 0, -2*np.pi*300*(1 + 1j/2), -2*np.pi*300*(1 - 1j/2)),
    -1e-8 * (2*np.pi*300)**2 / (2*np.pi*800), Hz=False)

filtPRCL = ctrl.Filter(
    -2*np.pi*10,
    ctrl.catzp(0, -2*np.pi*(20 + 10j), -2*np.pi*(20 - 10j)),
    -1e-5, Hz=False)

filtSRCL = ctrl.Filter([], ctrl.catzp(0, -2*np.pi*20), -4e-5, Hz=False)

filtMICH = ctrl.Filter(
    -2*np.pi*10,
    ctrl.catzp(0, -2*np.pi*(10 + 10j), -2*np.pi*(10 - 10j), -2*np.pi*300),
    0.25, Hz=False)

filtMICH_FF = ctrl.Filter([], [], 2.5e-3)

cs = ctrl.ControlSystem()

cs.addDOF('DARM', 'OMC_DIFF', DARM, 'pos')
cs.addDOF('PRCL', 'POP_If1', 'PR', 'pos')
cs.addDOF('SRCL', 'POP_If2', 'SR', 'pos')
cs.addDOF('MICH', 'POP_Qf2', MICH, 'pos')

cs.addFilter('DARM', 'DARM', filtDARM)
cs.addFilter('PRCL', 'PRCL', filtPRCL)
cs.addFilter('SRCL', 'SRCL', filtSRCL)
cs.addFilter('MICH', 'MICH', filtMICH)
cs.addFilter('DARM', 'MICH', ctrl.catfilt(filtMICH_FF, filtMICH))

cs.setPyTicklePlant(opt)

cs.run()

dofs = cs.dofs.keys()


def test_oltfs():
    rslt = []
    for dof_to in dofs:
        for dof_from in dofs:
            rslt.append(np.allclose(cs.getOLTF(dof_to, dof_from, 'err'),
                                    data['oltfs'][()][dof_to][dof_from]))
    assert all(rslt)


def test_cltfs():
    rslt = []
    for dof_to in dofs:
        for dof_from in dofs:
            rslt.append(np.allclose(cs.getCLTF(dof_to, dof_from, 'err'),
                                    data['cltfs'][()][dof_to][dof_from]))
    assert all(rslt)


def test_sense_closed():
    rslt = []
    for dof_to in dofs:
        for probe in cs.probes:
            tf = cs.getTF(dof_to, 'err', probe, 'sens', closed=True)
            rslt.append(np.allclose(
                tf, data['sense_closed'][()][dof_to][probe]))
    assert all(rslt)


def test_sense_open():
    rslt = []
    for dof_to in dofs:
        for probe in cs.probes:
            tf = cs.getTF(dof_to, 'err', probe, 'sens', closed=False)
            rslt.append(np.allclose(
                tf, data['sense_open'][()][dof_to][probe]))
    assert all(rslt)


def test_drive_closed():
    rslt = []
    for dof_to in dofs:
        for drive in cs.drives:
            tf = cs.getTF(dof_to, 'err', drive, 'drive', closed=True)
            rslt.append(np.allclose(
                tf, data['drive_closed'][()][dof_to][drive]))
    assert all(rslt)


def test_drive_open():
    rslt = []
    for dof_to in dofs:
        for drive in cs.drives:
            tf = cs.getTF(dof_to, 'err', drive, 'drive', closed=False)
            rslt.append(np.allclose(
                tf, data['drive_open'][()][dof_to][drive]))
    assert all(rslt)


def test_cal_closed():
    rslt = []
    for dof_to in dofs:
        for dof_from in dofs:
            tf = cs.getTF(dof_to, 'err', dof_from, 'cal', closed=True)
            rslt.append(np.allclose(
                tf, data['calTFs_closed'][()][dof_to][dof_from]))
    assert all(rslt)


def test_cal_open():
    rslt = []
    for dof_to in dofs:
        for dof_from in dofs:
            tf = cs.getTF(dof_to, 'err', dof_from, 'cal', closed=False)
            rslt.append(np.allclose(
                tf, data['calTFs_open'][()][dof_to][dof_from]))
    assert all(rslt)


def test_sensing_noise():
    shotNoise = {probe: opt.getQuantumNoise(probe) for probe in cs.probes}
    sensingNoise = {}
    calDARM = cs.getTF('DARM', 'err', 'DARM', 'cal', closed=False)
    cltfDARM = cs.getCLTF('DARM', 'DARM', 'err')
    for probe, sn in shotNoise.items():
        tf = cs.getTF('DARM', 'err', probe, 'sens')/cltfDARM
        sensingNoise[probe] = np.abs(tf*sn/calDARM)
    rslt = [np.allclose(noise, data['sensingNoise'][()][probe])
            for probe, noise in sensingNoise.items()]
    assert all(rslt)
