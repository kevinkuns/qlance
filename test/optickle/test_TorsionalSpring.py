"""
Unit tests for optickle torsional spring
"""

import matlab.engine
import numpy as np
import qlance.optickle as pyt
import qlance.controls as ctrl
import qlance.plant as plant
import os
import close
import pytest

eng = matlab.engine.start_matlab()
pyt.addOpticklePath(eng)

data = np.load('data/optickle_TorsionalSpring_data.npz')

fmod = 11e3
gmod = 0.1
Pin = 10e3
Ti = 0.014
Lcav = 40e3
Ri = 34e3
Re = 36e3

gi = 1 - Lcav/Ri
ge = 1 - Lcav/Re
r = 2/((gi - ge) + np.sqrt((gi - ge)**2 + 4))

I = 25
f0 = 1
Q = 100
poles = np.array(ctrl.resRoots(2*np.pi*f0, Q, Hz=False))

vRF = np.array([-fmod, 0, fmod])

opt = pyt.Optickle(eng, 'opt', vRF)
# opt = pyt.Optickle(eng, 'opt')

# make the cavity
opt.addMirror('EX', Chr=1/Re)
opt.addMirror('IX', Thr=Ti, Chr=1/Ri)
opt.addLink('IX', 'fr', 'EX', 'fr', Lcav)
opt.addLink('EX', 'fr', 'IX', 'fr', Lcav)
opt.setCavityBasis('IX', 'EX')

# set the pitch response
opt.setMechTF('EX', [], poles, 1/I, doftype='pitch')
opt.setMechTF('IX', [], poles, 1/I, doftype='pitch')

# add input
opt.addSource('Laser', np.sqrt(Pin)*(vRF == 0))
# opt.addSource('Laser', np.sqrt(Pin))
opt.addRFmodulator('Mod', fmod, 1j*gmod)  # RF modulator for PDH sensing
opt.addLink('Laser', 'out', 'Mod', 'in', 0)
opt.addLink('Mod', 'out', 'IX', 'bk', 0)
# opt.addLink('Laser', 'out', 'IX', 'bk', 0)

# add DC and RF photodiodes
opt.addSink('REFL')
opt.addLink('IX', 'bk', 'REFL', 'in', 0)
opt.addReadout('REFL', fmod, 0)

opt.monitorBeamSpotMotion('EX', 'fr')
opt.monitorBeamSpotMotion('IX', 'fr')


HARD = {'IX': -1, 'EX': r}
SOFT = {'IX': r, 'EX': 1}

fmin = 1e-1
fmax = 30
npts = 1000
ff = np.logspace(np.log10(fmin), np.log10(fmax), npts)
opt.run(ff, doftype='pitch', noise=False)

opt.save('test_torsional_spring.hdf5')
opt2 = plant.OpticklePlant()
opt2.load('test_torsional_spring.hdf5')
os.remove('test_torsional_spring.hdf5')


def test_REFLI_HARD():
    hard = ctrl.DegreeOfFreedom(HARD, 'pitch')
    hard2 = ctrl.DegreeOfFreedom(HARD, 'pitch', probes='REFL_I')
    tf1 = opt.getTF('REFL_I', HARD, doftype='pitch')
    tf2 = opt.getTF('REFL_I', hard)
    tf3 = opt.getTF(hard2)
    ref = data['tf_REFLI_HARD']
    c1 = close.allclose(tf1, ref)
    c2 = close.allclose(tf2, ref)
    c3 = close.allclose(tf3, ref)
    assert np.all([c1, c2, c3])


def test_REFLI_SOFT():
    tf = opt.getTF('REFL_I', SOFT, doftype='pitch')
    ref = data['tf_REFLI_SOFT']
    assert close.allclose(tf, ref)


def test_mech_HARD():
    tf = opt.getMechTF(HARD, HARD, doftype='pitch')
    ref = data['mech_HARD']
    assert close.allclose(tf, ref)


def test_mech_HARD2():
    hard = ctrl.DegreeOfFreedom(HARD, 'pitch')
    tf = opt.getMechTF(hard, hard)
    ref = data['mech_HARD']
    assert close.allclose(tf, ref)


def test_mech_SOFT():
    tf = opt.getMechTF(SOFT, SOFT, doftype='pitch')
    ref = data['mech_SOFT']
    assert close.allclose(tf, ref)


def test_mech_SOFT2():
    soft = ctrl.DegreeOfFreedom(SOFT, 'pitch')
    tf1 = opt.getMechTF(SOFT, soft, doftype='pitch')
    tf2 = opt.getMechTF(soft, SOFT, doftype='pitch')
    ref = data['mech_SOFT']
    c1 = close.allclose(tf1, ref)
    c2 = close.allclose(tf2, ref)
    assert np.all([c1, c2])


def test_mMech_EX_EX():
    mMech = opt.getMechMod('EX', 'EX', doftype='pitch')
    ref = data['mMech_EX_EX']
    assert close.allclose(mMech, ref)


def test_mMech_EX_EX2():
    ex = ctrl.DegreeOfFreedom('EX', doftype='pitch')
    mMech1 = opt.getMechMod('EX', ex, doftype='pitch')
    mMech2 = opt.getMechMod(ex, 'EX', doftype='pitch')
    mMech3 = opt.getMechMod(ex, ex, doftype='pitch')
    mMech4 = opt.getMechMod(ex, ex)
    ref = data['mMech_EX_EX']
    c1 = close.allclose(mMech1, ref)
    c2 = close.allclose(mMech2, ref)
    c3 = close.allclose(mMech3, ref)
    c4 = close.allclose(mMech4, ref)
    assert np.all([c1, c2, c3, c4])


def test_mMech_IX_EX():
    mMech = opt.getMechMod('IX', 'EX', doftype='pitch')
    ref = data['mMech_IX_EX']
    assert close.allclose(mMech, ref)


def test_mMech_IX_EX2():
    ex = ctrl.DegreeOfFreedom('EX', 'pitch')
    ix = ctrl.DegreeOfFreedom('IX', 'pitch')
    mMech = opt.getMechMod(ix, ex)
    ref = data['mMech_IX_EX']
    assert close.allclose(mMech, ref)


def test_bsm_EX_IX():
    bsm = opt.computeBeamSpotMotion('EX', 'fr', 'IX', 'pitch')
    ref = data['bsm_EX_IX']
    assert close.allclose(bsm, ref)


def test_bsm_EX_EX():
    bsm = opt.computeBeamSpotMotion('EX', 'fr', 'EX', 'pitch')
    ref = data['bsm_EX_EX']
    assert close.allclose(bsm, ref)


##############################################################################
# test reloaded plants
##############################################################################

def test_load_REFLI_HARD():
    tf = opt2.getTF('REFL_I', HARD, doftype='pitch')
    ref = data['tf_REFLI_HARD']
    assert close.allclose(tf, ref)


def test_load_REFLI_SOFT():
    tf = opt2.getTF('REFL_I', SOFT, doftype='pitch')
    ref = data['tf_REFLI_SOFT']
    assert close.allclose(tf, ref)


def test_load_mech_HARD():
    tf = opt2.getMechTF(HARD, HARD, doftype='pitch')
    ref = data['mech_HARD']
    assert close.allclose(tf, ref)


def test_load_mech_SOFT():
    tf = opt2.getMechTF(SOFT, SOFT, doftype='pitch')
    ref = data['mech_SOFT']
    assert close.allclose(tf, ref)


def test_load_mMech_EX_EX():
    mMech = opt2.getMechMod('EX', 'EX', doftype='pitch')
    ref = data['mMech_EX_EX']
    assert close.allclose(mMech, ref)


def test_load_mMech_IX_EX():
    mMech = opt2.getMechMod('IX', 'EX', doftype='pitch')
    ref = data['mMech_IX_EX']
    assert close.allclose(mMech, ref)


def test_load_bsm_EX_IX():
    bsm = opt2.computeBeamSpotMotion('EX', 'fr', 'IX', 'pitch')
    ref = data['bsm_EX_IX']
    assert close.allclose(bsm, ref)


def test_load_bsm_EX_EX():
    bsm = opt2.computeBeamSpotMotion('EX', 'fr', 'EX', 'pitch')
    ref = data['bsm_EX_EX']
    assert close.allclose(bsm, ref)
