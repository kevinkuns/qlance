"""
Unit tests for finesse torsional spring
"""

import matlab.engine
import numpy as np
import pytickle.optickle as pyt
import pytickle.controls as ctrl
import pytickle.plant as plant
import os
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

opt = pyt.PyTickle(eng, 'opt', vRF)
# opt = pyt.PyTickle(eng, 'opt')

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
    tf1 = opt.getTF('REFL_I', HARD, doftype='pitch')
    tf2 = opt.getTF('REFL_I', hard)
    ref = data['tf_REFLI_HARD']
    c1 = np.allclose(tf1, ref)
    c2 = np.allclose(tf2, ref)
    assert np.all([c1, c2])


def test_REFLI_SOFT():
    tf = opt.getTF('REFL_I', SOFT, doftype='pitch')
    ref = data['tf_REFLI_SOFT']
    assert np.allclose(tf, ref)


def test_mech_HARD():
    tf = opt.getMechTF(HARD, HARD, doftype='pitch')
    ref = data['mech_HARD']
    assert np.allclose(tf, ref)


def test_mech_SOFT():
    tf = opt.getMechTF(SOFT, SOFT, doftype='pitch')
    ref = data['mech_SOFT']
    assert np.allclose(tf, ref)


def test_mMech_EX_EX():
    mMech = opt.getMechMod('EX', 'EX', doftype='pitch')
    ref = data['mMech_EX_EX']
    assert np.allclose(mMech, ref)


def test_mMech_IX_EX():
    mMech = opt.getMechMod('IX', 'EX', doftype='pitch')
    ref = data['mMech_IX_EX']
    assert np.allclose(mMech, ref)


def test_bsm_EX_IX():
    bsm = opt.computeBeamSpotMotion('EX', 'fr', 'IX', 'pitch')
    ref = data['bsm_EX_IX']
    assert np.allclose(bsm, ref)


def test_bsm_EX_EX():
    bsm = opt.computeBeamSpotMotion('EX', 'fr', 'EX', 'pitch')
    ref = data['bsm_EX_EX']
    assert np.allclose(bsm, ref)


##############################################################################
# test reloaded plants
##############################################################################

def test_load_REFLI_HARD():
    tf = opt2.getTF('REFL_I', HARD, doftype='pitch')
    ref = data['tf_REFLI_HARD']
    assert np.allclose(tf, ref)


def test_load_REFLI_SOFT():
    tf = opt2.getTF('REFL_I', SOFT, doftype='pitch')
    ref = data['tf_REFLI_SOFT']
    assert np.allclose(tf, ref)


def test_load_mech_HARD():
    tf = opt2.getMechTF(HARD, HARD, doftype='pitch')
    ref = data['mech_HARD']
    assert np.allclose(tf, ref)


def test_load_mech_SOFT():
    tf = opt2.getMechTF(SOFT, SOFT, doftype='pitch')
    ref = data['mech_SOFT']
    assert np.allclose(tf, ref)


def test_load_mMech_EX_EX():
    mMech = opt2.getMechMod('EX', 'EX', doftype='pitch')
    ref = data['mMech_EX_EX']
    assert np.allclose(mMech, ref)


def test_load_mMech_IX_EX():
    mMech = opt2.getMechMod('IX', 'EX', doftype='pitch')
    ref = data['mMech_IX_EX']
    assert np.allclose(mMech, ref)


def test_load_bsm_EX_IX():
    bsm = opt2.computeBeamSpotMotion('EX', 'fr', 'IX', 'pitch')
    ref = data['bsm_EX_IX']
    assert np.allclose(bsm, ref)


def test_load_bsm_EX_EX():
    bsm = opt2.computeBeamSpotMotion('EX', 'fr', 'EX', 'pitch')
    ref = data['bsm_EX_EX']
    assert np.allclose(bsm, ref)
