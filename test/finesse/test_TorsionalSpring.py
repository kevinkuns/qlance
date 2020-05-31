"""
Unit tests for finesse torsional spring
"""

import numpy as np
import pytickle.finesse as fin
import pytickle.controls as ctrl
import pytickle.plant as plant
import pykat
import os
import pytest

data = np.load('data/finesse_TorsionalSpring_data.npz')

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

kat = pykat.finesse.kat()

# make the cavity
fin.addMirror(kat, 'EX', Chr=1/Re)
fin.addMirror(kat, 'IX', Thr=Ti, Chr=1/Ri)
fin.addSpace(kat, 'IX_fr', 'EX_fr', Lcav)
fin.setCavityBasis(kat, 'IX_fr', 'EX_fr')

# set the pitch response
fin.setMechTF(kat, 'EX', [], poles, 1/I, dof='pitch')
fin.setMechTF(kat, 'IX', [], poles, 1/I, dof='pitch')

# add input
fin.addLaser(kat, 'Laser', Pin)
fin.addModulator(kat, 'Mod', fmod, gmod, 1, 'pm')
fin.addSpace(kat, 'Laser_out', 'Mod_in', 0)
fin.addSpace(kat, 'Mod_out', 'IX_bk', 0)

# add DC and RF photodiodes
fin.addReadout(kat, 'REFL', 'IX_bk', fmod, 0, dof='pitch')

fin.monitorMotion(kat, 'EX', dof='pitch')
fin.monitorMotion(kat, 'IX', dof='pitch')

kat.phase = 2
kat.maxtem = 1

katTF = fin.KatFR(kat)

HARD = {'IX': -1, 'EX': r}
SOFT = {'IX': r, 'EX': 1}

fmin = 1e-1
fmax = 30
npts = 1000
katTF.run(fmin, fmax, npts, dof='pitch')

katTF.save('test_torsional_spring.hdf5')
katTF2 = plant.FinessePlant()
katTF2.load('test_torsional_spring.hdf5')
os.remove('test_torsional_spring.hdf5')

def test_REFLI_HARD():
    tf = katTF.getTF('REFL_I', HARD, dof='pitch')
    ref = data['tf_REFLI_HARD']
    assert np.allclose(tf, ref)


def test_REFLI_SOFT():
    tf = katTF.getTF('REFL_I', SOFT, dof='pitch')
    ref = data['tf_REFLI_SOFT']
    assert np.allclose(tf, ref)


def test_mech_HARD():
    tf = katTF.getMechTF(HARD, HARD, dof='pitch')
    ref = data['mech_HARD']
    assert np.allclose(tf, ref)


def test_mech_SOFT():
    tf = katTF.getMechTF(SOFT, SOFT, dof='pitch')
    ref = data['mech_SOFT']
    assert np.allclose(tf, ref)


def test_mMech_EX_EX():
    mMech = katTF.getMechMod('EX', 'EX', dof='pitch')
    ref = data['mMech_EX_EX']
    assert np.allclose(mMech, ref)


def test_mMech_IX_EX():
    mMech = katTF.getMechMod('IX', 'EX', dof='pitch')
    ref = data['mMech_IX_EX']
    assert np.allclose(mMech, ref)


##############################################################################
# test reloaded plants
##############################################################################

def test_load_REFLI_HARD():
    tf = katTF2.getTF('REFL_I', HARD, dof='pitch')
    ref = data['tf_REFLI_HARD']
    assert np.allclose(tf, ref)


def test_load_REFLI_SOFT():
    tf = katTF2.getTF('REFL_I', SOFT, dof='pitch')
    ref = data['tf_REFLI_SOFT']
    assert np.allclose(tf, ref)


def test_load_mech_HARD():
    tf = katTF2.getMechTF(HARD, HARD, dof='pitch')
    ref = data['mech_HARD']
    assert np.allclose(tf, ref)


def test_load_mech_SOFT():
    tf = katTF2.getMechTF(SOFT, SOFT, dof='pitch')
    ref = data['mech_SOFT']
    assert np.allclose(tf, ref)


def test_load_mMech_EX_EX():
    mMech = katTF2.getMechMod('EX', 'EX', dof='pitch')
    ref = data['mMech_EX_EX']
    assert np.allclose(mMech, ref)


def test_load_mMech_IX_EX():
    mMech = katTF2.getMechMod('IX', 'EX', dof='pitch')
    ref = data['mMech_IX_EX']
    assert np.allclose(mMech, ref)
