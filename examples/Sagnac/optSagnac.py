"""
Create a Sagnac model with polarization.

This is copied almost exactly from the optPolSag.m example from Optickle
in order to test pytickle
"""

from __future__ import division
import numpy as np
import pytickle as pyt
import scipy.constants as scc


def optSagnac(eng, opt, sqzAng=0, sqdB=0, antidB=0):
    """Create a Sagnac Optickle model

    Inputs:
      eng: matlab engine
      opt: name of the optickle model
      sqzAng: squeeze angle [deg]
      sqzdB: squeezing amplitude in dB
      antidB: anti-squeezing in dB

    Returns:
      opt: the pytickle model
    """
    lambda0 = 1064e-9
    Pin = 100
    gamma = 0.1

    PBSleakS = 0.01
    PBSleakP = 0.01
    # Tpbs = [[PBSleakS, lambda0, 1],
    #         [1 - PBSleakP, lambda0, 0]]
    Tbs = 0.5
    Tin = 0.01
    Tend = 10e-6
    lCav = 4e3

    fmod = 20e6
    vRF = np.array([-fmod, 0, fmod])
    vRF = np.concatenate((vRF, vRF))
    pol = np.array(['S']*3 + ['P']*3)

    opt = pyt.PyTickle(eng, opt, vRF, lambda0, pol)

    ######################################################################
    # Add components
    ######################################################################

    # Add laser with all power in the P carrier
    inds = np.logical_and(opt.vRF == 0, opt.pol == 'P')
    opt.addSource('Laser', inds*np.sqrt(Pin))

    # Modulators
    opt.addModulator('AM', 1)
    opt.addModulator('PM', 1j)
    opt.addRFmodulator('Mod1', fmod, gamma*1j)

    opt.addBeamSplitter('BS', aoi=45, Thr=Tbs)
    opt.addPBS('PBS', aoi=45, transS=PBSleakS, reflP=PBSleakP, BS=True)

    opt.addWaveplate('WPX_A', 0.25, 45)
    opt.addWaveplate('WPX_B', 0.25, -45)
    opt.addWaveplate('WPY_A', 0.25, 45)
    opt.addWaveplate('WPY_B', 0.25, -45)

    opt.addMirror('IX', Thr=Tin)
    opt.addMirror('EX', Chr=0.7/lCav, Thr=Tend)
    opt.addMirror('IY', Thr=Tin)
    opt.addMirror('EY', Chr=0.7/lCav, Thr=Tend)

    ######################################################################
    # Add links
    ######################################################################

    # input
    opt.addLink('Laser', 'out', 'AM', 'in', 0)
    opt.addLink('AM', 'out', 'PM', 'in', 0)
    opt.addLink('PM', 'out', 'Mod1', 'in', 0)
    opt.addLink('Mod1', 'out', 'BS', 'frA', 0)

    # beam splitters - links going forward (A sides)
    opt.addLink('BS', 'frA', 'PBS', 'bkA', 0)
    opt.addLink('BS', 'bkA', 'PBS', 'bkB',  0)
    opt.addLink('PBS', 'frA', 'WPX_A', 'in', 0)
    opt.addLink('PBS', 'frB', 'WPY_A', 'in', 0)

    opt.addLink('WPY_A', 'out', 'IY', 'bk', 0)
    opt.addLink('WPX_A', 'out', 'IX', 'bk', 0)

    # X-arm
    opt.addLink('IX', 'fr', 'EX', 'fr', lCav)
    opt.addLink('EX', 'fr', 'IX', 'fr', lCav)

    # Y-arm
    opt.addLink('IY', 'fr', 'EY', 'fr', lCav)
    opt.addLink('EY', 'fr', 'IY', 'fr', lCav)

    # beam splitters - links going backward (B sides)
    opt.addLink('IY', 'bk', 'WPY_B', 'in', 0)
    opt.addLink('IX', 'bk', 'WPX_B', 'in', 0)

    opt.addLink('WPX_B', 'out', 'PBS', 'frB', 0)
    opt.addLink('WPY_B', 'out', 'PBS', 'frA', 0)
    opt.addLink('PBS', 'bkB', 'BS', 'frB', 0)
    opt.addLink('PBS', 'bkA', 'BS', 'bkB', 0)

    ######################################################################
    # Mechanical
    ######################################################################

    w = 2*np.pi*0.7
    mI = 40
    mE = 40

    w_pit = 2*np.pi*0.5
    rTM = 0.17
    tTM = 0.2
    iTM = (3*rTM**2 + tTM**2) / 12

    iI = mE*iTM
    iE = mE*iTM

    dampRes = np.array([0.01 + 1j, 0.01 - 1j])
    opt.setMechTF('IX', -w*dampRes, 1/mI)
    opt.setMechTF('EX', -w*dampRes, 1/mE)
    opt.setMechTF('IY', -w_pit*dampRes, 1/iI, 'pitch')
    opt.setMechTF('EY', -w_pit*dampRes, 1/iE, 'pitch')

    ######################################################################
    # Squeezer
    ######################################################################

    if sqdB > 0:
        opt.addSqueezer('Sqz', pol='P', sqzAng=sqzAng, sqdB=sqdB,
                        antidB=antidB)
        opt.addLink('Sqz', 'out', 'BS', 'bkA', 0)

    ######################################################################
    # Probes
    ######################################################################

    opt.addSink('REFL')
    opt.addLink('BS', 'frB', 'REFL', 'in', 0)

    phi = 0
    opt.addReadout('REFL', fmod, phi)

    opt.addHomodyneReadout('AS', 90, LOpower=1, pol='P', nu=scc.c/lambda0,
                           BnC=False)
    opt.addLink('BS', 'bkB', 'AS_BS', 'fr', 0)

    opt.setCavityBasis('IX', 'EX')

    return opt
