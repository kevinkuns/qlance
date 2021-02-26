"""
Make a basic Finesse FPMI model
"""

import numpy as np
import qlance.finesse as fin
import pykat


def katFPMI(par):
    """Make a Finesse Fabry FPMI

    Inputs:
      par: parameter dictionary

    Returns:
      kat: the model
    """
    fmod = par['Mod']['fmod']
    gmod = par['Mod']['gmod']

    kat = pykat.finesse.kat()
    kat.lambda0 = par['lambda0']  # set the laser wavelength

    # Add optics and set mechanical plants
    mirrors = ['EX', 'IX', 'EY', 'IY']
    splitters = ['BS']

    for mirror in mirrors:
        fin.addMirror(kat, mirror, **par[mirror]['opt'])

    for splitter in splitters:
        fin.addBeamSplitter(kat, splitter, **par[splitter]['opt'])

    for optic in mirrors + splitters:
        pmech = par[optic]['mech']
        fin.setMechTF(kat, optic, pmech['zs'], pmech['ps'], pmech['k'])

    # need to microscopically tune the BS to make the AS port dark due to
    # the Finesse convention that transmission gets a 90 deg phase shift
    # Setting, for example, kat.EX.phi = 90, kat.IX.phi = 90, and kat.BS.phi = 0
    # would also work
    kat.BS.phi = np.sqrt(2) * 45

    # Add input
    fin.addLaser(kat, 'Laser', par['Pin'])
    fin.addModulator(kat, 'Mod', fmod, gmod, 1, 'pm')
    fin.addSpace(kat, 'Laser_out', 'Mod_in', 0)

    # Add Faraday isolator
    fin.addFaradayIsolator(kat, 'FI')
    fin.addSpace(kat, 'Mod_out', 'FI_fr_in', 0)
    fin.addSpace(kat, 'FI_fr_out', 'BS_frI', 1)

    # Add links
    plen = par['Length']

    # X arm
    fin.addSpace(kat, 'BS_bkT', 'IX_bk', plen['lx'])
    fin.addSpace(kat, 'IX_fr', 'EX_fr', plen['Lx'])

    # Y arm
    fin.addSpace(kat, 'BS_frR', 'IY_bk', plen['ly'])
    fin.addSpace(kat, 'IY_fr', 'EY_fr', plen['Ly'])

    # Add output probes
    fin.addSpace(kat, 'FI_bk_out', 'REFL_in', 1, new_comp='REFL')
    fin.addSpace(kat, 'BS_bkO', 'AS_in', 1, new_comp='AS')
    # demod phases chosen to maximize CARM in REFL_I and DARM in AS_Q
    fin.addReadout(kat, 'REFL', 'REFL_in', fmod, 125)
    fin.addReadout(kat, 'AS', 'AS_in', fmod, 11)

    # add probes to monitor the motion of the test masses and beamsplitter
    fin.monitorMotion(kat, 'EX')
    fin.monitorMotion(kat, 'EY')
    fin.monitorMotion(kat, 'BS')

    # add probes to compute the shotnoise of all probes in the model
    fin.monitorAllQuantumNoise(kat)

    return kat
