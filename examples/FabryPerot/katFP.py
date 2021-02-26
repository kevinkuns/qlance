"""
Basic Finesse Fabry Perot models
"""

import numpy as np
import qlance.finesse as fin
import pykat
from qlance.controls import resRoots


def katFP(par):
    """Make a Finesse Fabry Perot cavity

    Inputs:
      par: parameter dictionary

    Returns:
      kat: the model
    """
    fmod = par['Mod']['fmod']
    gmod = par['Mod']['gmod']

    kat = pykat.finesse.kat()
    kat.lambda0 = par['lambda0']  # set the laser wavelength

    # Make the cavity
    # In this case it's very simple, but in more complicated models you can
    # easily loop through all of the optics if you've defined the parameters
    # in a dictionary
    mirrors = ['EX', 'IX']
    for mirror in mirrors:
        # add the mirror and set the optic properties
        fin.addMirror(kat, mirror, **par[mirror]['opt'])

        # set the mechanical response
        pmech = par[mirror]['mech']
        poles = np.array(resRoots(2*np.pi*pmech['f0'], pmech['Q'], Hz=False))
        fin.setMechTF(kat, mirror, [], poles, 1/pmech['mass'])

    fin.addSpace(kat, 'IX_fr', 'EX_fr', par['Lcav'])
    fin.setCavityBasis(kat, 'IX_fr', 'EX_fr')

    # add input
    fin.addLaser(kat, 'Laser', par['Pin'])
    fin.addModulator(kat, 'Mod', fmod, gmod, 1, 'pm')
    fin.addSpace(kat, 'Laser_out', 'Mod_in', 0)
    fin.addSpace(kat, 'Mod_out', 'IX_bk', 0)

    # add DC and RF photodiodes
    fin.addReadout(kat, 'REFL', 'IX_bk', fmod, 0)

    return kat
