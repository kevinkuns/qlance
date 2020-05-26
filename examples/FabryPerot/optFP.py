"""
Basic Optickle Fabry Perot models
"""

import numpy as np
import pytickle.optickle as pyt
from pytickle.controls import resRoots


def optFP(eng, opt_name, par):
    """Make an Optickle Fabry Perot cavity

    Inputs:
      eng: the matlab engine
      opt_name: name of the optickle model
      par: parameter dictionary

    Returns:
      opt: the model
    """
    fmod = par['Mod']['fmod']
    gmod = par['Mod']['gmod']

    vRF = np.array([-fmod, 0, fmod])

    opt = pyt.PyTickle(eng, opt_name, vRF=vRF, lambda0=par['lambda0'])

    # Make the cavity
    # In this case it's very simple, but in more complicated models you can
    # easily loop through all of the optics if you've defined the parameters
    # in a dictionary
    mirrors = ['EX', 'IX']
    for mirror in mirrors:
        # add the mirror and set the optic properties
        opt.addMirror(mirror, **par[mirror]['opt'])

        # set the mechanical response
        pmech = par[mirror]['mech']
        poles = np.array(resRoots(2*np.pi*pmech['f0'], pmech['Q'], Hz=False))
        opt.setMechTF(mirror, [], poles, 1/pmech['mass'])

    opt.addLink('IX', 'fr', 'EX', 'fr', par['Lcav'])
    opt.addLink('EX', 'fr', 'IX', 'fr', par['Lcav'])
    opt.setCavityBasis('IX', 'EX')

    # add input
    opt.addSource('Laser', np.sqrt(par['Pin'])*(vRF == 0))
    opt.addRFmodulator('Mod', fmod, 1j*gmod)
    opt.addLink('Laser', 'out', 'Mod', 'in', 0)
    opt.addLink('Mod', 'out', 'IX', 'bk', 0)

    # add DC and RF photodiodes
    opt.addSink('REFL')
    opt.addLink('IX', 'bk', 'REFL', 'in', 0)
    opt.addReadout('REFL', fmod, 0)

    return opt
