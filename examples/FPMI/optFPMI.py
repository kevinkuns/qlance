"""
Make a basic Optickle FPMI model
"""

import numpy as np
import qlance.optickle as pyt


def optFPMI(eng, opt_name, par):
    """Make an Optickle FPMI

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

    opt = pyt.Optickle(eng, opt_name, vRF=vRF, lambda0=par['lambda0'])

    # Add optics and set mechanical plants
    mirrors = ['EX', 'IX', 'EY', 'IY']
    splitters = ['BS']

    for mirror in mirrors:
        opt.addMirror(mirror, **par[mirror]['opt'])

    for splitter in splitters:
        opt.addBeamSplitter(splitter, **par[splitter]['opt'])

    for optic in mirrors + splitters:
        pmech = par[optic]['mech']
        opt.setMechTF(optic, pmech['zs'], pmech['ps'], pmech['k'])

    # Add input
    opt.addSource('Laser', np.sqrt(par['Pin'])*(vRF == 0))
    opt.addRFmodulator('Mod', fmod, 1j*gmod)
    opt.addLink('Laser', 'out', 'Mod', 'in', 0)
    opt.addLink('Mod', 'out', 'BS', 'frA', 1)

    # Add links
    plen = par['Length']

    # X arm
    opt.addLink('BS', 'bkA', 'IX', 'bk', plen['lx'])
    opt.addLink('IX', 'fr', 'EX', 'fr', plen['Lx'])
    opt.addLink('EX', 'fr', 'IX', 'fr', plen['Lx'])
    opt.addLink('IX', 'bk', 'BS', 'bkB', plen['lx'])

    # Y arm
    opt.addLink('BS', 'frA', 'IY', 'bk', plen['ly'])
    opt.addLink('IY', 'fr', 'EY', 'fr', plen['Ly'])
    opt.addLink('EY', 'fr', 'IY', 'fr', plen['Ly'])
    opt.addLink('IY', 'bk', 'BS', 'frB', plen['ly'])

    # Add output probes
    opt.addSink('REFL')
    opt.addSink('AS')
    opt.addLink('BS', 'frB', 'REFL', 'in', 1)
    opt.addLink('BS', 'bkB', 'AS', 'in', 1)
    # demod phases chosen to maximize CARM in REFL_I and DARM in AS_Q
    opt.addReadout('REFL', fmod, -11)
    opt.addReadout('AS', fmod, -11)

    return opt
