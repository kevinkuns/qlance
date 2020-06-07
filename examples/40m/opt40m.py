'''
Create a 40m Optickle model.
'''

import numpy as np
import pytickle.optickle as pyt


def opt40m(eng, opt, par, phi=0, zeta=0, Pin=1):
    """Create a 40m Optickle model.

    Inputs:
    eng: matlab engine
    par: parameters of the model
    phi: one-way SRC detuning [deg]
    zeta: homodyne angle [deg]
    Pin: input power incident on the back of PRM [W]

    Two probes AS_DIFF and AS_SUM measure the homodyne difference and sum
    signals at the AS port respectively.

    All angles use BnC conventions:
        phi = 0 deg is ESR
        phi = 90 deg is ERSE
        zeta = 0 deg is the phase quadrature (.i.e. b2)
        zeta = 90 deg is the amplitude quadrature (i.e. b1)
    """

    opt = pyt.PyTickle(eng, opt)

    ##########################################################################
    # Add optics.
    ##########################################################################

    splitters = ['BS', 'PR2', 'PR3', 'SR2', 'SR3']
    mirrors = ['IX', 'IY', 'EX', 'EY', 'PRM', 'SRM']
    for splitter in splitters:
        p = par[splitter]
        opt.addBeamSplitter(
            splitter, p['aoi'], p['Chr'], p['Thr'], p['Lhr'],
            p['Rar'], p['Lmd'])
        opt.setPosOffset(splitter, p['pos'])

    for mirror in mirrors:
        p = par[mirror]
        opt.addMirror(
            mirror, p['aoi'], p['Chr'], p['Thr'], p['Lhr'],
            p['Rar'], p['Lmd'])
        opt.setPosOffset(mirror, p['pos'])

    # Set SRM detuning.
    opt.setPosOffset('SRM', -phi*par['lambda']/360)

    ##########################################################################
    # Mechanical.
    ##########################################################################

    # Set transfer functions.
    dampRes = np.array([-0.1 + 1j, -0.1 - 1j])
    poles = par['w'] * dampRes
    # optics = ['EX', 'EY', 'IX', 'IY']
    optics = mirrors + splitters
    for optic in optics:
        opt.setMechTF(optic, [], poles, 1/par[optic]['mass'])
        opt.setMechTF(optic, [], poles, 1/par[optic]['mass'], 'pitch')
        opt.setMechTF(optic, [], poles, 1/par[optic]['mass'], 'yaw')

    ##########################################################################
    # Input.
    ##########################################################################

    # Main laser.
    opt.addSource('Laser', np.sqrt(Pin))

    # Modulators for amplitude and phase noise.
    opt.addModulator('AM', 1)
    opt.addModulator('PM', 1j)

    ##########################################################################
    # Add links.
    ##########################################################################

    # Input.
    opt.addLink('Laser', 'out', 'AM', 'in', 0)
    opt.addLink('AM', 'out', 'PM', 'in', 0)
    opt.addLink('PM', 'out', 'PRM', 'bk', 0)

    # PRC.
    opt.addLink('PRM', 'fr', 'PR2', 'frA', par['Length']['PR1'])
    opt.addLink('PR2', 'frA', 'PR3', 'frA', par['Length']['PR2'])
    opt.addLink('PR3', 'frA', 'BS', 'frA', par['Length']['PR3'])
    opt.addLink('BS', 'frB', 'PR3', 'frB', par['Length']['PR3'])
    opt.addLink('PR3', 'frB', 'PR2', 'frB', par['Length']['PR2'])
    opt.addLink('PR2', 'frB', 'PRM', 'fr', par['Length']['PR1'])

    # X arm.
    opt.addLink('BS', 'frA', 'IX', 'bk', par['Length']['BS_X'])
    opt.addLink('IX', 'fr', 'EX', 'fr', par['Length']['Xarm'])
    opt.addLink('EX', 'fr', 'IX', 'fr', par['Length']['Xarm'])
    opt.addLink('IX', 'bk', 'BS', 'frB', par['Length']['BS_X'])

    # Y arm.
    opt.addLink('BS', 'bkA', 'IY', 'bk', par['Length']['BS_Y'])
    opt.addLink('IY', 'fr', 'EY', 'fr', par['Length']['Yarm'])
    opt.addLink('EY', 'fr', 'IY', 'fr', par['Length']['Yarm'])
    opt.addLink('IY', 'bk', 'BS', 'bkB', par['Length']['BS_Y'])

    # Output and SRC.
    opt.addLink('BS', 'bkB', 'SR3', 'frB', par['Length']['SR3'])
    opt.addLink('SR3', 'frB', 'SR2', 'frB', par['Length']['SR2'])
    opt.addLink('SR2', 'frB', 'SRM', 'fr', par['Length']['SR1'])
    opt.addLink('SRM', 'fr', 'SR2', 'frA', par['Length']['SR1'])
    opt.addLink('SR2', 'frA', 'SR3', 'frA', par['Length']['SR2'])
    opt.addLink('SR3', 'frA', 'BS', 'bkA', par['Length']['SR3'])

    opt.setCavityBasis('IX', 'EX')
    opt.setCavityBasis('IY', 'EY')

    ##########################################################################
    # Add Readout.
    ##########################################################################

    opt.addHomodyneReadout('AS', zeta, par['qe'], LOpower=0)

    # Pick off LO from PR2
    opt.addLink('PR2', 'bkA', 'AS_LOphase', 'fr', 0)

    # set homodyne phase
    homoPhase = ((zeta + phi)/2 + 45)*par['lambda']/360
    opt.setPosOffset('AS_LOphase', homoPhase)

    # Connect signal to homodyne detector
    opt.addLink('SRM', 'bk', 'AS_BS', 'fr', 0)

    return opt
