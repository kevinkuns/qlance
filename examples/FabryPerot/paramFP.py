"""
Parameters for a basic Fabry Perot cavity

In more complicated models these dictionaries can be partially generated
from, for example, a gwinc yaml file
"""


def parFP():
    """
    """
    par = {}

    par['lambda0'] = 1064e-9  # laser wavelength [m]
    par['Pin'] = 1  # input laser power [W]
    par['Mod'] = dict(fmod=11e3,  # modulation frequency [Hz]
                      gmod=0.1)   # modulation depth

    Mtm = 300  # testmass mass [m]
    Re = 36e3  # ETM radius of curvature
    Ri = 34e3  # ITM radius of curvature
    Lhr = 0    # HR surface loss

    optics = ['EX', 'IX']
    for optic in optics:
        par[optic] = dict(opt=dict(), mech=dict())

    par['EX']['opt'] = dict(Thr=0, Chr=1/Re, Lhr=Lhr)
    par['IX']['opt'] = dict(Thr=0.014, Chr=1/Ri, Lhr=Lhr)

    par['EX']['mech'] = dict(mass=Mtm, f0=1, Q=100)
    par['IX']['mech'] = dict(mass=Mtm, f0=1, Q=100)

    par['Lcav'] = 40e3  # cavity length [m]

    return par
