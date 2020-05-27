"""
Define a dictionary of parameters for both the Opticke and Finesse FPMI
models.

Defining dictionaries is often useful and can be organized in whatever
manner is convenient.
"""


def parFPMI(zs, ps, k):
    """Define a dictionary of parameters for an FPMI

    Takes as arguments a zpk model for the mechanical response of
    the optics to forces acting on them. All of the optics will
    get the same response in this example. The zeros and poles
    should be given in the s-domain.

    Inputs:
      zs: the zeros [rad/s]
      ps: the poles [rad/s]
      k: the gain

    Returns:
      par: the parameter dictionary
    """
    par = {}

    par['lambda0'] = 1064e-9
    par['Pin'] = 100
    par['Mod'] = dict(fmod=55e6, gmod=0.1)

    Larm = 400  # arm length [m]
    lm = 3      # mean distance between BS and ITMs
    lsch = 0.7  # Schnupp asymmetry
    par['Length'] = {}
    par['Length']['Lx'] = Larm
    par['Length']['Ly'] = Larm
    par['Length']['lx'] = lm + lsch/2
    par['Length']['ly'] = lm - lsch/2

    par['mirrors'] = ['EX', 'IX', 'EY', 'IY']
    par['splitters'] = ['BS']

    for optic in par['mirrors'] + par['splitters']:
        par[optic] = dict(opt=dict(), mech=dict())

    Re = 2190  # ETM radius of curvature [m]
    Ri = 1970  # ITM radius of curvature [m]

    Ti = 0.014  # ITM transmissivity
    Lhr = 0     # HR surface loss

    par['EX']['opt'] = dict(Thr=0, Chr=1/Re, Lhr=Lhr)
    par['EY']['opt'] = dict(Thr=0, Chr=1/Re, Lhr=Lhr)
    par['IX']['opt'] = dict(Thr=Ti, Chr=1/Ri, Lhr=Lhr)
    par['IY']['opt'] = dict(Thr=Ti, Chr=1/Ri, Lhr=Lhr)
    par['BS']['opt'] = dict(Thr=0.5, Lhr=Lhr)

    for optic in par['mirrors'] + par['splitters']:
        par[optic]['mech'] = dict(zs=zs, ps=ps, k=k)

    return par
