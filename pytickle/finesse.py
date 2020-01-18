import numpy as np
import pykat
import pykat.components as kcmp
import pykat.detectors as kdet
import pykat.commands as kcmd
from . import controls as ctrl
from . import plotting
from .utils import (append_str_if_unique, add_conjugates,
                    remove_conjugates, siPrefix, assertType)
from numbers import Number
import pandas as pd
from tqdm import tqdm
from itertools import compress
from pykat.external.peakdetect import peakdetect


def addMirror(
        kat, name, aoi=0, Chr=0, Thr=0, Lhr=0, Rar=0, Lmd=0, Nmd=1.45, phi=0,
        pitch=0, yaw=0, dh=0, comp=False):
    # FIXME: deal with "mirrors" with non-zero aoi's correctly
    if Chr == 0:
        Rcx = None
        Rcy = None
    else:
        Rcx = 1/Chr
        Rcy = 1/Chr

    # phi = 2*np.pi * pos/kat.lambda0
    fr = name + '_fr'
    bk = name + '_bk'
    fr_s = fr + '_s'
    bk_s = bk + '_s'

    if comp:
        kat.add(kcmp.mirror(
            name, fr, fr_s, T=Thr, L=Lhr, phi=phi, Rcx=Rcx, Rcy=Rcy))
        kat.add(kcmp.mirror(
            name + '_AR', bk_s, bk, R=Rar, L=0, phi=phi))
        kat.add(kcmp.space(name + '_sub', bk_s, fr_s, dh, Nmd))

    else:
        kat.add(kcmp.mirror(
            name, fr, bk, T=Thr, L=Lhr, phi=phi, Rcx=Rcx, Rcy=Rcy))


def addBeamSplitter(
        kat, name, aoi=45, Chr=0, Thr=0.5, Lhr=0, Rar=0, Lmd=0, Nmd=1.45,
        phi=0, dh=0, comp=False):
    if Chr == 0:
        Rcx = None
        Rcy = None
    else:
        Rcx = 1/Chr
        Rcy = 1/Chr

    # phi = 2*np.pi * pos/kat.lambda0
    frI = name + '_frI'
    frR = name + '_frR'
    bkT = name + '_bkT'
    bkO = name + '_bkO'
    bkT_s = bkT + '_s'
    bkO_s = bkO + '_s'
    frT_s = name + '_frT_s'
    frO_s = name + '_frO_s'

    if comp:
        kat.add(kcmp.beamSplitter(
            name, frI, frR, frT_s, frO_s, T=Thr, L=Lhr, alpha=aoi, phi=phi,
            Rcx=Rcx, Rcy=Rcy))
        kat.add(kcmp.mirror(
            name + '_AR_T', bkT_s, bkT, R=Rar, L=0, phi=phi))
        kat.add(kcmp.mirror(
            name + '_AR_O', bkO_s, bkO, R=Rar, L=0, phi=phi))
        kat.add(kcmp.space(name + '_subT', frT_s, bkT_s, dh, Nmd))
        kat.add(kcmp.space(name + '_subO', frO_s, bkO_s, dh, Nmd))

    else:
        kat.add(kcmp.beamSplitter(
            name, frI, frR, bkT, bkO, T=Thr, L=Lhr, alpha=aoi, phi=phi,
            Rcx=Rcx, Rcy=Rcy))


def addProbe(kat, name, node, freq, phase, freqresp=True, alternate_beam=False,
             dof='pos'):
    """Add a probe to a finesse model

    Inputs:
      kat: the finesse model
      name: name of the probe
      node: node the probe probes
      freq: demodulation frequency
      phase: demodulation phase
      freqresp: if True, the probe is used for a frequency response
        measurement (Default: True)
      alternate_beam: if True, the alternate beam is probed (Default: False)
      dof: which DOF is probed: pos, pitch, or yaw (Default: pos)

    Examples:
      addProbe(kat, 'REFL_DC', 'REFL_in', 0, 0)
      addProbe(kat, 'REFL_Q', 'REFL_in', fmod, 90)
    """
    if dof == 'pos':
        pdtype = None
    elif dof == 'pitch':
        pdtype = 'y-split'
    elif dof == 'yaw':
        pdtype = 'x-split'
    else:
        raise ValueError('Unrecognized dof ' + dof)
    kwargs = {'pdtype': pdtype, 'alternate_beam': alternate_beam}

    if freq:
        kwargs.update({'f1': freq, 'phase1': phase})
        if freqresp:
            kwargs['f2'] = 1
        kat.add(kdet.pd(name, 1 + freqresp, node, **kwargs))

    else:
        if freqresp:
            kwargs['f1'] = 1
        kat.add(kdet.pd(name, 0 + freqresp, node, **kwargs))


def addReadout(kat, name, node, freqs, phases, freqresp=True, alternate_beam=False,
               dof='pos', fnames=None):
    """Add RF and DC probes to a detection port

    Inputs:
      kat: the finesse model
      name: name of the port
      node: node that the readout probes
      freqs: demodulation frequencies [Hz]
      phases: demodulation phases [deg]
      freqresp: same as for add_probe
      alternate_beam: same as for add_probe
      dof: same as for add_probe
      fnames: suffixes for RF probe names (Optional)
        If blank and there are multiple demod frequencies, the suffixes
        1, 2, 3, ... are added

    Examples:
      * addReadout(kat, 'REFL', 'REFL_in', f1, 0)
      adds the probes 'REFL_DC', 'REFL_I', and 'REFL_Q' at demod frequency
      f1 and phases 0 and 90 to the node REFL_in

      * addReadout(kat, 'POP', 'POP_in', [11e6, 55e6], [0, 30],
                    fnames=['11', '55'])
      adds the probes POP_DC, POP_I11, POP_Q11, POP_I55, and POP_Q55 at
      demod frequency 11 MHz w/ phases 0 and 90 and at demod phase 55 MHz and
      phases 30 and 120 to the node POP_in
    """
    # Get demod frequencies and phases
    if isinstance(freqs, Number):
        freqs = np.array([freqs])
    if isinstance(phases, Number):
        phases = np.array([phases])
    if len(freqs) != len(phases):
        raise ValueError(
            'Frequency and phase vectors are not the same length.')

    # Figure out naming scheme
    if fnames is None:
        if len(freqs) > 1:
            fnames = (np.arange(len(freqs), dtype=int) + 1).astype(str)
        else:
            fnames = ['']
    elif isinstance(fnames, str):
        fnames = [fnames]

    # Add the probes
    addProbe(kat, name + '_DC', node, 0, 0, freqresp=freqresp,
             alternate_beam=alternate_beam, dof=dof)
    for freq, phase, fname in zip(freqs, phases, fnames):
        nameI = name + '_I' + fname
        nameQ = name + '_Q' + fname
        addProbe(
            kat, nameI, node, freq, phase, freqresp=freqresp,
            alternate_beam=alternate_beam, dof=dof)
        addProbe(
            kat, nameQ, node, freq, phase + 90, freqresp=freqresp,
            alternate_beam=alternate_beam, dof=dof)


def addGouyReadout(kat, name, phaseA, dphaseB=90):
    """Add Gouy phases for WFS readout

    Inputs:
      kat: the finesse model
      name: base name of the probes
      phaseA: Gouy phase of the A probe [deg]
      dphaseB: additional Gouy phase of the B probe relative to the
        A probe [deg] (Default: 90 deg)
    """
    phaseB = phaseA + dphaseB
    bs_name = name + '_WFS_BS'
    addBeamSplitter(kat, bs_name, Thr=0.5, aoi=45)
    kat.add(kcmp.space(
        name + '_A', bs_name + '_frR', name + '_A', 0, gx=phaseA, gy=phaseA))
    kat.add(kcmp.space(
        name + '_B', bs_name + '_bkT', name + '_B', 0, gx=phaseB, gy=phaseB))


def addModulator(kat, name, fmod, gmod, order, modtype, phase=0):
    """Add a modulator to a finesse model

    Adds a modulator with input node name_in and output node name_out

    Inputs:
      kat: the finesse model
      name: name of the modulator
      fmod: modulation frequency [Hz]
      gmod: modulation depth
      order: number of sidebands to model
      modtype: type of modulation:
        * 'pm' for phase modulation
        * 'am' for amplitude modulation
      phase: phase of modulation [deg] (Default: 0)

    Example:
        addModulator(kat, 'Modf1', 11e6, 0.1, 3, 'pm')
      Adds the phase modulator named 'Modf1' at 11 MHz with modulation
      depth 0.1. The the input node is 'Modf1_in' and the output noide is
      'Modf1_out'. 3 sidebands are tracked in the model.
    """
    kat.add(kcmp.modulator(
        name, name + '_in', name + '_out', fmod, gmod, order,
        modulation_type=modtype, phase=phase))


def addLaser(kat, name, P, f=0, phase=0):
    """Add a laser to a finesse model

    Adds a laser with output node name_out

    Inputs:
      kat: the finesse model
      name: name of the laser
      P: laser power [W]
      f: frequency offset from the carrier [Hz] (Defualt: 0)
      phase: phase of the laser [deg] (Defualt: 0)

    Example:
        addLaser(kat, 'Laser', Pin)
      adds the laser named 'Laser' with input power Pin. The output node
      is 'Laser_out'
    """
    kat.add(kcmd.laser(name, name + '_out', P=P, f=f, phase=phase))


def set_probe_response(kat, name, resp):
    """Set whether a probe will be used to measure DC or frequency response

    Useful for changing how an existing probe will be used. For example
    probes added to a model for DC sweeps can be changed to measure
    transfer functions.

    Inputs:
      kat: the finesse model
      name: name of the probe to set
      resp: type of response:
        'fr': frequency response
        'dc': DC response for sweeps and DC operating point

    Examples:
      * If a probe was added for frequency response as
          addProbe(kat, 'REFL_I', 'REFL_in', fmod, 0)
        it can be used for sweeps by
          set_probe_response(kat, 'REFL_I', 'dc')
      * If a probe was added for DC measurements as
          addProbe(kat, 'REFL_I', 'REFL_in', fmod, 0, freqresp=False)
        it can be used for measuring transfer functions by
          set_probe_response(kat, 'REFL_I', 'fr')
    """
    if resp == 'dc':
        kwargs = {'freqresp': False}
    elif resp == 'fr':
        kwargs = {'freqresp': True}
    else:
        raise ValueError('Unrecognized signal type ' + resp)

    # get the original parameters
    det = kat.detectors[name]
    node = det.node.name
    kwargs['alternate_beam'] = det.alternate_beam

    if det.pdtype is None:
        kwargs['dof'] = 'pos'
    elif det.pdtype == 'y-split':
        kwargs['dof'] = 'pitch'
    elif det.pdtype == 'x-split':
        kwargs['dof'] = 'yaw'
    else:
        raise ValueError('Unrecognized pdtype ' + det.pdtype)

    # figure out what kind of detector it was originally
    if det.num_demods == 0:
        # DC probe for DC response
        freq = 0
        phase = 0

    elif det.num_demods == 1:
        phase = det.phase1.value
        if phase is None:
            # DC probe for freq response
            freq = 0
            phase = 0

        else:
            # RF probe for DC response
            freq = det.f1.value

    elif det.num_demods == 2:
        # RF probe for freq response
        freq = det.f1.value
        phase = det.phase1.value

    else:
        raise ValueError('Too many demodulations {:d}'.format(det.num_demods))

    # Remove the old probe
    det.remove()

    # Add the new one
    addProbe(kat, name, node, freq, phase, **kwargs)


def set_all_probe_response(kat, resp):
    """Set whether every probe in the model will measure DC or frquency response

    Inputs:
      kat: the finesse model
      resp: type of response:
        'fr': frequency response
        'dc': DC response for sweeps and DC operating point

    Example:
      * To prepare all the probes in a model to measure a transfer function:
          set_all_probe_response(kat, 'fr')
      * To prepare all the probes in a model to measure a DC response:
          set_all_probe_response(kat, 'dc')
    """
    for det in kat.detectors.values():
        if isinstance(det, kdet.pd):
            set_probe_response(kat, det.name, resp)


def get_drive_dof(kat, drive, dof, force=False):
    """Return the component for a given degree of freedom

    Inputs:
      kat: the finesse model
      drive: the name of the drive
      dof: which DOF to drive pos, pitch, or yaw
      force: if True a force or torque is applied instead of the motion
        (Default: False)

    Returns:
      the component
    """
    if dof not in ['pos', 'pitch', 'yaw', 'amp', 'freq']:
        raise ValueError('Unrecognized dof ' + dof)

    if force:
        if dof == 'pos':
            return kat.components[drive].Fz
        elif dof == 'pitch':
            return kat.components[drive].Fry
        elif dof == 'yaw':
            return kat.components[drive].Frx
        elif dof == 'amp':
            return kat.components[drive].P
        elif dof == 'freq':
            return kat.components[drive].f

    else:
        if dof == 'pos':
            return kat.components[drive].phi
        elif dof == 'pitch':
            return kat.components[drive].ybeta
        elif dof == 'yaw':
            return kat.components[drive].xbeta
        elif dof == 'amp':
            return kat.components[drive].P
        elif dof == 'freq':
            return kat.components[drive].f


def has_dof(kat, drive, dof):
    """Check whether a component has a given degree of freedom
    """
    comp = kat.components[drive]
    if dof in ['pos', 'pitch', 'yaw']:
        if isinstance(comp, (kcmp.mirror, kcmp.beamSplitter)):
            return True
        else:
            return False

    elif dof in ['amp', 'freq']:
        if isinstance(comp, kcmp.laser):
            return True
        else:
            return False


def extract_zpk(comp, dof):
    """Extract the zpk of a mechanical transfer function for a kat component

    Extracts the zeros, poles, and gain of the longitudinal, pitch, or yaw
    response of a kat component.

    The zeros and poles are returned in the s-domain.

    Inputs:
      comp: the component
      dof: the degree of freedom (pos, pitch, or yaw)

    Returns:
      zs: the zeros
      ps: the poles
      k: the gain

    Examples:
      extract_zpk(kat.EX, 'pos')
      extract_zpk(kat.components['IX'], 'pitch')
    """
    if dof == 'pos':
        tf = comp.zmech.value
    elif dof == 'pitch':
        tf = comp.rymech.value
    elif dof == 'yaw':
        tf = comp.rxmech.value
    else:
        raise ValueError('Unrecognized dof ' + dof)

    if tf is None:
        # There's no mechanical transfer function defined, so the response is
        # either 1/(M Omega^2) for pos or 1/(I Omega^2) for pitch or yaw
        zs = []
        ps = [0, 0]
        if dof == 'pos':
            kinv = comp.mass.value
        elif dof == 'pitch':
            kinv = comp.Iy.value
        elif dof == 'yaw':
            kinv = comp.Ix.value

        if kinv is None:
            k = 1
        else:
            k = 1/kinv

    elif isinstance(tf, kcmd.tf):
        zs = []
        ps = []
        for z in tf.zeros:
            zs.extend(ctrl.resRoots(2*np.pi*z.f, z.Q, Hz=False))
        for p in tf.poles:
            ps.extend(ctrl.resRoots(2*np.pi*p.f, p.Q, Hz=False))
        zs = np.array(zs)
        ps = np.array(ps)
        k = tf.gain

    elif isinstance(tf, kcmd.tf2):
        zs = add_conjugates(tf.zeros)
        ps = add_conjugates(tf.poles)
        k = tf.gain

    else:
        raise ValueError('Unrecognized transfer function')

    return zs, ps, k


def setMechTF(kat, name, zs, ps, k, dof='pos'):
    """Set the mechanical transfer function of an optic

    The transfer function is from radiation pressure to one of the degrees
    of freedom position, pitch, or yaw.

    The zeros and poles should be in the s-domain.

    NOTE: Unlike with pure finesse code, the mechanical transfer function is
    defined with the specified zpk model: ALL ZEROS AND POLES MUST BE GIVEN.

    Inputs:
      kat: the finesse model
      name: name of the optic
      zs: the zeros
      ps: the poles
      k: the gain
      dof: degree of freedom: pos, pitch, or yaw (Default: pos)
    """
    # define the transfer function and add it to the model
    tf_name = '{:s}TF_{:s}'.format(dof, name)
    tf = kcmd.tf2(tf_name)
    for z in remove_conjugates(zs):
        tf.addZero(z)
    for p in remove_conjugates(ps):
        tf.addPole(p)
    tf.gain = k
    kat.add(tf)

    # get the component and set the mechanical TF
    # masses and moments of inertia must be set to 1 for the gain to be correct
    comp = kat.components[name]

    if dof == 'pos':
        comp.mass = 1
        comp.zmech = tf
    elif dof == 'pitch':
        comp.Iy = 1
        comp.rymech = tf
    elif dof == 'yaw':
        comp.Ix = 1
        comp.rxmech = tf
    else:
        raise ValueError('Unrecognized dof ' + dof)


def setCavityBasis(kat, node_name1, node_name2):
    """Specify a two mirror cavity to use for the Gaussian mode basis

    Inputs:
      kat: the finesse model
      node_name1: the name of the node of the first mirror
      node_name2: the name of the node of the second mirror

    Example:
      To use the basis defined by the front surfaces of IX and EX
        setCavityBasis(kat, 'IX_fr', 'EX_fr')
      This is equivalent to
        kat.add(kcmd.cavity('cav_IX_EX', kat.IX, kat.IX_fr, kat.EX, kat.EX_fr))
    """
    # Get the list of nodes in this model
    nodes = kat.nodes.getNodes()

    def get_mirror_node(node_name):
        """Get the node and optic connected to a given node name

        Inputs:
          node_name: the name of the node

        Returns:
          node: the node object
          mirr: the mirror object connected to this node
        """
        try:
            node = nodes[node_name]
        except KeyError:
            raise ValueError(node_name + ' is not a node in this model')

        components = kat.nodes.getNodeComponents(node)
        mirr_inds = [isinstance(comp, kcmp.mirror) for comp in components]
        space_inds = [isinstance(comp, kcmp.space) for comp in components]

        # make sure that this node is connected to one mirror and one space
        num_mirrs = sum(mirr_inds)
        num_spaces = sum(space_inds)
        if num_mirrs != 1 or num_spaces != 1:
            raise ValueError(
                'A cavity should be two mirrors connected by a space but'
                + ' node {:s} has {:d} mirrors and {:d} spaces.'.format(
                    node_name, num_mirrs, num_spaces))

        return node, list(compress(components, mirr_inds))[0]

    # set the cavity basis
    node1, mirr1 = get_mirror_node(node_name1)
    node2, mirr2 = get_mirror_node(node_name2)
    cav_name = 'cav_{:s}_{:s}'.format(mirr1.name, mirr2.name)
    kat.add(kcmd.cavity(cav_name, mirr1, node1, mirr2, node2))


def add_lock(kat, name, probe, drive, gain, tol, offset=0, dof='pos'):
    """Add a lock to a finesse model

    Inputs:
      kat: the finesse model
      name: name of the lock
      probe: name of the probe used for the error signal
      drive: name of the drive the error signal is fed back to
      gain: gain of the lock
      tol: tolerance (or accuracy) of the lock
      offset: offset in the error point (Default: 0)
      dof: degree of freedom to drive (Default 'pos')
    """
    # FIXME: add functionality for linear combinations of probes and drives
    if probe not in kat.detectors.keys():
        raise ValueError('Probe {:s} does not exist'.format(probe))

    lock_name = name + '_lock'
    err_sig = '${:s}_err'.format(name)
    kat.parse('set {:s}_err0 {:s} re'.format(name, probe))
    off_sign = np.sign(offset)
    off_abs = np.abs(offset)
    func = '${0}_err0 + ({1}) * ({2})'.format(name, off_sign, off_abs)
    kat.add(kcmd.func(name + '_err', func))
    kat.add(kcmd.lock(lock_name, err_sig, gain, tol))
    comp = get_drive_dof(kat, drive, dof, force=False)
    comp.put(kat.commands[lock_name].output)


def remove_all_locks(kat):
    """Remove all lock commands from a finesse model

    Inputs:
      kat: the finesse model
    """
    for cmd in kat.commands.values():
        if isinstance(cmd, kcmd.lock):
            cmd.remove()


def set_tunings(kat, tunings, dof='pos'):
    for drive, pos in tunings.items():
        comp = get_drive_dof(kat, drive, dof, force=False)
        comp.value = pos


def showfDC(basekat, freqs, verbose=False):
    kat = basekat.deepcopy()
    if isinstance(freqs, Number):
        freqs = np.array([freqs])
    freqs.sort()

    links = []
    for comp in kat.components.values():
        if isinstance(comp, kcmp.space):
            name = comp.name
            end_node = comp.nodes[0].name
            links.append(name)
            for fi, freq in enumerate(freqs):
                probe_name = '{:s}_f{:d}'.format(name, fi)
                kat.add(kdet.ad(probe_name, freq, end_node))

    kat.noxaxis = True
    kat.verbose = verbose
    out = kat.run()

    def link2key(link):
        space = kat.components[link]
        key = '{:s}  -->  {:s}'.format(space.nodes[0].name,
                                       space.nodes[1].name)
        return key

    fDC = {link2key(link): [] for link in links}
    for link in links:
        for fi, freq in enumerate(freqs):
            power = np.abs(out['{:s}_f{:d}'.format(link, fi)])**2
            pow_str = '{:0.1f} {:s}W'.format(*siPrefix(power)[::-1])
            # key = '{:s} -> {:s}'.format(space.nodes[0].name,
            #                             space.nodes[1].name)
            fDC[link2key(link)].append(pow_str)

    index = ['{:0.0f} {:s}Hz'.format(*siPrefix(freq)[::-1]) for freq in freqs]
    fDC = pd.DataFrame(fDC, index=index).T
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None):
        display(fDC)


def showsigDC(basekat, verbose=False):
    kat = basekat.deepcopy()
    set_all_probe_response(kat, 'dc')
    kat.noxaxis = True
    kat.verbose = verbose
    out = kat.run()

    sigDC = {}
    for det_name, det in kat.detectors.items():
        if isinstance(det, kdet.pd):
            power = np.abs(out[det_name])
            sigDC[det_name] = '{:0.1f} {:s}W'.format(*siPrefix(power)[::-1])

    sigDC = pd.DataFrame(sigDC, index=['Power']).T
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None):
        display(sigDC)


class KatFR:
    def __init__(self, kat, all_drives=True):
        self._dofs = ['pos', 'pitch', 'yaw', 'amp', 'freq']
        self.kat = kat.deepcopy()
        set_all_probe_response(self.kat, 'fr')

        # populate the list of drives and position detectors if necessary
        if all_drives:
            comps = (kcmp.mirror, kcmp.beamSplitter, kcmp.laser)
            self.drives = [
                name for name, comp in kat.components.items()
                if isinstance(comp, comps)]

            self.pos_detectors = [name for name, det in kat.detectors.items()
                                  if isinstance(det, kdet.xd)]
        else:
            self.drives = []
            self.pos_detectors = []

        # populate the list of probes
        self.probes = [name for name, det in kat.detectors.items()
                       if isinstance(det, (kdet.pd, kdet.hd))]
        self.amp_detectors = [name for name, det in kat.detectors.items()
                              if isinstance(det, kdet.ad)]

        # initialize response dictionaries
        self.freqresp = {dof: {probe: {} for probe in self.probes}
                         for dof in self._dofs}
        self.mechmod = {dof: {drive: {} for drive in self.pos_detectors}
                        for dof in self._dofs}
        self._mechTF = {dof: {drive: {} for drive in self.pos_detectors}
                        for dof in self._dofs}

        self.ff = None

    def tickle(self, fmin, fmax, npts, dof='pos', linlog='log', rtype='both',
               verbose=1):
        if dof not in self._dofs:
            raise ValueError('Unrecognized dof ' + dof)

        if rtype not in ['opt', 'mech', 'both']:
            raise ValueError('Unrecognized response type ' + rtype)

        drives = [drive for drive in self.drives if has_dof(
            self.kat, drive, dof)]

        if verbose:
            pbar = tqdm(total=len(drives))

        for drive in drives:
            # make a seperate kat for each drive
            kat = self.kat.deepcopy()
            if verbose <= 1:
                kat.verbose = False

            kat.signals.f = 1
            kat.add(
                kcmd.xaxis(linlog, [fmin, fmax], kat.signals.f, npts))

            # apply the signal to each photodiode to compute the
            # optomechanical plant
            if rtype in ['opt', 'both']:
                for probe in self.probes:
                    det = kat.detectors[probe]
                    if det.num_demods == 0:
                        raise ValueError(
                            '{:s} has no demodulations'.format(probe))
                    if det.num_demods == 1:
                        det.f1.put(kat.xaxis.x)
                    elif det.num_demods == 2:
                        det.f2.put(kat.xaxis.x)
                    else:
                        raise ValueError(
                            '{:s} has too many demodulations'.format(probe))

            # run the simulation for this drive and store the results
            kat.parse('yaxis re:im')

            # compute the optomechanical plant
            if rtype in ['opt', 'both']:
                kat_opt = kat.deepcopy()
                kat_opt.signals.apply(
                    get_drive_dof(kat, drive, dof, force=False), 1, 0)
                if dof == 'pos':
                    kat_opt.parse('scale meter')
                out = kat_opt.run()

                for probe in self.probes:
                    self.freqresp[dof][probe][drive] = out[probe]

            # compute the radiation pressure modifications to drives
            if rtype in ['mech', 'both']:
                kat_mech = kat.deepcopy()
                kat_mech.signals.apply(
                    get_drive_dof(kat, drive, dof, force=True), 1, 0)
                out = kat_mech.run()

                # extract the mechanical plant for this drive
                comp = kat_mech.components[drive]
                if dof in ['pos', 'pitch', 'yaw']:
                    plant = ctrl.Filter(*extract_zpk(comp, dof), Hz=False)
                    tf = plant.computeFilter(out.x)
                else:
                    tf = np.ones_like(out.x)

                for drive_out in self.pos_detectors:
                    self.mechmod[dof][drive_out][drive] = out[drive_out] / tf
                    self._mechTF[dof][drive_out][drive] = out[drive_out]

            if verbose:
                pbar.update()

        if verbose:
            pbar.close()

        self.ff = out.x

    def getTF(self, probes, drives, dof='pos'):
        if dof not in self._dofs:
            raise ValueError('Unrecognized dof ' + dof)

        # figure out the shape of the TF
        if isinstance(self.ff, Number):
            # TF is at a single frequency
            tf = 0
        else:
            # TF is for a frequency vector
            tf = np.zeros(len(self.ff), dtype='complex')

        if isinstance(drives, str):
            drives = {drives: 1}

        if isinstance(probes, str):
            probes = {probes: 1}

        # loop through the drives and probes to compute the TF
        for probe, pc in probes.items():
            for drive, drive_pos in drives.items():
                # add the contribution from this drive
                tf += pc * drive_pos * self.freqresp[dof][probe][drive]

        return tf

    def getMechTF(self, outDrives, inDrives, dof='pos'):
        """Compute a mechanical transfer function

        Inputs:
          outDrives: name of the output drives
          inDrives: name of the input drives
          dof: degree of freedom: pos, pitch, or yaw (Default: pos)

        Returns:
          tf: the transfer function
            * In units of [m/N] for position
            * In units of [rad/(N m)] for pitch and yaw
        """
        if dof not in ['pos', 'pitch', 'yaw']:
            raise ValueError('Unrecognized degree of freedom {:s}'.format(dof))

        # figure out the shape of the TF
        if isinstance(self.ff, Number):
            # TF is at a single frequency
            tf = 0
        else:
            # TF is for a frequency vector
            tf = np.zeros(len(self.ff), dtype='complex')

        if isinstance(outDrives, str):
            outDrives = {outDrives: 1}

        if isinstance(inDrives, str):
            inDrives = {inDrives: 1}

        # loop through drives to compute the TF
        for inDrive, c_in in inDrives.items():
            # get the default mechanical plant of the optic being driven
            comp = self.kat.components[inDrive]
            plant = ctrl.Filter(*extract_zpk(comp, dof), Hz=False)

            for outDrive, c_out in outDrives.items():
                mmech = self.getMechMod(outDrive, inDrive, dof=dof)
                tf += c_in * c_out * plant.computeFilter(self.ff) * mmech

        return tf

    def getMechMod(self, drive_out, drive_in, dof='pos'):
        if dof not in self._dofs:
            raise ValueError('Unrecognized dof ' + dof)

        out_det = drive_out + '_' + dof
        if out_det not in self.pos_detectors:
            raise ValueError(out_det + ' is not a detector in this model')

        return self.mechmod[dof][out_det][drive_in]

    def getQuantumNoise(self, probeName, dof):
        qnoise = list(self.freqresp[dof][probeName + '_shot'].values())[0]
        if np.any(np.iscomplex(qnoise)):
            print('Warning: some quantum noise spectra are complex')
        return np.real(qnoise)

    def addDrives(self, drives):
        append_str_if_unique(self.drives, drives)

    def removeDrives(self, drives):
        if isinstance(drives, str):
            drives = [drives]

        for drive in drives:
            try:
                self.drives.remove(drive)
            except ValueError:
                pass

    def plotTF(self, probeName, driveNames, mag_ax=None, phase_ax=None,
               dof='pos', dB=False, phase2freq=False, **kwargs):
        """Plot a transfer function.

        See documentation for plotTF in plotting
        """
        ff = self.ff
        tf = self.getTF(probeName, driveNames, dof=dof)
        fig = plotting.plotTF(
            ff, tf, mag_ax=mag_ax, phase_ax=phase_ax, dB=dB,
            phase2freq=phase2freq, **kwargs)
        return fig


class KatSweep:
    def __init__(self, kat, drives, dof='pos', relative=True):
        self.kat = kat.deepcopy()
        set_all_probe_response(self.kat, 'dc')
        self.drives = drives
        if isinstance(self.drives, str):
            self.drives = {self.drives: 1}
        self.dof = dof
        self.sigs = dict.fromkeys(kat.detectors)
        self.lock_sigs = {}
        for cmd in self.kat.commands.values():
            if isinstance(cmd, kcmd.lock):
                self.lock_sigs[cmd.name] = None
        self.poses = dict.fromkeys(assertType(drives, dict))
        self.poses[''] = None
        self._relative = relative
        self.offsets = dict.fromkeys(assertType(drives, dict))

    def sweep(
            self, spos, epos, npts, linlog='lin', verbose=False):
        kat = self.kat.deepcopy()
        kat.verbose = verbose
        kat.add(kcmd.variable('sweep', 1))
        kat.parse('set sweepre sweep re')
        kat.add(kcmd.xaxis(linlog, [spos, epos], 're', npts, comp='sweep'))

        for drive, coeff in self.drives.items():
            comp = get_drive_dof(kat, drive, self.dof, force=False)

            if self._relative:
                self.offsets[drive] = comp.value
            else:
                self.offsets[drive] = 0

            csign = np.sign(coeff)
            cabs = np.abs(coeff)
            p0sign = np.sign(self.offsets[drive])
            p0abs = np.abs(self.offsets[drive])
            name = drive + '_sweep'
            func = '({0}) * ({1}) * $sweepre + ({2}) * ({3})'.format(
                csign, cabs, p0sign, p0abs)
            kat.add(kcmd.func(name, func))
            comp.put(kat.commands[name].output)

        kat.parse('yaxis re:im')
        out = kat.run()

        for probe in self.sigs.keys():
            self.sigs[probe] = out[probe]

        for lock in self.lock_sigs.keys():
            self.lock_sigs[lock] = out[lock]

        for drive, coeff in self.drives.items():
            self.poses[drive] = coeff * out.x + self.offsets[drive]
        self.poses[''] = out.x

    def getSweepSignal(self, probeName, driveName, func=None):
        sig = self.sigs[probeName]
        if func:
            if func == 'abs':
                sig = np.abs(sig)
            elif func == 're':
                sig = np.real(sig)
            elif func == 'im':
                sig = np.imag(sig)
            else:
                raise ValueError('Unrecognized function ' + func)

        return self.poses[driveName], sig

    def find_peak(self, probeName, driveName, minmax, func=None):
        """
        based off of pykat's find_peak
        """
        pos, sig = self.getSweepSignal(probeName, driveName, func=func)
        maxima, minima = peakdetect(sig, pos, 1)

        if minmax == 'min':
            if len(minima) == 0:
                raise ValueError('No minima were found')

            minima = np.array(minima)
            ind = np.argmin(minima[:, 1])
            pos, peak = minima[ind]

        elif minmax == 'max':
            if len(maxima) == 0:
                raise ValueError('No maxima were found')

            maxima = np.array(maxima)
            ind = np.argmax(maxima[:, 1])
            pos, peak = maxima[ind]

        else:
            raise ValueError('Unrecognized option ' + minmax)

        return pos, peak

    def scan_to_precision(
            self, probeName, p0, dp, prec, minmax, npts=400, func=None):
        """
        based off of pykat's scan_to_precision
        """
        pos = p0
        while dp > prec:
            self.sweep(pos - 1.5*dp, pos + 1.5*dp, npts)
            try:
                pos, peak = self.find_peak(probeName, '', minmax, func=func)
            except ValueError as exc:
                print(exc)
                print('pos ' + str(pos))
                print('dp ' + str(pos))
            dp = self.poses[''][1] - self.poses[''][0]
        return pos, peak

    def get_tuning_dict(self, pos):
        tunings = {drive: pos * coeff + self.offsets[drive]
                   for drive, coeff in self.drives.items()}
        return tunings
