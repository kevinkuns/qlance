import numpy as np
import pykat
import pykat.components as kcmp
import pykat.detectors as kdet
import pykat.commands as kcmd
from . import controls as ctrl
from . import plotting
from .utils import append_str_if_unique, add_conjugates
from numbers import Number
from tqdm import tqdm


def addMirror(
        kat, name, aoi=0, Chr=0, Thr=0, Lhr=0, Rar=0, Lmd=0, Nmd=1.45, pos=0,
        pitch=0, yaw=0, dh=0):
    # FIXME: deal with "mirrors" with non-zero aoi's correctly
    if Chr == 0:
        Rcx = None
        Rcy = None
    else:
        Rcx = 1/Chr
        Rcy = 1/Chr

    phi = 2*np.pi * pos/kat.lambda0
    fr = name + '_fr'
    bk = name + '_bk'
    fr_s = fr + '_s'
    bk_s = fr + '_s'
    kat.add(kcmp.mirror(
        name, fr, bk, T=Thr, L=Lhr, phi=phi, Rcx=Rcx, Rcy=Rcy))
    # kat.add(kcmp.mirror(
    #     name, fr, fr_s, T=Thr, L=Lhr, phi=phi, Rcx=Rcx, Rcy=Rcy))
    # kat.add(name + '_AR', bk_s, bk, R=Rar)


def addBeamSplitter(kat, name, aoi=45, Chr=0, Thr=0.5, Lhr=0, pos=0):
    if Chr == 0:
        Rcx = None
        Rcy = None
    else:
        Rcx = 1/Chr
        Rcy = 1/Chr

    phi = 2*np.pi * pos/kat.lambda0
    frI = name + '_frI'
    frR = name + '_frR'
    bkT = name + '_bkT'
    bkO = name + '_bkO'
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
    if dof not in ['pos', 'pitch', 'yaw']:
        raise ValueError('Unrecognized dof ' + dof)

    if force:
        if dof == 'pos':
            return kat.components[drive].Fz
        elif dof == 'pitch':
            return kat.components[drive].Fry
        elif dof == 'yaw':
            return kat.components[drive].Frx

    else:
        if dof == 'pos':
            return kat.components[drive].phi
        elif dof == 'pitch':
            return kat.components[drive].ybeta
        elif dof == 'yaw':
            return kat.components[drive].xbeta


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

    if isinstance(tf, kcmd.tf):
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


class KatFR:
    def __init__(self, kat, all_drives=True):
        self._dofs = ['pos', 'pitch', 'yaw', 'amp', 'freq']
        self.kat = kat

        # populate the list of drives and position detectors if necessary
        if all_drives:
            self.drives = [
                name for name, comp in kat.components.items()
                if isinstance(comp, (kcmp.mirror, kcmp.beamSplitter))]
            self.pos_detectors = [name for name, det in kat.detectors.items()
                                  if isinstance(det, kdet.xd)]
        else:
            self.drives = []
            self.pos_detectors = []

        # populate the list of probes
        self.probes = [name for name, det in kat.detectors.items()
                   if isinstance(det, kdet.pd)]
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

        if verbose:
            pbar = tqdm(total=len(self.drives))

        for drive in self.drives:
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
                    if det.num_demods == 1:
                        det.f1.put(kat.xaxis.x)
                    elif det.num_demods == 2:
                        det.f2.put(kat.xaxis.x)
                    else:
                        raise ValueError('Too many demodulations')

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
                comp = kat_mech.components[drive]
                plant = ctrl.Filter(*extract_zpk(comp, dof), Hz=False)
                tf = plant.computeFilter(out.x)

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
