import numpy as np
import pykat
import pykat.components as kcomp
import pykat.detectors as kdet
import pykat.commands as kcom
from . import plotting
from numbers import Number
from tqdm import tqdm


def addMirror(kat, name, aoi=0, Chr=0, Thr=0, Lhr=0, pos=0):
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
    kat.add(kcomp.mirror(
        name, fr, bk, T=Thr, L=Lhr, phi=phi, Rcx=Rcx, Rcy=Rcy))


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
    kat.add(kcomp.beamSplitter(
        name, frI, frR, bkT, bkO, T=Thr, L=Lhr, alpha=aoi, phi=phi,
        Rcx=Rcx, Rcy=Rcy))


def add_probe(kat, name, node, freq, phase, freqresp=True, alternate_beam=False,
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
      add_probe(kat, 'REFL_DC', 'REFL_in', 0, 0)
      add_probe(kat, 'REFL_Q', 'REFL_in', fmod, 90)
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


def add_readout(kat, name, node, freqs, phases, freqresp=True, alternate_beam=False,
                dof='pos', fnames=None):
    """Add RF and DC probes to a detection port

    Inputs:
      kat: the finesse model
      name: name of the port
      node: node that the readout probes
      freqs: demodulation frequencies
      phases: demodulation phases
      freqresp: same as for add_probe
      alternate_beam: same as for add_probe
      dof: same as for add_probe
      fnames: suffixes for RF probe names (Optional)
        If blank and there are multiple demod frequencies, the suffixes
        1, 2, 3, ... are added

    Examples:
      * add_readout(kat, 'REFL', 'REFL_in', f1, 0)
      adds the probes 'REFL_DC', 'REFL_I', and 'REFL_Q' at demod frequency
      f1 and phases 0 and 90 to the node REFL_in

      * add_readout(kat, 'POP', 'POP_in', [11e6, 55e6], [0, 30],
                    fnames=['11', '55'])
      adds the probes POP_DC, POP_I11, POP_Q11, POP_I55, and POP_Q55 at
      demod frequency 11 MHz w/ phases 0 and 90 and at demod phase 55 MHz and
      phases 30 and 100 to the node POP_in
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
    add_probe(kat, name + '_DC', node, 0, 0, freqresp=freqresp,
              alternate_beam=alternate_beam, dof=dof)
    for freq, phase, fname in zip(freqs, phases, fnames):
        nameI = name + '_I' + fname
        nameQ = name + '_Q' + fname
        add_probe(
            kat, nameI, node, freq, phase, freqresp=freqresp,
            alternate_beam=alternate_beam, dof=dof)
        add_probe(
            kat, nameQ, node, freq, phase + 90, freqresp=freqresp,
            alternate_beam=alternate_beam, dof=dof)


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


class KatFR:
    def __init__(self, kat):
        self._dofs = ['pos', 'pitch', 'yaw']
        self.kat = kat
        self.drives = [name for name, comp in kat.components.items()
                       if isinstance(comp, (kcomp.mirror, kcomp.beamSplitter))]
        self.probes = [name for name, det in kat.detectors.items()
                       if isinstance(det, kdet.pd)]
        self.pos_detectors = [name for name, det in kat.detectors.items()
                              if isinstance(det, kdet.xd)]
        self.amp_detectors = [name for name, det in kat.detectors.items()
                              if isinstance(det, kdet.ad)]
        self.freqresp = {dof: {probe: {} for probe in self.probes}
                         for dof in self._dofs}
        self.mechmod = {dof: {drive: {} for drive in self.pos_detectors}
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
                kcom.xaxis(linlog, [fmin, fmax], kat.signals.f, npts))

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

                for drive_out in self.pos_detectors:
                    self.mechmod[dof][drive_out][drive] = out[drive_out]

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

    def add_drive(self, drive, dof='pos'):
        if dof not in self._dofs:
            raise ValueError('Unrecognized dof ' + dof)

        if drive in self.drives[dof]:
            print('drive {:} with dof {:s} is already added'.format(drive, dof))
        else:
            self.drives[dof].append(drive)
