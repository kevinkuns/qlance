import numpy as np
import pykat
import pykat.detectors as kdet
import pykat.commands as kcom
from . import plotting
from numbers import Number
from tqdm import tqdm


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


class KatFR:
    def __init__(self, kat):
        self._dofs = ['pos', 'pitch', 'yaw']
        self.kat = kat
        self.drives = {dof: [] for dof in self._dofs}
        self.probes = list(kat.detectors.keys())
        self.freqresp = {dof: {probe: {} for probe in self.probes}
                         for dof in self._dofs}
        self.ff = None

    def tickle(self, fmin, fmax, npts, dof='pos', linlog='log', verbose=1):
        if dof not in self._dofs:
            raise ValueError('Unrecognized dof ' + dof)

        if verbose:
            pbar = tqdm(total=len(self.drives[dof]))

        for drive in self.drives[dof]:
            # make a seperate kat for each drive
            kat = self.kat.deepcopy()
            if verbose <= 1:
                kat.verbose = False

            kat.signals.f = 1
            kat.add(
                kcom.xaxis(linlog, [fmin, fmax], kat.signals.f, npts))

            # apply the signal to each detector
            for det in kat.detectors.values():
                if det.num_demods == 1:
                    det.f1.put(kat.xaxis.x)
                elif det.num_demods == 2:
                    det.f2.put(kat.xaxis.x)
                else:
                    raise ValueError('Too many demodulations')

            # run the simulation for this drive
            # kat.signals.apply(kat.components[drive].phi, 1, 0)
            kat.signals.apply(self._get_drive_dof(kat, drive, dof), 1, 0)
            kat.parse('yaxis lin re:im')
            kat.parse('scale meter')
            out = kat.run()

            # store the results
            for probe in self.probes:
                self.freqresp[dof][probe][drive] = out[probe]

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

    @staticmethod
    def _get_drive_dof(kat, drive, dof):
        if dof == 'pos':
            return kat.components[drive].phi
        elif dof == 'pitch':
            return kat.components[drive].ybeta
        elif dof == 'yaw':
            return kat.components[drive].xbeta
        else:
            raise ValueError('Unrecognized dof ' + dof)
