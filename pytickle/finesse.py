import numpy as np
import pykat
import pykat.detectors as kdet
import pykat.commands as kcom
from . import plotting
from numbers import Number
from tqdm import tqdm


def add_probe(kat, name, node, freq, phase, freqresp=True):
    """Add a probe to a finesse model

    Inputs:
      kat: the finesse model
      name: name of the probe
      node: node the probe probes
      freq: demodulation frequency
      phase: demodulation phase
      freqresp: if True, the probe is used for a frequency response
        measurement (Default: True)

    Examples:
      add_probe(kat, 'REFL_DC', 'REFL_in', 0, 0)
      add_probe(kat, 'REFL_Q', 'REFL_in', fmod, 90)
    """
    if freq:
        kwargs = {'f1': freq, 'phase1': phase}
        if freqresp:
            kwargs['f2'] = 1
        kat.add(kdet.pd(name, 1 + freqresp, node, **kwargs))

    else:
        kwargs = {}
        if freqresp:
            kwargs['f1'] = 1
        kat.add(kdet.pd(name, 0 + freqresp, node, **kwargs))


def add_readout(kat, name, node, freqs, phases, freqresp=True, fnames=None):
    """Add RF and DC probes to a detection port

    Inputs:
      kat: the finesse model
      name: name of the port
      node: node that the readout probes
      freqs: demodulation frequencies
      phases: demodulation phases
      freqresp: same as for add_probe
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
      demod frequency 11 w/ phases 0 and 90 and at demod phase 55 MHz and
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
    add_probe(kat, name + '_DC', node, 0, 0)
    for freq, phase, fname in zip(freqs, phases, fnames):
        nameI = name + '_I' + fname
        nameQ = name + '_Q' + fname
        add_probe(kat, nameI, node, freq, phase, freqresp=freqresp)
        add_probe(kat, nameQ, node, freq, phase + 90, freqresp=freqresp)


class KatFR:
    def __init__(self, kat, drives):
        self.kat = kat
        self.drives = drives
        self.probes = kat.detectors.keys()
        self.freqresp = {probe: {} for probe in self.probes}
        self.ff = None

    def tickle(self, fmin, fmax, npts, linlog='log', verbose=1):
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

            # apply the signal to each detector
            for det in kat.detectors.values():
                if det.num_demods == 1:
                    det.f1.put(kat.xaxis.x)
                elif det.num_demods == 2:
                    det.f2.put(kat.xaxis.x)
                else:
                    raise ValueError('Too many demodulations')

            # run the simulation for this drive
            kat.signals.apply(kat.components[drive].phi, 1, 0)
            kat.parse('yaxis lin re:im')
            kat.parse('scale meter')
            out = kat.run()

            # store the results
            for probe in self.probes:
                self.freqresp[probe][drive] = out[probe]

            if verbose:
                pbar.update()

        if verbose:
            pbar.close()

        self.ff = out.x

    def getTF(self, probes, drives):
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
                tf += pc * drive_pos * self.freqresp[probe][drive]

        return tf

    def plotTF(self, probeName, driveNames, mag_ax=None, phase_ax=None,
               dB=False, phase2freq=False, **kwargs):
        """Plot a transfer function.

        See documentation for plotTF in plotting
        """
        ff = self.ff
        tf = self.getTF(probeName, driveNames)
        fig = plotting.plotTF(
            ff, tf, mag_ax=mag_ax, phase_ax=phase_ax, dB=dB,
            phase2freq=phase2freq, **kwargs)
        return fig
