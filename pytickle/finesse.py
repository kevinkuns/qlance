import numpy as np
import pykat
import pykat.detectors as kdet
import pykat.commands as kcom
from . import plotting
from numbers import Number
from tqdm import tqdm


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
