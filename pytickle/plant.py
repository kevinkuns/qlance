'''
Code for general optomechanical plants
'''

import numpy as np
import matplotlib.pyplot as plt
from . import controls as ctrl
from . import plotting
from . import utils
from numbers import Number


class OpticklePlant:
    def __init__(self):
        self.probes = []
        self.drives = []
        self.vRF = None
        self.lambda0 = None
        self.ff = None
        self.fDC = None
        self.sigAC = {}
        self.sigDC_tickle = None
        self.mMech = {}
        self.mOpt = {}
        self.noiseAC = {}
        self.poses = None
        self.sigDC_sweep = None
        self.fDC_sweep = None
        self.qq = None

    def getTF(self, probes, drives, dof='pos', optOnly=False):
        """Compute a transfer function

        Inputs:
          probes: name of the probes at which the TF is calculated
          drives: names of the drives from which the TF is calculated
          dof: degree of freedom of the drives (Default: pos)
          optOnly: if True, only return the optical TF with no mechanics
            (Default: False)

        Returns:
          tf: the transfer function
            * In units of [W/m] if drive is an optic with dof pos
            * In units of [W/rad] if drive is an optic with dof pitch or yaw
            * In units of [W/rad] if drive is a PM modulator with dof drive
            * In units of [W/RAM] if drive is an AM modulator with dof drive
            * In units of [W/rad] if drive is an RF modulator with dof phase
            * In units of [W/RAM] if drive is an RF modulator with dof amp
              modulation of an RF modulator.

        Note: To convert [W/RAM] to [W/RIN], divide by 2 since RIN = 2*RAM

        Examples:
          * If only a single drive is used, the drive name can be a string.
            To compute the phase transfer function in reflection from a FP
            cavity
              tf = opt.getTF('REFL', 'PM', 'drive')

          * If multiple drives are used, the drive names should be a dict.
            To compute the DARM transfer function to the AS_DIFF homodyne PD
              DARM = {'EX': 1/2, 'EY': -1/2}
              tf = opt.getTF('AS_DIFF', DARM)
            [Note: Since DARM is defined as Lx - Ly, to get 1 m of DARM
            requires 0.5 m of both Lx and Ly]
        """
        if dof not in ['pos', 'pitch', 'yaw', 'drive', 'amp', 'phase']:
            raise ValueError('Unrecognized degree of freedom {:s}'.format(dof))

        # figure out the shape of the TF
        if isinstance(self.ff, Number):
            # TF is at a single frequency
            tf = 0
        else:
            # TF is for a frequency vector
            tf = np.zeros(len(self.ff), dtype='complex')

        if optOnly:
            tfData = self.mOpt
        else:
            tfData = self.sigAC

        if tfData is None:
            msg = 'Must run tickle for the appropriate DOF before ' \
                  + 'calculating a transfer function.'
            raise RuntimeError(msg)

        # figure out which raw output matrix to use
        if dof in ['pos', 'drive', 'amp', 'phase']:
            tfData = tfData['pos']
        elif dof == 'pitch':
            tfData = tfData['pitch']
        elif dof == 'yaw':
            tfData = tfData['yaw']

        if isinstance(drives, str):
            drives = {drives: 1}

        if isinstance(probes, str):
            probes = {probes: 1}

        # loop through the drives and probes to compute the TF
        for probe, pc in probes.items():
            # get the probe index
            probeNum = self.probes.index(probe)

            for drive, drive_pos in drives.items():
                # get the drive index
                driveNum = self._getDriveIndex(drive, dof)

                # add the contribution from this drive
                try:
                    tf += pc * drive_pos * tfData[probeNum, driveNum]
                except IndexError:
                    tf += pc * drive_pos * tfData[driveNum]

        return tf

    def getMechMod(self, drive_out, drive_in, dof='pos'):
        """Get the radiation pressure modifications to drives

        Inputs:
          drive_out: name of the output drive
          drive_in: name of the input drive
          dof: degree of freedom: pos, pitch, or yaw (Default: pos)
        """
        if dof not in ['pos', 'pitch', 'yaw', 'drive', 'amp', 'phase']:
            raise ValueError('Unrecognized degree of freedom {:s}'.format(dof))

        # figure out which raw output matrix to use
        if dof in ['pos', 'drive', 'amp', 'phase']:
            mMech = self.mMech['pos']
        elif dof == 'pitch':
            mMech = self.mMech['pitch']
        elif dof == 'yaw':
            mMech = self.mMech['yaw']

        if mMech is None:
            msg = 'Must run tickle for the appropriate DOF before ' \
                  + 'calculating a transfer function.'
            raise RuntimeError(msg)

        driveInNum = self._getDriveIndex(drive_in, dof)
        driveOutNum = self._getDriveIndex(drive_out, dof)

        return mMech[driveOutNum, driveInNum]

    def getQuantumNoise(self, probeName, dof='pos'):
        """Compute the quantum noise at a probe

        Returns the quantum noise at a given probe in [W/rtHz]
        """
        probeNum = self.probes.index(probeName)
        try:
            qnoise = self.noiseAC[dof][probeNum, :]
        except IndexError:
            qnoise = self.noiseAC[dof][probeNum]
        return qnoise

    def plotTF(self, probeName, driveNames, mag_ax=None, phase_ax=None,
               dof='pos', optOnly=False, **kwargs):
        """Plot a transfer function.

        See documentation for plotTF in plotting
        """
        ff = self.ff
        tf = self.getTF(probeName, driveNames, dof=dof, optOnly=optOnly)
        fig = plotting.plotTF(
            ff, tf, mag_ax=mag_ax, phase_ax=phase_ax, **kwargs)
        return fig

    def plotQuantumASD(self, probeName, driveNames, fig=None, **kwargs):
        """Plot the quantum ASD of a probe

        Plots the ASD of a probe referenced the the transfer function for
        some signal, such as DARM.

        Inputs:
          probeName: name of the probe
          driveNames: names of the drives from which the TF to refer the
            noise to
          fig: if not None, an existing figure to plot the noise on
            (Default: None)
          **kwargs: extra keyword arguments to pass to the plot
        """
        ff = self.ff
        if driveNames:
            tf = self.getTF(probeName, driveNames)
            noiseASD = np.abs(self.getQuantumNoise(probeName)/tf)
        else:
            noiseASD = np.abs(self.getQuantumNoise(probeName))

        if fig is None:
            newFig = True
            fig = plt.figure()
        else:
            newFig = False

        fig.gca().loglog(ff, noiseASD, **kwargs)
        fig.gca().set_ylabel('Noise')
        fig.gca().set_xlim([min(ff), max(ff)])
        fig.gca().set_xlabel('Frequency [Hz]')
        fig.gca().xaxis.grid(True, which='both', alpha=0.5)
        fig.gca().xaxis.grid(alpha=0.25, which='minor')
        fig.gca().yaxis.grid(True, which='both', alpha=0.5)
        fig.gca().yaxis.grid(alpha=0.25, which='minor')

        if newFig:
            return fig

    def plotSweepSignal(self, probeName, driveName, fig=None, **kwargs):
        """Plot the signal from sweeping drives

        Inputs:
          probeName: name of the probe
          driveName: name of the drives
          fig: if not None, an existing figure to plot the signal on
            (Default: None)
          **kwargs: extra keyword arguments to pass to the plot
        """
        if fig is None:
            newFig = True
            fig = plt.figure()
        else:
            newFig = False
        ax = fig.gca()

        poses, sig = self.getSweepSignal(probeName, driveName)
        ax.plot(poses, sig, **kwargs)
        ax.set_xlim(poses[0], poses[-1])
        ax.grid(True, alpha=0.5)
        if newFig:
            return fig


    def _getDriveIndex(self, name, dof):
        """Find the drive index of a given drive and degree of freedom
        """
        if dof in ['pos', 'pitch', 'yaw']:
            driveNum = self.drives.index(name + '.pos')
        elif dof in ['drive', 'amp', 'phase']:
            driveNum = self.drives.index('{:s}.{:s}'.format(name, dof))
        return driveNum

    def _getSidebandInd(self, freq, lambda0=1064e-9, ftol=1, wltol=1e-10):
        """Find the index of an RF sideband frequency

        Inputs:
          freq: the frequency of the desired sideband
          lambda0: wavelength of desired sideband [m] (Default: 1064 nm)
          ftol: tolerance of the difference between freq and the RF sideband
            of the model [Hz] (Default: 1 Hz)
          wltol: tolerance of the difference between lambda0 and the RF
            sideband wavelength of the model [m] (Default: 100 pm)

        Returns:
          nRF: the index of the RF sideband
        """
        # FIXME: add support for multiple polarizations
        ind = np.nonzero(np.logical_and(
            np.isclose(self.vRF, freq, atol=ftol),
            np.isclose(self.lambda0, lambda0, atol=wltol)))[0]

        if len(ind) == 0:
            msg = 'There are no sidebands with frequency '
            msg += '{:0.0f} {:s}Hz'.format(
                *utils.siPrefix(freq)[::-1])
            raise ValueError(msg)

        elif len(ind) > 1:
            msg = 'There are {:d} sidebands with '.format(len(ind))
            msg += 'frequency {:0.0f} {:s}Hz'.format(
                *utils.siPrefix(freq)[::-1])
            raise ValueError(msg)

        else:
            return int(ind)

    def _dof2opt(self, dof):
        """Convert degrees of freedom to 1s, 2s, and 3s for Optickle
        """
        if dof == 'pos':
            nDOF = 1
        elif dof == 'pitch':
            nDOF = 2
        elif dof == 'yaw':
            nDOF = 3
        else:
            raise ValueError('Unrecognized degree of freedom ' + str(dof)
                             + '. Choose \'pos\', \'pitch\', or \'yaw\'.')

        return nDOF

    def _pol2opt(self, pol):
        """Convert S and P polarizations to 1s and 0s for Optickle
        """
        if pol == 'S':
            nPol = 1
        elif pol == 'P':
            nPol = 0
        else:
            raise ValueError('Unrecognized polarization ' + str(pol)
                             + '. Use \'S\' or \'P\'')
        return nPol
