from __future__ import division
import numpy as np
from scipy.linalg import inv
from collections import OrderedDict
from utils import assertType
from functools import partial
from numbers import Number
import plotting


def append_str_if_unique(array, elements):
    """Append elements to an array only if that element is unique

    Inputs:
      array: the array to which the elements should be appended
      elements: the elements to append to the array
    """
    if isinstance(elements, str):
        elements = [elements]
    elif isinstance(elements, dict) or isinstance(elements, OrderedDict):
        elements = elements.keys()

    for element in elements:
        if element not in array:
            array.append(element)


def zpk(zs, ps, k, ff):
    """Return the function specified by zeros, poles, and a gain

    Inputs:
      zs: the zeros
      ps: the poles
      k: the gain
      ff: the frequencies at which to evaluate the function [Hz]

    Returns:
      filt: the function
    """
    def assertArr(arr):
        """Ensure that the input is an array
        """
        if isinstance(arr, Number):
            return [arr]
        elif isinstance(arr, list) or isinstance(arr, np.ndarray):
            return arr
        else:
            raise ValueError('Unrecognized data dtype')

    if not isinstance(k, Number):
        raise ValueError('The gain should be a scalar')

    if isinstance(ff, Number):
        filt = k
    else:
        filt = k * np.ones(len(ff), dtype=complex)

    for z in assertArr(zs):
        filt *= (ff - z)

    for p in assertArr(ps):
        filt /= (ff - p)

    return filt


class DegreeOfFreedom:
    def __init__(self, name, probes, drives):
        self.name = name
        self.probes = assertType(probes, dict)
        self.drives = assertType(drives, dict)
        self.inDOFs = []
        self.outDOFs = []

    def probes2dof(self, probeList):
        """Make a vector of probes
        """
        dofArr = np.zeros(len(probeList))
        for probe, coeff in self.probes.items():
            probeInd = probeList.index(probe)
            dofArr[probeInd] = coeff
        return dofArr

    def dof2drives(self, driveList):
        """Make a vector of drives
        """
        driveArr = np.zeros(len(driveList))
        for drive, coeff in self.drives.items():
            driveInd = driveList.index(drive)
            driveArr[driveInd] = coeff
        return driveArr


class Filter:
    """A class representing a generic filter

    Inputs:
      The filter can be specified in one of two ways:
        1) Giving a callable function
        2) Giving the zeros, poles, and gain

    Attributes:
      filt: the filter function
    """
    def __init__(self, *args):
        if len(args) == 1:
            self.filt = args[0]
            if not callable(self.filt):
                raise ValueError('One argument filters should be functions')
        elif len(args) == 3:
            zs = args[0]
            ps = args[1]
            k = args[2]
            self.filt = partial(zpk, zs, ps, k)
        else:
            msg = 'Incorrect number of arguments. Input can be either\n'
            msg += '1) A single argument which is the filter function\n'
            msg += '2) Three arguments representing the zeros, poles, and gain'
            raise ValueError(msg)

    def plotFilter(self, ff, mag_ax=None, phase_ax=None, dB=False, **kwargs):
        """Plot the filter

        See documentation for plotting.plotTF
        """
        ss = 2j*np.pi*ff
        fig = plotting.plotTF(
            ff, self.filt(ss), mag_ax=mag_ax, phase_ax=phase_ax, dB=dB,
            **kwargs)
        return fig


class ControlSystem:
    def __init__(self):
        self.opt = None
        self._ss = None
        self._dofs = OrderedDict()
        self._filters = []
        self._probes = []
        self._drives = []
        self._opticalPlant = None
        self._ctrlMat = None
        self._oltf = None
        self._cltf = None
        self._inMat = None
        self._outMat = None

    def setPyTicklePlant(self, opt):
        """Set a PyTickle model to be the plant

        Inputs:
          opt: the PyTickle instance to be used as the plant
        """
        if self.opt is not None:
            raise ValueError('A PyTickle model is already set as the plant')
        else:
            self.opt = opt
            self._ss = 2j*np.pi*opt._ff

    def tickle(self):
        self._inMat = self._computeInputMatrix()
        self._outMat = self._computeOutputMatrix()
        self._plant = self._computePlant()
        self._ctrlMat = self._computeController()
        self._oltf = {sig: self._computeOLTF(sig) for sig in ['err', 'ctrl']}
        self._cltf = {sig: self._computeCLTF(oltf)
                      for sig, oltf in self._oltf.items()}

    def addDOF(self, name, probes, drives):
        """Add a degree of freedom to the model

        Inputs:
          name: name of the DOF
          probes: the probes used to sense the DOF
          drives: the drives used to define the DOF

        Example:
          For DARM where L = Lx - Ly and it is sensed by AS_DIFF
          cs.addDOF('DARM', 'AS_DIFF', {'EX.pos': 1, 'EY.pos': -1})
        """
        if name in self._dofs.keys():
            raise ValueError(
                'Degree of freedom {:s} already exists'.format(name))

        self._dofs[name] = DegreeOfFreedom(name, probes, drives)
        append_str_if_unique(self._probes, probes)
        append_str_if_unique(self._drives, drives)

    def _computePlant(self):
        """Compute the PyTickle plant from drives to probes

        Returns:
          plant: The plant a (nProbes, nDrives, nFreq) array
            with the transfer functions from drives to probes
        """
        nProbes = len(self._probes)
        nDrives = len(self._drives)
        nff = len(self.opt._ff)
        plant = np.zeros((nProbes, nDrives, nff), dtype=complex)

        for pi, probe in enumerate(self._probes):
            for di, drive in enumerate(self._drives):
                tf = self.opt.getTF(probe, drive)
                plant[pi, di, :] = tf
        return plant

    def _computeInputMatrix(self):
        """Compute the sensing matrix from probes to DOFs

        Returns:
          sensMat: a (nDOF, nProbes) array
        """
        nProbes = len(self._probes)
        nDOF = len(self._dofs)
        sensMat = np.zeros((nDOF, nProbes))
        for di, dof in enumerate(self._dofs.values()):
            sensMat[di, :] = dof.probes2dof(self._probes)
        return sensMat

    def _computeOutputMatrix(self):
        """Compute the actuation matrix from DOFs to drives

        Returns:
          actMat: a (nDrives, nDOF) array
        """
        nDOF = len(self._dofs)
        nDrives = len(self._drives)
        actMat = np.zeros((nDrives, nDOF))
        for di, dof in enumerate(self._dofs.values()):
            actMat[:, di] = dof.dof2drives(self._drives)
        return actMat

    def _computeController(self):
        """Compute the control matrix from DOFs to DOFs

        Returns:
          ctrlMat: a (nDOF, nDOF, nff) array
        """
        if self._ss is None:
            raise RuntimeError('There is no associated PyTickle model')
        nDOF = len(self._dofs)
        ctrlMat = np.zeros((nDOF, nDOF, len(self._ss)), dtype=complex)
        for (dofTo, dofFrom, filt) in self._filters:
            toInd = self._dofs.keys().index(dofTo)
            fromInd = self._dofs.keys().index(dofFrom)
            ctrlMat[toInd, fromInd, :] = filt.filt(self._ss)
        return ctrlMat

    def _computeOLTF(self, sig='err'):
        """Compute the OLTF from DOFs to DOFs

        Inputs:
          sig: which signal to compute the TF for:
            1) 'err': error signal (Default)
            2) 'ctrl': control signal
        """
        if sig not in ['err', 'ctrl']:
            raise ValueError('The signal must be either err or ctrl')

        if sig == 'err':
            oltf = np.einsum(
                'ij,jkf,kl,lmf->imf', self._inMat, self._plant, self._outMat,
                self._ctrlMat)

        elif sig == 'ctrl':
            oltf = np.einsum(
                'ijf,jk,klf,lm->imf', self._ctrlMat, self._inMat, self._plant,
                self._outMat)

        return oltf

    def _computeCLTF(self, oltf):
        """Compute the CLTF from DOFs to DOFs

        Inputs:
          sig: which signal to compute the TF for:
            1) 'err': error signal (Default)
            2) 'ctrl': control signal
        """
        cltf = np.zeros_like(oltf)
        nDOF = cltf.shape[0]
        for fi in range(cltf.shape[-1]):
            cltf[:, :, fi] = inv(np.identity(nDOF) - oltf[:, :, fi])
        return cltf

    def addFilter(self, dofTo, dofFrom, *args):
        """Add a filter between two DOFs to the controller

        Inputs:
          dofTo: output DOF
          dofFrom: input DOF
          *args: arguments used to define a Filter instance
        """
        self._filters.append((dofTo, dofFrom, Filter(*args)))

    def getFilter(self, dofTo, dofFrom):
        """Get the filter between two DOFs

        Inputs:
          dofTo: output DOF
          dofFrom: input DOF

        Returns:
          filt: the filter
        """
        dofsTo = np.array([filt[0] for filt in self._filters])
        dofsFrom = np.array([filt[1] for filt in self._filters])
        inds = np.logical_and(dofTo == dofsTo, dofFrom == dofsFrom)
        nfilts = np.count_nonzero(inds)
        if nfilts == 0:
            raise ValueError('There is no filter from {:s} to {:s}'.format(
                dofFrom, dofTo))
        elif nfilts > 1:
            raise ValueError('There are multiple filters')
        else:
            return self._filters[inds.nonzero()[0][0]][-1]

    def plotFilter(self, dofTo, dofFrom, mag_ax=None, phase_ax=None, dB=False,
                   **kwargs):
        """Plot a filter function

        See documentation for plotFilter in Filter class
        """
        filt = self.getFilter(dofTo, dofFrom)
        return filt.plotFilter(self.opt._ff, mag_ax, phase_ax, dB, **kwargs)

    def getOLTF(self, dofTo, dofFrom, sig='err'):
        """Compute the OLTF from DOFs to DOFs

        Inputs:
          dofTo: output DOF
          dofFrom: input DOF
          sig: which signal to compute the TF for:
            1) 'err': error signal (Default)
            2) 'ctrl': control signal
        """
        if sig not in ['err', 'ctrl']:
            raise ValueError('The signal must be either err or ctrl')

        oltf = self._oltf[sig]
        dofToInd = self._dofs.keys().index(dofTo)
        dofFromInd = self._dofs.keys().index(dofFrom)
        return oltf[dofToInd, dofFromInd, :]

    def getCLTF(self, dofTo, dofFrom, sig='err'):
        """Compute the CLTF from DOFs to DOFs

        Inputs:
          dofTo: output DOF
          dofFrom: input DOF
          sig: which signal to compute the TF for:
            1) 'err': error signal (Default)
            2) 'ctrl': control signal
        """
        if sig not in ['err', 'ctrl']:
            raise ValueError('The signal must be either err or ctrl')

        cltf = self._cltf[sig]
        dofToInd = self._dofs.keys().index(dofTo)
        dofFromInd = self._dofs.keys().index(dofFrom)
        return cltf[dofToInd, dofFromInd, :]

    def getCalibration(self, dofTo, dofFrom, sig='err'):
        """Compute the CLTF between two DOFs to use for calibration

        Computes the TF necessary to signal-refer noise to. For example, noise
        is usually plotted DARM referred in which case the raw noise should be
        divided by the CLTF from the DARM drives to the DARM probes.

        Inputs:
          dofTo: output DOF
          dofFrom: input DOF
          sig: which signal to compute the TF for:
            1) 'err': error signal (Default)
            2) 'ctrl': control signal

        Returns:
          tf: the transfer function
        """
        cltf = self._cltf[sig]

        if sig == 'err':
            tf = np.einsum(
                'ijf,jk,klf,lm->imf', cltf, self._inMat, self._plant,
                self._outMat)

        elif sig == 'ctrl':
            tf = np.einsum(
                'ijf,jkf,kl,lmf,mn->inf', cltf, self._ctrlMat, self._inMat,
                self._plant, self._outMat)

        dofToInd = self._dofs.keys().index(dofTo)
        dofFromInd = self._dofs.keys().index(dofFrom)
        return tf[dofToInd, dofFromInd, :]

    def plotOLTF(self, dofTo, dofFrom, sig='err', mag_ax=None, phase_ax=None,
                 dB=False, **kwargs):
        """Plot an OLTF

        See documentation for getOLTF and plotting.plotTF
        """
        oltf = self.getOLTF(dofTo, dofFrom, sig=sig)
        fig = plotting.plotTF(
            self.opt._ff, oltf, mag_ax=mag_ax, phase_ax=phase_ax, dB=dB,
            **kwargs)
        return fig

    def getSensingNoise(self, dof, probe, asd=None, sig='err'):
        """Compute the sensing noise from a probe to a DOF

        Inputs:
          dof: DOF name
          probe: probe name
          asd: Noise ASD to use. If None (Default) noise is calculated
            from the Optickle model
          sig: which signal to compute the TF for:
            1) 'err': error signal (Default)
            2) 'ctrl': control signal

        Returns:
          sensNoise: the sensing noise
            [W/rtHz] if sig='err'
            [m/rtHz] if sig='ctrl'
        """
        if sig not in ['err', 'ctrl']:
            raise ValueError('The signal must be either err or ctrl')

        cltf = self._cltf[sig]
        if sig == 'err':
            noiseTF = np.einsum('ijf,jk->ikf', cltf, self._inMat)
        elif sig == 'ctrl':
            noiseTF = np.einsum('ijf,jkf,kl->ilf', cltf, self._ctrlMat,
                                self._inMat)

        if asd is None:
            qnoise = self.opt.getQuantumNoise(probe)
        else:
            qnoise = asd

        dofInd = self._dofs.keys().index(dof)
        probeInd = self._probes.index(probe)
        sensNoise = np.abs(noiseTF[dofInd, probeInd, :]) * qnoise
        return sensNoise

    def getDisplacementNoise(self, dof, drive, asd, sig='err'):
        pass

    def getForceNoise(self, dof, drive, asd, sig='err'):
        pass

    def getSensingFunction(self, dofTo, dofFrom):
        """Computes the sensing function from DOFs to DOFs

        Computes the sensing function between two degrees of freedom.
        This includes the optomechanical plant as well as the actuation
        and sensing matrices.

        Inputs:
          dofTo: output DOF
          dofFrom: input DOF

        Returns:
          sensFunc: the sensing function [W/m]
        """
        sensFunc = np.einsum(
            'ij,jkf,kl->ilf', self._inMat, self._plant, self._outMat)
        dofTo = self._dofs.keys().index(dofTo)
        dofFrom = self._dofs.keys().index(dofFrom)
        return sensFunc[dofTo, dofFrom, :]

    def plotCLTF(self, dofTo, dofFrom, sig='err', mag_ax=None, phase_ax=None,
                 dB=False, **kwargs):
        """Plot a CLTF

        See documentation for getCLTF and plotting.plotTF
        """
        cltf = self.getCLTF(dofTo, dofFrom, sig=sig)
        fig = plotting.plotTF(
            self.opt._ff, cltf, mag_ax=mag_ax, phase_ax=phase_ax, dB=dB,
            **kwargs)
        return fig
