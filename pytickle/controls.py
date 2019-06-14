from __future__ import division
import numpy as np
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
    def assertArr(arr):
        if isinstance(arr, Number):
            return [arr]
        elif isinstance(arr, list) or isinstance(arr, np.ndarray):
            return arr
        else:
            raise ValueError('Unrecognized data dtype')

    if not isinstance(k, Number):
        raise ValueError('The gain should be a scalar')

    filt = k * np.ones(len(ff), dtype=complex)
    for z in assertArr(zs):
        filt *= (ff - z)
    for p in assertArr(ps):
        filt /= (ff - p)
    return filt


def zpk2func(zs, ps, k):
    return partial(zpk, zs, ps, k)


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

        See documentation for plotTF in plotting
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
        self._dofNames = []
        self._driveNames = []
        self._probes = []
        self._drives = []

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

    def computePlant(self):
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

    def computeSensingMatrix(self):
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

    def computeActuationMatrix(self):
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

    def computeController(self):
        if self._ss is None:
            raise RuntimeError('There is no associated PyTickle model')
        nDOF = len(self._dofs)
        ctrlMat = np.zeros((nDOF, nDOF, len(self._ss)), dtype=complex)
        for (dofTo, dofFrom, filt) in self._filters:
            toInd = self._dofs.keys().index(dofTo)
            fromInd = self._dofs.keys().index(dofFrom)
            ctrlMat[toInd, fromInd, :] = filt.filt(self._ss)
        return ctrlMat

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

    def getOLTF(self, dofTo, dofFrom):
        pass

    def getCLTF(self, dofTo, dofFrom):
        pass
