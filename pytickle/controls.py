from __future__ import division
import numpy as np
from collections import OrderedDict
from utils import assertType


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


class ControlSystem:
    def __init__(self):
        self.opt = None
        self._dofs = OrderedDict()
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
        plant = np.zeros((nProbes, nDrives, nff))

        for pi, probe in enumerate(self._probes):
            for di, drive in enumerate(self.drives):
                tf = self.opt.getTF(probe, drive)
                plant[pi, di, :] = tf

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
        pass

    def addFilter(self, dofTo, dofFrom, z, p, k):
        pass

    def getOLTF(self, dofTo, dofFrom):
        pass

    def getCLTF(self, dofTo, dofFrom):
        pass
