"""
Control system calculations
"""

import numpy as np
from numpy.linalg import inv
import pandas as pd
from collections import OrderedDict
from .utils import (assertType, siPrefix, append_str_if_unique,
                    get_default_kwargs)
from itertools import cycle
import IIRrational.AAA as AAA
import matplotlib.pyplot as plt


def multiplyMat(mat1, mat2):
    """Multiply two matrices
    """
    if np.isscalar(mat1) or np.isscalar(mat2):
        return mat1*mat2

    dim1 = len(mat1.shape)
    dim2 = len(mat2.shape)
    str1 = ''.join(['i', 'j', 'f'][:dim1])
    str2 = ''.join(['j', 'k', 'f'][:dim2])
    str3 = ''.join(['i', 'k', 'f'][:max(dim1, dim2)])
    cmd = '{:s},{:s}->{:s}'.format(str1, str2, str3)
    return np.einsum(cmd, mat1, mat2)


def compute_phase_margin(ff, tf):
    """Compute the UGF and phase margin of an OLTF

    Inputs:
      ff: the frequency vector [Hz]
      tf: the OLTF

    Returns:
      ugf: the UGF [Hz]
      pm: phase margin [deg]
    """
    ind = np.argmin(np.abs(np.abs(tf) - 1))
    ugf = ff[ind]
    pm = np.angle(tf[ind], True)
    return ugf, pm


def compute_stability_margin(ff, tf):
    """Compute the stability margin and frequency of maximum stability

    Computes the stability margin s is defined as the shortest distance to the
    critical point of the OLTF G = 1. If M is the maximum of the loop supresion
    function 1/(1 - G) then s = 1/M.

    Inputs:
      ff: the frequency vector [Hz]
      tf: the loop suppression (or stability function) i.e. 1/(1 - G)
        where

    Returns:
      fms: the frequency at where the loop is most sensitive [Hz]
      sm: the stability margin
    """
    tf = np.abs(tf)
    ind = np.argmax(tf)
    fms = ff[ind]
    sm = 1/np.max(tf)
    return fms, sm


def plotNyquist(oltf, rmax=3):
    """Make a Nyquist plot

    Inputs:
      oltf: the open loop transfer function
      rmax: maximum radius for the plot (Default: 3)

    Returns:
      fig: the figure
    """
    npts = len(oltf)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    rr = np.abs(oltf)
    ph = np.angle(oltf)
    ax.plot(ph, rr, 'C0-')
    ax.plot(-ph, rr, 'C0-')
    ax.set_ylim(0, rmax)
    ax.plot(np.linspace(0, 2*np.pi, npts), np.ones(npts), c='C1')

    # plot the arrows
    # find the index of the halfway point of one branch of the curve
    ind = np.argmin(np.abs(rr - min(np.max(rr), rmax)/2))
    # using ax.arrow sometimes looks funky with polar plots
    ax.annotate(
        '', xy=(ph[ind + 1], rr[ind + 1]), xytext=(ph[ind], rr[ind]),
        arrowprops=dict(fc='C0', ec='C0'))
    ax.annotate(
        '', xy=(-ph[ind], rr[ind]), xytext=(-ph[ind + 1], rr[ind + 1]),
        arrowprops=dict(fc='C0', ec='C0'))

    return fig


class DegreeOfFreedom:
    """A generic degree of freedom

    Inputs:
      drives: the drives defining the DOF (a dict or single string)
      doftype: type of DOF: pos, pitch, or yaw (Default: pos)
      probes: the probes used to sense the DOF (a dict or single string)
      name: name of the DOF

    Example:
      For DARM where L = Lx - Ly and it is sensed by AS_DIFF
        DegreeOfFreedom('DARM', 'AS_DIFF', {'EX': 1, 'EY': -1})
    """

    def __init__(self, drives=None, doftype='pos', probes=None, name=''):
        if drives is None:
            raise ValueError('drives must either be a string or a dictionary')
        self._name = name
        self._doftype = doftype

        if probes is None:
            self._probes = None
        else:
            self._probes = assertType(probes, dict).copy()
        # self._drives = OrderedDict()

        # append the type of dof to the names of the drives
        in_drives = assertType(drives, dict).copy()
        # self._drives = OrderedDict(
        #     {k + '.' + doftype: v for k, v in in_drives.items()})
        self._drives = OrderedDict()
        for drive, cc in in_drives.items():
            drive_data = drive.split('.')
            if len(drive_data) == 1:
                name = drive_data[0] + '.' + doftype
            elif len(drive_data) == 2:
                name = drive
            else:
                raise ValueError('Incorrect drive specificiation')
            self._drives[name] = cc

    @property
    def name(self):
        """DOF name
        """
        return self._name

    @property
    def probes(self):
        """Dictionary of probes used to sense the DOF
        """
        return self._probes

    @property
    def drives(self):
        """Dictionary of drives defining the DOF
        """
        return self._drives

    @property
    def doftype(self):
        """Type of DOF
        """
        return self._doftype

    def dofs(self):
        """Iterator over the drives returned as DegreeOfFreedom instances
        """
        for drive, cc in self.drives.items():
            name, doftype = drive.split('.')
            dof = DegreeOfFreedom(
                drives=name, doftype=doftype, name=name, probes=self.probes)
            yield dof, cc

    def probes2dof(self, probeList):
        """Make a vector of probes
        """
        dofArr = np.zeros(len(probeList))
        for probe, coeff in self.probes.items():
            probeInd = probeList.index(probe)
            dofArr[probeInd] = coeff
        return dofArr

    def dofs2drives(self, driveList):
        """Make a vector of drives
        """
        driveArr = np.zeros(len(driveList))
        for drive, coeff in self.drives.items():
            driveInd = driveList.index(drive)
            driveArr[driveInd] = coeff
        return driveArr


class ControlSystem:
    """A general MIMO control system
    """
    def __init__(self):
        self._plant_model = None
        self._dofs = OrderedDict()
        self._filters = []
        self._probes = []
        self._drives = []
        self._actFilts = []
        self._compFilts = []
        self._plant = None
        self._ctrlMat = None
        self._oltf = None
        self._cltf = None
        self._inMat = None
        self._outMat = None
        self._compMat = None
        self._actMat = None
        self._mMech = None
        self._drive2bsm = None
        self._outComp = None

    @property
    def plant_model(self):
        """The Optickle or Finesse optomechanical plant
        """
        return self._plant_model

    @property
    def ff(self):
        """Frequency vector [Hz]
        """
        return self.plant_model.ff

    @property
    def ss(self):
        """Imaginary frequency vector [rad/s]
        """
        return 2j*np.pi*self.ff

    @property
    def dofs(self):
        """Dictionary of degrees of freedom
        """
        return self._dofs

    @property
    def probes(self):
        """List of probes
        """
        return self._probes

    @property
    def drives(self):
        """List of drives
        """
        return self._drives

    def setOptomechanicalPlant(self, plant_model):
        """Set an Optickle or Finesse model to be the optomechanical plant

        Inputs:
          plant_model: the OpticklePlant or FinessePlant instance
        """
        if self.plant_model is not None:
            raise ValueError('Another model is already set as the plant')
        else:
            self._plant_model = plant_model

    def run(self, drive2bsm=False, mechmod=True):
        """Compute the control system dynamics

        Inputs:
          drive2bsm: Compute the drive to beamspot motion response
            (Default: False)
          mechmod: Compute the radiation pressure mechanical modifications
            (Default: True)

        Note: The spot test point cannot be used if drive2bsm is False
        and the pos test point cannot be used if mechmod is False
        """
        self._inMat = self._computeInputMatrix()
        self._outMat = self._computeOutputMatrix()
        self._plant = self._computePlant()
        self._ctrlMat = self._computeController()
        self._compMat = self._computeCompensator()
        self._actMat = self._computeResponse()
        if mechmod:
            self._mMech = self._computeMechMod()
        self._outComp = np.einsum('ijf,jk->ikf', self._compMat, self._outMat)
        self._oltf = {tp: self._computeOLTF(tp)
                     for tp in ['err', 'ctrl', 'act', 'drive', 'sens']}
        self._cltf = {sig: self._computeCLTF(oltf)
                     for sig, oltf in self._oltf.items()}
        if drive2bsm:
            self._computeBeamSpotMotion()

    def addDOF(self, *args, **kwargs):
        """Add a degree of freedom to the model

        Inputs:
          The DOF can be add in one of two ways
           1) Specifying the name of the DOF, the probes used to sense it, and
              the drives that define it. The optional keyword argument doftype
              defines the doftype (pos, pitch, or yaw) and is 'pos' by default
           2) Giving a DegreOfFreedom instance

        Example:
          For DARM where L = Lx - Ly and it is sensed by AS_DIFF
            cs.addDOF('DARM', 'AS_DIFF', {'EX': 1, 'EY': -1})
          or equivalently
            DARM = DegreOfFreedom(
                name='DARM', probes='AS_DIFF', drives={'EX': 1, 'EY': -1})
            cs.addDOF(DARM)
        """
        if len(args) == 1:
            if isinstance(args[0], DegreeOfFreedom):
                dof = args[0]
                name = dof.name
                if len(name) == 0:
                    raise ValueError(
                        'the degree of freedom must have a name to be added')

                if dof.probes is None:
                    raise ValueError('DOF must have probes defined')

            else:
                raise TypeError(
                    'When adding a DOF with a single argument, ' \
                    + 'it must be a DegreeOfFreedom instance')

        elif len(args) == 3:
            name, probes, drives = args
            doftype = get_default_kwargs(kwargs, doftype='pos')['doftype']
            dof = DegreeOfFreedom(
                name=name, probes=probes, drives=drives, doftype=doftype)

        elif len(args) == 4:
            name, probes, drives, doftype = args
            dof = DegreeOfFreedom(
                name=name, probes=probes, drives=drives, doftype=doftype)

        else:
            msg = 'Incorrect number of arguments. Input can be either\n' \
                + '1) A single DegreeOfFreedom argument\n' \
                + '2) Three arguments representing the name, probes,' \
                + 'and drives with an optional keyword for the doftype'
            raise TypeError(msg)

        if name in self.dofs.keys():
            raise ValueError(
                'Degree of freedom {:s} already exists'.format(name))

        self._dofs[name] = dof
        append_str_if_unique(self._probes, dof.probes)
        append_str_if_unique(self._drives, dof.drives)

    def _computePlant(self):
        """Compute the qlance plant from drives to probes

        Returns:
          plant: The plant a (nProbes, nDrives, nFreq) array
            with the transfer functions from drives to probes
        """
        nProbes = len(self.probes)
        nDrives = len(self.drives)
        nff = len(self.ff)
        plant = np.zeros((nProbes, nDrives, nff), dtype=complex)

        for pi, probe in enumerate(self.probes):
            for di, drive in enumerate(self.drives):
                driveData = drive.split('.')
                driveName = driveData[0]
                doftype = driveData[-1]
                tf = self.plant_model.getTF(probe, driveName, doftype=doftype)
                plant[pi, di, :] = tf
        return plant

    def _computeInputMatrix(self):
        """Compute the input matrix from probes to DOFs

        Returns:
          sensMat: a (nDOF, nProbes) array
        """
        nProbes = len(self.probes)
        nDOF = len(self.dofs)
        sensMat = np.zeros((nDOF, nProbes))
        for di, dof in enumerate(self.dofs.values()):
            sensMat[di, :] = dof.probes2dof(self.probes)
        return sensMat

    def _computeOutputMatrix(self):
        """Compute the actuation matrix from DOFs to drives

        Returns:
          actMat: a (nDrives, nDOF) array
        """
        nDOF = len(self.dofs)
        nDrives = len(self.drives)
        actMat = np.zeros((nDrives, nDOF))
        for di, dof in enumerate(self.dofs.values()):
            actMat[:, di] = dof.dofs2drives(self.drives)
        return actMat

    def _computeController(self):
        """Compute the control matrix from DOFs to DOFs

        Returns:
          ctrlMat: a (nDOF, nDOF, nff) array
        """
        if self.ss is None:
            raise RuntimeError('There is no associated qlance model')
        nDOF = len(self.dofs)
        ctrlMat = np.zeros((nDOF, nDOF, len(self.ss)), dtype=complex)
        for (dof_to, dof_from, filt) in self._filters:
            toInd = list(self.dofs.keys()).index(dof_to)
            fromInd = list(self.dofs.keys()).index(dof_from)
            ctrlMat[toInd, fromInd, :] = filt(self.ff)
        return ctrlMat

    def _computeCompensator(self):
        nff = len(self.ss)
        ndrives = len(self.drives)
        ones = np.ones(nff)
        compMat = np.zeros((ndrives, ndrives, nff), dtype=complex)
        compdrives = [cf[0] for cf in self._compFilts]
        for di, drive in enumerate(self.drives):
            try:
                ind = compdrives.index(drive)
                compMat[di, di, :] = self._compFilts[ind][-1](self.ff)
            except ValueError:
                compMat[di, di, :] = ones
        return compMat

    def _computeResponse(self):
        nff = len(self.ss)
        ndrives = len(self.drives)
        ones = np.ones(nff)
        actMat = np.zeros((ndrives, ndrives, nff), dtype=complex)
        actdrives = [rf[0] for rf in self._actFilts]
        for di, drive in enumerate(self.drives):
            try:
                ind = actdrives.index(drive)
                actMat[di, di, :] = self._actFilts[ind][-1](self.ff)
            except ValueError:
                actMat[di, di, :] = ones
        return actMat

    def _computeMechMod(self):
        nDrives = len(self.drives)
        nff = len(self.ff)
        mMech = np.zeros((nDrives, nDrives, nff), dtype=complex)
        for dit, driveTo in enumerate(self.drives):
            driveData = driveTo.split('.')
            driveToName = driveData[0]
            dofToType = driveData[-1]
            for djf, driveFrom in enumerate(self.drives):
                driveData = driveFrom.split('.')
                driveFromName = driveData[0]
                dofFromType = driveData[-1]
                if dofFromType != dofToType:
                    msg = 'Input and output drives should be the same ' \
                          + 'degree of freedom (pos, pitch, or yaw)'
                    raise ValueError(msg)
                mMech[dit, djf, :] = self.plant_model.getMechMod(
                    driveToName, driveFromName, dofToType)
        return mMech

    def _computeBeamSpotMotion(self):
        nDrives = len(self.drives)
        nff = len(self.ff)
        drive2bsm = np.zeros((nDrives, nDrives, nff), dtype=complex)
        for si, spot_drive in enumerate(self.drives):
            opticName = spot_drive.split('.')[0]
            for di, drive in enumerate(self.drives):
                driveData = drive.split('.')
                driveName = driveData[0]
                doftype = driveData[-1]
                # FIXME: make work with finesse and non-front surfaces
                try:
                    drive2bsm[si, di] = self.plant_model.computeBeamSpotMotion(
                        opticName, 'fr', driveName, doftype)
                except KeyError:
                    drive2bsm[si, di] = self.plant_model.computeBeamSpotMotion(
                        opticName, 'frA', driveName, doftype)

        self._drive2bsm = drive2bsm

    def _computeOLTF(self, tstpnt):
        """Compute the OLTF

        Inputs:
          tstpnt: the test point
        """
        if tstpnt == 'err':
            oltf = self.getTF('', 'err', '', 'ctrl', closed=False)
            oltf = multiplyMat(oltf, self._ctrlMat)

        elif tstpnt == 'sens':
            oltf = self.getTF('', 'sens', '', 'err', closed=False)
            oltf = multiplyMat(oltf, self._inMat)

        elif tstpnt == 'drive':
            oltf = self.getTF('', 'drive', '', 'sens', closed=False)
            oltf = multiplyMat(oltf, self._plant)

        elif tstpnt == 'act':
            oltf = self.getTF('', 'act', '', 'drive', closed=False)
            oltf = multiplyMat(oltf, self._actMat)

        elif tstpnt == 'ctrl':
            oltf = self.getTF('', 'ctrl', '', 'act', closed=False)
            oltf = multiplyMat(oltf, self._outComp)

        return oltf

    def _computeCLTF(self, oltf):
        """Compute the CLTF from an OLTF
        """
        # transpose so that numpy can invert without a loop
        oltf = np.einsum('ijf->fij', oltf)

        # make the 3D identity matrix
        ident = np.zeros_like(oltf)
        inds = np.arange(oltf.shape[-1])
        ident[:, inds, inds] = 1

        # invert and transpose back
        cltf = np.einsum('fij->ijf', inv(ident - oltf))

        return cltf

    def addFilter(self, dof_to, dof_from, filt):
        """Add a filter between two DOFs to the controller

        Inputs:
          dof_to: output DOF
          dof_from: input DOF
          filt: Filter instance for the filter
        """
        dofs_to = np.array([filt[0] for filt in self._filters])
        dofs_from = np.array([filt[1] for filt in self._filters])
        inds = [ind for ind, (d1, d2) in enumerate(zip(dofs_to, dofs_from))
                if d1 == dof_to and d2 == dof_from]

        if len(inds) > 0:
            raise ValueError(
                'There is already a filter from {:s} to {:s}'.format(
                    dof_from, dof_to))

        self._filters.append((dof_to, dof_from, filt))

    def addCompensator(self, drive, doftype, filt):
        """Add a compensation filter

        Inputs:
          drive: which drive to compensate
          doftype: type of drive (pos, pitch, or yaw)
          filt: Filter instance for the filter
        """
        drive += '.' + doftype
        if drive in [cf[0] for cf in self._compFilts]:
            raise ValueError(
                'A compensator is already set for drive {:s}'.format(drive))

        self._compFilts.append((drive, filt))

    def setActuator(self, drive, doftype, filt):
        """Set the actuation plant for a drive

        Inputs:
          drive: which drive to actuate
          doftype: type of drive (pos, pitch, or yaw)
          filt: Filter instance for the plant
        """
        drive += '.' + doftype
        if drive in [rf[0] for rf in self._actFilts]:
            raise ValueError(
                'A response is already set for drive {:s}'.format(drive))

        self._actFilts.append((drive, filt))

    def getFilter(self, dof_to, dof_from):
        """Get the filter between two DOFs

        Inputs:
          dof_to: output DOF
          dof_from: input DOF

        Returns:
          filt: the filter
        """
        dofs_to = np.array([filt[0] for filt in self._filters])
        dofs_from = np.array([filt[1] for filt in self._filters])
        inds = [ind for ind, (d1, d2) in enumerate(zip(dofs_to, dofs_from))
                if d1 == dof_to and d2 == dof_from]
        nfilts = len(inds)

        if nfilts == 0:
            raise ValueError('There is no filter from {:s} to {:s}'.format(
                dof_from, dof_to))
        elif nfilts > 1:
            raise ValueError('There are multiple filters')
        else:
            return self._filters[inds[0]][-1]

    def getActuator(self, drive):
        """Get the actuation filter for a drive

        Inputs:
          drive: the drive

        Returns:
          filt: the filter
        """
        drives = [rf[0] for rf in self._actFilts]
        try:
            ind = drives.index(drive)
            return self._actFilts[ind][-1]
        except ValueError:
            raise ValueError('No actuator is set for ' + drive)

    def getCompensator(self, drive):
        """Get the compensation filter for a drive

        Inputs:
          drive: the drive

        Returns:
          filt: the filter
        """
        drives = [rf[0] for rf in self._actFilts]
        try:
            ind = drives.index(drive)
            return self._actFilts[ind][-1]
        except ValueError:
            raise ValueError(drive + ' does not have a compensation filter')

    def getOLTF(self, sig_to, sig_from, tstpnt, fit=False):
        """Compute an OLTF

        Inputs:
          sig_to: output signal
          sig_from: input signal
          tstpnt: which test point to compute the TF for
          fit: if True, returns a FitTF fit to the transfer function
            (Default: False)
        """
        oltf = self._oltf[tstpnt]
        if sig_from:
            from_ind = self._getIndex(sig_from, tstpnt)
            oltf = oltf[:, from_ind]
        if sig_to:
            to_ind = self._getIndex(sig_to, tstpnt)
            oltf = oltf[to_ind]

        if fit:
            return FitTF(self.ff, oltf)
        else:
            return oltf

    def getCLTF(self, sig_to, sig_from, tstpnt, fit=False):
        """Compute a CLTF

        Inputs:
          sig_to: output signal
          sig_from: input signal
          tstpnt: which test point to compute the TF for
          fit: if True, returns a FitTF fit to the transfer function
            (Default: False)
        """
        cltf = self._cltf[tstpnt]
        if sig_from:
            from_ind = self._getIndex(sig_from, tstpnt)
            cltf = cltf[:, from_ind]
        if sig_to:
            to_ind = self._getIndex(sig_to, tstpnt)
            cltf = cltf[to_ind]

        if fit:
            return FitTF(self.ff, cltf)
        else:
            return cltf

    def compute_margins(self, sig_to, sig_from, tstpnt):
        """Compute stability margins for a loop

        Inputs:
          sig_to: output signal
          sig_from: input signal
          tstpnt: test point

        Returns:
          ugf: UGF of the loop [Hz]
          pm: phase margin [deg]
          fms: frequency of maximum sensitivity [Hz]
          sm: stability margin
        """
        oltf = self.getOLTF(sig_to, sig_from, tstpnt)
        cltf = self.getCLTF(sig_to, sig_from, tstpnt)
        ugf, pm = compute_phase_margin(self.ff, oltf)
        fms, sm = compute_stability_margin(self.ff, cltf)
        return ugf, pm, fms, sm

    def check_stability(self, tstpnt):
        """Check the stability of all of the loops

        Inputs:
          tstpnt: test point
        """
        stability = OrderedDict()
        for dof in self.dofs.keys():
            cltf = self.getCLTF(dof, dof, tstpnt)
            fit = AAA.tfAAA(self.ff, cltf)
            z, p, k = fit.zpk
            stability[dof] = np.all(np.real(p) < 0)

        for dof, stab in stability.items():
            print(dof, stab)

    def print_margins(self, tstpnt):
        """Print the stability margins for all loops

        Input:
          tstpnt: test point
        """
        margins = {}
        for dof in self.dofs.keys():
            ugf, pm, fms, sm = self.compute_margins(dof, dof, tstpnt)
            margins[dof] = ['{:0.2f} {:s}Hz'.format(*siPrefix(ugf)),
                            '{:0.2f} deg'.format(pm),
                            '{:0.2f} {:s}Hz'.format(*siPrefix(fms)),
                            '{:0.2f}'.format(sm)]

        margins = pd.DataFrame(
            margins,
            index=['UGF', 'phase margin', 'max sensitivity', 'stability margin']).T
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None):
            display(margins)

    def getTF(self, sig_to, tp_to, sig_from, tp_from, closed=True, fit=False):
        """Get a transfer function between two test points in a loop

        If sig_to is the empty string '', the vector to all signals is returned
        If sig_from is the empty string '', the vector of all signals is returned

        Inputs:
          sig_to: output signal
          sig_from: input signal
          tp_to: output test point
          tp_from: input test point
          fit: if True, returns a FitTF fit to the transfer function
            (Default: False)
        """
        def cltf_or_unity(tp_to):
            """Returns the CLTF if a closed loop TF is necessary and the identity
            matrix if an open loop TF is necessary
            """
            if closed:
                return self._cltf[tp_to]
            else:
                return 1

        # test points and their matrices in cyclic order for computing TFs
        # around the main loop
        tstpnts = ['err', 'sens', 'drive', 'act', 'ctrl']
        mats = [self._inMat, self._plant, self._actMat, self._outComp,
                self._ctrlMat]

        # Main loop if the input test point is not a calibration test point
        if tp_from in tstpnts:

            # If the test points are the same, this is just a CLTF
            if tp_to == tp_from:
                tf = cltf_or_unity(tp_to)

            # Main loop if the output test point is not pos or beam spot motion
            elif tp_to in tstpnts:
                start = False
                tf = cltf_or_unity(tp_to)
                for tstpnt, mat in zip(cycle(tstpnts), cycle(mats)):
                    if tstpnt != tp_to and not start:
                        continue
                    start = True
                    if tstpnt == tp_from:
                        break
                    tf = multiplyMat(tf, mat)

            # If the output test point is pos or beam spot motion, first compute
            # the TF to drive and then prepend the corrections
            elif tp_to in ['pos', 'spot']:
                # Get the loop transfer function to drives
                if tp_from == 'drive':
                    loopTF = cltf_or_unity('drive')
                else:
                    loopTF = self.getTF(
                        '', 'drive', '', tp_from, closed=closed)

                # for pos, apply mechanical modification
                if tp_to == 'pos':
                    tf = multiplyMat(self._mMech, loopTF)

                # for beam spot motion, apply conversion from drives to motion
                elif tp_to == 'spot':
                    tf = multiplyMat(self._drive2bsm, loopTF)

            # No valid output test point
            else:
                raise ValueError('Unrecognized test point to ' + tp_to)

        # If the input test point is a calibration test point, first compute
        # the TF from the corresponding drive and append the output matrix
        elif tp_from == 'cal':
            loopTF = self.getTF('', tp_to, '', 'drive', closed=closed)
            tf = multiplyMat(loopTF, self._outMat)

        # No valid input test point
        else:
            raise ValueError('Unrecognized test point from ' + tp_from)

        # Reduce size of returned TF if necessary
        if sig_from:
            from_ind = self._getIndex(sig_from, tp_from)
            tf = tf[:, from_ind]

        if sig_to:
            to_ind = self._getIndex(sig_to, tp_to)
            tf = tf[to_ind]

        # Return the TF fitting it if necessary
        if fit:
            if not (sig_to and sig_from):
                raise ValueError(
                    'Fits can only be returned for single transfer functions')
            else:
                return FitTF(self.ff, tf)
        else:
            return tf

    def getTotalNoiseTo(self, sig_to, tp_to, tp_from, noiseASDs, closed=True):
        """Compute the total noise at a test point due to multiple sources

        Inputs:
          sig_to: the output signal
          tp_to: the output test point
          tp_from: the input test point
          noiseASDs: a dictionary of noise ASDs with keys the name of the
            signal from and values the ASDs [u/rtHz]

        Returns:
          totalASD: the total ASD [u/rtHz]
        """
        if tp_to == 'cal':
            ampTF = inv(self._outMat)
            to_ind = self._getIndex(sig_to, 'cal')
            ampTF = ampTF[to_ind]
        else:
            ampTF = self.getTF(sig_to, tp_to, '', tp_from, closed=closed)
        if len(ampTF.shape) == 1:
            ampTF = np.einsum('i,j->ij', ampTF, np.ones_like(self.ff))
        powTF = np.abs(ampTF)**2
        totalPSD = np.zeros(powTF.shape[-1])
        for dof_from, noiseASD in noiseASDs.items():
            from_ind = self._getIndex(dof_from, tp_from)
            totalPSD += powTF[from_ind] * noiseASD**2
        return np.sqrt(totalPSD)

    # FIXME:
    # def getTotalNoiseFrom(self, sig_to, sig_from, tp_from, noiseASDs):
    #     ampTF = self.getTF('', sig_to, sig_from, tp_from)
    #     powTF = np.abs(ampTF)**2
    #     totalPSD = np.zeros(powTF.shape[-1])
    #     for dof_to, noiseASD in noiseASDs.items():
    #         to_ind = self._getIndex(dof_to, sig_to)
    #         totalPSD += powTF[to_ind] * noiseASD**2
    #     return np.sqrt(totalPSD)

    def plotNyquist(self, sig_to, sig_from, tstpnt, rmax=3):
        """Make a Nyquist plot

        Inputs:
          sig_to: output signal
          sig_from: input signal
          tstpnt: which test point to compute the TF for
          rmax: maximum radius for the plot (Default: 3)
        """
        oltf = self.getOLTF(sig_to, sig_from, tstpnt)
        return plotNyquist(oltf, rmax=rmax)

    def _getIndex(self, name_or_dof, tstpnt):
        """Get the index of a signal

        Inputs:
          name: name of the signal
          tstpnt: type of test point
        """
        if tstpnt in ['err', 'ctrl', 'cal']:
           sig_list = list(self.dofs.keys())
        elif tstpnt in ['act', 'drive', 'pos', 'spot']:
            sig_list = self.drives
        elif tstpnt == 'sens':
            sig_list = self.probes
        else:
            raise ValueError('Unrecognized test point ' + tstpnt)

        if isinstance(name_or_dof, DegreeOfFreedom):
            if tstpnt in ['act', 'drive', 'pos', 'spot']:
                # the DOF should be a single drive
                # take the drive instead of the name so that it has the
                # doftype appended
                drives_keys = list(name_or_dof.drives.keys())
                if len(drives_keys) == 1:
                    name = drives_keys[0]
                else:
                    raise ValueError(
                        tstpnt + ' test point should only have one drive')

            elif tstpnt == 'sens':
                # probes are never DOFs
                raise ValueError('sens test point cannot be a DegreeOfFreedom')

            elif tstpnt in ['err', 'ctrl', 'cal']:
                # the DOF should be a control system DOF
                name = name_or_dof.name

        else:
            name = name_or_dof

        try:
            ind = sig_list.index(name)
        except ValueError:
            raise ValueError(
                '{:s} is not a {:s} test point'.format(name, tstpnt))

        return ind
