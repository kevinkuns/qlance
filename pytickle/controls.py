import numpy as np
from scipy.linalg import inv
import pandas as pd
from collections import OrderedDict
from .utils import assertType, siPrefix, append_str_if_unique
from functools import partial
from numbers import Number
from itertools import cycle, zip_longest
import matplotlib.pyplot as plt
from . import plotting


def assertArr(arr):
    """Ensure that the input is an array
    """
    if isinstance(arr, Number):
        return [arr]
    elif isinstance(arr, list) or isinstance(arr, np.ndarray):
        return arr
    elif isinstance(arr, tuple):
        return list(arr)
    else:
        raise ValueError('Unrecognized data dtype')


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


def zpk(zs, ps, k, ss):
    """Return the function specified by zeros, poles, and a gain

    Inputs:
      zs: the zeros
      ps: the poles
      k: the gain
      ss: the frequencies at which to evaluate the function [rad/s]

    Returns:
      filt: the function
    """
    if not isinstance(k, Number):
        raise ValueError('The gain should be a scalar')

    if isinstance(ss, Number):
        filt = k
    else:
        filt = k * np.ones(len(ss), dtype=complex)

    # for z in assertArr(zs):
    #     filt *= (ss - z)

    # for p in assertArr(ps):
    #     filt /= (ss - p)

    # Do this with pole/zero pairs instead of all the zeros and then all the
    # poles to avoid numerical issues when dividing huge numerators by huge
    # denominators for filters with many poles and zeros
    for z, p in zip_longest(assertArr(zs), assertArr(ps)):
        if z is not None:
            filt *= (ss - z)

        if p is not None:
            filt /= (ss - p)

    return filt


def resRoots(f0, Q, Hz=True):
    """Compute the complex roots of a TF from the resonance frequency and Q

    Inputs:
      f0: the resonance frequency [Hz or rad/s]
      Q: the quality factor
      Hz: If True the roots are in the frequency domain and f0 is in Hz
        If False, the roots are in the s-domain and f0 is in rad/s
        (Default: True)

    Returns:
      r1, r2: the two complex roots
    """
    a = (-1)**(not Hz)
    rr = np.sqrt(1 - 4*Q**2 + 0j)
    r1 = a * f0/(2*Q) * (1 + rr)
    r2 = a * f0/(2*Q) * (1 - rr)
    return r1, r2


def res_from_roots(rr, Hz=True):
    """Compute the resonance frequency and Q from a complex pole

    Inputs:
      rr: the complex pole
      Hz: If True the roots are in the frequency domain and f0 is in Hz
        If False, the roots are in the s-domain and f0 is in rad/s
        (Default: True)

    Returns:
      f0: the resonance frequency
      Q: the Q factor
    """
    rr = (-1)**(not Hz) * rr
    f0 = np.abs(rr)
    Q = 1/(2*np.cos(np.angle(rr)))
    return f0, Q


def catzp(*args):
    """Concatenate a list of zeros or poles

    Useful in conjunction with resRoots. For example, a pole at 1 Hz and a
    complex pair of poles with frequency 50 Hz and Q 10 can be defined with
        catzp(1, resRoots(50, 10))

    Inputs:
      The zeros or poles

    Returns:
      zp: a list of the zeros or poles
    """
    zp = []
    for arg in args:
        zp.extend(assertArr(arg))
    return zp


def catfilt(*args):
    """Concatenate a list of Filters

    Returns a new Filter instance which is the product of the input filters

    Inputs:
      args: a list of Filter instances to multiply

    Returns:
      newFilt: a Filter instance which is the product of the inputs
    """
    def newFilt(ss):
        out = 1
        for filt in args:
            out *= filt.filt(ss)
        return out

    return Filter(newFilt)


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


class DegreeOfFreedom:
    def __init__(self, name, probes, drives, dofType='pos'):
        self.name = name
        self.probes = assertType(probes, dict).copy()
        self.drives = OrderedDict()
        self.dofType = dofType

        # append the type of dof to the names of the drives
        in_drives = assertType(drives, dict).copy()
        self.drives = OrderedDict(
            {k + '.' + dofType: v for k, v in in_drives.items()})

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


class Filter:
    """A class representing a generic filter

    Inputs:
      The filter can be specified in one of two ways:
        1) Giving a callable function
        2) Giving the zeros, poles, and gain
        3) Giving the zeros, poles, and gain at a specific frequency
      Hz: If True, the zeros and poles are in the frequency domain and in Hz
        If False, the zeros and poles are in the s-domain and in rad/s
        (Default: True)

    Attributes:
      filt: the filter function
    """

    def __init__(self, *args, **kwargs):
        if 'Hz' in kwargs:
            if kwargs['Hz']:
                a = -2*np.pi
            else:
                a = 1
        else:
            a = -2*np.pi

        if len(args) == 1:
            self.filt = args[0]
            if not callable(self.filt):
                raise ValueError('One argument filters should be functions')

        elif len(args) == 3:
            zs = a*np.array(args[0])
            ps = a*np.array(args[1])
            k = args[2]
            if not isinstance(k, Number):
                raise ValueError('The gain should be a scalar')

            self.filt = partial(zpk, zs, ps, k)

        elif len(args) == 4:
            zs = a*np.array(args[0])
            ps = a*np.array(args[1])
            g = args[2]
            s0 = np.abs(a)*1j*args[3]
            if not (isinstance(g, Number) and isinstance(s0, Number)):
                raise ValueError(
                    'The gain and reference frequency should be scalars')

            k = g / np.abs(zpk(zs, ps, 1, s0))
            self.filt = partial(zpk, zs, ps, k)

        else:
            msg = 'Incorrect number of arguments. Input can be either\n' \
                  + '1) A single argument which is the filter function\n' \
                  + '2) Three arguments representing the zeros, poles,' \
                  + ' and gain\n' \
                  + '3) Four arguments representing the zeros, poles,' \
                  + ' and gain at a specific frequency'
            raise ValueError(msg)

    def computeFilter(self, ff):
        ss = 2j*np.pi*ff
        return self.filt(ss)

    def plotFilter(self, ff, mag_ax=None, phase_ax=None, dB=False, **kwargs):
        """Plot the filter

        See documentation for plotting.plotTF
        """
        fig = plotting.plotTF(
            ff, self.computeFilter(ff), mag_ax=mag_ax, phase_ax=phase_ax,
            dB=dB, **kwargs)
        return fig


class ControlSystem:
    def __init__(self):
        self.opt = None
        self.ss = None
        self.dofs = OrderedDict()
        self.filters = []
        self.probes = []
        self.drives = []
        self.respFilts = []
        self.compFilts = []
        self.plant = None
        self.ctrlMat = None
        self.oltf = None
        self.cltf = None
        self.inMat = None
        self.outMat = None
        self.compMat = None
        self.respMat = None
        self.mMech = None
        self.drive2bsm = None
        self.outComp = None

    def setPyTicklePlant(self, opt):
        """Set a PyTickle model to be the plant

        Inputs:
          opt: the PyTickle instance to be used as the plant
        """
        if self.opt is not None:
            raise ValueError('A PyTickle model is already set as the plant')
        else:
            self.opt = opt
            self.ss = 2j*np.pi*opt.ff

    def tickle(self, drive2bsm=False, mechmod=True):
        """Compute the control system dynamics

        Inputs:
          drive2bsm: Compute the drive to beamspot motion response
            (Default: False)
          mechmod: Compute the radiation pressure mechanical modifications
            (Default: True)

        Note: The spot test point cannot be used if drive2bsm is False
        and the pos test point cannot be used if mechmod is False
        """
        self.inMat = self.computeInputMatrix()
        self.outMat = self.computeOutputMatrix()
        self.plant = self.computePlant()
        self.ctrlMat = self.computeController()
        self.compMat = self.computeCompensator()
        self.respMat = self.computeResponse()
        if mechmod:
            self.mMech = self.computeMechMod()
        # self.respMat = np.einsum('ijf,jkf->ikf', self.mMech, response)
        self.outComp = np.einsum('ijf,jk->ikf', self.compMat, self.outMat)
        self.oltf = {tp: self.computeOLTF(tp)
                     for tp in ['err', 'ctrl', 'comp', 'drive', 'sens']}
        self.cltf = {sig: self.computeCLTF(oltf)
                     for sig, oltf in self.oltf.items()}
        if drive2bsm:
            self.computeBeamSpotMotion()

    def addDOF(self, name, probes, drives, dofType='pos'):
        """Add a degree of freedom to the model

        Inputs:
          name: name of the DOF
          probes: the probes used to sense the DOF
          drives: the drives used to define the DOF

        Example:
          For DARM where L = Lx - Ly and it is sensed by AS_DIFF
          cs.addDOF('DARM', 'AS_DIFF', {'EX.pos': 1, 'EY.pos': -1})
        """
        if name in self.dofs.keys():
            raise ValueError(
                'Degree of freedom {:s} already exists'.format(name))

        dof = DegreeOfFreedom(name, probes, drives, dofType=dofType)
        self.dofs[name] = dof
        append_str_if_unique(self.probes, dof.probes)
        append_str_if_unique(self.drives, dof.drives)

    def computePlant(self):
        """Compute the PyTickle plant from drives to probes

        Returns:
          plant: The plant a (nProbes, nDrives, nFreq) array
            with the transfer functions from drives to probes
        """
        nProbes = len(self.probes)
        nDrives = len(self.drives)
        nff = len(self.opt.ff)
        plant = np.zeros((nProbes, nDrives, nff), dtype=complex)

        for pi, probe in enumerate(self.probes):
            for di, drive in enumerate(self.drives):
                driveData = drive.split('.')
                driveName = driveData[0]
                dofType = driveData[-1]
                tf = self.opt.getTF(probe, driveName, dof=dofType)
                plant[pi, di, :] = tf
        return plant

    def computeInputMatrix(self):
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

    def computeOutputMatrix(self):
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

    def computeController(self):
        """Compute the control matrix from DOFs to DOFs

        Returns:
          ctrlMat: a (nDOF, nDOF, nff) array
        """
        if self.ss is None:
            raise RuntimeError('There is no associated PyTickle model')
        nDOF = len(self.dofs)
        ctrlMat = np.zeros((nDOF, nDOF, len(self.ss)), dtype=complex)
        for (dof_to, dof_from, filt) in self.filters:
            toInd = list(self.dofs.keys()).index(dof_to)
            fromInd = list(self.dofs.keys()).index(dof_from)
            ctrlMat[toInd, fromInd, :] = filt.filt(self.ss)
        return ctrlMat

    def computeCompensator(self):
        nff = len(self.ss)
        ndrives = len(self.drives)
        ones = np.ones(nff)
        compMat = np.zeros((ndrives, ndrives, nff), dtype=complex)
        compdrives = [cf[0] for cf in self.compFilts]
        for di, drive in enumerate(self.drives):
            try:
                ind = compdrives.index(drive)
                compMat[di, di, :] = self.compFilts[ind][-1].filt(self.ss)
            except ValueError:
                compMat[di, di, :] = ones
        return compMat

    def computeResponse(self):
        nff = len(self.ss)
        ndrives = len(self.drives)
        ones = np.ones(nff)
        respMat = np.zeros((ndrives, ndrives, nff), dtype=complex)
        respdrives = [rf[0] for rf in self.respFilts]
        for di, drive in enumerate(self.drives):
            try:
                ind = respdrives.index(drive)
                respMat[di, di, :] = self.respFilts[ind][-1].filt(self.ss)
            except ValueError:
                respMat[di, di, :] = ones
        return respMat

    def computeMechMod(self):
        nDrives = len(self.drives)
        nff = len(self.opt.ff)
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
                mMech[dit, djf, :] = self.opt.getMechMod(
                    driveToName, driveFromName, dofToType)
        return mMech

    def computeBeamSpotMotion(self):
        nDrives = len(self.drives)
        nff = len(self.opt.ff)
        drive2bsm = np.zeros((nDrives, nDrives, nff), dtype=complex)
        for si, spot_drive in enumerate(self.drives):
            opticName = spot_drive.split('.')[0]
            for di, drive in enumerate(self.drives):
                driveData = drive.split('.')
                driveName = driveData[0]
                dofType = driveData[-1]
                drive2bsm[si, di] = self.opt.computeBeamSpotMotion(
                    opticName, driveName, dofType)
                # drive2bsm[si, di] = self.opt.computeBeamSpotMotion(
                #     opticName, 'fr', driveName, dofType)

        self.drive2bsm = drive2bsm

    def computeOLTF(self, tstpnt):
        """Compute the OLTF from DOFs to DOFs

        Inputs:
          sig: which signal to compute the TF for:
            1) 'err': error signal (Default)
            2) 'ctrl': control signal
        """
        if tstpnt == 'err':
            oltf = self.getTF('', 'err', '', 'ctrl', closed=False)
            oltf = multiplyMat(oltf, self.ctrlMat)

        elif tstpnt == 'sens':
            oltf = self.getTF('', 'sens', '', 'err', closed=False)
            oltf = multiplyMat(oltf, self.inMat)

        elif tstpnt == 'drive':
            oltf = self.getTF('', 'drive', '', 'sens', closed=False)
            oltf = multiplyMat(oltf, self.plant)

        elif tstpnt == 'comp':
            oltf = self.getTF('', 'comp', '', 'drive', closed=False)
            oltf = multiplyMat(oltf, self.respMat)

        elif tstpnt == 'ctrl':
            oltf = self.getTF('', 'ctrl', '', 'comp', closed=False)
            oltf = multiplyMat(oltf, self.outComp)

        return oltf

    def computeCLTF(self, oltf):
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

    def addFilter(self, dof_to, dof_from, *args):
        """Add a filter between two DOFs to the controller

        Inputs:
          dof_to: output DOF
          dof_from: input DOF
          *args: Filter instance or arguments used to define a Filter instance
        """
        if len(args) == 1 and isinstance(args[0], Filter):
            # args is already a Filter instance
            self.filters.append((dof_to, dof_from, args[0]))

        else:
            # args defines a new Filter
            self.filters.append((dof_to, dof_from, Filter(*args)))

    def addCompensator(self, drive, driveType, *args):
        drive += '.' + driveType
        if drive in [cf[0] for cf in self.compFilts]:
            raise ValueError(
                'A compensator is already set for drive {:s}'.format(drive))

        if len(args) == 1 and isinstance(args[0], Filter):
            self.compFilts.append((drive, args[0]))

        else:
            self.compFilts.append((drive, Filter(*args)))

    def setResponse(self, drive, driveType, *args):
        drive += '.' + driveType
        if drive in [rf[0] for rf in self.respFilts]:
            raise ValueError(
                'A response is already set for drive {:s}'.format(drive))

        if len(args) == 1 and isinstance(args[0], Filter):
            self.respFilts.append((drive, args[0]))

        else:
            self.respFilts.append((drive, Filter(*args)))

    def getFilter(self, dof_to, dof_from):
        """Get the filter between two DOFs

        Inputs:
          dof_to: output DOF
          dof_from: input DOF

        Returns:
          filt: the filter
        """
        dofs_to = np.array([filt[0] for filt in self.filters])
        dofs_from = np.array([filt[1] for filt in self.filters])
        inds = np.logical_and(dof_to == dofs_to, dof_from == dofs_from)
        nfilts = np.count_nonzero(inds)
        if nfilts == 0:
            raise ValueError('There is no filter from {:s} to {:s}'.format(
                dof_from, dof_to))
        elif nfilts > 1:
            raise ValueError('There are multiple filters')
        else:
            return self.filters[inds.nonzero()[0][0]][-1]

    def plotFilter(self, dof_to, dof_from, mag_ax=None, phase_ax=None, dB=False,
                   **kwargs):
        """Plot a filter function

        See documentation for plotFilter in Filter class
        """
        filt = self.getFilter(dof_to, dof_from)
        return filt.plotFilter(self.opt.ff, mag_ax, phase_ax, dB, **kwargs)

    def getOLTF(self, dof_to, dof_from, tstpnt):
        """Compute the OLTF from DOFs to DOFs

        Inputs:
          dof_to: output DOF
          dof_from: input DOF
          tstpnt: which test pont to compute the TF for:
        """
        oltf = self.oltf[tstpnt]
        if dof_from:
            from_ind = self._getIndex(dof_from, tstpnt)
            oltf = oltf[:, from_ind]
        if dof_to:
            to_ind = self._getIndex(dof_to, tstpnt)
            oltf = oltf[to_ind]
        return oltf

    def getCLTF(self, dof_to, dof_from, tstpnt):
        """Compute the CLTF from DOFs to DOFs

        Inputs:
          dof_to: output DOF
          dof_from: input DOF
          tstpnt: which test point to compute the TF for:
        """
        cltf = self.cltf[tstpnt]
        if dof_from:
            from_ind = self._getIndex(dof_from, tstpnt)
            cltf = cltf[:, from_ind]
        if dof_to:
            to_ind = self._getIndex(dof_to, tstpnt)
            cltf = cltf[to_ind]
        return cltf

    def compute_margins(self, dof_to, dof_from, tstpnt):
        oltf = self.getOLTF(dof_to, dof_from, tstpnt)
        cltf = self.getCLTF(dof_to, dof_from, tstpnt)
        ugf, pm = compute_phase_margin(self.opt.ff, oltf)
        fms, sm = compute_stability_margin(self.opt.ff, cltf)
        return ugf, pm, fms, sm

    def print_margins(self, tstpnt):
        margins = {}
        for dof in self.dofs.keys():
            ugf, pm, fms, sm = self.compute_margins(dof, dof, tstpnt)
            margins[dof] = ['{:0.2f} {:s}Hz'.format(*siPrefix(ugf)[::-1]),
                            '{:0.2f} deg'.format(pm),
                            '{:0.2f} {:s}Hz'.format(*siPrefix(fms)[::-1]),
                            '{:0.2f}'.format(sm)]

        margins = pd.DataFrame(
            margins,
            index=['UGF', 'phase margin', 'max sensitivity', 'stability margin']).T
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None):
            display(margins)

    def getTF(self, dof_to, tp_to, dof_from, tp_from, closed=True):
        """Get a transfer function between two test points in a loop

        If dof_to is the empty string '', the vector to all dofs is returned
        If dof_from is the empty string '', the vector of all dofs is returned
        """
        def cltf_or_unity(tp_to):
            """Returns the CLTF if a closed loop TF is necessary and the identity
            matrix if an open loop TF is necessary
            """
            if closed:
                return self.cltf[tp_to]
            else:
                return 1

        # test points and their matrices in cyclic order for computing TFs
        # around the main loop
        tstpnts = ['err', 'sens', 'drive', 'comp', 'ctrl']
        mats = [self.inMat, self.plant, self.respMat, self.outComp,
                self.ctrlMat]

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
                    tf = multiplyMat(self.mMech, loopTF)

                # for beam spot motion, apply conversion from drives to motion
                elif tp_to == 'spot':
                    tf = multiplyMat(self.drive2bsm, loopTF)

            # No valid output test point
            else:
                raise ValueError('Unrecognized test point to ' + tp_to)

        # If the input test point is a calibration test point, first compute
        # the TF from the corresponding drive and append the output matrix
        elif tp_from == 'cal':
            loopTF = self.getTF('', tp_to, '', 'drive', closed=closed)
            tf = multiplyMat(loopTF, self.outMat)

        # No valid input test point
        else:
            raise ValueError('Unrecognized test point from ' + tp_from)

        # Reduce size of returned TF if necessary
        if dof_from:
            from_ind = self._getIndex(dof_from, tp_from)
            tf = tf[:, from_ind]

        if dof_to:
            to_ind = self._getIndex(dof_to, tp_to)
            tf = tf[to_ind]

        return tf

    def getTotalNoiseTo(self, dof_to, tp_to, tp_from, noiseASDs, closed=True):
        """Compute the total noise at a test point due to multiple sources

        Inputs:
          dof_to: the output DOF
          tp_to: the output test point
          tp_from: the input test point
          noiseASDs: a dictionary of noise ASDs with keys the name of the
            dofs from and values the ASDs [u/rtHz]

        Returns:
          totalASD: the total ASD [u/rtHz]
        """
        if tp_to == 'cal':
            ampTF = inv(self.outMat)
            to_ind = self._getIndex(dof_to, 'cal')
            ampTF = ampTF[to_ind]
        else:
            ampTF = self.getTF(dof_to, tp_to, '', tp_from, closed=closed)
        if len(ampTF.shape) == 1:
            ampTF = np.einsum('i,j->ij', ampTF, np.ones_like(self.opt.ff))
        powTF = np.abs(ampTF)**2
        totalPSD = np.zeros(powTF.shape[-1])
        for dof_from, noiseASD in noiseASDs.items():
            from_ind = self._getIndex(dof_from, tp_from)
            totalPSD += powTF[from_ind] * noiseASD**2
        return np.sqrt(totalPSD)

    def getTotalNoiseFrom(self, sig_to, dof_from, tp_from, noiseASDs):
        ampTF = self.getTF('', sig_to, dof_from, tp_from)
        powTF = np.abs(ampTF)**2
        totalPSD = np.zeros(powTF.shape[-1])
        for dof_to, noiseASD in noiseASDs.items():
            to_ind = self._getIndex(dof_to, sig_to)
            totalPSD += powTF[to_ind] * noiseASD**2
        return np.sqrt(totalPSD)

    def plotNyquist(self, dof_to, dof_from, tstpnt):
        oltf = self.getOLTF(dof_to, dof_from, tstpnt)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        ax.plot(np.angle(oltf), np.abs(oltf))
        npts = len(oltf)
        ax.plot(np.linspace(0, 2*np.pi, npts), np.ones(npts))
        ax.set_ylim(0, 3)
        return fig

    def _getIndex(self, name, tstpnt):
        if tstpnt in ['err', 'ctrl', 'cal', 'cal']:
            ind = list(self.dofs.keys()).index(name)
        elif tstpnt in ['comp', 'drive', 'pos', 'spot']:
            ind = self.drives.index(name)
        elif tstpnt == 'sens':
            ind = self.probes.index(name)
        else:
            raise ValueError('Unrecognized test point ' + tstpnt)
        return ind
