"""
Control system calculations
"""

import numpy as np
from numpy.linalg import inv
import pandas as pd
from collections import OrderedDict
from .utils import assertType, siPrefix, append_str_if_unique
from . import io
from functools import partial
from numbers import Number
from itertools import cycle, zip_longest
import scipy.signal as sig
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
    # check if any of the filters have not been defined with zpk
    zpk_defined = True
    for filt in args:
        if filt._ps is None:
            zpk_defined = False
            break

    # if all filters have zpk, define a new one combining this information
    if zpk_defined:
        zs = []
        ps = []
        k = 1
        for filt in args:
            zf, pf, kf = filt.get_zpk(Hz=False)
            zs.extend(assertArr(zf))
            ps.extend(assertArr(pf))
            k *= kf
        return Filter(zs, ps, k, Hz=False)

    # otherwise just make a new function
    else:
        def newFilt(ss):
            out = 1
            for filt in args:
                out *= filt._filt(ss)
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
      name: name of the DOF
      probes: the probes used to sense the DOF (a dict or single string)
      drives: the drives defining the DOF (a dict or single string)
      dofType: type of DOF: pos, pitch, or yaw (Default: pos)

    Example:
      For DARM where L = Lx - Ly and it is sensed by AS_DIFF
        DegreeOfFreedom('DARM', 'AS_DIFF', {'EX.pos': 1, 'EY.pos': -1})
    """

    def __init__(self, name, probes, drives, dofType='pos'):
        self._name = name
        self._probes = assertType(probes, dict).copy()
        # self._drives = OrderedDict()
        self._dofType = dofType

        # append the type of dof to the names of the drives
        in_drives = assertType(drives, dict).copy()
        self._drives = OrderedDict(
            {k + '.' + dofType: v for k, v in in_drives.items()})

    @property
    def name(self):
        return self._name

    @property
    def probes(self):
        return self._probes

    @property
    def drives(self):
        return self._drives

    @property
    def dofType(self):
        return self._dofType

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


def filt_from_hdf5(path, h5file):
    """Define a filter from a dictionary stored in an hdf5 file

    Inputs:
      path: path to the dictionary
      h5file: the hdf5 file

    Returns:
      filt: the filter instance
    """
    zpk_dict = dict(
        zs=np.array(io.hdf5_to_possible_none(path + '/zs', h5file)),
        ps=np.array(io.hdf5_to_possible_none(path + '/ps', h5file)),
        k=h5file[path + '/k'][()])
    return Filter(zpk_dict, Hz=False)


class Filter:
    """A class representing a generic filter

    Inputs:
      The filter can be specified in one of two ways:
        1) Giving a callable function that is the s-domain filter
        2) Giving the zeros, poles, and gain
        3) Giving the zeros, poles, and gain at a specific frequency
        4) Giving a dictionary specifying the zeros, poles, and gain
      Hz: If True, the zeros and poles are in the frequency domain and in Hz
          If False, the zeros and poles are in the s-domain and in rad/s
          (Default: True)

      Examples:
        All of the following define the same single pole low-pass filter with
        1 Hz corner frequency:
          Filter([], 1, 1)
          Filter([], -2*np.pi, 1, Hz=False)
          Filter(lambda s: 1/(s + 2*np.pi))
          Filter(dict(zs=[], ps=1, k=1))
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
            if isinstance(args[0], dict):
                zs = a*np.array(args[0]['zs'])
                ps = a*np.array(args[0]['ps'])
                k = args[0]['k']
                self._filt = partial(zpk, zs, ps, k)
                self._zs = zs
                self._ps = ps
                self._k = k

            elif callable(args[0]):
                self._filt = args[0]
                self._zs = None
                self._ps = None
                self._k = None

            else:
                raise ValueError(
                    'One argument filters should be dictionaries or functions')

        elif len(args) == 3:
            zs = a*np.array(args[0])
            ps = a*np.array(args[1])
            k = args[2]
            if not isinstance(k, Number):
                raise ValueError('The gain should be a scalar')

            self._filt = partial(zpk, zs, ps, k)
            self._zs = zs
            self._ps = ps
            self._k = k

        elif len(args) == 4:
            zs = a*np.array(args[0])
            ps = a*np.array(args[1])
            g = args[2]
            s0 = np.abs(a)*1j*args[3]
            if not (isinstance(g, Number) and isinstance(s0, Number)):
                raise ValueError(
                    'The gain and reference frequency should be scalars')

            k = g / np.abs(zpk(zs, ps, 1, s0))
            self._filt = partial(zpk, zs, ps, k)
            self._zs = zs
            self._ps = ps
            self._k = k

        else:
            msg = 'Incorrect number of arguments. Input can be either\n' \
                  + '1) A single argument which is the filter function\n' \
                  + '2) Three arguments representing the zeros, poles,' \
                  + ' and gain\n' \
                  + '3) Four arguments representing the zeros, poles,' \
                  + ' and gain at a specific frequency'
            raise ValueError(msg)

    def computeFilter(self, ff):
        """Compute the filter

        Inputs:
          ff: frequency vector at which to compute the filter [Hz]
        """
        ss = 2j*np.pi*ff
        return self._filt(ss)

    def get_zpk(self, Hz=False, as_dict=False):
        """Get the zeros, poles, and gain of this filter

        A ValueError is raised if the filter was defined as a function instead
        of giving zeros, poles, and gain.

        Inputs:
          Hz: If True, the zeros and poles are in the frequency domain and in Hz
              If False, the zeros and poles are in the s-domain and in rad/s
              (Default: False)
          as_dict: If True, returns a dictionary instead (Default: False)

        Returns:
          if as_dict is False:
            zs: the zeros
            ps: the poles
            k: the gain
          if as dict is True:
            zpk_dict: dictionary with keys 'zs', 'ps', and 'k'
        """
        if self._ps is None:
            raise ValueError(
                'This filter was not defined with zeros, poles, and a gain')
        a = (-2*np.pi)**Hz

        if as_dict:
            zpk_dict = dict(zs=self._zs/a, ps=self._ps/a, k=self._k)
            return zpk_dict

        else:
            return self._zs/a, self._ps/a, self._k

    def get_state_space(self):
        """Get a state space representation of this filter

        Returns:
          ss: a scipy.signal state space representation of the filter
        """
        zs, ps, k = self.get_zpk(Hz=False)
        return sig.StateSpace(*sig.zpk2ss(assertArr(zs), assertArr(ps), k))

    def plotFilter(self, ff, mag_ax=None, phase_ax=None, dB=False, **kwargs):
        """Plot the filter

        See documentation for plotting.plotTF
        """
        fig = plotting.plotTF(
            ff, self.computeFilter(ff), mag_ax=mag_ax, phase_ax=phase_ax,
            dB=dB, **kwargs)
        return fig

    def to_hdf5(self, path, h5file):
        zs, ps, k = self.get_zpk(Hz=False)
        io.possible_none_to_hdf5(zs, path + '/zs', h5file)
        io.possible_none_to_hdf5(ps, path + '/ps', h5file)
        h5file[path + '/k'] = k


class ControlSystem:
    """A class for a general MIMO control system
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
        return self._plant_model

    @property
    def ff(self):
        return self.plant_model.ff

    @property
    def ss(self):
        return 2j*np.pi*self.ff

    @property
    def dofs(self):
        return self._dofs

    @property
    def probes(self):
        return self._probes

    @property
    def drives(self):
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

    def _computePlant(self):
        """Compute the PyTickle plant from drives to probes

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
                dofType = driveData[-1]
                tf = self.plant_model.getTF(probe, driveName, dof=dofType)
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
            raise RuntimeError('There is no associated PyTickle model')
        nDOF = len(self.dofs)
        ctrlMat = np.zeros((nDOF, nDOF, len(self.ss)), dtype=complex)
        for (dof_to, dof_from, filt) in self._filters:
            toInd = list(self.dofs.keys()).index(dof_to)
            fromInd = list(self.dofs.keys()).index(dof_from)
            ctrlMat[toInd, fromInd, :] = filt._filt(self.ss)
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
                compMat[di, di, :] = self._compFilts[ind][-1]._filt(self.ss)
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
                actMat[di, di, :] = self._actFilts[ind][-1]._filt(self.ss)
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
                dofType = driveData[-1]
                # FIXME: make work with finesse and non-front surfaces
                drive2bsm[si, di] = self.plant_model.computeBeamSpotMotion(
                    opticName, 'fr', driveName, dofType)

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

    def addCompensator(self, drive, driveType, filt):
        """Add a compensation filter

        Inputs:
          drive: which drive to compensate
          driveType: type of drive (pos, pitch, or yaw)
          filt: Filter instance for the filter
        """
        drive += '.' + driveType
        if drive in [cf[0] for cf in self._compFilts]:
            raise ValueError(
                'A compensator is already set for drive {:s}'.format(drive))

        self._compFilts.append((drive, filt))

    def setActuator(self, drive, driveType, filt):
        """Set the actuation plant for a drive

        Inputs:
          drive: which drive to actuate
          driveType: type of drive (pos, pitch, or yaw)
          filt: Filter instance for the plant
        """
        drive += '.' + driveType
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

    def getOLTF(self, sig_to, sig_from, tstpnt):
        """Compute an OLTF

        Inputs:
          sig_to: output signal
          sig_from: input signal
          tstpnt: which test point to compute the TF for
        """
        oltf = self._oltf[tstpnt]
        if sig_from:
            from_ind = self._getIndex(sig_from, tstpnt)
            oltf = oltf[:, from_ind]
        if sig_to:
            to_ind = self._getIndex(sig_to, tstpnt)
            oltf = oltf[to_ind]
        return oltf

    def getCLTF(self, sig_to, sig_from, tstpnt):
        """Compute a CLTF

        Inputs:
          sig_to: output signal
          sig_from: input signal
          tstpnt: which test point to compute the TF for
        """
        cltf = self._cltf[tstpnt]
        if sig_from:
            from_ind = self._getIndex(sig_from, tstpnt)
            cltf = cltf[:, from_ind]
        if sig_to:
            to_ind = self._getIndex(sig_to, tstpnt)
            cltf = cltf[to_ind]
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

    def print_margins(self, tstpnt):
        """Print the stability margins for all loops

        Input:
          tstpnt: test point
        """
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

    def getTF(self, sig_to, tp_to, sig_from, tp_from, closed=True):
        """Get a transfer function between two test points in a loop

        If sig_to is the empty string '', the vector to all signals is returned
        If sig_from is the empty string '', the vector of all signals is returned
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

    def _getIndex(self, name, tstpnt):
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

        try:
            ind = sig_list.index(name)
        except ValueError:
            raise ValueError(
                '{:s} is not a {:s} test point'.format(name, tstpnt))

        return ind
