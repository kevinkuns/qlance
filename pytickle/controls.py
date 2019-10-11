import numpy as np
from scipy.linalg import inv
from collections import OrderedDict
from .utils import assertType
from functools import partial
from numbers import Number
from . import plotting


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

    for z in assertArr(zs):
        filt *= (ss - z)

    for p in assertArr(ps):
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
        self.sigs = ['err', 'ctrl', 'comp', 'drive', 'sens']

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

    def tickle(self):
        self.inMat = self.computeInputMatrix()
        self.outMat = self.computeOutputMatrix()
        self.plant = self.computePlant()
        self.ctrlMat = self.computeController()
        self.compMat = self.computeCompensator()
        response = self.computeResponse()
        mMech = self.computeMechMod()
        self.respMat = np.einsum('ijf,jkf->ikf', mMech, response)
        self.oltf = {sig: self.computeOLTF(sig) for sig in self.sigs}
        self.cltf = {sig: self.computeCLTF(oltf)
                      for sig, oltf in self.oltf.items()}

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
                tf = self.opt.getTF(probe, driveName, dof=dofType, optOnly=True)
                plant[pi, di, :] = tf
        return plant

    def computeInputMatrix(self):
        """Compute the sensing matrix from probes to DOFs

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
        for (dofTo, dofFrom, filt) in self.filters:
            toInd = list(self.dofs.keys()).index(dofTo)
            fromInd = list(self.dofs.keys()).index(dofFrom)
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


    def computeOLTF(self, sig='err'):
        """Compute the OLTF from DOFs to DOFs

        Inputs:
          sig: which signal to compute the TF for:
            1) 'err': error signal (Default)
            2) 'ctrl': control signal
        """
        if sig not in self.sigs:
            raise ValueError('Unrecognized signal')

        if sig == 'err':
            oltf = np.einsum(
                'ij,jkf,klf,lmf,mn,npf->ipf', self.inMat, self.plant,
                self.respMat, self.compMat, self.outMat, self.ctrlMat)

        elif sig == 'ctrl':
            oltf = np.einsum(
                'ijf,jk,klf,lmf,mnf,np->ipf', self.ctrlMat, self.inMat,
                self.plant, self.respMat, self.compMat, self.outMat)

        elif sig == 'comp':
            oltf = np.einsum(
                'ijf,jk,klf,lm,mnf,npf->ipf', self.compMat, self.outMat,
                self.ctrlMat, self.inMat, self.plant, self.respMat)

        elif sig == 'drive':
            oltf = np.einsum(
                'ijf,jkf,kl,lmf,mn,npf->ipf', self.respMat, self.compMat,
                self.outMat, self.ctrlMat, self.inMat, self.plant)

        elif sig == 'sens':
            oltf = np.einsum(
                'ijf,jkf,klf,lm,mnf,np->ipf', self.plant, self.respMat,
                self.compMat, self.outMat, self.ctrlMat, self.inMat)

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

    def addFilter(self, dofTo, dofFrom, *args):
        """Add a filter between two DOFs to the controller

        Inputs:
          dofTo: output DOF
          dofFrom: input DOF
          *args: Filter instance or arguments used to define a Filter instance
        """
        if len(args) == 1 and isinstance(args[0], Filter):
            # args is already a Filter instance
            self.filters.append((dofTo, dofFrom, args[0]))

        else:
            # args defines a new Filter
            self.filters.append((dofTo, dofFrom, Filter(*args)))

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

    def getFilter(self, dofTo, dofFrom):
        """Get the filter between two DOFs

        Inputs:
          dofTo: output DOF
          dofFrom: input DOF

        Returns:
          filt: the filter
        """
        dofsTo = np.array([filt[0] for filt in self.filters])
        dofsFrom = np.array([filt[1] for filt in self.filters])
        inds = np.logical_and(dofTo == dofsTo, dofFrom == dofsFrom)
        nfilts = np.count_nonzero(inds)
        if nfilts == 0:
            raise ValueError('There is no filter from {:s} to {:s}'.format(
                dofFrom, dofTo))
        elif nfilts > 1:
            raise ValueError('There are multiple filters')
        else:
            return self.filters[inds.nonzero()[0][0]][-1]

    def plotFilter(self, dofTo, dofFrom, mag_ax=None, phase_ax=None, dB=False,
                   **kwargs):
        """Plot a filter function

        See documentation for plotFilter in Filter class
        """
        filt = self.getFilter(dofTo, dofFrom)
        return filt.plotFilter(self.opt.ff, mag_ax, phase_ax, dB, **kwargs)

    def getOLTF(self, dofTo, dofFrom, sig='err'):
        """Compute the OLTF from DOFs to DOFs

        Inputs:
          dofTo: output DOF
          dofFrom: input DOF
          sig: which signal to compute the TF for:
            1) 'err': error signal (Default)
            2) 'ctrl': control signal
        """
        # if sig not in ['err', 'ctrl']:
        #     raise ValueError('The signal must be either err or ctrl')

        oltf = self.oltf[sig]
        dofToInd = list(self.dofs.keys()).index(dofTo)
        dofFromInd = list(self.dofs.keys()).index(dofFrom)
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
        # if sig not in ['err', 'ctrl']:
        #     raise ValueError('The signal must be either err or ctrl')

        cltf = self.cltf[sig]
        dofToInd = list(self.dofs.keys()).index(dofTo)
        dofFromInd = list(self.dofs.keys()).index(dofFrom)
        return cltf[dofToInd, dofFromInd, :]

    def getTF(self, dofTo, sigTo, dofFrom, sigFrom, closed=True):
        """Get a transfer function between two test points in a loop

        If dofTo is the empty string '', the vector to all dofs is returned
        If dofFrom is the empty string '', the vector of all dofs is returned
        """
        def cltf_or_unity(sigTo):
            """Returns the CLTF if a closed loop TF is necessary and the identity
            matrix if an open loop TF is necessary
            """
            if closed:
                return self.cltf[sigTo]
            else:
                unity = np.zeros_like(self.cltf[sigTo])
                inds = np.arange(self.cltf[sigTo].shape[0])
                unity[inds, inds, :] = 1
                return unity

        if sigTo == sigFrom:
            tf = cltf_or_unity(sigTo)

        elif sigTo == 'err':
            cltf = cltf_or_unity(sigTo)
            if sigFrom == 'sens':
                tf = np.einsum(
                    'ijf,jk->ikf', cltf, self.inMat)
            elif sigFrom == 'drive':
                tf = np.einsum(
                    'ijf,jk,klf->ilf', cltf, self.inMat, self.plant)
            elif sigFrom == 'cal':
                tf = np.einsum(
                    'ijf,jk,klf,lm->imf', cltf, self.inMat, self.plant,
                    self.outMat)
            elif sigFrom == 'comp':
                tf = np.einsum(
                    'ijf,jk,klf,lmf->imf', cltf, self.inMat, self.plant,
                    self.respMat)
            elif sigFrom == 'ctrl':
                tf = np.einsum(
                    'ijf,jk,klf,lmf,mnf,np->ipf', cltf, self.inMat,
                    self.plant, self.respMat, self.compMat, self.outMat)

        elif sigTo == 'ctrl':
            cltf = cltf_or_unity(sigTo)
            if sigFrom == 'err':
                tf = np.einsum(
                    'ijf,jkf->ikf', cltf, self.ctrlMat)
            elif sigFrom == 'sens':
                tf = np.einsum(
                    'ijf,jkf,kl->ilf', cltf, self.ctrlMat, self.inMat)
            elif sigFrom == 'drive':
                tf = np.einsum(
                    'ijf,jkf,kl,lmf->imf', cltf, self.ctrlMat, self.inMat,
                    self.plant)
            elif sigFrom == 'cal':
                tf = np.einsum(
                    'ijf,jkf,kl,lmf,mn->inf', cltf, self.ctrlMat, self.inMat,
                    self.plant, self.outMat)
            elif sigFrom == 'comp':
                tf = np.einsum(
                    'ijf,jkf,kl,lmf,mn->inf', cltf, self.ctrlMat, self.inMat,
                    self.plant, self.respMat)

        elif sigTo == 'comp':
            cltf = cltf_or_unity(sigTo)
            if sigFrom == 'ctrl':
                tf = np.einsum(
                    'ijf,jkf,kl->ilf', cltf, self.compMat, self.outMat)
            elif sigFrom == 'err':
                tf = np.einsum(
                    'ijf,jkf,kl,lmf->imf', cltf, self.compMat, self.outMat,
                    self.ctrlMat)
            elif sigFrom == 'sens':
                tf = np.einsum(
                    'ijf,jkf,kl,lmf,mn->inf', cltf, self.compMat,
                    self.outMat, self.ctrlMat, self.inMat)
            elif sigFrom == 'drive':
                tf = np.einsum(
                    'ijf,jk,klf,lm,mnf->inf', cltf, self.compMat,
                    self.outMat, self.ctrlMat, self.inMat, self.plant)
            elif sigFrom == 'cal':
                tf = np.einsum(
                    'ijf,jk,klf,lm,mnf,np->ipf', cltf, self.compMat,
                    self.outMat, self.ctrlMat, self.inMat, self.plant,
                    self.outMat)

        elif sigTo == 'drive':
            cltf = cltf_or_unity(sigTo)
            if sigFrom == 'cal':
                tf = np.einsum('ijf,jkf,kl->ilf',
                               cltf, self.oltf['drive'], self.outMat)
            elif sigFrom == 'comp':
                tf = np.einsum(
                    'ijf,jkf->ikf', cltf, self.respMat)
            elif sigFrom == 'ctrl':
                tf = np.einsum(
                    'ijf,jkf,klf,lm->imf', cltf, self.respMat, self.compMat,
                    self.outMat)
            elif sigFrom == 'err':
                tf = np.einsum(
                    'ijf,jkf,klf,lm,mnf->inf', cltf, self.respMat,
                    self.compMat, self.outMat, self.ctrlMat)
            elif sigFrom == 'sens':
                tf = np.einsum(
                    'ijf,jkf,klf,lm,mnf,np->ipf', cltf, self.respMat,
                    self.compMat, self.outMat, self.ctrlMat, self.inMat)

        elif sigTo == 'sens':
            cltf = cltf_or_unity(sigTo)
            if sigFrom == 'drive':
                tf = np.einsum(
                    'ijf,jkf->ikf', cltf, self.plant)
            if sigFrom == 'cal':
                tf = np.einsum(
                    'ijf,jkf,kl->ilf', cltf, self.plant, self.outMat)
            elif sigFrom == 'comp':
                tf = np.einsum(
                    'ijf,jkf,klf->ilf', cltf, self.plant, self.respMat)
            elif sigFrom == 'ctrl':
                tf = np.einsum(
                    'ijf,jkf,klf,lmf,mn->inf', cltf, self.plant,
                    self.respMat, self.compMat, self.outMat)
            elif sigFrom == 'err':
                tf = np.einsum(
                    'ijf,jkf,klf,lmf,mn,npf->ipf', cltf, self.plant,
                    self.respMat, self.compMat, self.outMat, self.ctrlMat)

        elif sigTo == 'spot':
            # Get the conversion from drives to beam spot motion
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
                        opticName, 'fr', driveName, dofType)

            # Get the loop transfer function to drives
            if sigFrom == 'drive':
                loopTF = cltf_or_unity('drive')
            else:
                loopTF = self.getTF('', 'drive', '', sigFrom)
            tf = np.einsum('ijf,jkf->ikf', drive2bsm, loopTF)

        if dofFrom:
            fromInd = self._getIndex(dofFrom, sigFrom)
            tf = tf[:, fromInd]

        if dofTo:
            toInd = self._getIndex(dofTo, sigTo)
            tf = tf[toInd]

        return tf

    def getTotalNoiseTo(self, dofTo, sigTo, sigFrom, noiseASDs):
        ampTF = self.getTF(dofTo, sigTo, '', sigFrom)
        powTF = np.abs(ampTF)**2
        totalPSD = np.zeros(powTF.shape[-1])
        for dofFrom, noiseASD in noiseASDs.items():
            fromInd = self._getIndex(dofFrom, sigFrom)
            totalPSD += powTF[fromInd] * noiseASD**2
        return np.sqrt(totalPSD)

    def getTotalNoiseFrom(self, sigTo, dofFrom, sigFrom, noiseASDs):
        ampTF = self.getTF('', sigTo, dofFrom, sigFrom)
        powTF = np.abs(ampTF)**2
        totalPSD = np.zeros(powTF.shape[-1])
        for dofTo, noiseASD in noiseASDs.items():
            toInd = self._getIndex(dofTo, sigTo)
            totalPSD += powTF[toInd] * noiseASD**2
        return np.sqrt(totalPSD)

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

        cltf = self.cltf[sig]
        if sig == 'err':
            noiseTF = np.einsum('ijf,jk->ikf', cltf, self.inMat)
        elif sig == 'ctrl':
            noiseTF = np.einsum('ijf,jkf,kl->ilf', cltf, self.ctrlMat,
                                self.inMat)

        if asd is None:
            qnoise = self.opt.getQuantumNoise(probe)
        else:
            qnoise = asd

        dofInd = list(self.dofs.keys()).index(dof)
        probeInd = self.probes.index(probe)
        sensNoise = np.abs(noiseTF[dofInd, probeInd, :]) * qnoise
        return sensNoise

    def _getIndex(self, name, sig):
        if sig in ['err', 'ctrl', 'cal']:
            ind = list(self.dofs.keys()).index(name)
        elif sig in ['comp', 'drive', 'spot']:
            ind = self.drives.index(name)
        elif sig == 'sens':
            ind = self.probes.index(name)
        return ind
