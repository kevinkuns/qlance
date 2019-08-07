'''
Provides code for calling Optickle from within python
'''

from __future__ import division
import numpy as np
import matlab
import matplotlib.pyplot as plt
import scipy.constants as scc
from numbers import Number
from collections import OrderedDict
import plotting
import utils


def mat2py(mat_arr):
    """Convert matlab arrays into python arrays.
    """
    return np.squeeze(np.array(mat_arr))


def py2mat(py_arr, is_complex=False):
    """Convert python arrays into matlab arrays.
    """
    if isinstance(py_arr, list):
        mat_arr = py_arr
    elif isinstance(py_arr, np.ndarray):
        mat_arr = py_arr.tolist()
        if isinstance(mat_arr, Number):
            mat_arr = [mat_arr]
    elif isinstance(py_arr, Number):
        mat_arr = [py_arr]
    else:
        raise TypeError('Invalid type ' + str(type(py_arr)) + ' for py_arr')
    return matlab.double(mat_arr, is_complex=is_complex)


def str2mat(string):
    """Ensure that strings are properly formatted for matlab.
    """
    return ''.join("'{0}'".format(string))


def addOpticklePath(eng, path=None):
    """Add the Optickle path to the matlab path

    This must be run once everytime a matlab engine is initialized

    Inputs:
      eng: the matlab engine
      path: If None (default) the path is taken from 'OPTICKLE_PATH'
            if a string, adds that string to the path
    """
    if path is None:
        cmd = "addpath(genpath(getenv('OPTICKLE_PATH')));"
    else:
        cmd = "addpath(genpath({:s}));".format(str2mat(path))
    eng.eval(cmd, nargout=0)


class PyTickle:
    """A class for an Optickle model

    Inputs:
      eng: the python matlab engine
      optName: the name for the optical model
      vRF: vector of RF frequencies [Hz] (Default: 0)
      lambda0: carrier wavelength [m] (Default: 1064e-9)
      pol: polarization (Default: 'S')

    Attributes:
      eng: same as input
      optName: same as input
      vRF: same as input
      lambda0: same as input
      pol: same as input
      probes: list of probe names associated with the model
      drives: list of drive names associated with the model
    """
    def __init__(self, eng, optName, vRF=0, lambda0=1064e-9, pol='S'):
        # convert RF and polarization information
        if isinstance(vRF, Number):
            vRF = [vRF]
        if isinstance(pol, str):
            pol = [pol] * len(vRF)
        if len(vRF) != len(pol):
            raise ValueError(
                'RF vector and polarization vector must be the same length')

        # convert S and P definitions to 1s and 0s for Optickle
        matPol = [self._pol2opt(k) for k in pol]

        # initialize the Optickle model in the matlab engine
        self.eng = eng
        self.optName = optName
        self.eng.workspace['vRF'] = py2mat(vRF)
        self.eng.workspace['lambda0'] = py2mat(lambda0)
        self.eng.workspace['pol'] = py2mat(matPol)
        self.eng.eval(
            self.optName + " = Optickle(vRF, lambda0, pol);", nargout=0)

        # initialize pytickle class variables
        self.lambda0 = lambda0
        self.vRF = mat2py(self.eng.eval(self.optName + ".vFrf"))
        self.pol = np.array(pol)
        self.nRF = OrderedDict()
        for ri, fRF in enumerate(vRF):
            pref, num = utils.siPrefix(round(fRF))
            if num == 0:
                key = 'DC'
            else:
                key = str(int(round(num))) + ' ' + pref + 'Hz'
            self.nRF[key] = ri

        self.probes = []
        self.drives = []

        self._ff = None
        self._fDC = None
        self._sigAC = None
        self._sigAC_pitch = None
        self._sigAC_yaw = None
        self._sigDC_tickle = None
        self._mMech = None
        self._mMech_pitch = None
        self._mMech_yaw = None
        self._noiseAC = None
        self._poses = None
        self._sigDC_sweep = None
        self._fDC_sweep = None
        self._rotatedBasis = False

        # Track whether tickle or sweepLinear has been run on this model
        # If it has, do not let a matlab model be associated with this
        # PyTickle instance to avoid potentially confusing results
        self._tickled = False

    def loadMatModel(self):
        """Loads a matlab model with the same name as this class
        """
        if self._tickled:
            msg = 'This pytickle model has already been run\n'
            msg += 'Initialize a new model and load the Matlab model again'
            raise RuntimeError(msg)
        self.lambda0 = mat2py(self.eng.eval(self.optName + ".lambda"))
        self.vRF = mat2py(self.eng.eval(self.optName + ".vFrf"))
        # FIXME: add polarization
        self._updateNames()

    def tickle(self, ff=None, noise=True, computeAC=True, dof='pos'):
        """Compute the optomechanical response and quantum noise

        Computes the optomechanical response and quantum noise of this model
        by running Optickle's tickle function. Angular response is computed
        with Optickle's tickle01 and tickle10 functions.

        Inputs:
          ff: frequency vector [Hz].
            If ff is not given or is None, only the DC signals are computed
            which reduces runtime if the AC signals are not needed.
          noise: If False, the quantum noise is not computed which can decrease
            runtime. (Default: True)
          computeAC: If False, only the DC signals are computed which can
            decrease runtime. (Default: True)
          dof: which degree of freedom or mode to compute the response for:
            pos: position or TEM00 mode
            pitch: pitch or TEM01 mode
            yaw: yaw or TEM10 mode
            (Default: pos)
        """
        self._tickled = True

        if dof not in ['pos', 'pitch', 'yaw']:
            msg = 'Unrecognized degree of freedom {:s}'.format(dof)
            msg += '. Choose \'pos\', \'pitch\', or \'yaw\'.'
            raise ValueError(msg)

        if ff is not None:
            self._ff = ff
            self.eng.workspace['f'] = py2mat(self._ff)

        if dof == 'pos':
            if ff is None:
                output = "[fDC, sigDC_tickle]"
                cmd = "{:s} = {:s}.tickle([], 0);".format(output, self.optName)
                self.eng.eval(cmd, nargout=0)
                self._fDC = mat2py(self.eng.workspace['fDC'])
                self._sigDC_tickle = mat2py(self.eng.workspace['sigDC_tickle'])

            else:
                if noise:
                    output = "[fDC, sigDC_tickle, sigAC, mMech, noiseAC]"
                else:
                    output = "[fDC, sigDC_tickle, sigAC, mMech]"
                cmd = "{:s} = {:s}.tickle([], f);".format(output, self.optName)

                self.eng.eval(cmd, nargout=0)
                self._fDC = mat2py(self.eng.workspace['fDC'])
                self._sigDC_tickle = mat2py(self.eng.workspace['sigDC_tickle'])
                self._sigAC = mat2py(self.eng.workspace['sigAC'])
                self._mMech = mat2py(self.eng.workspace['mMech'])
                if noise:
                    self._noiseAC = mat2py(self.eng.workspace['noiseAC'])

        elif dof == 'pitch':
            if self._sigDC_tickle is None:
                # Compute the DC fields since they haven't been computed yet
                self.tickle(ff=None)
            cmd = "[sigAC_pitch, mMech_pitch] = {:}.tickle01([], f);".format(
                self.optName)
            self.eng.eval(cmd, nargout=0)
            self._sigAC_pitch = mat2py(self.eng.workspace['sigAC_pitch'])
            self._mMech_pitch = mat2py(self.eng.workspace['mMech_pitch'])

        elif dof == 'yaw':
            if self._sigDC_tickle is None:
                # Compute the DC fields since they haven't been computed yet
                self.tickle(ff=None)
            cmd = "[sigAC_yaw, mMech_yaw] = {:}.tickle10([], f);".format(
                self.optName)
            self.eng.eval(cmd, nargout=0)
            self._sigAC_yaw = mat2py(self.eng.workspace['sigAC_yaw'])
            self._mMech_yaw = mat2py(self.eng.workspace['mMech_yaw'])

    def sweeepLinear(self, startPos, endPos, npts):
        """Run Optickle's sweepLinear function

        Inputs:
          startPos: a dictionary with the starting positions of the drives
            to sweep
          endPos: a dictionary with the ending positions of the drives
            to sweep
          npts: number of points to sweep
        """
        self._tickled = True
        if isinstance(startPos, str):
            startPos = {startPos: 1}
        if isinstance(endPos, str):
            endPos = {endPos: 1}

        sPosMat = np.zeros(len(self.drives))
        ePosMat = np.zeros(len(self.drives))
        for driveName, drivePos in startPos.items():
            driveNum = self.drives.index(driveName)
            sPosMat[driveNum] = drivePos
        for driveName, drivePos in endPos.items():
            driveNum = self.drives.index(driveName)
            ePosMat[driveNum] = drivePos

        self.eng.workspace['startPos'] = py2mat(sPosMat)
        self.eng.workspace['endPos'] = py2mat(ePosMat)
        self.eng.workspace['npts'] = matlab.double([npts])

        cmd = "[poses, sigDC, fDC] = " + self.optName
        cmd += ".sweepLinear(startPos, endPos, npts);"

        self.eng.eval(cmd, nargout=0)
        self._poses = mat2py(self.eng.workspace['poses'])
        self._sigDC_sweep = mat2py(self.eng.workspace['sigDC'])
        self._fDC_sweep = mat2py(self.eng.workspace['fDC'])

    def getTF(self, probeName, driveNames):
        """Compute a transfer function

        Inputs:
          probeName: name of the probe at which the TF is calculated
          driveNames: names of the drives from which the TF is calculated

        Returns:
          tf: the transfer function

        Examples:
          * If only a single drive is used, the drive names can be a string
            To compute the phase transfer function in reflection from a FP
            cavity
              tf = opt.getTF('REFL', 'PM.drive')

          * If multiple drives are used, the drive names should be a dict
            To compute the DARM transfer function to the AS_DIFF PD
              DARM = {'EX.pos': 1, 'EY.pos': -1}
              tf = opt.getTF('AS_DIFF', DARM)
        """
        if self._sigAC is None:
            raise RuntimeError(
                'Must run tickle before calculating a transfer function.')

        probeNum = self.probes.index(probeName)
        if isinstance(driveNames, str):
            driveNames = {driveNames: 1}
        if isinstance(self._ff, Number):
            tf = 0
        else:
            tf = np.zeros(len(self._ff), dtype='complex')

        for driveName, drivePos in driveNames.items():
            driveNum = self.drives.index(driveName)
            try:
                tf += drivePos * self._sigAC[probeNum, driveNum]
            except IndexError:
                tf += drivePos * self._sigAC[driveNum]

        return tf

    def getAngularTF(self, probeName, driveName, dof, spotName=None,
                     spotPort=None):
        """Compute an angular transfer function

        Computes an angular transfer function from angle to beam-spot motion
        Note that this function includes the conversion to beam-spot motion
        which is not automatically included in Optickle

        Inputs:
          probeName: name of the probe at which the TF is calculated
          driveName: name of the drive from which the TF is calculated
          spotName: name of the optic which is probed
          spotPort: name of the port

        Returns:
          tf: the transfer function [rad/m]
        """
        if dof == 'pitch':
            sigAC = self._sigAC_pitch
        elif dof == 'yaw':
            sigAC = self._sigAC_yaw
        else:
            msg = 'Unrecognized angular degree of freedom {:s}'.format(dof)
            msg += '. Choose \'pitch\', or \'yaw\'.'
            raise ValueError(msg)

        # Should we convert to beam-spot motion?
        if not(spotName and spotPort):
            if (spotName is not None) or (spotPort is not None):
                msg = 'To convert to beam-spot motion, both the spot name' \
                      + ' and the spot port must be given'
                raise ValueError(msg)
            bsm = False
        else:
            bsm = True

        if sigAC is None:
            msg = 'Must run tickle for dof {:s}'.format(dof)
            msg += ' before calculating an angular transfer function'
            raise RuntimeError(msg)

        probeNum = self.probes.index(probeName)
        driveNum = self.drives.index(driveName)
        tf = sigAC[probeNum, driveNum]

        # Convert to beam-spot motion if necessary
        if bsm:
            spotNum = self.drives.index(spotName + '.pos')
            w, _, _, _ = self.getBeamProperties(spotName, spotPort)
            Pdc = self._sigDC_tickle[spotNum]
            tf *= w/(2*Pdc)

        return tf

    def getMechMod(self, driveOutName, driveInName):
        if self.mMech is None:
            raise ValueError(
                'Must run tickle before calculating mechanical modifications')

        driveInNum = self.drives.index(driveInName)
        driveOutNum = self.drives.index(driveOutName)
        return self._mMech[driveOutNum, driveInNum]

    def getQuantumNoise(self, probeName):
        """Compute the quantum noise at a probe

        Returns the quantum noise at a given probe in [W/rtHz]
        """
        probeNum = self.probes.index(probeName)
        try:
            qnoise = self._noiseAC[probeNum, :]
        except IndexError:
            qnoise = self._noiseAC[probeNum]
        return qnoise

    def plotTF(self, probeName, driveNames, mag_ax=None, phase_ax=None,
               dB=False, phase2freq=False, **kwargs):
        """Plot a transfer function.

        See documentation for plotTF in plotting
        """
        ff = self._ff
        tf = self.getTF(probeName, driveNames)
        fig = plotting.plotTF(ff, tf, mag_ax=mag_ax, phase_ax=phase_ax, dB=dB,
                              phase2freq=phase2freq, **kwargs)
        return fig

    def plotQuantumASD(self, probeName, driveNames, Larm=None, mass=None,
                       **kwargs):
        """Plot the quantum ASD of a probe

        Plots the ASD of a probe referenced the the transfer function for
        some signal, such as DARM. Optionally plots the SQL.

        Inputs:
          probeName: name of the probe
          driveNames: names of the drives from which the TF to refer the
            noise to
          Larm: If not None (default), divide by the arm length to convert
            displacement noise to strain noise
          mass: If not None (default) plots the SQL with the given mass [kg]
          **kwargs: extra keyword arguments to pass to the plot
        """
        ff = self._ff
        tf = self.getTF(probeName, driveNames)
        noiseASD = np.abs(self.getQuantumNoise(probeName)/tf)

        fig = plt.figure()
        if Larm is None:
            fig.gca().set_ylabel(
                r'Displacement $[\mathrm{m}/\mathrm{Hz}^{1/2}]$')
            Larm = 1
        else:
            fig.gca().set_ylabel(
                r'Strain $[1/\mathrm{Hz}^{1/2}]$')
        fig.gca().loglog(ff, noiseASD/Larm, **kwargs)

        if mass:
            hSQL = np.sqrt(8*scc.hbar/(mass*(2*np.pi*ff)**2))
            fig.gca().loglog(ff, hSQL/Larm, 'k--', label='SQL', alpha=0.7)
            fig.gca().legend()

        fig.gca().set_xlim([min(ff), max(ff)])
        fig.gca().set_xlabel('Frequency [Hz]')
        fig.gca().xaxis.grid(True, which='both', alpha=0.5)
        fig.gca().xaxis.grid(alpha=0.25, which='minor')
        fig.gca().yaxis.grid(True, which='both', alpha=0.5)
        fig.gca().yaxis.grid(alpha=0.25, which='minor')

        return fig

    def plotErrSig(self, probeName, driveName, quad=False, ax=None):
        if self._sigDC_sweep is None:
            raise ValueError(
                'Must run sweepLinear before plotting an error signal')
        if quad:
            probeNames = [probeName + '_I', probeName + '_Q']
        else:
            probeNames = [probeName]
        driveNum = self.drives.index(driveName)
        poses = self.poses[driveNum, :]
        if ax is None:
            fig = plt.figure()
            ax = fig.gca()
        # fig = plt.figure()
        for probeName in probeNames:
            probeNum = self.probes.index(probeName)
            sigDC = self._sigDC_[probeNum, :]
            ax.plot(poses, sigDC, label=probeName)
        # ax.plot(poses, np.zeros(len(poses)), 'C7--', alpha=0.5)
        ax.xaxis.grid(True, which='major', alpha=0.4)
        ax.yaxis.grid(True, which='major', alpha=0.4)
        # ax.axvline(0, color='C7', linestyle='--', alpha=0.5)
        ax.set_xlabel('drive position [m]')
        ax.set_ylabel('probe power [W]')
        ax.set_title('Sweeping ' + driveName)
        ax.legend()
        ax.set_xlim([min(poses), max(poses)])
        # return fig

    def getSweepPower(self, driveName, linkStart, linkEnd, fRF=0):
        """Get the power along a link between two optics as a drive is swept

        Inputs:
          linkStart: name of the start of the link
          linkEnd: name of the end of the link
          fRF: frequency of the sideband power to return [Hz]
            (Default: 0, i.e. the carrier)

        Returns:
          poses: the drive positions
          power: the power at those positions [W]
        """
        if self.fDCsweep is None:
            raise ValueError(
                'Must run sweepLinear before calculating sweep power')

        # find the link and the drive
        linkNum = self._getLinkNum(linkStart, linkEnd)
        driveNum = self.drives.index(driveName)

        poses = self.poses[driveNum, :]
        if self.vRF.size == 1:
            power = np.abs(self._fDC_sweep[linkNum, :])**2
        else:
            nRF = self._getSidebandInd(fRF)
            power = np.abs(self._fDC_sweep[linkNum, nRF, :])**2

        return poses, power

    def getSweepSignal(self, probeName, driveName):
        """Compute the signals as a drive is swept

        Inputs:
          probeName: name of the probe
          driveName: name of the drive

        Returns:
          poses: the positions of the drive
          sig: signal power at those positions [W]
        """
        probeNum = self.probes.index(probeName)
        driveNum = self.drives.index(driveName)

        poses = self._poses[driveNum, :]
        try:
            sig = self._sigDC_sweep[probeNum, :]
        except IndexError:
            sig = self._sigDC_sweep

        return poses, sig

    def getSigDC(self, probeName):
        """Get the DC power on a probe

        Inputs:
          probeName: the probe name

        Returns:
          power: the DC power on the probe [W]
        """
        probeNum = self.probes.index(probeName)
        return self._sigDC_tickle[probeNum]

    def getDCpower(self, linkStart, linkEnd, fRF=0):
        """Get the DC power along a link

        Inputs:
          linkStart: name of the start of the link
          linkEnd: name of the end of the link
          fRF: frequency of the sideband power to return [Hz]
            (Default: 0, i.e. the carrier)

        Returns:
          power: the DC power [W]
        """
        if self._fDC is None:
            raise ValueError(
                'Must run tickle before getting the DC power levels.')

        linkNum = self._getLinkNum(linkStart, linkEnd)
        if self.vRF.size == 1:
            power = np.abs(self._fDC[linkNum])**2
        else:
            nRF = self._getSidebandInd(fRF)
            power = np.abs(self._fDC[linkNum, nRF])**2

        return power

    def showfDC(self):
        """Print the DC power at each link in the model
        """
        pad2 = 7
        nLink = int(self.eng.eval(self.optName + ".Nlink;"))
        links = []
        for ii in range(nLink):
            sourceName = self._getSourceName(ii + 1)
            sinkName = self._getSinkName(ii + 1)
            links.append(sourceName + ' --> ' + sinkName)
        pad = max([len(link) for link in links])
        l1 = pad
        try:
            nRF = len(self.vRF)
        except TypeError:
            nRF = 1
        l2 = (pad2 + 3)*nRF
        print('{:<{l1}s}|'.format('Link', l1=l1) + utils.printHeader(
            self.vRF, pad2 - 1))
        for li, link in enumerate(links):
            line = '{:{pad}s}|'.format(link, pad=pad)
            if nRF == 1:
                line += utils.printLine([self._fDC[li]], pad2)
            else:
                line += utils.printLine(self._fDC[li, :], pad2)
            if li % 5 == 0:
                print('{:_<{length}}'.format('', length=int(l1 + l2 + 1)))
            print(line)

    def showsigDC(self):
        """Print the DC power at each probe
        """
        pad1 = max([len('Probe '),
                    max([len(probe) for probe in self.probes])])
        pad2 = 10
        print('{:<{pad1}s}| {:<{pad2}s}|'.format(
            'Probe', 'Power', pad1=pad1, pad2=pad2))
        print('{:_<{length}}'.format('', length=int(pad1 + pad2 + 3)))
        if isinstance(self.probes, str):
            probes = [self.probes]
        else:
            probes = self.probes
        for pi, probe in enumerate(probes):
            try:
                pref, num = utils.siPrefix(self._sigDC_tickle[pi])
            except IndexError:
                pref, num = utils.siPrefix(self._sigDC_tickle)
            pad3 = pad2 - len(pref) - 2
            print('{:{pad1}s}| {:{pad3}.1f} {:s}W|'.format(
                probe, num, pref, pad1=pad1, pad3=pad3))

    def addMirror(self, name, aoi=0, Chr=0, Thr=0, Lhr=0,
                  Rar=0, Lmd=0, Nmd=1.45):
        """Add a mirror to the model

        Inputs:
          name: Name of the mirror
          aoi: angle of incidence [deg] (Default: 0)
          Chr: inverse radius of curvature (Default: 0)
          Thr: power transmisivity (Default: 0)
          Lhr: HR loss (Default: 0)
          Rar: AR reflectivity (Default: 0)
          Lmd: loss through one pass of the material (Default: 0)
          Nmd: index of refraction (Default: 1.45)
        """
        args = OrderedDict([
            ('aoi', aoi), ('Chr', Chr), ('Thr', Thr), ('Lhr', Lhr),
            ('Rar', Rar), ('Lmd', Lmd), ('Nmd', Nmd)])
        cmd = self.optName + ".addMirror(" + str2mat(name)
        for key, val in args.items():
            self.eng.workspace[key] = py2mat(val)
            cmd += ", " + key
        cmd += ");"
        self.eng.eval(cmd, nargout=0)
        self._updateNames()

    def addBeamSplitter(self, name, aoi=45, Chr=0, Thr=0.5, Lhr=0,
                        Rar=0, Lmd=0, Nmd=1.45):
        """Add a beamsplitter to the model

        Inputs:
          name: Name of the mirror
          aoi: angle of incidence [deg] (Default: 45)
          Chr: inverse radius of curvature (Default: 0)
          Thr: power transmisivity (Default: 0.5)
          Lhr: HR loss (Default: 0)
          Rar: AR reflectivity (Default: 0)
          Lmd: loss through one pass of the material (Default: 0)
          Nmd: index of refraction (Default: 1.45)
        """
        args = OrderedDict([
            ('aoi', aoi), ('Chr', Chr), ('Thr', Thr), ('Lhr', Lhr),
            ('Rar', Rar), ('Lmd', Lmd), ('Nmd', Nmd)])
        cmd = self.optName + ".addBeamSplitter(" + str2mat(name)
        for key, val in args.items():
            self.eng.workspace[key] = py2mat(val)
            cmd += ", " + key
        cmd += ");"
        self.eng.eval(cmd, nargout=0)
        self._updateNames()

    def addPBS(self, name, aoi=45, Chr=0, transS=0, reflP=0, Lhr=0, Rar=0,
               Lmd=0, Nmd=1.45, BS=False):
        """Add a polarizing beamsplitter.

        Need to make this function a bit more general to handle the case
        of dichroic pytickle models

        Inputs:
          reflS: power reflectivity of S-pol (default: 0)
          transP: power transmisivity of P-pol (default: 0)
          BS: if True, adds the PBS as a BeamSplitter
            if False, adds it as a mirror (default: False)
        """
        Thr = [[transS, self.lambda0, 1],
               [1 - reflP, self.lambda0, 0]]
        if BS:
            self.addBeamSplitter(name, aoi, Chr, Thr, Lhr, Rar, Lmd, Nmd)
        else:
            self.addMirror(name, aoi, Chr, Thr, Lhr, Rar, Lmd, Nmd)

    def addLink(self, nameOut, portOut, nameIn, portIn, linkLen):
        """Add a link

        Inputs:
          nameOut: name of the outgoing optic
          portOut: name of the outgoing port
          nameIn: name of the ingoing optic
          portIn: name of the ingoing port
          linkLen: length of the link [m]
        """
        cmd = self.optName + ".addLink("
        cmd += str2mat(nameOut) + ", " + str2mat(portOut)
        cmd += ", " + str2mat(nameIn) + ", " + str2mat(portIn)
        cmd += ", " + str(linkLen) + ");"
        self.eng.eval(cmd, nargout=0)

    def addSource(self, name, ampRF):
        """Add a laser

        Inputs:
          name: name of the laser
          ampRF: amplitude of each RF component [sqrt(W)]
        """
        if isinstance(ampRF, np.ndarray):
            self.eng.workspace['ampRF'] = py2mat(ampRF)
        else:
            self.eng.workspace['ampRF'] = py2mat([ampRF])
        cmd = self.optName + ".addSource("
        cmd += str2mat(name) + ", ampRF);"
        self.eng.eval(cmd, nargout=0)
        self._updateNames()

    def addSqueezer(self, name, lambda0=1064e-9, fRF=0, pol='S', sqAng=0,
                    **sqzKwargs):
        """Add a squeezer

        Inputs:
          name: name of the squeezer
          lambda0: wavelength to be squeezed [m] (Default: 1064e-9)
          fRF: RF sideband to be squeezed [Hz] (Default: 0)
          pol: polarization to be squeezed (Default: 'S')
          sqAng: squeezing angle [deg] (Default: 0, i.e. amplitude)
          sqdB: amount of squeezing in dB at OPO output (Default: 10 dB)
          antidB: amount of antisqueezing in dB at OPO output (Default 10 dB)
          x: normalized nonlinear interaction strength (Default: 1)
          escEff: escape efficiency (Default: 1)

        The squeezer can be added in one of two ways:
        1) Specify the squeezing (sqdB) and anti-squeezing (antidB) amplitudes
        2) Specify the normalized nonlinear interaction strength (x) and
           escape efficiency (escEff)

        Default is to specify squeezing levels. If only sqdB is given, the
        squeezing and anti-squeezing levels are the same.

        The angle convention is that 0 deg is amplitude and 90 deg is phase.

        Examples:
        1) add a 10 dB amplitude squeezed field:
            opt.addSqueezer('sqz', sqdB=10)
        2) add a 10 db squeezed, 15 db anti-squeezed field in phase quadrature:
            opt.addSqueezer('sqz', sqdB=10, antidB=15, sqAng=90)
        3) add an amplitude squeezed field with escape efficiency 0.9
           and nonlinear gain 0.5 at 45 deg:
            opt.addSqueezer('sqz', sqAng=45, x=0.5, escEff=0.9)
        """
        # parse squeezing definitions
        sqAng = sqAng * np.pi/180
        for kwarg in sqzKwargs:
            if kwarg not in ['sqdB', 'antidB', 'x', 'escEff']:
                raise ValueError('Unrecognized keyword ' + str(kwarg))
        msg = 'Cannot specify both sqdB/antidB and x/escEff'
        if len(sqzKwargs) == 0:
            sqzOption = 0
            sqdB = 10
            antidB = 10

        if ('sqdB' in sqzKwargs) or ('antidB' in sqzKwargs):
            if ('x' in sqzKwargs) or ('escEff' in sqzKwargs):
                raise ValueError(msg)
            sqzOption = 0
            try:
                sqdB = sqzKwargs['sqdB']
            except KeyError:
                sqdB = 10
            try:
                antidB = sqzKwargs['antidB']
            except KeyError:
                antidB = sqdB

        elif ('x' in sqzKwargs) or ('escEff' in sqzKwargs):
            if ('sqdB' in sqzKwargs) or ('antidB' in sqzKwargs):
                raise ValueError(msg)
            sqzOption = 1
            try:
                x = sqzKwargs['x']
            except KeyError:
                x = 0.5195
            try:
                escEff = sqzKwargs['escEff']
            except KeyError:
                escEff = 1

        # add squeezer to model
        self.eng.workspace['lambda0'] = py2mat(lambda0)
        self.eng.workspace['fRF'] = py2mat(fRF)
        self.eng.workspace['npol'] = py2mat(self._pol2opt(pol))
        self.eng.workspace['sqAng'] = py2mat(sqAng)
        self.eng.workspace['sqzOption'] = py2mat(sqzOption)

        cmd = self.optName + ".addSqueezer(" + str2mat(name)
        cmd += ", lambda0, fRF, npol, sqAng, "

        if sqzOption == 0:
            self.eng.workspace['sqdB'] = py2mat(sqdB)
            self.eng.workspace['antidB'] = py2mat(antidB)
            cmd += "sqdB, antidB, sqzOption);"
        elif sqzOption == 1:
            self.eng.workspace['x'] = py2mat(x)
            self.eng.workspace['escEff'] = py2mat(escEff)
            cmd += "x, escEff, sqzOption);"
        self.eng.eval(cmd, nargout=0)

    def addWaveplate(self, name, lfw, theta):
        """Add a waveplate

        Need to make this function a bit more general to handle the case
        of dichroic pytickle models

        Inputs:
          name: name of the waveplate
          lfw: waveplate fraction of a wavelength
          theta: rotation angle of waveplate [deg]
        """
        self.eng.workspace['lfw'] = py2mat(lfw)
        self.eng.workspace['theta'] = py2mat(theta)
        cmd = self.optName + ".addWaveplate("
        cmd += str2mat(name) + ", lfw, theta);"
        self.eng.eval(cmd, nargout=0)
        self._updateNames()

    def addModulator(self, name, cMod):
        """Add a modulator

        Inputs:
          name: name of the modulator
          cMod: modulation index (amplitude: 1; phase: 1j)
        """
        cmd = self.optName + ".addModulator(" + str2mat(name)
        cmd += ", " + str(cMod) + ");"
        self.eng.eval(cmd, nargout=0)
        self._updateNames()

    def addRFmodulator(self, name, fMod, aMod):
        """Add an RF modulator

        Inputs:
          name: name of the modulator
          fMod: frequency of the modulated RF component [Hz]
          aMod: modulation amplitude (real for amplitude; complex for phase)
        """
        cmd = self.optName + ".addRFmodulator(" + str2mat(name)
        cmd += ", " + str(fMod) + ", " + str(aMod) + ");"
        self.eng.eval(cmd, nargout=0)
        self._updateNames()

    def addSink(self, name, loss=1):
        cmd = self.optName + ".addSink(" + str2mat(name)
        cmd += ", " + str(loss) + ");"
        self.eng.eval(cmd, nargout=0)
        self._updateNames()

    def addProbeIn(self, name, sinkName, sinkPort, freq, phase):
        if self._rotatedBasis:
            msg = ('The probe basis has already been rotated to do homodyne'
                   + ' detection. No further probes can be added.\n'
                   + 'Rewrite the model so that all probes are added before'
                   + ' the probe basis is rotated. (Some instances of '
                   + ' addHomodyneReadout may have to be called with '
                   + 'rotateBasis=False.')
            raise RuntimeError(msg)
        cmd = self.optName + ".addProbeIn("
        cmd += str2mat(name) + ", " + str2mat(sinkName)
        cmd += ", " + str2mat(sinkPort) + ", "
        cmd += str(freq) + ", " + str(phase) + ");"
        self.eng.eval(cmd, nargout=0)
        self._updateNames()

    def addReadout(self, sinkName, freqs, phases, names=None):
        # Figure out the naming scheme.
        if isinstance(freqs, Number):
            freqs = np.array([freqs])
        if isinstance(phases, Number):
            phases = np.array([phases])
        if len(freqs) != len(phases):
            raise ValueError(
                'Frequency and phase vectors are not the same length.')
        if names is None:
            if len(freqs) > 1:
                names = (np.arange(len(freqs), dtype=int) + 1).astype(str)
            else:
                names = ['']
        elif isinstance(names, str):
            names = [names]
        # Add the probes.
        self.addProbeIn(sinkName + '_DC', sinkName, 'in', 0, 0)
        for freq, phase, name in zip(freqs, phases, names):
            nameI = sinkName + '_I' + name
            nameQ = sinkName + '_Q' + name
            self.addProbeIn(nameI, sinkName, 'in', freq, phase)
            self.addProbeIn(nameQ, sinkName, 'in', freq, phase + 90)

    def setPosOffset(self, name, dist):
        cmd = self.optName + ".setPosOffset(" + str2mat(name)
        cmd += ", " + str(dist) + ");"
        self.eng.eval(cmd, nargout=0)

    def setMechTF(self, name, z, p, k, dof='pos'):
        """Set the mechanical transfer function of an optic

        The transfer function is from radiation pressure to one of the degrees
        of freedom position, pitch, or yaw.

        Inputs:
          name: name of the optic
          p: poles
          k: gain
          dof: degree of freedom: pos, pitch, or yaw (default: pos)
        """
        if dof == 'pos':
            nDOF = 1
        elif dof == 'pitch':
            nDOF = 2
        elif dof == 'yaw':
            nDOF = 3
        else:
            msg = 'Unrecognized degree of freedom ' + str(dof)
            msg += '. Choose \'pos\', \'pitch\', or \'yaw\'.'
            raise ValueError(msg)

        self.eng.workspace['z'] = py2mat(z, is_complex=True)
        self.eng.workspace['p'] = py2mat(p, is_complex=True)
        self.eng.workspace['k'] = py2mat(k, is_complex=True)
        self.eng.workspace['nDOF'] = py2mat(nDOF)

        self.eng.eval("tf = zpk(z, p, k);", nargout=0)
        cmd = self.optName + ".setMechTF(" + str2mat(name) + ", tf, nDOF);"
        self.eng.eval(cmd, nargout=0)

    def addHomodyneReadout(
            self, name, phase, qe=1, BnC=True, LOpower=1, nu=None, pol='S',
            rotateBasis=True):
        # Add homodyne optics.
        self.addMirror(name + '_BS', Thr=0.5, aoi=45)
        self.addSink(name + '_attnA', 1 - qe)
        self.addSink(name + '_attnB', 1 - qe)
        self.addSink(name + '_A', 1)
        self.addSink(name + '_B', 1)
        if LOpower == 0:
            self.addMirror(name + '_LOphase', aoi=0, Thr=0)
        # Add LO if necessary.
        if LOpower > 0:
            if nu is None:
                freqs = mat2py(self.eng.eval(self.optName + ".nu;"))
                if len(freqs.shape) == 0:
                    nu = freqs
                else:
                    nu = freqs[0]
            npol = self._pol2opt(pol)
            self.eng.workspace['nu'] = py2mat(nu)
            cmd = "Optickle.matchFreqPol(" + self.optName + ", nu"
            cmd += ", " + str(npol) + ");"
            ampRF = self.eng.eval(cmd)
            ampRF = np.sqrt(LOpower)*ampRF
            self.addSource(name + '_LO', ampRF)
            self.setHomodynePhase(name + '_LO', phase, BnC)
        # Add detectors.
        if LOpower == 0:
            self.addLink(name + '_LOphase', 'fr', name + '_BS', 'bk', 0)
        else:
            self.addLink(name + '_LO', 'out', name + '_BS', 'bk', 0)
        self.addLink(name + '_BS', 'fr', name + '_attnA', 'in', 0)
        self.addLink(name + '_attnA', 'out', name + '_A', 'in', 0)
        self.addLink(name + '_BS', 'bk', name + '_attnB', 'in', 0)
        self.addLink(name + '_attnB', 'out', name + '_B', 'in', 0)
        self.addProbeIn(name + '_SUM', name + '_A', 'in', 0, 0)
        self.addProbeIn(name + '_DIFF', name + '_B', 'in', 0, 0)
        # Rotate probe basis to do homodyne detection if necessary.
        if rotateBasis:
            self.rotateHomodyneBasis(name)

    def setHomodynePhase(self, LOname, phase, BnC=False):
        if BnC:
            phase = np.pi*(90 - phase)/180
        else:
            phase = np.pi*phase/180
        cmd = "LO = " + self.optName + ".getOptic(" + str2mat(LOname) + ");"
        self.eng.eval(cmd, nargout=0)
        self.eng.eval("RFamp = abs(LO.vArf);", nargout=0)
        self.eng.eval("LO.vArf = RFamp * exp(1i*" + str(phase) + ");",
                      nargout=0)

    def rotateHomodyneBasis(self, probes):
        if isinstance(probes, str):
            probes = [probes]
        mProbeOut = np.identity(len(self.probes))
        for probe in probes:
            nA = self.probes.index(probe + '_SUM')
            nB = self.probes.index(probe + '_DIFF')
            mProbeOut[nA, nA] = 1
            mProbeOut[nB, nA] = -1
            mProbeOut[nA, nB] = 1
            mProbeOut[nB, nB] = 1
        self.setProbeBasis(mProbeOut)

    def setProbeBasis(self, mProbeOut):
        """Set the probe basis
        """
        self.eng.workspace['mProbeOut'] = py2mat(mProbeOut)
        self.eng.eval(self.optName + ".mProbeOut = mProbeOut;", nargout=0)
        self._rotatedBasis = True

    def setCavityBasis(self, name1, name2):
        """Set the cavity basis for a two mirror cavity

        Inputs:
          name1, name2: the names of the optics
        """
        cmd = self.optName + ".setCavityBasis("
        cmd += str2mat(name1) + ", " + str2mat(name2) + ");"
        self.eng.eval(cmd, nargout=0)

    def setOpticParam(self, name, param, val):
        """Set the value of an optic's parameter

        Inputs:
          name: name of the optic
          param: name of the parameter
          val: value of the paramter
        """
        cmd = self.optName + ".setOpticParam("
        cmd += str2mat(name) + ", " + str2mat(param)
        cmd += ", " + str(val) + ");"
        self.eng.eval(cmd, nargout=0)

    def getOpticParam(self, name, param):
        """Get the value of an optic's paramter

        Inputs:
          name: name of the optic
          param: name of the parameter
        """
        cmd = self.optName + ".getOptic(" + str2mat(name) + ");"
        self.eng.eval(cmd, nargout=0)
        return self.eng.eval("ans." + param + ";")

    def getProbePhase(self, name):
        """Get the phase of a probe

        Inputs:
          name: name of the probe

        Returns:
          phase: phase of the probe [deg]
        """
        cmd = self.optName + ".getProbePhase(" + str2mat(name) + ");"
        return self.eng.eval(cmd, nargout=1)

    def setProbePhase(self, name, phase):
        """Set the phase of a probe

        Inputs:
          name: name of the probe
          phase: phase of the probe [deg]
        """
        cmd = self.optName + ".setProbePhase("
        cmd += str2mat(name) + ", " + str(phase) + ");"
        self.eng.eval(cmd, nargout=0)

    def getLinkLength(self, linkStart, linkEnd):
        """Get the length of a link

        Inputs:
          linkStart: name of the start of the link
          linkEnd: name of the end of the link

        Returns:
          linkLen: the length of the link [m]
        """
        linkLens = self.eng.eval(self.optName + ".getLinkLengths", nargout=1)
        linkLens = mat2py(linkLens)
        linkNum = self._getLinkNum(linkStart, linkEnd)
        return linkLens[linkNum]

    def setLinkLength(self, linkStart, linkEnd, linkLen):
        """Set a link length after the model is defined

        Inputs:
          linkStart: name of the start of the link
          linkEnd: name of the end of the link
          linkLen: new length of the link [m]
        """
        # add 1 to convert to Matlab's 1-based system
        linkNum = self._getLinkNum(linkStart, linkEnd) + 1
        self.eng.workspace['linkNum'] = py2mat(linkNum)
        self.eng.workspace['linkLen'] = py2mat(linkLen)
        self.eng.eval(
            self.optName + ".setLinkLength(linkNum, linkLen);", nargout=0)

    def getBeamProperties(self, name, port):
        """Compute the properties of a Gaussian beam at an optic

        Inputs:
          name: name of the optic
          port: name of the port

        Returns:
          w: beam radius on the optic [m]
          z: distance from the beam waist to the optic [m]
            Negative values indicate that the optic is before the waist.
          z0: Rayleigh range of the beam [m]
          R: radius of curvature of the phase front on the optic [m]
        """
        cmd = "[w, z0, z, R] = {:s}.getBeamSize({:s}, {:s})".format(
            self.optName, str2mat(name), str2mat(port))
        self.eng.eval(cmd, nargout=0)
        w = mat2py(self.eng.workspace['w'])
        z0 = mat2py(self.eng.workspace['z0'])
        z = mat2py(self.eng.workspace['z'])
        R = mat2py(self.eng.workspace['R'])
        return w, z0, z, R

    def _getSourceName(self, linkNum):
        """Find the name of the optic that is the source for a link

        Inputs:
          linkNum: the link number
        """
        cmd = self.optName + ".getSourceName(" + str(linkNum) + ");"
        return self.eng.eval(cmd, nargout=1)

    def _getSinkName(self, linkNum):
        """Find the name of the optic that is the sink for a link

        Inputs:
          linkNum: the link number
        """
        cmd = self.optName + ".getSinkName(" + str(linkNum) + ");"
        return self.eng.eval(cmd, nargout=1)

    def _getLinkNum(self, linkStart, linkEnd):
        """Find the link number of a particular link in the Optickle model

        Inputs:
          linkStart: name of the start of the link
          linkEnd: name of the end of the link

        Returns:
          linkNum: the link number converted to pytickle's 0-based system
        """
        linkStart = str2mat(linkStart)
        linkEnd = str2mat(linkEnd)
        linkNum = self.eng.eval(
            self.optName + ".getLinkNum(" + linkStart + ", " + linkEnd + ");")
        try:
            linkNum = int(linkNum) - 1
        except TypeError:
            msg = "Link from " + linkStart + " to " + linkEnd
            msg += " does not appear to exist."
            raise ValueError(msg)
        return linkNum

    def _updateNames(self):
        """Refresh the pytickle model's list of probe and drive names
        """
        self.probes = self.eng.eval(self.optName + ".getProbeName")
        self.drives = self.eng.eval(self.optName + ".getDriveNames")

    def _pol2opt(self, pol):
        """Convert S and P polarizations to 1s and 0s for Optickle
        """
        if pol == 'S':
            return 1
        elif pol == 'P':
            return 0
        else:
            raise ValueError('Unrecognized polarization ' + str(pol)
                             + '. Use \'S\' or \'P\'')

    def _getSidebandInd(self, freq, tol=1):
        """Find the index of an RF sideband frequency

        Inputs:
          freq: the frequency of the desired sideband
          tol: tolerance of the difference between freq and the RF sideband
            of the model [Hz] (Default: 1 Hz)

        Returns:
          nRF: the index of the RF sideband
        """
        # FIXME: add support for multiple colors and polarizations
        ind = np.nonzero(np.isclose(self.vRF, freq, atol=tol))[0]
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
