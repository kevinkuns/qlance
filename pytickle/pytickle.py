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
import misc


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
      opt: the name for the optical model (Default: 'opt')
      vRF: vector of RF frequencies [Hz] (Default: 0)
      lambda0: carrier wavelength [m] (Default: 1064e-9)
      pol: polarization (Default: 'S')
    """
    def __init__(self, eng, opt='opt', vRF=0, lambda0=1064e-9, pol='S'):
        # convert RF and polarization information
        if isinstance(vRF, Number):
            vRF = [vRF]
        if isinstance(pol, str) or isinstance(pol, Number):
            pol = [pol]
        if len(vRF) != len(pol):
            raise ValueError(
                'RF vector and polarization vector must be the same length')

        # convert S and P definitions to 1s and 0s for Optickle
        matPol = [self._pol2opt(k) for k in pol]

        # initialize the Optickle model in the matlab engine
        self.eng = eng
        self.opt = opt
        self.eng.workspace['vRF'] = py2mat(vRF)
        self.eng.workspace['lambda0'] = py2mat(lambda0)
        self.eng.workspace['pol'] = py2mat(matPol)
        self.eng.eval(self.opt + " = Optickle(vRF, lambda0, pol);", nargout=0)

        # initialize pytickle class variables
        self.lambda0 = lambda0
        self.vRF = mat2py(self.eng.eval(self.opt + ".vFrf"))
        self.pol = np.array(pol)
        self.nRF = OrderedDict()
        for ri, fRF in enumerate(vRF):
            pref, num = misc.siPrefix(round(fRF))
            if num == 0:
                key = 'DC'
            else:
                key = str(int(round(num))) + ' ' + pref + 'Hz'
            self.nRF[key] = ri
        self.f = None
        self.fDC = None
        self.sigAC = None
        self.sigDCtickle = None
        self.noiseAC = None
        self.poses = None
        self.sigDC = None
        self.fDCsweep = None
        self._rotatedBasis = False

    def tickle(self, f, noise=True):
        """Compute the Optickle model (Runs Optickle's tickle function)

        Inputs:
          f: frequency vector [Hz]
          noise: If False, the quantum noise is not computed which can decrease
            runtime. Default: True
        """
        self.f = f
        self.eng.workspace['f'] = py2mat(f)
        if noise:
            output = "[fDC, sigDCtickle, sigAC, ~, noiseAC] = "
        else:
            output = "[fDC, sigDCtickle, sigAC] = "
        self.eng.eval(output + self.opt + ".tickle([], f);", nargout=0)
        self.fDC = mat2py(self.eng.workspace['fDC'])
        self.sigDCtickle = mat2py(self.eng.workspace['sigDCtickle'])
        self.sigAC = mat2py(self.eng.workspace['sigAC'])
        if noise:
            self.noiseAC = mat2py(self.eng.workspace['noiseAC'])

    def sweeepLinear(self, driveNames, npts):
        """Run Optickle's sweepLinear function
        """
        if isinstance(driveNames, str):
            driveNames = {driveNames: 1}
        pos = np.zeros(len(self.drives))
        for driveName, drivePos in driveNames.items():
            driveNum = self.drives.index(driveName)
            pos[driveNum] = drivePos
        self.eng.workspace['pos'] = py2mat(pos)
        self.eng.workspace['npts'] = matlab.double([npts])
        output = "[poses, sigDC, fDC] = "
        self.eng.eval(
            output + self.opt + ".sweepLinear(-pos, pos, npts);", nargout=0)
        self.poses = mat2py(self.eng.workspace['poses'])
        self.sigDC = mat2py(self.eng.workspace['sigDC'])
        self.fDCsweep = mat2py(self.eng.workspace['fDC'])

    def getTF(self, probeName, driveNames, phase2freq=False):
        """Compute a transfer function

        Inputs:
          probeName: name of the probe at which the TF is calculated
          driveNames: names of the drives from which the TF is calculated
          phase2freq: If True, a phase transfer function is divided by the
            frequency to return a frequency transfer function (Default: False)

        Returns:
          tf: the transfer function

        Examples:
          * If only a single drive is used, the drive names can be a string
            To compute the phase transfer function in reflection from a FP
            cavity
              tf = opt.getTF('REFL', 'PM.drive')
            To get the frequency transfer function, call getTF with
            phase2freq=True

          * If multiple drives are used, the drive names should be a dict
            To compute the DARM transfer function to the AS_DIFF PD
              DARM = {'EX.pos': 1, 'EY.pos': -1}
              tf = opt.getTF('AS_DIFF', DARM)
        """
        if self.sigAC is None:
            raise ValueError(
                'Must run tickle before calculating a transfer function.')

        probeNum = self.probes.index(probeName)
        if isinstance(driveNames, str):
            driveNames = {driveNames: 1}
        if isinstance(self.f, Number):
            tf = 0
        else:
            tf = np.zeros(len(self.f), dtype='complex')

        for driveName, drivePos in driveNames.items():
            driveNum = self.drives.index(driveName)
            try:
                tf += drivePos * self.sigAC[probeNum, driveNum]
            except IndexError:
                tf += drivePos * self.sigAC[driveNum]

        if phase2freq:
            tf = tf/self.f

        return tf

    def getQuantumNoise(self, probeName):
        """Compute the quantum noise at a probe

        Returns the quantum noise at a given probe in [W/rtHz]
        """
        probeNum = self.probes.index(probeName)
        try:
            qnoise = self.noiseAC[probeNum, :]
        except IndexError:
            qnoise = self.noiseAC[probeNum]
        return qnoise

    def plotTF(self, probeName, driveNames, mag_ax=None, phase_ax=None,
               dB=False, phase2freq=False, **kwargs):
        """Plot a transfer function.

        See documentation for plotTF in plotting
        """
        f = self.f
        tf = self.getTF(probeName, driveNames)
        fig = plotting.plotTF(f, tf, mag_ax=mag_ax, phase_ax=phase_ax, dB=dB,
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
        f = self.f
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
        fig.gca().loglog(f, noiseASD/Larm, **kwargs)

        if mass:
            hSQL = np.sqrt(8*scc.hbar/(mass*(2*np.pi*f)**2))
            fig.gca().loglog(f, hSQL/Larm, 'k--', label='SQL', alpha=0.7)
            fig.gca().legend()

        fig.gca().set_xlim([min(f), max(f)])
        fig.gca().set_xlabel('Frequency [Hz]')
        fig.gca().xaxis.grid(True, which='both', alpha=0.5)
        fig.gca().xaxis.grid(alpha=0.25, which='minor')
        fig.gca().yaxis.grid(True, which='both', alpha=0.5)
        fig.gca().yaxis.grid(alpha=0.25, which='minor')

        return fig

    def plotErrSig(self, probeName, driveName, quad=False, ax=None):
        if self.sigDC is None:
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
            sigDC = self.sigDC[probeNum, :]
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

    def plotSweepPower(self, driveName, linkStart, linkEnd, nRFs='all',
                       wavelength=False, ax=None):
        """Plot the power along a link between two optics as a drive is swept

        Inputs:
          driveName: name of the drive to sweep
          linkStart: name of the start of the link
          linkEnd: name of the end of the link
          nRFs: which RF components to plot. (Default: 'all')
            If nRFs = 'all', all RF components are plotted
            If nRFs is a number or list of numbers, only those RF components
            are plotted
          wavelength: if True, the drive position is plotted in units of
            wavelength instead of meters (Default: False)
          ax: axis to plot the sweeps on. If None, a new figure is created.
            (Default: None)

        Returns:
          fig: If ax is None, returns new figure

        Example:
          To plot the power in the SRC as the SRM is swept:
            opt.sweepLinear({'SRM.pos': 1064e-9}, 1000)
            fig = opt.plotSweepPower('SRM.pos', 'SRM', 'SR2')
        """
        if self.fDCsweep is None:
            raise ValueError(
                'Must run sweepLinear before plotting sweep power')
        if nRFs == 'all':
            nRFs = range(self.vRF.size)
        elif isinstance(nRFs, Number):
            nRFs = [nRFs]

        # find the link and the drive
        linkStart = str2mat(linkStart)
        linkEnd = str2mat(linkEnd)
        linkNum = self.eng.eval(
            self.opt + ".getLinkNum(" + linkStart + ", " + linkEnd + ");")
        try:
            linkNum = int(linkNum) - 1
        except TypeError:
            msg = "Link from " + linkStart + " to " + linkEnd
            msg += " does not appear to exist."
            raise ValueError(msg)
        driveNum = self.drives.index(driveName)

        poses = self.poses[driveNum, :]
        if wavelength:
            poses = poses / self.lambda0

        if ax is None:
            fig = plt.figure()
            ax = fig.gca()
            newfig = False
        else:
            newfig = True

        if self.vRF.size == 1:
            power = np.abs(self.fDCsweep[linkNum, :])**2
            ax.semilogy(poses, power, label='0')
        else:
            for nRF in nRFs:
                power = np.abs(self.fDCsweep[linkNum, nRF, :])**2
                ax.semilogy(poses, power, label=str(nRF))
        if wavelength:
            ax.set_xlabel('Drive position [wavelengths]')
        else:
            ax.set_xlabel('Drive position [m]')
        ax.set_ylabel('Power [W]')
        ax.legend()
        ax.set_xlim(poses[0], poses[-1])
        ax.set_title('Sweeping {:s}; Power in link from {:s} to {:s}'.format(
            driveName, linkStart, linkEnd))

        if newfig:
            return fig

    def showfDC(self):
        """Print the DC power at each link in the model
        """
        pad2 = 7
        nLink = int(self.eng.eval(self.opt + ".Nlink;"))
        links = []
        for ii in range(nLink):
            sourceName = self.getSourceName(ii + 1)
            sinkName = self.getSinkName(ii + 1)
            links.append(sourceName + ' --> ' + sinkName)
        pad = max([len(link) for link in links])
        l1 = pad
        try:
            nRF = len(self.vRF)
        except TypeError:
            nRF = 1
        l2 = (pad2 + 3)*nRF
        print('{:<{l1}s}|'.format('Link', l1=l1) + misc.printHeader(
            self.vRF, pad2 - 1))
        for li, link in enumerate(links):
            line = '{:{pad}s}|'.format(link, pad=pad)
            if nRF == 1:
                line += misc.printLine([self.fDC[li]], pad2)
            else:
                line += misc.printLine(self.fDC[li, :], pad2)
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
                pref, num = misc.siPrefix(self.sigDCtickle[pi])
            except IndexError:
                pref, num = misc.siPrefix(self.sigDCtickle)
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
        cmd = self.opt + ".addMirror(" + str2mat(name)
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
        cmd = self.opt + ".addBeamSplitter(" + str2mat(name)
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
        cmd = self.opt + ".addLink("
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
        cmd = self.opt + ".addSource("
        cmd += str2mat(name) + ", ampRF);"
        self.eng.eval(cmd, nargout=0)
        self._updateNames()

    def addSqueezer(self, name, lambda0=1064e-9, fRF=0, pol='S', sqzAng=0,
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
            opt.addSqueezer('sqz', sqzAng=45, x=0.5, escEff=0.9)
        """
        # parse squeezing definitions
        sqzAng = sqzAng * np.pi/180
        for kwarg in sqzKwargs:
            if kwarg not in ['sqdB', 'antidB', 'x', 'escEff']:
                raise ValueError('Unrecognized keyword + ' + str(kwarg))
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
        self.eng.workspace['sqzAng'] = py2mat(sqzAng)
        self.eng.workspace['sqzOption'] = py2mat(sqzOption)

        cmd = self.opt + ".addSqueezer(" + str2mat(name)
        cmd += ", lambda0, fRF, npol, sqzAng, "

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
        cmd = self.opt + ".addWaveplate("
        cmd += str2mat(name) + ", lfw, theta);"
        self.eng.eval(cmd, nargout=0)
        self._updateNames()

    def addModulator(self, name, cMod):
        """Add a modulator

        Inputs:
          name: name of the modulator
          cMod: modulation index (amplitude: 1; phase: 1j)
        """
        cmd = self.opt + ".addModulator(" + str2mat(name)
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
        cmd = self.opt + ".addRFmodulator(" + str2mat(name)
        cmd += ", " + str(fMod) + ", " + str(aMod) + ");"
        self.eng.eval(cmd, nargout=0)
        self._updateNames()

    def addSink(self, name, loss=1):
        cmd = self.opt + ".addSink(" + str2mat(name)
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
        cmd = self.opt + ".addProbeIn("
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
        cmd = self.opt + ".setPosOffset(" + str2mat(name)
        cmd += ", " + str(dist) + ");"
        self.eng.eval(cmd, nargout=0)

    def setMechTF(self, name, p, k, dof='pos'):
        """Set the mechanical transfer function of an optic

        The transfer function is from radiation pressure to one of the degrees
        of freedom position, pitch, or yaw.

        Need to make this a bit more general to handle zeros as well.

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
        self.eng.workspace['p'] = py2mat(p, is_complex=True)
        self.eng.workspace['nDOF'] = py2mat(nDOF)
        self.eng.eval("tf = zpk([], p, " + str(k) + ");", nargout=0)
        cmd = self.opt + ".setMechTF(" + str2mat(name) + ", tf, nDOF);"
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
                freqs = mat2py(self.eng.eval(self.opt + ".nu;"))
                if len(freqs.shape) == 0:
                    nu = freqs
                else:
                    nu = freqs[0]
            npol = self._pol2opt(pol)
            self.eng.workspace['nu'] = py2mat(nu)
            cmd = "Optickle.matchFreqPol(" + self.opt + ", nu"
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
        cmd = "LO = " + self.opt + ".getOptic(" + str2mat(LOname) + ");"
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
            mProbeOut[nA, nA] = 1/2
            mProbeOut[nB, nA] = -1/2
            mProbeOut[nA, nB] = 1/2
            mProbeOut[nB, nB] = 1/2
        self.setProbeBasis(mProbeOut)

    def setProbeBasis(self, mProbeOut):
        """Set the probe basis
        """
        self.eng.workspace['mProbeOut'] = py2mat(mProbeOut)
        self.eng.eval(self.opt + ".mProbeOut = mProbeOut;", nargout=0)
        self._rotatedBasis = True

    def setCavityBasis(self, name1, name2):
        """Set the cavity basis for a two mirror cavity

        Inputs:
          name1, name2: the names of the optics
        """
        cmd = self.opt + ".setCavityBasis("
        cmd += str2mat(name1) + ", " + str2mat(name2) + ");"
        self.eng.eval(cmd, nargout=0)

    def setOpticParam(self, name, param, val):
        """Set the value of an optic's parameter

        Inputs:
          name: name of the optic
          param: name of the parameter
          val: value of the paramter
        """
        cmd = self.opt + ".setOpticParam("
        cmd += str2mat(name) + ", " + str2mat(param)
        cmd += ", " + str(val) + ");"
        self.eng.eval(cmd, nargout=0)

    def getOpticParam(self, name, param):
        """Get the value of an optic's paramter

        Inputs:
          name: name of the optic
          param: name of the parameter
        """
        cmd = self.opt + ".getOptic(" + str2mat(name) + ");"
        self.eng.eval(cmd, nargout=0)
        return self.eng.eval("ans." + param + ";")

    def getProbePhase(self, name):
        """Get the phase of a probe

        Inputs:
          name: name of the probe

        Returns:
          phase: phase of the probe [deg]
        """
        cmd = self.opt + ".getProbePhase(" + str2mat(name) + ");"
        return self.eng.eval(cmd, nargout=1)

    def setProbePhase(self, name, phase):
        """Set the phase of a probe

        Inputs:
          name: name of the probe
          phase: phase of the probe [deg]
        """
        cmd = self.opt + ".setProbePhase("
        cmd += str2mat(name) + ", " + str(phase) + ");"
        self.eng.eval(cmd, nargout=0)

    def getSourceName(self, linkNum):
        """Find the name of the optic that is the source for a link

        Inputs:
          linkNum: the link number
        """
        cmd = self.opt + ".getSourceName(" + str(linkNum) + ");"
        return self.eng.eval(cmd, nargout=1)

    def getSinkName(self, linkNum):
        """Find hte name of the optic that is the sink for a link

        Inputs:
          linkNum: the link number
        """
        cmd = self.opt + ".getSinkName(" + str(linkNum) + ");"
        return self.eng.eval(cmd, nargout=1)

    def _updateNames(self):
        """Refresh the pytickle model's list of probe and drive names
        """
        self.probes = self.eng.eval(self.opt + ".getProbeName")
        self.drives = self.eng.eval(self.opt + ".getDriveNames")

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
