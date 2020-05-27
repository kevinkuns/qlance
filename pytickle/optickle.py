'''
Provides code for calling Optickle from within PyTickle
'''

import numpy as np
import matlab
from . import plant
import scipy.constants as scc
from numbers import Number
from collections import OrderedDict
import pandas as pd
from . import controls as ctrl
from . import utils
from .matlab import mat2py, py2mat, str2mat, addOpticklePath
from .gaussian_beams import beam_properties_from_q


class PyTickle(plant.OpticklePlant):
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
        plant.OpticklePlant.__init__(self)
        # convert RF and polarization information
        if isinstance(vRF, Number):
            vRF = np.array([vRF])
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
        self._eval(
            self.optName + " = Optickle(vRF, lambda0, pol);", nargout=0)

        # initialize pytickle class variables
        self.lambda0 = mat2py(self.eng.eval(self.optName + ".lambda"))
        self.vRF = mat2py(self._eval(self.optName + ".vFrf", 1))
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

        # Track whether the probe basis has been rotated.
        # If it has, do not let any more probes be added
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

    def run(self, ff=None, noise=True, dof='pos'):
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
          dof: which degree of freedom or mode to compute the response for:
            pos: position or TEM00 mode
            pitch: pitch or TEM01 mode
            yaw: yaw or TEM10 mode
            (Default: pos)
        """
        self._tickled = True

        if ff is not None:
            self.ff = ff
            self.eng.workspace['f'] = py2mat(self.ff)

        if ff is None:
            output = "[fDC, sigDC_tickle]"
            cmd = "{:s} = {:s}.tickle([], 0);".format(output, self.optName)
            self._eval(cmd, nargout=0)
            self.fDC = mat2py(self.eng.workspace['fDC'])
            self.sigDC_tickle = mat2py(self.eng.workspace['sigDC_tickle'])

        else:
            if noise:
                output = "[fDC, sigDC_tickle, mOpt, mMech, noiseAC]"
            else:
                output = "[fDC, sigDC_tickle, mOpt, mMech]"
            cmd = "{:s} = {:s}.tickle2([], f, {:d});".format(
                output, self.optName, self._dof2opt(dof))

            self._eval(cmd, nargout=0)
            self._eval("sigAC = getProdTF(mOpt, mMech);", nargout=0)

            self.fDC = mat2py(self.eng.workspace['fDC'])
            self.sigDC_tickle = mat2py(self.eng.workspace['sigDC_tickle'])
            self.mOpt[dof] = mat2py(self.eng.workspace['mOpt'])
            self.sigAC[dof] = mat2py(self.eng.workspace['sigAC'])
            self.mMech[dof] = mat2py(self.eng.workspace['mMech'])
            if noise:
                self.noiseAC[dof] = mat2py(
                    self.eng.workspace['noiseAC'])

        # get the field basis if the dof is pitch or yaw
        if dof in ['pitch', 'yaw']:
            self.qq = mat2py(
                self._eval(self.optName + ".getAllFieldBases()", 1))[:, -1]

    def sweepLinear(self, startPos, endPos, npts):
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
            driveNum = self._getDriveIndex(driveName, 'pos')
            sPosMat[driveNum] = drivePos
        for driveName, drivePos in endPos.items():
            driveNum = self._getDriveIndex(driveName, 'pos')
            ePosMat[driveNum] = drivePos

        self.eng.workspace['startPos'] = py2mat(sPosMat)
        self.eng.workspace['endPos'] = py2mat(ePosMat)
        self.eng.workspace['npts'] = matlab.double([npts])

        cmd = "[poses, sigDC, fDC] = " + self.optName
        cmd += ".sweepLinear(startPos, endPos, npts);"

        self.eng.eval(cmd, nargout=0)
        self.poses = mat2py(self.eng.workspace['poses'])
        self.sigDC_sweep = mat2py(self.eng.workspace['sigDC'])
        self.fDC_sweep = mat2py(self.eng.workspace['fDC'])

    def computeBeamSpotMotion(self, opticName, driveName, dof):
        """Compute the beam spot motion on one optic due to angular motion of another

        The beam spot motion must have been monitored by calling monitorBeamSpotMotion
        before tickling the model.

        Inputs:
          opticName: name of the optic to compute the BSM on
          driveName: drive name of the optic from which to compute the BSM from
          dof: degree of freedom of the optic driving the BSM

        Returns:
          bsm: the beam spot motion [m/rad]

        Example:
          To compute the beam spot motion on the front of EX due to pitch
          motion of IX
            opt.monitorBeamSpotMotion('EX', 'fr')
            bsm = opt.computeBeamSpotMotion('EX', 'IX', 'pitch')
        """
        # figure out monitoring probe information
        probeName = opticName + '_DC'
        probe_info = self._eval(
            "opt.getSinkName(opt.getFieldProbed('{:s}'))".format(probeName), 1)
        spotPort = probe_info.split('<-')[-1]

        # get TF to monitoring probe power [W/rad]
        tf = self.getTF(probeName, driveName, dof)

        # DC power on the monitoring probe
        Pdc = self.getSigDC(probeName)

        # get the beam size on the optic
        w, _, _, _, _, _ = self.getBeamProperties(opticName, spotPort)
        w = np.abs(w[self.vRF == 0])

        # convert to spot motion [m/rad]
        bsm = w/Pdc * tf
        return bsm

    def addMirror(self, name, aoi=0, Chr=0, Thr=0, Lhr=0,
                  Rar=0, Lmd=0, Nmd=1.45):
        """Add a mirror to the model

        Inputs:
          name: Name of the mirror
          aoi: angle of incidence [deg] (Default: 0)
          Chr: inverse radius of curvature [1/m] (Default: 0)
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
        self._eval(cmd, nargout=0)
        self._updateNames()

    def addBeamSplitter(self, name, aoi=45, Chr=0, Thr=0.5, Lhr=0,
                        Rar=0, Lmd=0, Nmd=1.45):
        """Add a beamsplitter to the model

        Inputs:
          name: Name of the mirror
          aoi: angle of incidence [deg] (Default: 45)
          Chr: inverse radius of curvature [1/m] (Default: 0)
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
        self._eval(cmd, nargout=0)
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
        try:
            lambda0 = self.lambda0[0]
        except IndexError:
            lambda0 = self.lambda0
        Thr = [[transS, lambda0, 1],
               [1 - reflP, lambda0, 0]]

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
        self._eval(cmd, nargout=0)

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
        self._eval(cmd, nargout=0)
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
        self._eval(cmd, nargout=0)

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
        self._eval(cmd, nargout=0)
        self._updateNames()

    def addModulator(self, name, cMod):
        """Add a modulator

        Inputs:
          name: name of the modulator
          cMod: modulation index (amplitude: 1; phase: 1j)
        """
        cmd = self.optName + ".addModulator(" + str2mat(name)
        cmd += ", " + str(cMod) + ");"
        self._eval(cmd, nargout=0)
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
        self._eval(cmd, nargout=0)
        self._updateNames()

    def addSink(self, name, loss=1):
        """Add a sink

        Inputs:
          name: name of the sink
          loss: loss going from the in port to the out port (Default: 1)
        """
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
        self._eval(cmd, nargout=0)
        self._updateNames()

    def addReadout(self, sinkName, freqs, phases, names=None):
        """Add RF and DC probes to a detection port

        Inputs:
          sinkName: the sink name
          freqs: demodulation frequencies [Hz]
          phases: demodulation phases [deg]
          names: suffixes for RF probe names (Optional).
            If blank and there are multiple demod frequencies, the suffixes
            1, 2, 3, ... are added

        Examples:
          * self.addReadout('REFL', f1, 0)
          adds the probes 'REFL_DC', 'REFL_I', and 'REFL_Q' at demod frequency
          f1 and phases 0 and 90 to the REFL sink.

          * self.addReadout('POP', [11e6, 55e6], [0, 30], ['11', '55'])
          adds the probes POP_DC, POP_I11, POP_Q11, POP_I55, and POP_Q55 at
          demod frequency 11 MHz w/ phases 0 and 90 and at demod phase 55 MHz
          and phases 30 and 100 to the POP sink.
        """
        # Get demod frequencies and phases
        if isinstance(freqs, Number):
            freqs = np.array([freqs])
        if isinstance(phases, Number):
            phases = np.array([phases])
        if len(freqs) != len(phases):
            raise ValueError(
                'Frequency and phase vectors are not the same length.')

        # Figure out the naming scheme
        if names is None:
            if len(freqs) > 1:
                names = (np.arange(len(freqs), dtype=int) + 1).astype(str)
            else:
                names = ['']
        elif isinstance(names, str):
            names = [names]

        # Add the probes
        self.addProbeIn(sinkName + '_DC', sinkName, 'in', 0, 0)
        for freq, phase, name in zip(freqs, phases, names):
            nameI = sinkName + '_I' + name
            nameQ = sinkName + '_Q' + name
            self.addProbeIn(nameI, sinkName, 'in', freq, phase)
            self.addProbeIn(nameQ, sinkName, 'in', freq, phase + 90)

    def addGouyPhase(self, name, phase):
        """Add a Gouy phase optic

        Inputs:
          name: name of the Gouy phase optic
          phase: Gouy phase to add [deg]
        """
        phase *= np.pi/180
        cmd = self.optName + ".addGouyPhase(" + str2mat(name) + ", " \
            + str(phase) + ");"
        self._eval(cmd, nargout=0)

    def addGouyReadout(self, name, phaseA, dphaseB=90):
        """Add Gouy phases and sinks for WFS readout

        Inputs:
          name: base name of the probes
          phaseA: Gouy phase of the A probe [deg]
          dphaseB: additional Gouy phase of the B probe relative to the
            A probe [deg] (Default: 90 deg)
        """
        # add a beamsplitter and two sinks for the two RF PDs
        self.addMirror(name + '_WFS_BS', Thr=0.5, aoi=45)
        self.addSink(name + '_A')
        self.addSink(name + '_B')

        # set the Gouy phases of the two RF PDs
        self.addGouyPhase(name + '_GouyA', phaseA)
        self.addGouyPhase(name + '_GouyB', phaseA + dphaseB)

        # Links
        self.addLink(name + '_WFS_BS', 'fr', name + '_GouyA', 'in', 0)
        self.addLink(name + '_WFS_BS', 'bk', name + '_GouyB', 'in', 0)
        self.addLink(name + '_GouyA', 'out', name + '_A', 'in', 0)
        self.addLink(name + '_GouyB', 'out', name + '_B', 'in', 0)

    def monitorBeamSpotMotion(self, opticName, spotPort):
        """Add a DC probe to an optic to monitor beam spot motion

        NOTE: The probe basis cannot be rotated before monitoring beam spot
        motion.

        Inputs:
          opticName: name of the optic to monitor BSM on
          spotPort: port of the optic to monitor the BSM at

        Example:
          To monitor beam spot motion on the front of EX
            opt.monitorBeamSpotMotion('EX', 'fr')
        """
        # add a sink at the optic 'optic_DC'
        name = opticName + '_DC'
        # self.addSink(name)

        # add an unphisical probe 'optic_Dc'
        # self.addLink(opticName, spotPort, name, 'in', 0)
        self.addProbeIn(name, opticName, spotPort, 0, 0)

    def setPosOffset(self, name, dist):
        cmd = self.optName + ".setPosOffset(" + str2mat(name)
        cmd += ", " + str(dist) + ");"
        self._eval(cmd, nargout=0)

    def setMechTF(self, name, z, p, k, dof='pos'):
        """Set the mechanical transfer function of an optic

        The transfer function is from radiation pressure to one of the degrees
        of freedom position, pitch, or yaw.

        The zeros and poles should be in the s-domain.

        Inputs:
          name: name of the optic
          z: zeros
          p: poles
          k: gain
          dof: degree of freedom: pos, pitch, or yaw (default: pos)
        """
        nDOF = self._dof2opt(dof)

        self.eng.workspace['z'] = py2mat(z, is_complex=True)
        self.eng.workspace['p'] = py2mat(p, is_complex=True)
        self.eng.workspace['k'] = py2mat(k, is_complex=True)
        self.eng.workspace['nDOF'] = py2mat(nDOF)

        self._eval("tf = zpk(z, p, k);", nargout=0)
        cmd = self.optName + ".setMechTF(" + str2mat(name) + ", tf, nDOF);"
        self._eval(cmd, nargout=0)

    def getMechTF(self, outDrives, inDrives, dof='pos'):
        """Compute a mechanical transfer function

        Inputs:
          outDrives: name of the output drives
          inDrives: name of the input drives
          dof: degree of freedom: pos, pitch, or yaw (Default: pos)

        Returns:
          tf: the transfer function
            * In units of [m/N] for position
            * In units of [rad/(N m)] for pitch and yaw
        """
        if dof not in ['pos', 'pitch', 'yaw']:
            raise ValueError('Unrecognized degree of freedom {:s}'.format(dof))

        # figure out the shape of the TF
        if isinstance(self.ff, Number):
            # TF is at a single frequency
            tf = 0
        else:
            # TF is for a frequency vector
            tf = np.zeros(len(self.ff), dtype='complex')

        if isinstance(outDrives, str):
            outDrives = {outDrives: 1}

        if isinstance(inDrives, str):
            inDrives = {inDrives: 1}

        # loop through drives to compute the TF
        for inDrive, c_in in inDrives.items():
            # get the default mechanical plant of the optic being driven
            plant = ctrl.Filter(*self.extract_zpk(inDrive, dof), Hz=False)

            for outDrive, c_out in outDrives.items():
                mmech = self.getMechMod(outDrive, inDrive, dof=dof)
                tf += c_in * c_out * plant.computeFilter(self.ff) * mmech

        return tf

    def plotMechTF(self, outDrives, inDrives, mag_ax=None, phase_ax=None,
                   dof='pos', **kwargs):
        """Plot a mechanical transfer function

        See documentation for plotTF in plotting
        """
        ff = self.ff
        tf = self.getMechTF(outDrives, inDrives, dof=dof)
        fig = plotting.plotTF(
            ff, tf, mag_ax=mag_ax, phase_ax=phase_ax, **kwargs)
        return fig

    def extract_zpk(self, name, dof='pos'):
        """Get the mechanical transfer function of an optic

        Returns the zeros, poles, and gain of the mechanical transfer function
        of an optic.

        The zeros and poles are returned in the s-domain.

        Inputs:
          name: name of the optic
          dof: degree of freedom: pos, pitch, or yaw (defualt: pos)

        Returns:
          zs: the zeros
          ps: the poles
          k: the gain
        """
        self._eval("obj = {:s}.getOptic({:s});".format(
            self.optName, str2mat(name)))
        if dof == 'pos':
            self._eval("tf = obj.mechTF;")
        elif dof == 'pitch':
            self._eval("tf = obj.mechTFpit;")
        elif dof == 'yaw':
            self._eval("tf = obj.mechTFyaw;")
        else:
            raise ValueError('Unrecognized dof ' + dof)

        if self._eval("isempty(tf);", nargout=1):
            print('No TF is set for dof ' + dof)
            zs = []
            ps = []
            k = 0

        else:
            zs = mat2py(self._eval("tf.Z", nargout=1))
            ps = mat2py(self._eval("tf.P", nargout=1))
            k = float(mat2py(self._eval("tf.K", nargout=1)))

        return zs, ps, k

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
            ampRF = self._eval(cmd, 1)
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
        self._eval(cmd, nargout=0)
        self._eval("RFamp = abs(LO.vArf);", nargout=0)
        self._eval("LO.vArf = RFamp * exp(1i*" + str(phase) + ");",
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
        self._eval(cmd, nargout=0)

    def setOpticParam(self, name, param, val):
        """Set the value of an optic's parameter

        Inputs:
          name: name of the optic
          param: name of the parameter
          val: value of the paramter
        """
        # get the optic and load the value into the workspace
        self.eng.workspace['val'] = py2mat(val)
        self._eval("obj = {:s}.getOptic({:s});".format(
            self.optName, str2mat(name)))

        # Beamsplitters are write protected but can be tricked by modifying
        # the internal mirror directly
        if self._eval("isa(obj, 'BeamSplitter')", 1):
            self._eval("obj.mir." + str(param) + " = val;")

        else:
            self._eval("obj." + str(param) + " = val;")

    def getOpticParam(self, name, param):
        """Get the value of an optic's paramter

        Inputs:
          name: name of the optic
          param: name of the parameter
        """
        self._eval("obj = {:s}.getOptic({:s});".format(
            self.optName, str2mat(name)))

        # Need to access a beamsplitter's internal mirror to get the
        # true properties
        if self._eval("isa(obj, 'BeamSplitter')", 1):
            val = mat2py(self._eval("obj.mir." + str(param), 1))

        else:
            val = mat2py(self._eval("obj." + str(param), 1))

        return val

    def getProbePhase(self, name):
        """Get the phase of a probe

        Inputs:
          name: name of the probe

        Returns:
          phase: phase of the probe [deg]
        """
        cmd = self.optName + ".getProbePhase(" + str2mat(name) + ");"
        return self._eval(cmd, nargout=1)

    def setProbePhase(self, name, phase):
        """Set the phase of a probe

        Inputs:
          name: name of the probe
          phase: phase of the probe [deg]
        """
        cmd = self.optName + ".setProbePhase("
        cmd += str2mat(name) + ", " + str(phase) + ");"
        self._eval(cmd, nargout=0)

    def getGouyPhase(self, linkStart, linkEnd):
        """Compute the accumulated Gouy phase along a path

        Gouy phase is
          psi = arccot(Im(q)/Re(q)) = pi/2 - arctan(Im(q)/Re(q))
              = pi/2 - arg(q)

        Inputs:
          linkStart: name of the start of the path
          linkEnd: name of the end of the path

        Returns:
          dpsi: the accumulated Gouy phase along the path [deg]
        """
        linkNum = self._getLinkNum(linkStart, linkEnd)
        qq = self.qq[linkNum]
        dl = self.getLinkLength(linkStart, linkEnd)
        dphi = np.angle(qq - dl, deg=True) - np.angle(qq, deg=True)
        return dphi

    def setGouyPhase(self, name, phase):
        """Set the Gouy phase of an existing Gouy phase optic

        Inputs:
          name: name of the Gouy phase optic
          phase: phase [deg]
        """
        phase *= np.pi / 180
        cmd = "gouy = " + self.optName + ".getOptic(" + str2mat(name) + ");"
        self._eval(cmd, nargout=0)
        cmd = "gouy.setPhase(" + str(phase) + ");"
        self._eval(cmd, nargout=0)

    def getLinkLength(self, linkStart, linkEnd):
        """Get the length of a link

        Inputs:
          linkStart: name of the start of the link
          linkEnd: name of the end of the link

        Returns:
          linkLen: the length of the link [m]
        """
        linkLens = self._eval(self.optName + ".getLinkLengths", nargout=1)
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
        self._eval(
            self.optName + ".setLinkLength(linkNum, linkLen);", nargout=0)

    def getFieldBasis(self, optic, port=None, verbose=False):
        if port:
            qq = self.qq[self._getSinkNum(optic, port)]
            if verbose:
                print('Taking the basis determined by the model')

        else:
            self._eval("obj = {:s}.getOptic({:s});".format(
                self.optName, str2mat(optic)))

            if self._eval("isa(obj, 'Mirror')", 1):
                qq = mat2py(self._eval("obj.getFrontBasis();", 1))
                if np.shape(qq)[0]:
                    qq = qq[1]  # take pitch
                    if verbose:
                        print('Taking the front basis specified by the model')
                else:
                    print(optic + ' does not have a basis set')
                    qq = None
            else:
                qq = None
                if verbose:
                    print(optic + ' is not a mirror')

        return qq

    def getBeamProperties(self, name, port):
        """Compute the properties of a Gaussian beam at an optic

        Inputs:
          name: name of the optic
          port: name of the port

        Returns:
          w: beam radius on the optic [m]
          zR: Rayleigh range of the beam [m]
          z: distance from the beam waist to the optic [m]
            Negative values indicate that the optic is before the waist.
          w0: beam waist [m]
          R: radius of curvature of the phase front on the optic [m]
          psi: Gouy phase [deg]

        Example:
          opt.getBeamProperties('EX', 'fr')
        """
        qq = self.qq[self._getSinkNum(name, port)]
        return beam_properties_from_q(qq, lambda0=self.lambda0)

    def showBeamProperties(self):
        """Print the beam properties along each link of the model

        The beam properties are
          w: beam radius at the end of the link
          zR: Rayleigh range
          z - z0: distance of the end of the link past the waist
          w0: beam waist
          R: radius of curvature at the end of the link
          dpsi: accumulated Gouy phase along the link
        """
        nLink = int(self._eval(self.optName + ".Nlink", 1))
        sinks = [self._getSinkName(ii) for ii in range(1, nLink + 1)]
        sources = [self._getSourceName(ii) for ii in range(1, nLink + 1)]
        beam_properties = {}
        for source, sink in zip(sources, sinks):
            w, zR, z, w0, R, _ = self.getBeamProperties(*sink.split('<-'))
            dpsi = self.getGouyPhase(
                source.split('->')[0], sink.split('<-')[0])

            # get the carrier information
            try:
                ind = self._getSidebandInd(0)
                w = w[ind]
                w0 = w0[ind]
            except IndexError:
                pass

            beam_properties[source + ' --> ' + sink] = [
                '{:0.2f} {:s}m'.format(*utils.siPrefix(w)[::-1]),
                '{:0.2f} {:s}m'.format(*utils.siPrefix(zR)[::-1]),
                '{:0.2f} {:s}m'.format(*utils.siPrefix(z)[::-1]),
                '{:0.2f} {:s}m'.format(*utils.siPrefix(w0)[::-1]),
                '{:0.2f} {:s}m'.format(*utils.siPrefix(R)[::-1]),
                '{:0.0f} deg'.format(dpsi)]

        beam_properties = pd.DataFrame(
            beam_properties, index=['w', 'zR', 'z - z0', 'w0', 'R', 'dpsi']).T
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None):
            display(beam_properties)

    def getABCD(self, *args, dof='pitch'):
        """Get the ABCD matrix of an optic or path

        Returns the ABCD matrix of an optic if three arguments are supplied
        and the ABCD matrix of a path if two arguments are supplied

        Inputs (3 for optic):
          name: name of the optic
          inPort: input port of the transformation
          outPort: output port of the transformation
          dof: degree of freedom 'pitch' or 'yaw' (Default: 'pitch')

        Inputs (2 for path):
          linkStart: name of the start of the link
          linkEnd: name of the end of the link
          dof: degree of freedom 'pitch' or 'yaw' (Default: 'pitch')

        Returns:
          abcd: the ABCD matrix

        Examples:
          To compute the ABCD matrix for reflection from the front of EX:
            opt.getABCD('EX', 'fr', 'fr')
          To compute the ABCD matrix for propagation from IX to EX:
            opt.getABCD('IX', 'EX')
        """
        if len(args) == 3:
            name, inPort, outPort = args

            if dof == 'pitch':
                ax = "y"
            elif dof == 'yaw':
                ax = "x"

            self._eval("obj = {:s}.getOptic({:s});".format(
                self.optName, str2mat(name)))
            self._eval("nIn = obj.getInputPortNum({:s});".format(
                str2mat(inPort)))
            self._eval("nOut = obj.getOutputPortNum({:s});".format(
                str2mat(outPort)))
            self._eval("qm = obj.getBasisMatrix();")
            abcd = mat2py(self._eval("qm(nOut, nIn).{:s}".format(ax), 1))

        elif len(args) == 2:
            linkStart, linkEnd = args
            linkLen = self.getLinkLength(linkStart, linkEnd)
            abcd = np.array([[1, linkLen],
                             [0, 1]])

        else:
            raise ValueError('Incorrect number of arguments')

        return abcd

    def getSweepPower(self, driveName, linkStart, linkEnd, fRF=0,
                      lambda0=1064e-9):
        """Get the power along a link between two optics as a drive is swept

        Inputs:
          linkStart: name of the start of the link
          linkEnd: name of the end of the link
          fRF: frequency of the sideband power to return [Hz]
            (Default: 0, i.e. the carrier)
          lambda0: wavelength of the sideband power to return [m]
            (Default: 1064e-9)

        Returns:
          poses: the drive positions
          power: the power at those positions [W]
        """
        if self.fDC_sweep is None:
            raise ValueError(
                'Must run sweepLinear before calculating sweep power')

        # find the link and the drive
        linkNum = self._getLinkNum(linkStart, linkEnd)
        driveNum = self._getDriveIndex(driveName, 'pos')

        poses = self.poses[driveNum, :]
        if self.vRF.size == 1:
            power = np.abs(self.fDC_sweep[linkNum, :])**2
        else:
            nRF = self._getSidebandInd(fRF, lambda0)
            power = np.abs(self.fDC_sweep[linkNum, nRF, :])**2

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
        driveNum = self._getDriveIndex(driveName, 'pos')

        poses = self.poses[driveNum, :]
        try:
            sig = self.sigDC_sweep[probeNum, :]
        except IndexError:
            sig = self.sigDC_sweep

        return poses, sig

    def getSigDC(self, probeName):
        """Get the DC power on a probe

        Inputs:
          probeName: the probe name

        Returns:
          power: the DC power on the probe [W]
        """
        probeNum = self.probes.index(probeName)
        return self.sigDC_tickle[probeNum]

    def getDCpower(self, linkStart, linkEnd, fRF=0, lambda0=1064e-9):
        """Get the DC power along a link

        Inputs:
          linkStart: name of the start of the link
          linkEnd: name of the end of the link
          fRF: frequency of the sideband power to return [Hz]
            (Default: 0, i.e. the carrier)
          lambda0: wavelength of the sideband power to return [m]
            (Default: 1064e-9)

        Returns:
          power: the DC power [W]
        """
        if self.fDC is None:
            raise ValueError(
                'Must run tickle before getting the DC power levels.')

        linkNum = self._getLinkNum(linkStart, linkEnd)
        if self.vRF.size == 1:
            power = np.abs(self.fDC[linkNum])**2
        else:
            nRF = self._getSidebandInd(fRF, lambda0)
            power = np.abs(self.fDC[linkNum, nRF])**2

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
                line += utils.printLine([self.fDC[li]], pad2)
            else:
                line += utils.printLine(self.fDC[li, :], pad2)
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
                pref, num = utils.siPrefix(self.sigDC_tickle[pi])
            except IndexError:
                pref, num = utils.siPrefix(self.sigDC_tickle)
            pad3 = pad2 - len(pref) - 2
            print('{:{pad1}s}| {:{pad3}.1f} {:s}W|'.format(
                probe, num, pref, pad1=pad1, pad3=pad3))

    def _getSourceName(self, linkNum):
        """Find the name of the optic that is the source for a link

        Inputs:
          linkNum: the link number
        """
        cmd = self.optName + ".getSourceName(" + str(linkNum) + ");"
        return self._eval(cmd, nargout=1)

    def _getSinkName(self, linkNum):
        """Find the name of the optic that is the sink for a link

        Inputs:
          linkNum: the link number
        """
        cmd = self.optName + ".getSinkName(" + str(linkNum) + ");"
        return self._eval(cmd, nargout=1)

    def _getSinkNum(self, name, port):
        cmd = "{:s}.getFieldIn({:s}, {:s})".format(
            self.optName, str2mat(name), str2mat(port))
        sinkNum = self._eval(cmd, nargout=1)
        try:
            sinkNum = int(sinkNum) - 1
        except TypeError:
            raise ValueError(
                "There does not appear to be a sink at {:s} {:s}".format(
                    name, port))
        return sinkNum

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
        linkNum = self._eval(
            self.optName + ".getLinkNum(" + linkStart + ", " + linkEnd + ");", 1)
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
        self.probes = self._eval(self.optName + ".getProbeName", 1)
        self.drives = self._eval(self.optName + ".getDriveNames", 1)

    def _eval(self, cmd, nargout=0):
        """Evaluate a matlab command using the pytickle model's engine

        Inputs:
          cmd: the matlab command string
          nargout: the number of outputs to be returned (Defualt: 0)

        Returns:
          The outputs from matlab
        """
        return self.eng.eval(cmd, nargout=nargout)
