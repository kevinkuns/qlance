'''
Code for general optomechanical plants
'''

import numpy as np
import matplotlib.pyplot as plt
from . import controls as ctrl
from . import plotting
from . import utils
from . import io
import h5py
from numbers import Number
from .gaussian_beams import beam_properties_from_q


class OpticklePlant:
    """An Optickle optomechanical plant
    """
    def __init__(self):
        self._topology = OptickleTopology()
        self._vRF = None
        self._lambda0 = None
        self._pol = None
        self._probes = []
        self._drives = []
        self._ff = None
        self._fDC = None
        self._sigAC = {}
        self._sigDC_tickle = None
        self._mMech = {}
        self._mOpt = {}
        self._noiseAC = {}
        self._mech_plants = {}
        self._poses = None
        self._sigDC_sweep = None
        self._fDC_sweep = None
        self._qq = None

    @property
    def vRF(self):
        """Vector of RF sidebands computed [Hz]
        """
        return self._vRF

    @property
    def lambda0(self):
        """Wavelength [m]
        """
        return self._lambda0

    @property
    def pol(self):
        """Vector of polarizations for each field
        """
        return self._pol

    @property
    def drives(self):
        """List of drives
        """
        return self._drives

    @property
    def probes(self):
        """List of probes
        """
        return self._probes

    @property
    def ff(self):
        """Frequency vector [Hz]
        """
        return self._ff

    def getTF(self, probes, drives, dof='pos', optOnly=False):
        """Compute a transfer function

        Inputs:
          probes: name of the probes at which the TF is calculated
          drives: names of the drives from which the TF is calculated
          dof: degree of freedom of the drives (Default: pos)
          optOnly: if True, only return the optical TF with no mechanics
            (Default: False)

        Returns:
          tf: the transfer function
            * In units of [W/m] if drive is an optic with dof pos
            * In units of [W/rad] if drive is an optic with dof pitch or yaw
            * In units of [W/rad] if drive is a PM modulator with dof drive
            * In units of [W/RAM] if drive is an AM modulator with dof drive
            * In units of [W/rad] if drive is an RF modulator with dof phase
            * In units of [W/RAM] if drive is an RF modulator with dof amp
              modulation of an RF modulator.

        Note:
          * To convert phase noise in [W/rad] to frequency noise in [W/Hz],
            divide by 1j*ff
          * To convert amplitude noise in [W/RAM] to intensity noise in [W/RIN],
            divide by 2 since RIN = 2*RAM

        Examples:
          * If only a single drive is used, the drive name can be a string.
            To compute the phase transfer function in reflection from a FP
            cavity
              tf = opt.getTF('REFL', 'PM', 'drive')
            The frequency transfer function is then tf/(1j*ff)

          * If multiple drives are used, the drive names should be a dict.
            To compute the DARM transfer function to the AS_DIFF homodyne PD
              DARM = {'EX': 1/2, 'EY': -1/2}
              tf = opt.getTF('AS_DIFF', DARM)
            [Note: Since DARM is defined as Lx - Ly, to get 1 m of DARM
            requires 0.5 m of both Lx and Ly]
        """
        if dof not in ['pos', 'pitch', 'yaw', 'drive', 'amp', 'phase']:
            raise ValueError('Unrecognized degree of freedom {:s}'.format(dof))

        # figure out the shape of the TF
        if isinstance(self.ff, Number):
            # TF is at a single frequency
            tf = 0
        else:
            # TF is for a frequency vector
            tf = np.zeros(len(self.ff), dtype='complex')

        if optOnly:
            tfData = self._mOpt
        else:
            tfData = self._sigAC

        if tfData is None:
            msg = 'Must run tickle for the appropriate DOF before ' \
                  + 'calculating a transfer function.'
            raise RuntimeError(msg)

        # figure out which raw output matrix to use
        if dof in ['pos', 'drive', 'amp', 'phase']:
            tfData = tfData['pos']
        elif dof == 'pitch':
            tfData = tfData['pitch']
        elif dof == 'yaw':
            tfData = tfData['yaw']

        if isinstance(drives, str):
            drives = {drives: 1}

        if isinstance(probes, str):
            probes = {probes: 1}

        # loop through the drives and probes to compute the TF
        for probe, pc in probes.items():
            # get the probe index
            probeNum = self.probes.index(probe)

            for drive, drive_pos in drives.items():
                # get the drive index
                driveNum = self._getDriveIndex(drive, dof)

                # add the contribution from this drive
                try:
                    tf += pc * drive_pos * tfData[probeNum, driveNum]
                except IndexError:
                    tf += pc * drive_pos * tfData[driveNum]

        return tf

    def getMechMod(self, drive_out, drive_in, dof='pos'):
        """Get the radiation pressure modifications to drives

        Inputs:
          drive_out: name of the output drive
          drive_in: name of the input drive
          dof: degree of freedom: pos, pitch, or yaw (Default: pos)
        """
        if dof not in ['pos', 'pitch', 'yaw', 'drive', 'amp', 'phase']:
            raise ValueError('Unrecognized degree of freedom {:s}'.format(dof))

        # figure out which raw output matrix to use
        if dof in ['pos', 'drive', 'amp', 'phase']:
            mMech = self._mMech['pos']
        elif dof == 'pitch':
            mMech = self._mMech['pitch']
        elif dof == 'yaw':
            mMech = self._mMech['yaw']

        if mMech is None:
            msg = 'Must run tickle for the appropriate DOF before ' \
                  + 'calculating a transfer function.'
            raise RuntimeError(msg)

        driveInNum = self._getDriveIndex(drive_in, dof)
        driveOutNum = self._getDriveIndex(drive_out, dof)

        return mMech[driveOutNum, driveInNum]

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
            plant = self._mech_plants[dof][inDrive]

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

    def getQuantumNoise(self, probeName, dof='pos'):
        """Compute the quantum noise at a probe

        Returns the quantum noise at a given probe in [W/rtHz]
        """
        probeNum = self.probes.index(probeName)
        try:
            qnoise = self._noiseAC[dof][probeNum, :]
        except IndexError:
            qnoise = self._noiseAC[dof][probeNum]
        return qnoise

    def plotTF(self, probeName, driveNames, mag_ax=None, phase_ax=None,
               dof='pos', optOnly=False, **kwargs):
        """Plot a transfer function.

        See documentation for plotTF in plotting
        """
        ff = self.ff
        tf = self.getTF(probeName, driveNames, dof=dof, optOnly=optOnly)
        fig = plotting.plotTF(
            ff, tf, mag_ax=mag_ax, phase_ax=phase_ax, **kwargs)
        return fig

    def plotQuantumASD(self, probeName, driveNames, fig=None, **kwargs):
        """Plot the quantum ASD of a probe

        Plots the ASD of a probe referenced the the transfer function for
        some signal, such as DARM.

        Inputs:
          probeName: name of the probe
          driveNames: names of the drives from which the TF to refer the
            noise to
          fig: if not None, an existing figure to plot the noise on
            (Default: None)
          **kwargs: extra keyword arguments to pass to the plot
        """
        ff = self.ff
        if driveNames:
            tf = self.getTF(probeName, driveNames)
            noiseASD = np.abs(self.getQuantumNoise(probeName)/tf)
        else:
            noiseASD = np.abs(self.getQuantumNoise(probeName))

        if fig is None:
            newFig = True
            fig = plt.figure()
        else:
            newFig = False

        fig.gca().loglog(ff, noiseASD, **kwargs)
        fig.gca().set_ylabel('Noise')
        fig.gca().set_xlim([min(ff), max(ff)])
        fig.gca().set_xlabel('Frequency [Hz]')
        fig.gca().xaxis.grid(True, which='both', alpha=0.5)
        fig.gca().xaxis.grid(alpha=0.25, which='minor')
        fig.gca().yaxis.grid(True, which='both', alpha=0.5)
        fig.gca().yaxis.grid(alpha=0.25, which='minor')

        if newFig:
            return fig

    def plotSweepSignal(self, probeName, driveName, fig=None, **kwargs):
        """Plot the signal from sweeping drives

        Inputs:
          probeName: name of the probe
          driveName: name of the drives
          fig: if not None, an existing figure to plot the signal on
            (Default: None)
          **kwargs: extra keyword arguments to pass to the plot
        """
        if fig is None:
            newFig = True
            fig = plt.figure()
        else:
            newFig = False
        ax = fig.gca()

        poses, sig = self.getSweepSignal(probeName, driveName)
        ax.plot(poses, sig, **kwargs)
        ax.set_xlim(poses[0], poses[-1])
        ax.grid(True, alpha=0.5)
        if newFig:
            return fig

    def getSigDC(self, probeName):
        """Get the DC power on a probe

        Inputs:
          probeName: the probe name

        Returns:
          power: the DC power on the probe [W]
        """
        probeNum = self.probes.index(probeName)
        return self._sigDC_tickle[probeNum]

    def computeBeamSpotMotion(self, opticName, spotPort, driveName, dof):
        """Compute the beam spot motion on one optic due to angular motion of another

        The beam spot motion must have been monitored by calling
        monitorBeamSpotMotion before running the model.

        Inputs:
          opticName: name of the optic to compute the BSM on
          spotPort: port of the optic to compute the BSM on
          driveName: drive name of the optic from which to compute the BSM from
          dof: degree of freedom of the optic driving the BSM

        Returns:
          bsm: the beam spot motion [m/rad]

        Example:
          To compute the beam spot motion on the front of EX due to pitch
          motion of IX
            opt.monitorBeamSpotMotion('EX', 'fr')
            bsm = opt.computeBeamSpotMotion('EX', 'fr', 'IX', 'pitch')
        """
        # figure out monitoring probe information
        probeName = '_' + opticName + '_' + spotPort + '_DC'
        # probe_info = self._eval(
        #     "opt.getSinkName(opt.getFieldProbed('{:s}'))".format(probeName), 1)
        field_num = self._topology.get_field_probed(probeName)
        probe_info = self._topology.get_sink_name(field_num)
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
        # qq = self._qq[self._getSinkNum(name, port)]
        qq = self._qq[self._topology.get_sink_num(name, port)]
        return beam_properties_from_q(qq, lambda0=self.lambda0)

    def save(self, fname):
        """Export data to an hdf5 file

        Inputs:
          fname: file name to save data to
        """
        data = h5py.File(fname, 'w')
        self._topology.save(data)
        data['vRF'] = self.vRF
        data['lambda0'] = self.lambda0
        data['pol'] = io.str2byte(self.pol)
        data['probes'] = io.str2byte(self.probes)
        data['drives'] = io.str2byte(self.drives)
        data['ff'] = self.ff
        data['fDC'] = self._fDC
        io.dict_to_hdf5(self._sigAC, 'sigAC', data)
        data['sigDC_tickle'] = self._sigDC_tickle
        io.dict_to_hdf5(self._mMech, 'mMech', data)
        io.dict_to_hdf5(self._mOpt, 'mOpt', data)
        io.possible_none_to_hdf5(self._noiseAC, 'noiseAC', data)
        _mech_plants_to_hdf5(self._mech_plants, 'mech_plants', data)
        io.possible_none_to_hdf5(self._qq, 'qq', data)
        data.close()

    def load(self, fname):
        """Load stored data from an hdf5 file

        Inputs:
          fname: file name to read from
        """
        data = h5py.File(fname, 'r')
        self._topology.load(data)
        self._vRF = data['vRF'][()]
        self._lambda0 = data['lambda0'][()]
        self._pol = np.array(io.byte2str(data['pol'][()]))
        self._probes = io.byte2str(data['probes'][()])
        self._drives = io.byte2str(data['drives'][()])
        self._ff = data['ff'][()]
        self._fDC = data['fDC'][()]
        self._sigAC = io.hdf5_to_dict(data['sigAC'])
        self._sigDC_tickle = data['sigDC_tickle'][()]
        self._mMech = io.hdf5_to_dict(data['mMech'])
        self._mOpt = io.hdf5_to_dict(data['mOpt'])
        self._noiseAC = io.hdf5_to_possible_none('noiseAC', data)
        self._mech_plants = _mech_plants_from_hdf5(data['mech_plants'], data)
        self._qq = io.hdf5_to_possible_none('qq', data)
        data.close()

    def _getDriveIndex(self, name, dof):
        """Find the drive index of a given drive and degree of freedom
        """
        if dof in ['pos', 'pitch', 'yaw']:
            driveNum = self.drives.index(name + '.pos')
        elif dof in ['drive', 'amp', 'phase']:
            driveNum = self.drives.index('{:s}.{:s}'.format(name, dof))
        return driveNum

    def _getSidebandInd(
            self, freq, lambda0=1064e-9, pol='S', ftol=1, wltol=1e-10):
        """Find the index of an RF sideband frequency

        Inputs:
          freq: the frequency of the desired sideband
          lambda0: wavelength of desired sideband [m] (Default: 1064 nm)
          pol: polarization (Default: 'S')
          ftol: tolerance of the difference between freq and the RF sideband
            of the model [Hz] (Default: 1 Hz)
          wltol: tolerance of the difference between lambda0 and the RF
            sideband wavelength of the model [m] (Default: 100 pm)

        Returns:
          nRF: the index of the RF sideband
        """
        # indices of the relevent sidebands
        freq_inds = np.logical_and(
            np.isclose(self.vRF, freq, atol=ftol),
            np.isclose(self.lambda0, lambda0, atol=wltol))
        # find the right polarization
        ind = np.nonzero(np.logical_and(freq_inds, self.pol == pol))[0]

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

    def _dof2opt(self, dof):
        """Convert degrees of freedom to 1s, 2s, and 3s for Optickle
        """
        if dof == 'pos':
            nDOF = 1
        elif dof == 'pitch':
            nDOF = 2
        elif dof == 'yaw':
            nDOF = 3
        else:
            raise ValueError('Unrecognized degree of freedom ' + str(dof)
                             + '. Choose \'pos\', \'pitch\', or \'yaw\'.')

        return nDOF

    def _pol2opt(self, pol):
        """Convert S and P polarizations to 1s and 0s for Optickle
        """
        if pol == 'S':
            nPol = 1
        elif pol == 'P':
            nPol = 0
        else:
            raise ValueError('Unrecognized polarization ' + str(pol)
                             + '. Use \'S\' or \'P\'')
        return nPol


class OptickleTopology:
    def __init__(self):
        self._sink_nums = None
        self._sink_names = None
        self._fields_probed = None

    def update(self, sink_nums, sink_names, fields_probed):
        self._sink_nums = sink_nums
        self._sink_names = sink_names
        self._fields_probed = fields_probed

    def get_sink_num(self, name, port):
        return self._sink_nums[name + '<-' + port]

    def get_sink_name(self, link_num):
        return self._sink_names[link_num]

    def get_field_probed(self, probe):
        return self._fields_probed[probe]

    def save(self, data):
        io.dict_to_hdf5(self._sink_nums, 'topology/sink_nums', data)
        data['topology/sink_names'] = io.str2byte(self._sink_names)
        io.dict_to_hdf5(self._fields_probed, 'topology/fields_probed', data)

    def load(self, data):
        self._sink_nums = io.hdf5_to_dict(data['topology/sink_nums'])
        self._sink_names = io.byte2str(data['topology/sink_names'])
        self._fields_probed = io.hdf5_to_dict(data['topology/fields_probed'])


class FinessePlant:
    """A Finesse Optomechanical plant
    """
    def __init__(self):
        self._dofs = ['pos', 'pitch', 'yaw', 'amp', 'freq']
        self._drives = []
        self._probes = []
        self._amp_detectors = []
        self._pos_detectors = []
        self._bp_detectors = []
        self._freqresp = {}
        self._dcsigs = {}
        self._mechmod = {}
        self._mech_plants = {}
        self._ff = None
        self._lambda0 = None

    @property
    def drives(self):
        """List of drives
        """
        return self._drives

    @property
    def probes(self):
        """List of probes: photodiodes and homodyne detectors
        """
        return self._probes

    @property
    def amp_detectors(self):
        """List of amplitude detectors
        """
        return self._amp_detectors

    @property
    def pos_detectors(self):
        """List of position detectors
        """
        return self._pos_detectors

    @property
    def bp_detectors(self):
        """List of beam property detectors
        """
        return self._bp_detectors

    @property
    def ff(self):
        """Frequency vector [Hz]
        """
        return self._ff

    @property
    def lambda0(self):
        """Wavelength [m]
        """
        return self._lambda0

    def getTF(self, probes, drives, dof='pos'):
        """Compute a transfer function

        Inputs:
          probes: name of the probes at which the TF is calculated
          drives: names of the drives from which the TF is calculated
          dof: degree of freedom of the drives (Default: pos)

        Returns:
          tf: the transfer function
            * In units of [W/m] if drive is an optic with dof pos
            * In units of [W/rad] if drive is an optic with dof pitch or yaw
            * In units of [W/Hz] if drive is a laser with drive freq
            * In units of [W/RIN] if dirve is a laser with drive amp

        Note:
          * To convert frequency noise in [W/Hz] to phase noise in [W/rad],
            multiply by 1j*ff
          * To convert intensity noise in [W/RIN] to amplitude noise in [W/RAM],
            multiply by 2 since RIN = 2*RAM

        Examples:
          * If only a single drive is used, the drive name can be a string.
            To compute the frequency transfer function in reflection from a FP
            cavity
              tf = katFR.getTF('REFL', 'Laser', 'freq')
            The phase transfer function is then 1j*ff*tf

          * If multiple drives are used, the drive names should be a dict.
            To compute the DARM transfer function to the AS_DIFF homodyne PD
              DARM = {'EX': 1/2, 'EY': -1/2}
              tf = katFR.getTF('AS_DIFF', DARM)
            [Note: Since DARM is defined as Lx - Ly, to get 1 m of DARM
            requires 0.5 m of both Lx and Ly]
        """
        if dof not in self._dofs:
            raise ValueError('Unrecognized dof ' + dof)

        # figure out the shape of the TF
        if isinstance(self.ff, Number):
            # TF is at a single frequency
            tf = 0
        else:
            # TF is for a frequency vector
            tf = np.zeros(len(self.ff), dtype='complex')

        if isinstance(drives, str):
            drives = {drives: 1}

        if isinstance(probes, str):
            probes = {probes: 1}

        # loop through the drives and probes to compute the TF
        for probe, pc in probes.items():
            for drive, drive_pos in drives.items():
                # add the contribution from this drive
                tf += pc * drive_pos * self._freqresp[dof][probe][drive]

        return tf

    def getMechMod(self, drive_out, drive_in, dof='pos'):
        """Get the radiation pressure modifications to drives

        Inputs:
          drive_out: name of the output drive
          drive_in: name of the input drive
          dof: degree of freedom: pos, pitch, or yaw (Default: pos)
        """
        if dof not in self._dofs:
            raise ValueError('Unrecognized dof ' + dof)

        out_det = '_' + drive_out + '_' + dof
        if out_det not in self.pos_detectors:
            raise ValueError(out_det + ' is not a detector in this model')

        return self._mechmod[dof][out_det][drive_in]

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
            plant = self._mech_plants[dof][inDrive]

            for outDrive, c_out in outDrives.items():
                mmech = self.getMechMod(outDrive, inDrive, dof=dof)
                tf += c_in * c_out * plant.computeFilter(self.ff) * mmech

        return tf

    def getQuantumNoise(self, probeName, dof='pos'):
        """Compute the quantum noise at a probe

        Returns the quantum noise at a given probe in [W/rtHz]
        """
        shotName = '_' + probeName + '_shot'
        qnoise = list(self._freqresp[dof][shotName].values())[0]
        # without copying multiple calls will reduce noise
        qnoise = qnoise.copy()

        if dof == 'pos':
            # if this was computed from a position TF convert back to W/rtHz
            # 2*np.pi is correct here, not 360
            qnoise *= self.lambda0 / (2*np.pi)

        if np.any(np.iscomplex(qnoise)):
            print('Warning: some quantum noise spectra are complex')
        return np.real(qnoise)

    def getSigDC(self, probeName):
        """Get the DC signal from a probe

        Inputs:
          probeName: name of the probe
        """
        return self._dcsigs[probeName]

    def computeBeamSpotMotion(self, node, driveName, dof):
        """Compute the beam spot motion at a node

        The beam spot motion must have been monitored by calling
        monitorBeamSpotMotion before running the model. Both runDC and run for
        the degree of interest dof must have been called on the model.

        Inputs:
          node: name of the node
          driveName: drive name of the optic from which to compute the BSM from
          dof: degree of freedom of the optic driving the BSM

        Returns:
          bsm: the beam spot motion [m/rad]

        Example:
          To compute the beam spot motion on the front of EX due to pitch
          motion of IX
            monitorBeamSpotMotion(kat, 'EX_fr', 'pitch')
            katFR = KatFR(kat)
            katFR.runDC()
            katFR.run(fmin, fmax, npts, dof='pitch')
            bsm = katFR.computeBeamSpotMotion('EX_fr', 'IX', 'pitch')
        """
        if dof == 'pitch':
            direction = 'y'
        elif dof == 'yaw':
            direction = 'x'
        else:
            raise ValueError('Unrecognized degree of freedom ' + direction)

        probeName = '_' + node + '_bsm_' + direction

        # get TF to monitoring probe power [W/rad]
        tf = self.getTF(probeName, driveName, dof)

        # DC power on the monitoring probe
        Pdc = self.getSigDC('_' + node + '_DC')

        # get the beam size on the optic
        w, _, _, _, _, _ = self.getBeamProperties(node, dof)

        # convert to spot motion [m/rad]
        bsm = w/Pdc * tf
        return bsm

    def getBeamProperties(self, node, dof='pitch'):
        """Compute the properties of a Gaussian beam at a node

        Inputs:
          node: name of the node
          dof: which degree of freedom 'pitch' or 'yaw' (Defualt: pitch)

        Returns:
          w: beam radius on the optic [m]
          zR: Rayleigh range of the beam [m]
          z: distance from the beam waist to the optic [m]
            Negative values indicate that the optic is before the waist.
          w0: beam waist [m]
          R: radius of curvature of the phase front on the optic [m]
          psi: Gouy phase [deg]

        Example:
          katFR.getBeamProperties('EX_fr')
        """
        if dof == 'pitch':
            direction = 'y'
        elif dof == 'yaw':
            direction = 'x'
        else:
            raise ValueError('Unrecognized degree of freedom ' + direction)

        qq = self.getSigDC('_' + node + '_bp_' + direction)
        return beam_properties_from_q(qq, lambda0=self.lambda0)

    def plotTF(self, probeName, driveNames, mag_ax=None, phase_ax=None,
               dof='pos', **kwargs):
        """Plot a transfer function.

        See documentation for plotTF in plotting
        """
        ff = self.ff
        tf = self.getTF(probeName, driveNames, dof=dof)
        fig = plotting.plotTF(
            ff, tf, mag_ax=mag_ax, phase_ax=phase_ax, **kwargs)
        return fig

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

    def plotQuantumASD(self, probeName, driveNames, fig=None, **kwargs):
        """Plot the quantum ASD of a probe

        Plots the ASD of a probe referenced the the transfer function for
        some signal, such as DARM.

        Inputs:
          probeName: name of the probe
          driveNames: names of the drives from which the TF to refer the
            noise to
          fig: if not None, an existing figure to plot the noise on
            (Default: None)
          **kwargs: extra keyword arguments to pass to the plot
        """
        ff = self.ff
        if driveNames:
            tf = self.getTF(probeName, driveNames)
            noiseASD = np.abs(self.getQuantumNoise(probeName)/tf)
        else:
            noiseASD = np.abs(self.getQuantumNoise(probeName))

        if fig is None:
            newFig = True
            fig = plt.figure()
        else:
            newFig = False

        fig.gca().loglog(ff, noiseASD, **kwargs)
        fig.gca().set_ylabel('Noise')
        fig.gca().set_xlim([min(ff), max(ff)])
        fig.gca().set_xlabel('Frequency [Hz]')
        fig.gca().xaxis.grid(True, which='both', alpha=0.5)
        fig.gca().xaxis.grid(alpha=0.25, which='minor')
        fig.gca().yaxis.grid(True, which='both', alpha=0.5)
        fig.gca().yaxis.grid(alpha=0.25, which='minor')

        if newFig:
            return fig

    def save(self, fname):
        """Export data to an hdf5 file

        Inputs:
          fname: file name to save data to
        """
        data = h5py.File(fname, 'w')
        data['probes'] = io.str2byte(self.probes)
        data['amp_detectors'] = io.str2byte(self.amp_detectors)
        data['pos_detectors'] = io.str2byte(self.pos_detectors)
        data['bp_detectors'] = io.str2byte(self.bp_detectors)
        io.dict_to_hdf5(self._freqresp, 'freqresp', data)
        if len(self._mech_plants) > 0:
            # io.dict_to_hdf5(self.mechmod, 'mechmod',data)
            io.possible_none_to_hdf5(self._mechmod, 'mechmod', data)
            _mech_plants_to_hdf5(self._mech_plants, 'mech_plants', data)
        else:
            io.none_to_hdf5('mechmod', 'dict', data)
            io.none_to_hdf5('mech_plants', 'dict', data)
        if len(self._dcsigs) > 0:
            io.dict_to_hdf5(self._dcsigs, 'dcsigs', data)
        else:
            io.none_to_hdf5('dcsigs', 'dict', data)
        data['ff'] = self.ff
        data['lambda0'] = self.lambda0
        data.close()

    def load(self, fname):
        """Load stored data from an hdf5 file

        Inputs:
          fname: file name to read from
        """
        data = h5py.File(fname, 'r')
        self._probes = io.byte2str(data['probes'][()])
        self._amp_detectors = io.byte2str(data['amp_detectors'][()])
        self._pos_detectors = io.byte2str(data['pos_detectors'][()])
        self._bp_detectors = io.byte2str(data['bp_detectors'][()])
        self._freqresp = io.hdf5_to_dict(data['freqresp'])
        if isinstance(data['mech_plants'], h5py.Group):
            # self.mechmod = io.hdf5_to_dict(data['mechmod'])
            self._mechmod = io.hdf5_to_possible_none('mechmod', data)
            self._mech_plants = _mech_plants_from_hdf5(data['mech_plants'], data)
        else:
            self._mechmod = {}
            self._mech_plants = {}
        self._dcsigs = io.hdf5_to_possible_none('dcsigs', data)
        self._ff = data['ff'][()]
        self._lambda0 = data['lambda0'][()]
        data.close()


def _mech_plants_to_hdf5(dictionary, path, h5file):
    for key, val in dictionary.items():
        fpath = path + '/' + key
        if isinstance(val, dict):
            _mech_plants_to_hdf5(val, fpath, h5file)
            h5file[fpath].attrs['isfilter'] = False
        elif isinstance(val, ctrl.Filter):
            val.to_hdf5(path + '/' + key, h5file)
            h5file[fpath].attrs['isfilter'] = True
        else:
            raise ValueError('Something is wrong')


def _mech_plants_from_hdf5(h5_group, h5file):
    plants = {}
    for key, val in h5_group.items():
        if isinstance(val, h5py.Group):
            if val.attrs['isfilter']:
                plants[key] = ctrl.filt_from_hdf5(val.name, h5file)
            else:
                plants[key] = _mech_plants_from_hdf5(val, h5file)
        else:
            raise ValueError('Something is wrong')
    return plants
