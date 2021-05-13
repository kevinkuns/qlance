'''
Code for general optomechanical plants
'''

import numpy as np
import matplotlib.pyplot as plt
from . import controls as ctrl
from . import filters as filt
from . import plotting
from . import utils
from . import io
import h5py
from numbers import Number
from .utils import (beam_properties_from_q, get_default_kwargs,
                    append_str_if_unique)
from collections import OrderedDict


class Plant:
    """A generic plant
    """
    def __init__(self):
        self._plants = {}
        self._callable_plants = []
        self._plant_data = {}
        self._probes = []
        self._drives = []
        self._ff = None
        self._data_length = None

    @property
    def ff(self):
        """Frequency vector [Hz]
        """
        if self._ff is None:
            raise ValueError('No frequency vector has been defined')
        else:
            return self._ff

    @ff.setter
    def ff(self, ff):
        # the frequency vector should not be redefined if any plants have been
        # defined with data
        if self._data_length and self._ff is not None:
            raise ValueError(
                'Can\'t specify a new frequency vector if any plants are '
                + 'defined by data.')

        # check that the new frequency vector is compatible with existing data
        if self._data_length and self._data_length != len(ff):
            raise ValueError('Frequency vector does not have the same length '
                             + 'as existing data')

        # evaluate any callable plants for these new frequencies
        for probe, drive, plant in self._callable_plants:
            self._plants[probe][drive] = plant(ff)

        self._ff = ff

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

    def addPlant(self, probe, drive, plant):
        """Add a plant from a drive to a probe

        The plant can be specified in three ways
        1) Giving an array of data
        2) Giving a Filter instance
        3) Giving any callable function

        Inputs:
          probe: name of the probe
          drive: name of the drive
          plant: the plant
        """
        if probe in self._plants.keys():
            if drive in self._plants[probe].keys():
                raise ValueError(
                    'There is already a plant from {:s} to {:s}'.format(
                        drive, probe))
        else:
            self._plants[probe] = {}

        if callable(plant):
            self._callable_plants.append((probe, drive, plant))
            if self._ff:
                self._plants[probe][drive] = plant(self.ff)
            else:
                self._plants[probe][drive] = None

        else:
            if self._data_length is None:
                self._data_length = len(plant)
            else:
                if len(plant) != self._data_length:
                    raise ValueError(
                        'This data has a different length than existing data')
            self._plants[probe][drive] = plant

        append_str_if_unique(self._probes, probe)
        append_str_if_unique(self._drives, drive)

    def getTF(self, probes, drives, fit=False, **kwargs):
        """Compute a transfer function

        Inputs:
          probes: names of the probes at which the TF is calculated
          drives: names of the drives from which the TF is calculated
          fit: if True, returns a FitTF fit to the transfer function
            (Default: False)
        """
        if isinstance(drives, str):
            drives = {drives: 1}

        if isinstance(probes, str):
            probes = {probes: 1}

        tf = np.zeros(len(self.ff), dtype='complex')
        for probe, pc in probes.items():
            for drive, dc in drives.items():
                tf += pc*dc*self._plants[probe][drive]

        if fit:
            return filt.FitTF(self.ff, tf)
        else:
            return tf


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
        self._doftypes = ['pos', 'pitch', 'yaw', 'drive', 'amp', 'phase']
        self._optickle_sha = '???'

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

    @property
    def optickle_sha(self):
        """Optickle commit SHA used to compute
        """
        return self._optickle_sha

    def getTF(self, *args, **kwargs):
        """Compute a transfer function

        Inputs:
          Transfer functions can be computed in one of three ways:
          1) Specifying
              probes: name of the probes at which the TF is calculated
              drives: names of the drives from which the TF is calculated
              doftype: degree of freedom of the drives (Default: pos)
              optOnly: if True, only return the optical TF with no mechanics
                (Default: False)
              fit: if True, returns a FitTF fit to the transfer function
                (Default: False)
           2) Specifying probes and optOnly but using a DegreeOfFreedom to
              specify the drives and doftype
           3) Using a DegreeOfFreedom for the probes, drives, and doftype

        Returns:
          tf: the transfer function
            * In units of [W/m] if drive is an optic with doftype pos
            * In units of [W/rad] if drive is an optic with doftype pitch or yaw
            * In units of [W/rad] if drive is a PM modulator with doftype drive
            * In units of [W/RAM] if drive is an AM modulator with doftype drive
            * In units of [W/rad] if drive is an RF modulator with doftype phase
            * In units of [W/RAM] if drive is an RF modulator with doftype amp
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
              tf = opt.getTF('REFL', 'PM', doftype='drive')
            The frequency transfer function is then tf/(1j*ff). Equivalently
              PM = DegreeOfFreedom('PM', doftype='drive')
              tf = opt.getTF('REFL', PM)

          * If multiple drives are used, the drive names should be a dict.
            To compute the DARM transfer function to the AS_DIFF homodyne PD
              DARM = {'EX': 1/2, 'EY': -1/2}
              tf = opt.getTF('AS_DIFF', DARM)
            or equivalently
              DARM = DegreeOfFreedom({'EX': 1/2, 'EY': -1/2})
              tf = opt.getTF('AS_DIFF', DARM)
            or equivalently
              DARM = DegreeOfFreedom({'EX': 1/2, 'EY': -1/2}, probes='AS_DIFF')
              tf = opt.getTF(DARM)
            [Note: Since DARM is defined as Lx - Ly, to get 1 m of DARM
            requires 0.5 m of both Lx and Ly]

           * Note that if the probes are specified in a DegreeOfFreedom, these are
             the default probes. If different probes are specified in getTF, those
             will be used instead.
        """
        # figure out the kind of input and extract the probes and drives
        def get_dof_data(dof):
            doftype = dof.doftype
            drives = {
                key.split('.')[0]: val for key, val in dof.drives.items()}
            return doftype, drives

        if len(args) == 1:
            dof = args[0]
            if isinstance(dof, ctrl.DegreeOfFreedom):
                probes = dof.probes
                doftype, drives = get_dof_data(dof)
            else:
                raise TypeError(
                    'Single argument transfer functions must be a DegreeOfFreedom')
            kwargs = get_default_kwargs(kwargs, optOnly=False, fit=False)
            optOnly = kwargs['optOnly']
            fit = kwargs['fit']

        elif len(args) == 2:
            probes, drives = args
            if isinstance(drives, ctrl.DegreeOfFreedom):
                doftype, drives = get_dof_data(drives)
                kwargs = get_default_kwargs(kwargs, optOnly=False, fit=False)
                optOnly = kwargs['optOnly']
                fit = kwargs['fit']
            else:
                kwargs = get_default_kwargs(
                    kwargs, doftype='pos', optOnly=False, fit=False)
                doftype = kwargs['doftype']
                optOnly = kwargs['optOnly']
                fit = kwargs['fit']

        else:
            raise TypeError(
                'takes 2 positional arguments but ' + str(len(args)) + ' were given')

        if doftype not in self._doftypes:
            raise ValueError('Unrecognized degree of freedom {:s}'.format(doftype))

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
        if doftype in ['pos', 'drive', 'amp', 'phase']:
            tfData = tfData['pos']
        elif doftype == 'pitch':
            tfData = tfData['pitch']
        elif doftype == 'yaw':
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
                driveNum = self._getDriveIndex(drive, doftype)

                # add the contribution from this drive
                try:
                    tf += pc * drive_pos * tfData[probeNum, driveNum]
                except IndexError:
                    tf += pc * drive_pos * tfData[driveNum]

        if fit:
            return filt.FitTF(self.ff, tf)
        else:
            return tf

    def getMechMod(self, drive_out, drive_in, doftype='pos', fit=False):
        """Get the radiation pressure modifications to drives

        Inputs:
          drive_out: name of the output drive
          drive_in: name of the input drive
          doftype: degree of freedom: pos, pitch, or yaw (Default: pos)
          fit: if True, returns a FitTF fit to the transfer function
            (Default: False)
        """
        err_msg = ('Mechanical modifications can only be calculated for '
                   + 'degrees of freedom with one drive')
        if isinstance(drive_out, ctrl.DegreeOfFreedom):
            if len(drive_out.drives) != 1:
                raise ValueError(err_msg)
            for key in drive_out.drives.keys():
                drive_out, doftype_out = key.split('.')
        else:
            doftype_out = doftype

        if isinstance(drive_in, ctrl.DegreeOfFreedom):
            if len(drive_in.drives) != 1:
                raise ValueError(err_msg)
            for key in drive_in.drives.keys():
                drive_in, doftype_in = key.split('.')
        else:
            doftype_in = doftype

        if doftype_in not in self._doftypes:
            raise ValueError('Unrecognized in doftype ' + doftype_in)
        if doftype_out not in self._doftypes:
            raise ValueError('Unrecognized out doftype ' + doftype_out)

        # figure out which raw output matrix to use
        pos_types = ['pos', 'drive', 'amp', 'phase']
        if doftype_in in pos_types and doftype_out in pos_types:
            mMech = self._mMech['pos']
        elif doftype_in == 'pitch' and doftype_out == 'pitch':
            mMech = self._mMech['pitch']
        elif doftype_in == 'yaw' and doftype_out == 'yaw':
            mMech = self._mMech['yaw']
        else:
            raise ValueError('Input and output doftypes must be the same')

        if mMech is None:
            msg = 'Must run tickle for the appropriate DOF before ' \
                  + 'calculating a transfer function.'
            raise RuntimeError(msg)

        driveInNum = self._getDriveIndex(drive_in, doftype_in)
        driveOutNum = self._getDriveIndex(drive_out, doftype_out)

        tf = mMech[driveOutNum, driveInNum]
        if fit:
            return filt.FitTF(self.ff, tf)
        else:
            return tf

    def getMechTF(self, outDrives, inDrives, doftype='pos', fit=False):
        """Compute a mechanical transfer function

        Inputs:
          outDrives: name of the output drives
          inDrives: name of the input drives
          doftype: degree of freedom: pos, pitch, or yaw (Default: pos)
          fit: if True, returns a FitTF fit to the transfer function
            (Default: False)

        Returns:
          tf: the transfer function
            * In units of [m/N] for position
            * In units of [rad/(N m)] for pitch and yaw
        """
        if doftype not in self._doftypes:
            raise ValueError('Unrecognized degree of freedom {:s}'.format(doftype))

        # figure out the shape of the TF
        if isinstance(self.ff, Number):
            # TF is at a single frequency
            tf = 0
        else:
            # TF is for a frequency vector
            tf = np.zeros(len(self.ff), dtype='complex')

        if not isinstance(outDrives, ctrl.DegreeOfFreedom):
            outDrives = ctrl.DegreeOfFreedom(outDrives, doftype=doftype)
        if not isinstance(inDrives, ctrl.DegreeOfFreedom):
            inDrives = ctrl.DegreeOfFreedom(inDrives, doftype=doftype)

        # loop through drives to compute the TF
        for inDrive, c_in in inDrives.dofs():
            # get the default mechanical plant of the optic being driven
            dof_in = self._doftype2pos(inDrive.doftype)
            plant = self._mech_plants[dof_in][inDrive.name]
            if inDrive.doftype in ['drive', 'amp', 'phase']:
                z, p, k = plant.get_zpk()
                if k == 0:
                    plant = filt.ZPKFilter([], [], 1)

            for outDrive, c_out in outDrives.dofs():
                mmech = self.getMechMod(outDrive, inDrive)
                tf += c_in * c_out * plant.computeFilter(self.ff) * mmech

        if fit:
            return filt.FitTF(self.ff, tf)
        else:
            return tf

    def plotMechTF(self, outDrives, inDrives, mag_ax=None, phase_ax=None,
                   doftype='pos', **kwargs):
        """Plot a mechanical transfer function

        See documentation for plotTF in plotting
        """
        ff = self.ff
        tf = self.getMechTF(outDrives, inDrives, doftype=doftype)
        fig = plotting.plotTF(
            ff, tf, mag_ax=mag_ax, phase_ax=phase_ax, **kwargs)
        return fig

    def getQuantumNoise(self, probeName, doftype='pos', fit=False):
        """Compute the quantum noise at a probe

        Inputs:
          probeName: name of the probe
          doftype: degree of freedom (Default: pos)
          fit: if True, returns a FitTF fit to the transfer function
            (Default: False)

        Returns the quantum noise at a given probe in [W/rtHz]
        """
        probeNum = self.probes.index(probeName)
        try:
            qnoise = self._noiseAC[doftype][probeNum, :]
        except IndexError:
            qnoise = self._noiseAC[doftype][probeNum]

        if fit:
            return filt.FitTF(self.ff, qnoise)
        else:
            return qnoise

    def plotTF(self, *args, **kwargs):
        """Plot a transfer function.

        Transfer functions can be plotted by one of the three methods described
        in plotTF.

        By default a new figure is returned. The axes of an existing figure
        can be specified to plot multiple transfer functions on the same figure

        See documentation for plotTF in plotting

        Examples:
          Plot a transfer function on new figure specifying probes and drives
            DARM = {'EX': 1/2, 'EY': -1/2}
            fig = opt.plotTF('AS_DIFF', DARM)

          Plot on an existing figure 'fig' with a DegreeOfFreedom
            DARM = DegreeOfFreedom({'EX': 1/2, 'EY': -1/2}, probes='AS_DIFF')
            opt.plotTF('AS_DIFF', DARM, *fig.axes)

          Plot on an existing axis, using DegreeOfFreedom to specify probes
          and drives, and specify plot styles
            opt.plotTF(DARM, *fig.axes, label='DARM', color='xkcd:blue')

          Pass transfer function options with plot styles on an existing figure
            opt.plotTF('REFL', 'Laser', *fig.axes, doftype='freq', ls='-.')
        """
        tf_args, mag_ax, phase_ax = plotting._get_tf_args(*args)

        try:
            tf_kwargs = dict(doftype=kwargs.pop('doftype'))
        except KeyError:
            tf_kwargs = {}

        if 'optOnly' in kwargs:
            tf_kwargs['optOnly'] = kwargs.pop('optOnly')

        ff = self.ff
        tf = self.getTF(*tf_args, **tf_kwargs)
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

    def computeBeamSpotMotion(self, opticName, spotPort, driveName, doftype):
        """Compute the beam spot motion on one optic due to angular motion of another

        The beam spot motion must have been monitored by calling
        monitorBeamSpotMotion before running the model.

        Inputs:
          opticName: name of the optic to compute the BSM on
          spotPort: port of the optic to compute the BSM on
          driveName: drive name of the optic from which to compute the BSM from
          doftype: degree of freedom of the optic driving the BSM

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
        tf = self.getTF(probeName, driveName, doftype=doftype)

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
        if self._qq is None:
            raise RuntimeError(
                'Beam properties cannot be computed if a pitch or yaw response'
                + ' has not been calculated and an HG basis has not been set.')
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
        data['optickle_sha'] = self.optickle_sha
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
        try:
            self._optickle_sha = data['optickle_sha'][()]
        except KeyError:
            self._optickle_sha = '???'
        data.close()

    def _getDriveIndex(self, name, doftype):
        """Find the drive index of a given drive and degree of freedom
        """
        if doftype in ['pos', 'pitch', 'yaw']:
            driveNum = self.drives.index(name + '.pos')
        elif doftype in ['drive', 'amp', 'phase']:
            driveNum = self.drives.index('{:s}.{:s}'.format(name, doftype))
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

    def _doftype2opt(self, doftype):
        """Convert degrees of freedom to 1s, 2s, and 3s for Optickle
        """
        if doftype == 'pos':
            nDOF = 1
        elif doftype == 'pitch':
            nDOF = 2
        elif doftype == 'yaw':
            nDOF = 3
        else:
            raise ValueError('Unrecognized degree of freedom ' + str(doftype)
                             + '. Choose \'pos\', \'pitch\', or \'yaw\'.')

        return nDOF

    def _doftype2pos(self, doftype):
        """Convert doftypes into pos, pitch, or yaw
        """
        if doftype in ['pos', 'drive', 'amp', 'phase']:
            return 'pos'
        else:
            return doftype

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
        self._doftypes = ['pos', 'pitch', 'yaw', 'amp', 'freq']
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
        self._finesse_version = '?.?.?'

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

    @property
    def finesse_version(self):
        """Finesse version used to compute
        """
        return self._finesse_version

    def getTF(self, *args, **kwargs):
        """Compute a transfer function

        Inputs:
          Transfer functions can be computed in one of three ways:
          1) Specifying
              probes: name of the probes at which the TF is calculated
              drives: names of the drives from which the TF is calculated
              doftype: degree of freedom of the drives (Default: pos)
              fit: if True, returns a FitTF fit to the transfer function
                (Default: False)
           2) Specifying probes and optOnly but using a DegreeOfFreedom to
              specify the drives and doftype
           3) Using a DegreeOfFreedom for the probes, drives, and doftype

        Returns:
          tf: the transfer function
            * In units of [W/m] if drive is an optic with doftype pos
            * In units of [W/rad] if drive is an optic with doftype pitch or yaw
            * In units of [W/Hz] if drive is a laser with drive freq
            * In units of [W/RIN] if drive is a laser with drive amp

        Note:
          * To convert frequency noise in [W/Hz] to phase noise in [W/rad],
            multiply by 1j*ff
          * To convert intensity noise in [W/RIN] to amplitude noise in [W/RAM],
            multiply by 2 since RIN = 2*RAM

        Examples:
          * If only a single drive is used, the drive name can be a string.
            To compute the frequency transfer function in reflection from a FP
            cavity
              tf = katFR.getTF('REFL', 'Laser', doftype='freq')
            The phase transfer function is then 1j*ff*tf. Equivalently
              Laser = DegreeOfFreedom('Laser', doftype='freq')
              tf = katFR.getTF('REFL', Laser)

          * If multiple drives are used, the drive names should be a dict.
            To compute the DARM transfer function to the AS_DIFF homodyne PD
              DARM = {'EX': 1/2, 'EY': -1/2}
              tf = katFR.getTF('AS_DIFF', DARM)
            or equivalently
              DARM = DegreeOfFreedom({'EX': 1/2, 'EY': -1/2})
              tf = katFR.getTF('AS_DIFF', DARM)
            or equivalently
              DARM = DegreeOfFreedom({'EX': 1/2, 'EY': -1/2}, probes='AS_DIFF')
              tf = katFR.getTF(DARM)
            [Note: Since DARM is defined as Lx - Ly, to get 1 m of DARM
            requires 0.5 m of both Lx and Ly]

           * Note that if the probes are specified in a DegreeOfFreedom, these are
             the default probes. If different probes are specified in getTF, those
             will be used instead.
        """
        # figure out the kind of input and extract the probes and drives
        def get_dof_data(dof):
            doftype = dof.doftype
            drives = {
                key.split('.')[0]: val for key, val in dof.drives.items()}
            return doftype, drives

        doftype_msg = (
            'When specifying drives with a DegreeOfFreedom, the doftype should' \
            + 'be specified in the DegreeOfFreedom and not a keyword argument.')

        if len(args) == 1:
            dof = args[0]
            if isinstance(dof, ctrl.DegreeOfFreedom):
                probes = dof.probes
                doftype, drives = get_dof_data(dof)
            else:
                raise TypeError(
                    'Single argument transfer functions must be a DegreeOfFreedom')
            fit = get_default_kwargs(kwargs, fit=False)['fit']

        elif len(args) == 2:
            probes, drives = args
            if isinstance(drives, ctrl.DegreeOfFreedom):
                doftype, drives = get_dof_data(drives)
                fit = get_default_kwargs(kwargs, fit=False)['fit']
            else:
                kwargs = get_default_kwargs(kwargs, doftype='pos', fit=False)
                doftype = kwargs['doftype']
                fit = kwargs['fit']

        else:
            raise TypeError(
                'takes 2 positional arguments but ' + str(len(args)) + ' were given')

        if doftype not in self._doftypes:
            raise ValueError('Unrecognized doftype ' + doftype)

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
                tf += pc * drive_pos * self._freqresp[doftype][probe][drive]

        if fit:
            return filt.FitTF(self.ff, tf)
        else:
            return tf

    def getMechMod(self, drive_out, drive_in, doftype='pos', fit=False):
        """Get the radiation pressure modifications to drives

        Inputs:
          drive_out: name of the output drive
          drive_in: name of the input drive
          dof: degree of freedom: pos, pitch, or yaw (Default: pos)
          fit: if True, returns a FitTF fit to the transfer function
            (Default: False)

          doftype: degree of freedom: pos, pitch, or yaw (Default: pos)

        """
        err_msg = ('Mechanical modifications can only be calculated for '
                   + 'degrees of freedom with one drive')
        if isinstance(drive_out, ctrl.DegreeOfFreedom):
            if len(drive_out.drives) != 1:
                raise ValueError(err_msg)
            for key in drive_out.drives.keys():
                drive_out, doftype_out = key.split('.')
        else:
            doftype_out = doftype

        if isinstance(drive_in, ctrl.DegreeOfFreedom):
            if len(drive_in.drives) != 1:
                raise ValueError(err_msg)
            for key in drive_in.drives.keys():
                drive_in, doftype_in = key.split('.')
        else:
            doftype_in = doftype

        if doftype_in not in self._doftypes:
            raise ValueError('Unrecognized in doftype ' + doftype_in)
        if doftype_out not in self._doftypes:
            raise ValueError('Unrecognized out doftype ' + doftype_out)

        out_det = '_' + drive_out + '_' + doftype_out
        if out_det not in self.pos_detectors:
            raise ValueError(out_det + ' is not a detector in this model')

        tf = self._mechmod[doftype_in][out_det][drive_in]
        if fit:
            return filt.FitTF(self.ff, tf)
        else:
            return tf

    def getMechTF(self, outDrives, inDrives, doftype='pos', fit=False):
        """Compute a mechanical transfer function

        Inputs:
          outDrives: name of the output drives
          inDrives: name of the input drives
          doftype: degree of freedom: pos, pitch, or yaw (Default: pos)
          fit: if True, returns a FitTF fit to the transfer function
            (Default: False)

        Returns:
          tf: the transfer function
            * In units of [m/N] for position
            * In units of [rad/(N m)] for pitch and yaw
        """
        if doftype not in self._doftypes:
            raise ValueError('Unrecognized degree of freedom {:s}'.format(doftype))

        # figure out the shape of the TF
        if isinstance(self.ff, Number):
            # TF is at a single frequency
            tf = 0
        else:
            # TF is for a frequency vector
            tf = np.zeros(len(self.ff), dtype='complex')

        if not isinstance(outDrives, ctrl.DegreeOfFreedom):
            outDrives = ctrl.DegreeOfFreedom(outDrives, doftype=doftype)
        if not isinstance(inDrives, ctrl.DegreeOfFreedom):
            inDrives = ctrl.DegreeOfFreedom(inDrives, doftype=doftype)

        # loop through drives to compute the TF
        for inDrive, c_in in inDrives.dofs():
            # get the default mechanical plant of the optic being driven
            plant = self._mech_plants[inDrive.doftype][inDrive.name]

            for outDrive, c_out in outDrives.dofs():
                mmech = self.getMechMod(outDrive, inDrive)
                tf += c_in * c_out * plant.computeFilter(self.ff) * mmech

        if fit:
            return filt.FitTF(self.ff, tf)
        else:
            return tf

    def getQuantumNoise(self, probeName, doftype='pos', fit=False):
        """Compute the quantum noise at a probe

        Inputs:
          probeName: name of the probe
          doftype: degree of freedom (Default: pos)
          fit: if True, returns a FitTF fit to the transfer function
            (Default: False)

        Returns the quantum noise at a given probe in [W/rtHz]
        """
        shotName = '_' + probeName + '_shot'
        qnoise = list(self._freqresp[doftype][shotName].values())[0]
        # without copying multiple calls will reduce noise
        qnoise = qnoise.copy()

        if doftype == 'pos':
            # if this was computed from a position TF convert back to W/rtHz
            # 2*np.pi is correct here, not 360
            qnoise *= self.lambda0 / (2*np.pi)

        if np.any(np.iscomplex(qnoise)):
            print('Warning: some quantum noise spectra are complex')

        if fit:
            return filt.FitTF(self.ff, np.real(qnoise))
        else:
            return np.real(qnoise)

    def getSigDC(self, probeName):
        """Get the DC signal from a probe

        Inputs:
          probeName: name of the probe
        """
        return self._dcsigs[probeName]

    def computeBeamSpotMotion(self, node, driveName, doftype):
        """Compute the beam spot motion at a node

        The beam spot motion must have been monitored by calling
        monitorBeamSpotMotion before running the model. Both runDC and run for
        the degree of interest doftype must have been called on the model.

        Inputs:
          node: name of the node
          driveName: drive name of the optic from which to compute the BSM from
          doftype: degree of freedom of the optic driving the BSM

        Returns:
          bsm: the beam spot motion [m/rad]

        Example:
          To compute the beam spot motion on the front of EX due to pitch
          motion of IX
            monitorBeamSpotMotion(kat, 'EX_fr', 'pitch')
            katFR = KatFR(kat)
            katFR.runDC()
            katFR.run(fmin, fmax, npts, doftype='pitch')
            bsm = katFR.computeBeamSpotMotion('EX_fr', 'IX', 'pitch')
        """
        if doftype == 'pitch':
            direction = 'y'
        elif doftype == 'yaw':
            direction = 'x'
        else:
            raise ValueError('Unrecognized degree of freedom ' + direction)

        probeName = '_' + node + '_bsm_' + direction

        # get TF to monitoring probe power [W/rad]
        tf = self.getTF(probeName, driveName, doftype=doftype)

        # DC power on the monitoring probe
        Pdc = self.getSigDC('_' + node + '_DC')

        # get the beam size on the optic
        w, _, _, _, _, _ = self.getBeamProperties(node, doftype)

        # convert to spot motion [m/rad]
        bsm = w/Pdc * tf
        return bsm

    def getBeamProperties(self, node, doftype='pitch'):
        """Compute the properties of a Gaussian beam at a node

        Inputs:
          node: name of the node
          doftype: which degree of freedom 'pitch' or 'yaw' (Defualt: pitch)

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
        if doftype == 'pitch':
            direction = 'y'
        elif doftype == 'yaw':
            direction = 'x'
        else:
            raise ValueError('Unrecognized degree of freedom ' + direction)

        qq = self.getSigDC('_' + node + '_bp_' + direction)
        return beam_properties_from_q(qq, lambda0=self.lambda0)

    # def plotTF(self, *tf_args, mag_ax=None, phase_ax=None, **kwargs):
    def plotTF(self, *args, **kwargs):
        """Plot a transfer function.

        Transfer functions can be plotted by one of the three methods described
        in plotTF.

        By default a new figure is returned. The axes of an existing figure
        can be specified to plot multiple transfer functions on the same figure

        See documentation for plotTF in plotting

        Examples:
          Plot a transfer function on new figure specifying probes and drives
            DARM = {'EX': 1/2, 'EY': -1/2}
            fig = katFR.plotTF('AS_DIFF', DARM)

          Plot on an existing figure 'fig' with a DegreeOfFreedom
            DARM = DegreeOfFreedom({'EX': 1/2, 'EY': -1/2}, probes='AS_DIFF')
            katFR.plotTF('AS_DIFF', DARM, *fig.axes)

          Plot on an existing axis, using DegreeOfFreedom to specify probes
          and drives, and specify plot styles
            katFR.plotTF(DARM, *fig.axes, label='DARM', color='xkcd:blue')

          Pass transfer function options with plot styles on an existing figure
            katFR.plotTF('REFL', 'Laser', *fig.axes, doftype='freq', ls='-.')
        """
        tf_args, mag_ax, phase_ax = plotting._get_tf_args(*args)

        try:
            tf_kwargs = dict(doftype=kwargs.pop('doftype'))
        except KeyError:
            tf_kwargs = {}

        ff = self.ff
        tf = self.getTF(*tf_args, **tf_kwargs)
        fig = plotting.plotTF(
            ff, tf, mag_ax=mag_ax, phase_ax=phase_ax, **kwargs)

        return fig

    def plotMechTF(self, outDrives, inDrives, mag_ax=None, phase_ax=None,
                   doftype='pos', **kwargs):
        """Plot a mechanical transfer function

        See documentation for plotTF in plotting
        """
        ff = self.ff
        tf = self.getMechTF(outDrives, inDrives, doftype=doftype)
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
        data['finesse_version'] = self.finesse_version
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
        try:
            self._finesse_version = data['finesse_version'][()]
        except KeyError:
            self._finesse_version = '?.?.?'
        data.close()


def _mech_plants_to_hdf5(dictionary, path, h5file):
    for key, val in dictionary.items():
        fpath = path + '/' + key
        if isinstance(val, dict):
            _mech_plants_to_hdf5(val, fpath, h5file)
            h5file[fpath].attrs['isfilter'] = False
        elif isinstance(val, filt.Filter):
            val.to_hdf5(path + '/' + key, h5file)
            h5file[fpath].attrs['isfilter'] = True
        else:
            raise ValueError('Something is wrong')


def _mech_plants_from_hdf5(h5_group, h5file):
    plants = {}
    for key, val in h5_group.items():
        if isinstance(val, h5py.Group):
            if val.attrs['isfilter']:
                plants[key] = filt.filt_from_hdf5(val.name, h5file)
            else:
                plants[key] = _mech_plants_from_hdf5(val, h5file)
        else:
            raise ValueError('Something is wrong')
    return plants
