"""
Control filters
"""

import numpy as np
import scipy.signal as sig
# import IIRrational.AAA as AAA
# import wavestate.AAA as AAA
from . import AAA
from functools import partial
from .utils import assertArr
from . import io
from itertools import zip_longest
from collections import OrderedDict
from numbers import Number
import scipy.signal as sig
from abc import ABC, abstractmethod
from copy import deepcopy
import matplotlib.pyplot as plt
from . import plotting


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
    # check what kind of filters are to be concatenated
    nzpk = 0
    nsos = 0
    nfreq = 0
    for filt in args:
        if not isinstance(filt, Filter):
            raise ValueError('Can only concatenate filters')
        if isinstance(filt, ZPKFilter):
            nzpk += 1
        elif isinstance(filt, SOSFilter):
            nsos += 1
        elif isinstance(filt, FreqFilter):
            nfreq += 1

    # if all filters are ZPK, define a new one combining this information
    if nsos == 0 and nfreq == 0:
        zs = []
        ps = []
        k = 1
        for filt in args:
            zf, pf, kf = filt.get_zpk(Hz=False)
            zs.extend(assertArr(zf))
            ps.extend(assertArr(pf))
            k *= kf
        if len(args) == 0:
            k = 0

        return ZPKFilter(zs, ps, k, Hz=False)

    # if all filters are SOS, define a new one combining this information
    if nzpk == 0 and nfreq == 0:
        sos = np.empty((0, 6))
        for filt in args:
            sos = np.vstack((filt.sos, sos))

        return SOSFilter(sos, fs=filt.fs)

    # otherwise just make a new Freq
    else:
        def new_filter_func(ff):
            out = 1
            for filt in args:
                out *= filt(ff)
            return out

        return FreqFilter(new_filter_func, Hz=True)


def _plot_zp(zps, zp_type, Hz=True, fig=None):
    """Plots zeros or poles in the complex plane

    Inputs:
      zps: an array of the zeros or poles in the s-domain
      zp_type: 'zero' for zeros and 'pole' for poles
      Hz: whether the zeros and poles should be plotted in
        Hz (True) or rad/s (False)
      fig: if not None, existing figure to plot on

    Returns:
      fig: the figure
    """
    if fig is None:
        fig = plt.figure()
        w, h = fig.get_size_inches()
        dd = min([w, h])
        fig.set_size_inches(dd, dd)
        ax = fig.add_subplot(111)
    else:
        ax = fig.gca()

    a = (2*np.pi)**Hz
    zps = zps / a

    kwargs = dict(fillstyle='none', linestyle='none', markeredgewidth=3)
    if zp_type == 'zero':
        kwargs['marker'] = 'o'
        kwargs['color'] = 'xkcd:cerulean'
        kwargs['fillstyle'] = 'none'
    elif zp_type == 'pole':
        kwargs['marker'] = 'x'
        kwargs['color'] = 'xkcd:brick red'
    else:
        raise ValueError('the zp_type should only be pole or zero')

    ax.plot(zps.real, zps.imag, **kwargs)

    if Hz:
        freq_label = 'Frequency [Hz]'
    else:
        freq_label = 'Frequency [rad/s]'
    ax.set_xlabel(freq_label)
    ax.set_ylabel(freq_label)

    return fig


class Filter:

    def computeFilter(self, ff):
        """Compute the filter

        Inputs:
          ff: frequency vector at which to compute the filter [Hz]
        """
        return self(ff)

    def plotFilter(self, ff, mag_ax=None, phase_ax=None, dB=False, **kwargs):
        """Plot the filter

        See documentation for plotting.plotTF
        """
        fig = plotting.plotTF(
            ff, self(ff), mag_ax=mag_ax, phase_ax=phase_ax,
            dB=dB, **kwargs)
        return fig

    def fitZPK(self, ff, **kwargs):
        return FitTF(ff, self(ff), **kwargs)


class ZPKFilter(Filter):
    """A class representing a generic filter

    Inputs:
      The filter can be specified in one of four ways:
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
    def __init__(self, *args, Hz=True):
        if Hz:
            a = -2*np.pi
        else:
            a = 1

        if len(args) == 1:
            if isinstance(args[0], dict):
                zs = a*np.array(args[0]['zs'])
                ps = a*np.array(args[0]['ps'])
                k = args[0]['k']
                self._zs = zs
                self._ps = ps
                self._k = k

            else:
                raise ValueError(
                    'One argument filters should be dictionaries or functions')

        elif len(args) == 3:
            zs = a*np.array(args[0])
            ps = a*np.array(args[1])
            k = args[2]
            if not isinstance(k, Number):
                raise ValueError('The gain should be a scalar')

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

    @property
    def is_stable(self):
        """True if the filter is stable
        (the real part of all poles are negative)
        """
        return np.all(np.real(self._ps < 0))

    @property
    def is_mindelay(self):
        """True if the filter has minimum delay
        (the real part of all zeros are negative)
        """
        return np.all(np.real(self._zs < 0))

    def __call__(self, ff):
        ss = 2j*np.pi*ff
        return zpk(self._zs, self._ps, self._k, ss)

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

    def plot_poles(self, Hz=True):
        """Plot the poles of the filter

        Plots the poles of the filter in the complex plane

        Inputs:
          Hz: if true the poles are plotted in Hz, however stable
            poles are still plotted in the LHP (Default: True)

        Returns:
          fig: the figure
        """
        fig = _plot_zp(self._ps, 'pole', Hz=Hz)
        return fig

    def plot_zeros(self, Hz=True):
        """Plot the zeros of the filter

        Plots the zeros of the filter in the complex plane

        Inputs:
          Hz: if true the zeros are plotted in Hz, however minimum delay
            zeros are still plotted in the LHP (Default: True)

        Returns:
          fig: the figure
        """
        fig = _plot_zp(self._zs, 'zero', Hz=Hz)
        return fig

    def plot_zp(self, Hz=True):
        """Plot the zeros and poles of the filter

        Plots the zeros of the filter as blue circles and the poles as
        red crosses in the complex plane

        Inputs:
          Hz: if true the zeros and poles are plotted in Hz, however
            unstable poles and minimum delay zeros are still plotted in
            the LHP (Default: True)

        Returns:
          fig: the figure
        """
        fig = _plot_zp(self._zs, 'zero', Hz=Hz)
        _plot_zp(self._ps, 'pole', Hz=Hz, fig=fig)
        return fig

    def to_hdf5(self, path, h5file):
        """Save a filter to an hdf5 file

        Inputs:
          path: full path where to save the filter
          h5file: the hdf5file

        Example:
          To save the filter filt to an hdf5 file 'data.hdf5':
          h5file = h5py.File('data.hdf5', 'w')
          filt.to_hdf5('filt', h5file)
        """
        zs, ps, k = self.get_zpk(Hz=False)
        io.possible_none_to_hdf5(zs, path + '/zs', h5file)
        io.possible_none_to_hdf5(ps, path + '/ps', h5file)
        h5file[path + '/k'] = k

    @classmethod
    def from_hdf5(cls, path, h5file):
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
        return cls(zpk_dict, Hz=False)


class SOSFilter(Filter):

    empty_sos = np.array([[1, 0, 0, 1, 0, 0]], dtype=float)

    def __init__(self, sos, fs=16384):
        self._sos = deepcopy(sos)
        self._fs = fs

    @property
    def fs(self):
        """Sampling frequency [Hz]
        """
        return self._fs

    @fs.setter
    def fs(self, fs):
        self._fs = fs

    @property
    def sos(self):
        """SOS coefficients
        """
        return self._sos

    def __call__(self, ff):
        _, tf = sig.sosfreqz(self.sos, worN=ff, fs=self.fs)
        return tf


class FreqFilter(Filter):

    empty_filt = lambda ff: np.oneslike(ff)

    def __init__(self, filter_func, Hz=True):
        if Hz:
            self._filter_function = lambda ff: filter_func(ff)
        else:
            self._filter_function = lambda ff: filter_func(2j*np.pi*ff)

    def __call__(self, ff):
        return self._filter_function(ff)


class FitTF(ZPKFilter):
    """Filter class to fit transfer functions

    Fits transfer functions using the AAA algorithm

    Inputs:
      ff: frequency vector of data to be fit [Hz]
      data: transfer function data
      kwargs: keyword arguments to pass to the AAA fit
    """
    def __init__(self, ff, data, **kwargs):
        super().__init__([], [], 0)
        self._ff = ff
        self._data = data
        self._fit = AAA.tfAAA(ff, data, **kwargs)
        zs, ps, k = self.fit.zpk

        # convert to s-plane
        a = 2*np.pi
        self._zs = a * zs
        self._ps = a * ps

        # convert IIRrational's gain convention
        exc = len(ps) - len(zs)  # excess number of poles over zeros
        self._k = (2*np.pi)**exc * k

        self._filt = partial(zpk, self._zs, self._ps, self._k)

    @property
    def fit(self):
        """IIRrational AAA fit results
        """
        return self._fit

    @property
    def ff(self):
        """Frequency vector used for the fit
        """
        return self._ff

    @property
    def data(self):
        """Fit data
        """
        return self._data

    def __call__(self, ff):
        return self.fit(ff)

    def check_fit(self, ff, fit_first=True):
        """Plot data along with the interpolated fit

        Inputs:
          ff: frequency vector for interpolation [Hz]
          fit_first:
            if True, the data is plotted on top of the fit
            if False, the fit is plotted on top of the data
            (Default: True)

        Returns:
          fig: the figure
        """
        if fit_first:
            fig = self.plotFilter(ff, label='Fit')
            plotting.plotTF(
                self.ff, self.data, *fig.axes, ls='-.', label='Data')

        else:
            fig = plotting.plotTF(self.ff, self.data, label='Data')
            self.plotFilter(ff, *fig.axes, ls='-.', label='Fit')

        fig.axes[0].legend()
        return fig


class FilterBank(ABC):
    def __init__(self, name=''):
        self._filter_modules = []
        self._state = np.array([], dtype=bool)
        self.name = name
        self._gain = 1

    @property
    def filter_modules(self):
        return self._filter_modules

    @property
    def gain(self):
        """Overal gain of the filter bank
        """
        return self._gain

    @gain.setter
    def gain(self, val):
        assert isinstance(val, Number)
        self._gain = float(val)
        self._update()

    @property
    def nfilters(self):
        """The number of filters in the filter bank
        """
        return len(self._filter_modules)

    @property
    def engaged_filters(self):
        """Array of engaged filter modules
        """
        engaged_filters = np.array(self._filter_modules)[self._state]
        engaged_filters = [filt for _, filt in engaged_filters]
        return engaged_filters

    @property
    def num_engaged(self):
        """Number of engaged filters
        """
        return len(self.engaged_filters)

    def addFilterModule(self, filt, name=''):
        """Add a filter module to the filter bank

        Inputs:
          filt: the filter
          name: name of the filter module (Optional)
        """
        if not isinstance(filt, Filter):
            raise ValueError('Filter modules must be a type of filter')
        self._filter_modules.append((name, filt))
        self._state = np.concatenate((self._state, [False]))

    def turn_on(self, *args):
        """Turn on a list of filter modules indexed by their number

        Existing engaged filters are not turned off
        The numbers are the number of the filter bank, i.e. 1-based

        Example:
          Turn on filter banks 2, 5, and 8
            fbank.turn_on(2, 5, 8)
          if filter bank 4 was previously on, banks 2, 4, 5, an 8 are on
        """
        self._state[self._get_filter_inds(*args)] = True
        self._update()

    def turn_off(self, *args):
        """Turn off a list of filter modules indexed by their number

        The numbers are the number of the filter bank, i.e. 1-based

        Example:
          Turn off filter banks 2, 5, and 8
            fbank.turn_off(2, 5, 8)
          if filter bank 4 was previously on, bank 4 is the only bank on
        """
        self._state[self._get_filter_inds(*args)] = False
        self._update()

    def engage(self, *args):
        """Specify which filter modules are engaged indexed by their number

        This sets the state of the filter bank, so previously engaged modules
        modules are turned off and only those explicity listed are turned on
        The numbers are the number of the filter bank, i.e. 1-based

        Example:
          Engage only filter banks 2, 5, and 8
            fbank.engage(2, 5, 8)
        """
        self.turn_off('all')
        self.turn_on(*args)

    def toggle(self, *args):
        """Toggle the state of filter modules indexed by their number

        The numbers are the number of the filter bank, i.e. 1-based
        """
        state = self._state.astype(int)
        state[self._get_filter_inds(*args)] += 1
        self._state = np.mod(state, 2).astype(bool)
        self._update()

    @abstractmethod
    def _update(self):
        pass

    def __str__(self):
        fb_str = self.name + '\n'
        for num, (name, module) in enumerate(self.filter_modules):
            if self._state[num] == True:
                fm_state = 'On'
            else:
                fm_state = 'Off'
            fb_str += '{:<2d} | {:<3s} | {:s}\n'.format(num + 1, fm_state, name)
        fb_str += 'gain: ' + str(self.gain)
        return fb_str

    def __repr__(self):
        return self.__str__()

    def _get_filter_inds(self, *args):
        if self.nfilters == 0:
            raise ValueError('There are no filters defined')

        inds = np.array(args)

        if len(inds) == 1:
            if inds[0] == 'all':
                return np.ones(self.nfilters, dtype=bool)
            elif inds.dtype != int:
                raise ValueError('Unrecognized argument: ' + str(inds[0]))

        if inds.dtype != int:
            raise ValueError('Invalid arguments')

        if np.any(inds > self.nfilters):
            raise ValueError('There are only {:d} filters'.format(self.nfilters))

        if np.any(inds < 1):
            raise ValueError('Filters cannot have index less than 1')

        return inds - 1


class ZPKFilterBank(ZPKFilter, FilterBank):
    def __init__(self, name=''):
        ZPKFilter.__init__(self, [], [], 1)
        FilterBank.__init__(self, name=name)

    def _update(self):
        if self.num_engaged:
            new_filt = catfilt(*self.engaged_filters)
            self._zs = new_filt._zs
            self._ps = new_filt._ps
            self._k = new_filt._k
        else:
            self._zs = []
            self._ps = []
            self._k = 1
        self._k *= self.gain


class SOSFilterBank(SOSFilter, FilterBank):
    def __init__(self, name=''):
        SOSFilter.__init__(self, SOSFilter.empty_sos)
        FilterBank.__init__(self, name=name)

    @classmethod
    def from_foton_file(cls, file_name, filterbank_name):
        """Load a filter bank from a foton file

        Inputs:
          file_name: the path to the foton file
          filterbank_name: the name of the filter bank to load

        Returns:
          the filter bank

        Example:
          Load the H1 DARM1 filter
            fbank = SOSFilterBank.from_foton_file('H1OMC.txt', 'LSC_DARM1')
        """
        foton_filterbanks = io.read_foton_file(file_name)
        try:
            foton_filterbank = foton_filterbanks[filterbank_name]
        except KeyError:
            msg = filterbank_name + ' is not a filterbank in this file.'
            raise ValueError(msg)

        filterbank = cls()
        for filter_module in foton_filterbank.values():
            filt = SOSFilter(filter_module['sos_coeffs'], filter_module['fs'])
            filterbank.addFilterModule(filt, name=filter_module['name'])
        filterbank.name = filterbank_name
        return filterbank

    def _update(self):
        if self.num_engaged:
            new_filt = catfilt(*self.engaged_filters)
            self._sos = new_filt.sos
        else:
            self._sos = deepcopy(SOSFilter.empty_sos)
        self._sos[0, :3] *= self.gain


class FreqFilterBank(FreqFilter, FilterBank):
    def __init__(self, name=''):
        FreqFilter.__init__(self, FreqFilter.empty_filt)
        FilterBank.__init__(self, name=name)

    def _update(self):
        if self.num_engaged:
            new_filt = catfilt(*self.engaged_filters)
            self._filter_function = lambda ff: new_filt(ff) * self.gain
        else:
            self._filter_function = FreqFilter.empty_filt * self.gain
