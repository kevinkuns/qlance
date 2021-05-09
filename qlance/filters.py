"""
Control filters
"""

import numpy as np
import scipy.signal as sig
import IIRrational.AAA as AAA
from functools import partial
from .utils import assertArr
from . import io
from itertools import zip_longest
from collections import OrderedDict
from numbers import Number
import scipy.signal as sig
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


def catfilt(*args, as_filter=True):
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
        if not isinstance(filt, Filter):
            raise ValueError('Can only concatenate filters')
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
        if len(args) == 0:
            k = 0
        new_filter_func = partial(zpk, zs, ps, k)
        # return Filter(zs, ps, k, Hz=False)

    # otherwise just make a new function
    else:
        def new_filter_func(ss):
            out = 1
            for filt in args:
                out *= filt._filt(ss)
            return out

        zs = None
        ps = None
        k = None

    if as_filter:
        if zpk_defined:
            return Filter(zs, ps, k, Hz=False)
        else:
            return Filter(new_filter_func)

    else:
        return np.array(zs), np.array(ps), k, new_filter_func


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
        return self._filt(ss)

    def computeFilter(self, ff):
        """Compute the filter

        Inputs:
          ff: frequency vector at which to compute the filter [Hz]
        """
        return self.__call__(ff)

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


class SOSFilter:
    def __init__(self, sos, fs=16384):
        self._sos = sos
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

    def plotFilter(self, ff, mag_ax=None, phase_ax=None, dB=False, **kwargs):
        """Plot the filter

        See documentation for plotting.plotTF
        """
        fig = plotting.plotTF(
            ff, self(ff), mag_ax=mag_ax, phase_ax=phase_ax,
            dB=dB, **kwargs)
        return fig



class FitTF(Filter):
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


class FilterBank(Filter):
    def __init__(self):
        super().__init__([], [], 0)
        # self._filter_modules = OrderedDict()
        self._filter_modules = []
        self._state = np.array([], dtype=bool)

    @property
    def filter_modules(self):
        return self._filter_modules

    @property
    def nfilters(self):
        """The number of filters in the filter bank
        """
        # return len(self._filter_modules)
        return len(self._state)

    def get_filter(self, key_or_index):
        if isinstance(key_or_index, str):
            return self._filter_modules[key_or_index]
        elif isinstance(key_or_index, Number):
            keys = list(self._filter_modules.keys())
            key = keys[key_or_index]
            return self._filter_modules[key]

    def addFilterModule(self, filt, name=None):
        if not isinstance(filt, Filter):
            raise ValueError('Filter modules must be a type of filter')
        self._filter_modules.append((name, filt))
        self._state = np.concatenate((self._state, [False]))

    def turn_on(self, *args):
        self._state[self._get_filter_inds(*args)] = True
        # self._update()

    def turn_off(self, *args):
        self._state[self._get_filter_inds(*args)] = False
        # self._update()

    def toggle(self, *args):
        state = self._state.astype(int)
        state[self._get_filter_inds(*args)] += 1
        self._state = np.mod(state, 2).astype(bool)

    def _update(self):
        filters_on = np.array(self._filter_modules)[self._state]
        filters_on = [filt for _, filt in filters_on]
        self._zs, self._ps, self._k, self._filt = catfilt(
            *filters_on, as_filter=False)
        return filters_on

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
