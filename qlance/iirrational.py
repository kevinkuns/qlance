"""
Code for plant fitting and interfacing with IIRrational
"""

import numpy as np
from functools import partial
from IIRrational.v2 import data2filter
from . import controls as ctrl


class PlantFit(ctrl.Filter):
    """Filter class to fit plants from frequency response data

    Inputs:
      ff: frequency vector of data to be fit [Hz]
      data: plant data to fit
      kwargs: extra keywords to pass to IIRrational's data2filter fitter

    * By default the SNR is set to be 1e5 for all points. This can be
      overwritten by specifying an array with the 'SNR' keyword.
    * By default the initial fit order is set to be 20. This can be overwritten
      with the 'order_initial' keyword.

    Examples:
      * Find the plant for data tf evaluated at frequency vector ff starting at
        order 20:
          plant = PlantFit(ff, tf)
      * Find the plant starting at order 10:
          plant = PlantFit(ff, tf, order_initial=10)
    """
    def __init__(self, ff, data, **kwargs):
        ctrl.Filter.__init__(self, [], [], 0)
        self._ff = ff
        self._data = data

        if 'SNR' not in kwargs:
            kwargs['SNR'] = 1e5 * np.ones_like(ff)

        if 'order_initial' not in kwargs:
            kwargs['order_initial'] = 20

        self._fit = data2filter(data=data, F_Hz=ff, **kwargs)

    @property
    def ff(self):
        """Frequency vector used for the fit
        """
        return self._ff

    @property
    def data(self):
        """Data used for the fit
        """
        return self._data

    def _set_zpk(self, zs, ps, k, Hz=False):
        """Set the zpk of the underlying filter

        THIS SHOULD ONLY BE USED TO MANIPULATE THE FIT
        """
        a = (-2*np.pi)**Hz
        self._zs = a*zs
        self._ps = a*ps
        self._k = a*k
        self._filt = partial(ctrl.zpk, self._zs, self._ps, self._k)

    def _get_fit_zpk(self):
        """Get the zpk of the current fit

        Poles and zeros are returned in the s-domain
        """
        zpk_data = self._fit.as_ZPKrep()
        zs = 2*np.pi * zpk_data.zeros.fullplane
        ps = 2*np.pi * zpk_data.poles.fullplane
        k = zpk_data.gain
        exc = len(ps) - len(zs)  # excess number of poles over zeros
        k = (2*np.pi)**exc * k  # convert IIRrational's gain convention
        return zs, ps, k

    def choose(self, order):
        """Choose which fit order to use

        Inputs:
          order: fit order
        """
        self._fit.choose(order)
        zs, ps, k = self._get_fit_zpk()
        self._set_zpk(zs, ps, k, Hz=False)

    def investigate_order_plot(self):
        """Show IIRrational's order plot for this fit

        Returns:
          the figure
        """
        return self._fit.investigate_order_plot()

    def investigate_fit_plot(self):
        """Show IIRrational's fit plot for this fit order

        Returns:
          the figure
        """
        return self._fit.investigate_fit_plot()
