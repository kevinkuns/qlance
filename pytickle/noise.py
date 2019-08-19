from __future__ import division
import numpy as np


def resamplePSD(ffOld, ffNew, Sold):
    """Resample a PSD to a new frequency vector

    The power in each frequency bin is the integrated power of the PSD at the
    right end of that bin:
      P_n = S_n * df_{n - 1}; (df_{n} = f_n - f_{n -1})
    The integrated power from DC is the first frequency point is the
    integral of the first PSD value:
      P_0 = \int_0^{f_0} S(f) df,
    which is taken to be S(f_0) * (f_0 - 0)

    Inputs:
      ffOld: old frequency vector
      ffNew: new frequency vector
      Sold: old PSD evaluated at ffOld

    Returns:
      Snew: the resampled PSD evaluated at ffNew

    Based off of pickle's rebinSpec.m
    """
    # S has units of A**2 / Hz
    # make sure the old spectrum has a DC value
    if ffOld[0] != 0:
        ffDC = False
        ffOld = np.concatenate(([0], ffOld))
        Sold = np.concatenate(([0], Sold))
    else:
        ffDC = True

    # Old power in units of A**2
    Pold = np.diff(ffOld) * Sold[1:]
    Pold = np.concatenate(([0], Pold))
    Pnew = np.zeros_like(ffNew)

    # Ensure that the old frequency vector is as long as the new one
    if ffNew[-1] > ffOld[-1]:
        ffOld = np.concatenate((ffOld, [ffNew[-1]]))
        Pold = np.concatenate((Pold, [0]))

    # Make each new bin have the same power as the old bins it overlaps
    ffLast = 0
    oi = 0
    for ni, ffNewBin in enumerate(ffNew):
        # Loop through new bins
        if ffNewBin <= ffOld[oi + 1]:
            # Still in the last bin
            # add fraction of power Sold*df
            try:
                Pnew[ni] += Sold[oi + 1] * (ffNewBin - ffLast)
            except IndexError:
                print(ni, oi, ffNewBin, ffOld[oi])

        else:
            # Move on to the next bin
            # add fraction of power from the first old bin
            oi += 1
            Pnew[ni] += Sold[oi] * (ffOld[oi] - ffLast)

            # loop through old bins covered by this new bin
            while ffOld[oi + 1] <= ffNewBin:
                oi += 1
                Pnew[ni] += Pold[oi]

            # add fraction of power from the last old bin
            Pnew[ni] += Sold[oi] * (ffNewBin - ffOld[oi])

        ffLast = ffNewBin

    # New PSD in A**2 / Hz
    Snew = Pnew[1:] / np.diff(ffNew)
    if ffNew[0] != 0:
        # The new PSD doesn't have a DC value
        S0 = Pnew[0] / ffNew[0]
    else:
        if ffDC:
            # Both PSDs have a DC value
            S0 = Sold[0]
        else:
            # The new PSD has a DC value, but the old one doesn't
            S0 = Sold[0]
    Snew = np.concatenate(([S0], Snew))

    return Snew
