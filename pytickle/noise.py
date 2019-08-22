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
      ffOld: old frequency vector [Hz]
      ffNew: new frequency vector [Hz]
      Sold: old PSD evaluated at ffOld [A**2 / Hz, for some units A]

    Returns:
      Snew: the resampled PSD evaluated at ffNew [A**2 / Hz]

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
    if ffNew[-1] >= ffOld[-1]:
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
            while ffOld[oi + 1] < ffNewBin:
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


def convolvePSDs(ff, df, PSD1, PSD2):
    """Compute the convolution of two PSDs

    The two PSDs are resampled with a linear frequency vector, convolved,
    and the convolution resampled to the original (possibly logarithmic)
    frequency vector

    Inputs:
      ff: the frequencies at which the PSDs are evaluated [Hz]
      df: linear spacing for the intermediate frequency vector [Hz]
      PSD1: the first PSD [A**2 / Hz, for some units A]
      PSD2: the second PSD [A**2 / Hz]

    Returns:
      convPSD: the convolution [A**2 / Hz]

    Based off of pickle's convNoise
    """
    # resample PSDs to linear spacing
    ffLin = np.linspace(0, ff[-1], int(ff[-1]/df))
    nff = len(ffLin)
    linPSD1 = resamplePSD(ff, ffLin, PSD1)
    linPSD2 = resamplePSD(ff, ffLin, PSD2)

    # Convert to double-sided PSDs
    linPSD1 = 1/2 * np.concatenate((linPSD1[::-1], [], linPSD1))
    linPSD2 = 1/2 * np.concatenate((linPSD2[::-1], [], linPSD2))

    # Do the convolution
    linConv = np.convolve(linPSD1, linPSD2) * df

    # Take the central part of the convolution
    inds = np.arange(nff) + 2*(nff - 1)
    linConv = 2 * linConv[inds]

    # Return convolution resampled to original frequency vector
    return resamplePSD(ffLin, ff, linConv)


def computeRMS(ff, asd):
    """Compute the cumulative RMS

    Inputs:
        ff: the frequency vector [Hz]
        asd: the ASD of a quantity [A/rtHz, for some units A]

    Returns:
        rms: the cumulative RMS [A]
    """
    dS = np.flip(asd**2 * ff)
    rms = np.flip(np.sqrt(np.cumsum(dS)))
    return rms
