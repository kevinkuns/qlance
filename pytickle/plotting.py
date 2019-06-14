from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
from utils import mag2db


def plotTF(ff, tf, mag_ax=None, phase_ax=None, dB=False, phase2freq=False,
           **kwargs):
    """Plots a transfer function.

    Inputs:
        ff: the frequency at which the transfer function is evaluated [Hz]
        tf: the transfer function
        mag_ax: If not None, existing axis to plot the magnitude on
            (default: None)
        phase_ax: If not None, existing axis to plot the phase on
            (default: None)
        dB: If True, plots the magnitude in dB (default: False)
        phase2freq: If true, the transfer function tf/f is plotted instead
            of tf, which converts a phase TF to a frequency TF
            (default: False)
        **kwargs: any arguments (color, linestyle, label, etc.) to pass
            to the plot

    Returns:
      fig: if no mag_ax and phase_ax have been given, returns the new fig

    If mag_ax and phase_ax are None, then a new figure with the transfer
    function will be returned. However, if there are existing axes
    (for example from an existing TF plot) they can be passed to mag_ax
    and phase_ax and the new TF will be added to the existing axes. In this
    case no new figure is returned.

    Minimal labeling is done with this function. So, for example, to
    add a ylabel, title, and legend to a new TF
        fig = plotTF(ff, tf, label='legend label')
        mag_ax = fig.axes[0]
        mag_ax.set_xlabel('magnitude [W/m]')
        mag_ax.set_title('Plot Label')
        mag_ax.legend()
    Note that fig.axes[1] is phase_ax. A new TF can be plotted on this same
    figure by
        plotTF(ff, tf2, mag_ax=fig.axes[0], phase_ax=fig.axes[1], color='C1',
               label='second TF')
        mag_ax.legend()  # need to redraw legend
    """
    if not(mag_ax and phase_ax):
        if (mag_ax is not None) or (phase_ax is not None):
            msg = 'If one of the phase or magnitude axes is given,'
            msg += ' the other must be given as well.'
            raise ValueError(msg)
        newFig = True
    else:
        newFig = False

    if newFig:
        fig = plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.05)
        mag_ax = fig.add_subplot(gs[0])
        phase_ax = fig.add_subplot(gs[1], sharex=mag_ax)
    else:
        old_ylims = mag_ax.get_ylim()

    if phase2freq:
        tf = tf/ff

    if dB:
        mag_ax.semilogx(ff, mag2db(np.abs(tf)), **kwargs)
        mag_ax.set_ylabel('Magnitude [dB]')
        mag_ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    else:
        mag_ax.loglog(ff, np.abs(tf), **kwargs)
        mag_ax.set_ylabel('Magnitude')
        magTF = np.abs(tf)
        # If the TF is close to being constant magnitude, increase ylims
        # in order to show y tick labels and avoid a misleading plot.
        if np.abs(np.max(magTF)/np.min(magTF)) < 10:
            # mag_ax.set_yscale('linear')
            mag_ax.set_ylim(np.mean(magTF)/10.1, np.mean(magTF)*10.1)

    # If plotting ontop of an old TF, adjust the ylims so that the old TF
    # is still visible
    if not newFig:
        new_ylims = mag_ax.get_ylim()
        mag_ax.set_ylim(min(old_ylims[0], new_ylims[0]),
                        max(old_ylims[1], new_ylims[1]))

    mag_ax.set_xlim(min(ff), max(ff))
    phase_ax.set_ylim(-185, 185)
    # ticks = np.linspace(-180, 180, 7)
    ticks = np.arange(-180, 181, 45)
    phase_ax.yaxis.set_ticks(ticks)
    phase_ax.semilogx(ff, np.angle(tf, True), **kwargs)
    phase_ax.set_ylabel('Phase [deg]')
    phase_ax.set_xlabel('Frequency [Hz]')
    plt.setp(mag_ax.get_xticklabels(), visible=False)
    mag_ax.grid(True, which='both', alpha=0.5)
    mag_ax.grid(alpha=0.25, which='minor')
    phase_ax.grid(True, which='both', alpha=0.5)
    phase_ax.grid(alpha=0.25, which='minor')
    if newFig:
        return fig


def plotAbsTF(ff, tf, ax=None, dB=False, phase2freq=False, **kwargs):
    """Plots the magnitude of a transfer function

    Inputs:
    ff: the frequency at which the transfer function is evaluated [Hz]
    tf: the transfer function
    ax: If not None, existing axis to plot the transfer function on
        (default: None)
    dB: If True, plots the magnitude in dB (default: False)
    phase2freq: If true, the transfer function tf/f is plotted instead
        of tf, which converts a phase TF to a frequency TF
        (default: False)
    **kwargs: any arguments (color, linestyle, label, etc.) to pass
        to the plot

    Returns:
      fig: if no mag_ax and phase_ax have been given, returns the new fig

    See plotTF for more information.
    """
    if ax is None:
        newFig = True
    else:
        newFig = False

    if newFig:
        fig = plt.figure()
        ax = fig.gca()
    else:
        old_ylims = ax.get_ylim()

    if phase2freq:
        tf = tf/ff

    if dB:
        ax.semilogx(ff, mag2db(np.abs(tf)), **kwargs)
        ax.set_ylabel('Magnitude [dB]')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    else:
        ax.loglog(ff, np.abs(tf), **kwargs)
        ax.set_ylabel('Magnitude')
        magTF = np.abs(tf)
        # If the TF is close to being constant magnitude, increase ylims
        # in order to show y tick labels and avoid a misleading plot.
        if np.abs(np.max(magTF)/np.min(magTF)) < 10:
            # mag_ax.set_yscale('linear')
            ax.set_ylim(np.mean(magTF)/10.1, np.mean(magTF)*10.1)

    # If plotting ontop of an old TF, adjust the ylims so that the old TF
    # is still visible
    if not newFig:
        new_ylims = ax.get_ylim()
        ax.set_ylim(min(old_ylims[0], new_ylims[0]),
                    max(old_ylims[1], new_ylims[1]))

    ax.set_xlim(min(ff), max(ff))
    ax.set_xlabel('Frequency [Hz]')
    ax.grid(True, which='both', alpha=0.5)
    ax.grid(alpha=0.25, which='minor')
    if newFig:
        return fig
