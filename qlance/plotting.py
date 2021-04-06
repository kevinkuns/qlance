import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
from .utils import mag2db
# from .gaussian_beams import beam_properties_from_q


def plotTF(ff, tf, mag_ax=None, phase_ax=None, dB=False, **kwargs):
    """Plots a transfer function.

    Inputs:
        ff: the frequency at which the transfer function is evaluated [Hz]
        tf: the transfer function
        mag_ax: If not None, existing axis to plot the magnitude on
            (default: None)
        phase_ax: If not None, existing axis to plot the phase on
            (default: None)
        dB: If True, plots the magnitude in dB (default: False)
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
    The above can also be achieved with the shortcut
        plotTF(ff, tf2, *fig.axes, color='C1', label='second TF')
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
    mag_ax.grid(True, alpha=0.25, which='minor')
    phase_ax.grid(True, which='both', alpha=0.5)
    phase_ax.grid(True, alpha=0.25, which='minor')
    if newFig:
        return fig


# def plotBeamProperties(dist, qq, fig=None, optlocs=None, bkwd=False, **kwargs):
#     """Plot the beam properties along a beam path

#     Inputs:
#       dist: the distance from the initial point along the path
#       qq: the q parameter along the path
#       fig: if not None the beam path is plotted on this figure (Defualt: None)
#       bkwd: if True the beam is assumed to be backward propagating
#         (Defualt: False)
#       **kwargs: key word arguments for the plots

#     Returns:
#       fig: the figure if not given
#     """
#     if fig:
#         rad_ax = fig.axes[0]
#         roc_ax = fig.axes[1]
#         ph_ax = fig.axes[2]
#         newFig = False
#     else:
#         fig = plt.figure()
#         gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.05)
#         rad_ax = fig.add_subplot(gs[0])
#         roc_ax = fig.add_subplot(gs[1], sharex=rad_ax)
#         ph_ax = fig.add_subplot(gs[2], sharex=rad_ax)
#         newFig = True

#     if bkwd:
#         qq = qq[::-1]
#     w, _, _, _, R, psi = beam_properties_from_q(qq)
#     if bkwd:
#         # R = -R
#         # psi = -psi
#         pass

#     # Beam radius plot
#     rad_ax.plot(dist, w, **kwargs)
#     scale_axis(rad_ax.yaxis, 1e3)
#     rad_ax.set_ylabel('Beam radius [mm]')
#     rad_ax.grid(True, which='major', alpha=0.5)
#     plt.setp(rad_ax.get_xticklabels(), visible=False)

#     # Inverse ROC plot
#     roc_ax.plot(dist, 1/R, **kwargs)
#     roc_ax.set_ylabel('Inverse ROC [1/m]')
#     roc_ax.grid(True, which='major', alpha=0.5)
#     plt.setp(roc_ax.get_xticklabels(), visible=False)

#     # Gouy phase plot
#     ph_ax.plot(dist, psi, **kwargs)
#     ph_ax.set_ylabel('Gouy phase [deg]')
#     ph_ax.set_xlabel('Distance [m]')
#     ph_ax.grid(True, which='major', alpha=0.5)

#     if optlocs is not None:
#         for optloc in optlocs:
#             rad_ax.axvline(optloc, c='xkcd:forrest green', ls=':', alpha=0.5)
#             roc_ax.axvline(optloc, c='xkcd:forrest green', ls=':', alpha=0.5)
#             ph_ax.axvline(optloc, c='xkcd:forrest green', ls=':', alpha=0.5)

#     if newFig:
#         return fig


def scale_axis(ax, sf):
    ax.set_major_formatter(FuncFormatter(lambda x, p: '{:0.0f}'.format(sf*x)))


def _get_tf_args(*args):
    if len(args) == 1:
        # only DOF
        tf_args = args
        mag_ax = None
        phase_ax = None
    elif len(args) == 2:
        # probe, DOF/drives
        tf_args = args
        mag_ax = None
        phase_ax = None
    elif len(args) == 3:
        # probe, mag, phase
        tf_args = (args[0],)
        mag_ax, phase_ax = args[1:]
    elif len(args) == 4:
        # probe, DOF/probe, mag, phase
        tf_args = args[:2]
        mag_ax, phase_ax = args[2:]
    else:
        raise TypeError(
            'takes 4 positional arguments but ' + len(args) + ' were given')

    return tf_args, mag_ax, phase_ax
