import numpy as np
from collections import OrderedDict
from itertools import tee
from .matlab import mat2py, py2mat, str2mat
from . import plotting


def triples(iterable):
    """ Iterator for s -> (s0,s1,s2), (s1,s2,s3), (s2,s3,s4), ...

    based off of pairwise from
    https://docs.python.org/3.7/library/itertools.html#itertools-recipes
    """
    a, b, c = tee(iterable, 3)
    next(b, None)
    next(c, None)
    next(c, None)
    return zip(a, b, c)


def applyABCD(abcd, qi, ni=1, nf=1):
    """Transform a complex q parameter through an ABCD matrix

    Inputs:
      abcd: the ABCD matrix
      qi: the initial q parameter
      ni: initial index of refraction (Default: 1)
      nf: final index of refraction (Default: 1)

    Returns:
      qf: the final q parameter
    """
    qi = qi / ni
    qf = (abcd[0, 0]*qi + abcd[0, 1]) / (abcd[1, 0]*qi + abcd[1, 1])
    return qf * nf


def beam_properties_from_q(qq, lambda0=1064e-9):
    """Compute the properties of a Gaussian beam from a q parameter

    Inputs:
      qq: the complex q paramter
      lambda0: wavelength [m] (Default: 1064e-9)

    Returns:
      w: beam radius on the optic [m]
      zR: Rayleigh range of the beam [m]
      z: distance from the beam waist to the optic [m]
        Negative values indicate that the optic is before the waist.
      w0: beam waist [m]
      R: radius of curvature of the phase front on the optic [m]
      psi: Gouy phase [deg]
    """
    z = np.real(qq)
    zR = np.imag(qq)
    w0 = np.sqrt(lambda0*zR/np.pi)
    w = w0 * np.sqrt(1 + (z/zR)**2)
    R = zR**2 / z + z
    psi = (np.pi/2 - np.angle(qq)) * 180/np.pi
    return w, zR, z, w0, R, psi


def free_space_ABCD(length, n=1):
    """ABCD matrix for propagation through free space

    Inputs:
      length: propagation distance [m]
      n: index of refraction (Default: 1)
    """
    abcd = np.array([[1, length/n],
                     [0, 1]])
    return abcd


class GaussianPropagation:
    def __init__(self, opt, *optics, dof='pitch'):
        self.nopt = len(optics)
        if self.nopt < 2:
            raise ValueError('Need at least two optics to trace a beam')

        self.opt = opt
        self.optics = list(optics)
        self.dof = 'pitch'
        self.ABCDs = {direction: np.zeros((self.nopt, 2, 2))
                      for direction in ['fr', 'bk']}
        self.link_lens = {direction: np.zeros(self.nopt - 1)
                          for direction in ['fr', 'bk']}
        self.Chrs = OrderedDict(
            {optic: self.opt.getOpticParam(optic, 'Chr') for optic in optics})
        self.qq_opt = {}

        frwd_prop = self.propagateABCD(*optics)
        bkwd_prop = self.propagateABCD(*optics[::-1])
        self.link_lens['fr'], self.ABCDs['fr'], M_fr, self.qq_opt['fr'] = frwd_prop
        self.link_lens['bk'], self.ABCDs['bk'], M_bk, self.qq_opt['bk'] = bkwd_prop

        # Find the initial and final ABCD matrices
        if self.nopt == 1:
            port_fr_out, port_bk_in = self._getLinkPorts(optics[0], optics[-1])
            port_bk_out, port_fr_in = self._getLinkPorts(optics[-1], optics[0])

        else:
            port_fr_out, _ = self._getLinkPorts(optics[0], optics[1])
            _, port_fr_in = self._getLinkPorts(optics[-2], optics[-1])
            port_bk_out, _ = self._getLinkPorts(optics[-1], optics[-2])
            _, port_bk_in = self._getLinkPorts(optics[1], optics[0])

        self.ABCD_0 = self.opt.getABCD(
            optics[0], port_fr_in, port_fr_out, dof=self.dof)
        self.ABCD_n = self.opt.getABCD(
            optics[-1], port_bk_in, port_bk_out, dof=self.dof)

        # Compute the round trip ABCD matrix
        self.rt_ABCD = np.einsum(
            'ij,jk,kl,lm->im', self.ABCD_0, M_bk, self.ABCD_n, M_fr)

    def propagateABCD(self, *optics):
        """Compute the ABCD matrices along a beam path

        Inputs:
          *optics: a list of optics defining the path

        Returns:
          link_lens: the length of each link in the path
          path_ABCD: a (Noptic - 2, 2, 2) array of each ABCD matrix in the path
          tot_ABCD: the total ABCD matrix of the path

        Example:
          To compute the properties of the PRC from the ITM to the PRM
            prop = gp.propagateABCD('IX', 'BS', 'PR3', 'PR2', 'PR')
        """
        nopt = len(optics)  # optics not necessarily self.optics
        tot_ABCD= np.identity(2)
        path_ABCD = np.zeros((nopt - 2, 2, 2))
        link_lens = np.zeros(nopt - 1)
        qq_opt = {optic: {} for optic in optics}

        for ii, (opt0, opt1, opt2) in enumerate(triples(list(optics))):
            port0, port_in = self._getLinkPorts(opt0, opt1)
            port_out, port2 = self._getLinkPorts(opt1, opt2)
            opt_ABCD = self.opt.getABCD(opt1, port_in, port_out, dof=self.dof)
            path_ABCD[ii] = opt_ABCD
            link_lens[ii] = self.opt.getLinkLength(opt0, opt1)
            len_ABCD = free_space_ABCD(link_lens[ii])
            tot_ABCD = np.einsum(
                'ij,jk,kl->il', opt_ABCD, len_ABCD, tot_ABCD)
            qq_opt[opt1]['in'] = self.opt.getFieldBasis(opt1, port_in)
            qq_opt[opt1]['out'] = self.opt.getFieldBasis(opt1, port_out)
            if opt0 == optics[0]:
                qq_opt[opt0]['out'] = self.opt.getFieldBasis(opt0, port0)
            if opt2 == optics[-1]:
                qq_opt[opt2]['in'] = self.opt.getFieldBasis(opt2, port2)

        # add the length of the last link not accounted for with triples
        if nopt == 2:
            link_lens[0] = self.opt.getLinkLength(*optics)
            port0, port1 = self._getLinkPorts(*optics)
            qq_opt[optics[0]]['out'] = self.opt.getFieldBasis(optics[0], port0)
            qq_opt[optics[1]]['in'] = self.opt.getFieldBasis(optics[1], port1)
        else:
            link_lens[-1] = self.opt.getLinkLength(*optics[-2:])
        len_ABCD = free_space_ABCD(link_lens[-1])
        tot_ABCD = np.einsum('ij,jk->ik', len_ABCD, tot_ABCD)

        return link_lens, path_ABCD, tot_ABCD, qq_opt

    def traceBeam(self, qi, direction='fr', npts=100):
        """Compute the beam properties along the beam path

        Inputs:
          qi: the initial q paramter
          direction: beam direction 'fr' or 'bk' (Default: 'fr')
          npts: number of points to evaluate along each link in the path
            (Default: 100)

        Returns:
          dist: the distance from the inital point along the path
          qq: the q parameter along the path
        """
        qq_opt = OrderedDict({optic: {} for optic in self.optics})
        qq_opt[self.optics[0]]['out'] = qi
        dist = np.linspace(0, self.link_lens[direction][0], npts)
        qq = qi + dist
        qq_opt[self.optics[1]]['in'] = qq[-1]

        for li, link_len in enumerate(self.link_lens[direction][1:]):
            link_dist = np.linspace(0, link_len, npts)
            dist = np.concatenate((dist, dist[-1] + link_dist))
            qi = applyABCD(self.ABCDs[direction][li], qq[-1])
            qq_opt[self.optics[li + 1]]['out'] = qi
            qq = np.concatenate((qq, qi + link_dist))
            qq_opt[self.optics[li + 2]]['in'] = qq[-1]

        return dist, qq, qq_opt

    def plotBeamProperties(
            self, qi, bkwd=True, npts=100, plot_locs=True, plot_model=True):
        """Plot the beam properties along the beam path

        Inputs:
          qi: the initial q parameter
          bkwd: if True the beam is propagated backward as well (Default: True)
          npts: number of points to evaluate along each link in the path
            (Default: 100)
          plot_locs: if True the locations of the optics are marked with vertical
            dashed lines (Default: True)
          plot_model: if True, the beam parameters and mirror radii of curvature
            as determined by the optickle model are marked (Default: True)

        Returns:
          fig: the figure
        """
        dist, qq_fr, _ = self.traceBeam(qi, direction='fr', npts=npts)

        # locations of the optic in the path
        optlocs = np.cumsum(
            np.concatenate((np.array([0]), self.link_lens['fr'])))

        if plot_locs:
            fig = plotting.plotBeamProperties(dist, qq_fr, optlocs=optlocs)
        else:
            fig = plotting.plotBeamProperties(dist, qq_fr, optlocs=None)

        if bkwd:
            qi_bk = applyABCD(self.ABCD_n, qq_fr[-1])
            _, qq_bk, _ = self.traceBeam(qi_bk, direction='bk', npts=npts)
            plotting.plotBeamProperties(dist, qq_bk, fig, bkwd=True, ls='-.')

        if plot_model:
            rad_ax = fig.axes[0]
            roc_ax = fig.axes[1]
            ph_ax = fig.axes[2]
            marker = {'out': '>', 'in': '<'}
            c = {'fr': 'C0', 'bk': 'C1'}
            # qq_opt_fr = self.qq_opt['fr']
            for optic, optloc in zip(self.optics, optlocs):
                # qq_opt = qq_opt_fr[optic]
                Chr = self.Chrs[optic]
                fig.axes[1].plot(optloc, Chr, 'C0o')
                fig.axes[1].plot(optloc, -Chr, 'C0o', alpha=0.4)

                if bkwd:
                    directions = ['fr', 'bk']
                else:
                    directions = ['fr']
                for direction in directions:
                    qq_opt = self.qq_opt[direction][optic]
                    for port, qq in qq_opt.items():
                        w, _, _, _, R, psi = beam_properties_from_q(qq)
                        rad_ax.plot(
                            optloc, w, c=c[direction], marker=marker[port])
                        roc_ax.plot(
                            optloc, 1/R, c=c[direction], marker=marker[port])
                        ph_ax.plot(
                            optloc, psi, c=c[direction], marker=marker[port])

        return fig

    def getStability(self):
        """Compute the stability parameter m for the resonator

        For the round trip ABCD matrix,
          m = (A + D)/2
        and the resonator is stable if -1 <= m <= 1. This is only sensible
        if the path defined is actually a resonator.

        Note that for a two mirror resonator with g factors g1 and g2
          g1*g2 = (m + 1)/2
        """
        return self.rt_ABCD.trace() / 2

    def getRTGouyPhase(self):
        """Compute the round trip Gouy phase for the resonator

        The round trip Gouy phase is
          arccos((A + D)/2)
        where A and D characterize the round trip ABCD matrix.

        Note that for a two mirror resonator with g factors g1 and g2 this is
          2 * arccos(sqrt(g1*g2))

        Returns:
          dphi: the round trip Gouy phase [deg]
        """
        return np.arccos(self.getStability()) * 180/np.pi

    def _eval(self, cmd, nargout=0):
        """Evaluate a matlab command using the pytickle model's engine

        Inputs:
          cmd: the matlab command string
          nargout: the number of outputs to be returned (Defualt: 0)

        Returns:
          The outputs from matlab
        """
        return self.opt.eng.eval(cmd, nargout=nargout)

    def _getLinkPorts(self, link_start, link_end):
        """Get the ports of the start and end of a link
        """
        self._eval("nlink = {:s}.getLinkNum({:s}, {:s});".format(
            self.opt.optName, str2mat(link_start), str2mat(link_end)))
        self._eval("link = opt.link(nlink);")
        self._eval(("optStart = {opt}.getOptic({opt}".format(opt=self.opt.optName)
                    + ".getOpticName(link.snSource));"))
        self._eval(("optEnd = {opt}.getOptic({opt}".format(opt=self.opt.optName)
                    + ".getOpticName(link.snSink));"))
        start_info = self._eval("optStart.getOutputName(link.portSource)", 1)
        end_info = self._eval("optEnd.getInputName(link.portSink)", 1)
        start_name, start_port = start_info.split('->')
        end_name, end_port = end_info.split('<-')

        if start_name != link_start:
            raise ValueError('{:s} != {:s}'.format(link_start, start_name))
        if end_name != link_end:
            raise ValueError('{:s} != {:s}'.format(link_end, end_name))

        return start_port, end_port
