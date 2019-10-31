import numpy as np
from collections import OrderedDict
from itertools import tee
from .utils import mat2py, py2mat, str2mat


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


class GaussianPropagation:
    def __init__(self, opt, *optics):
        self.nopt = len(optics)
        if self.nopt < 2:
            raise ValueError('Need at least two optics to trace a beam')

        self.opt = opt
        self.optics = list(optics)
        self.fr_ABCDs = np.zeros((self.nopt, 2, 2))
        self.bk_ABCDs = np.zeros((self.nopt, 2, 2))
        self.fr_lens = np.zeros(self.nopt - 1)
        self.bk_lens = np.zeros(self.nopt - 1)

        self.fr_lens, self.fr_ABCDs, M_fr = self.propagateABCD(*optics)
        self.bk_lens, self.bk_ABCDs, M_bk = self.propagateABCD(*optics[::-1])

        # Find the initial and final ABCD matrices
        if self.nopt == 1:
            port_fr_out, port_bk_in = self._getLinkPorts(optics[0], optics[-1])
            port_bk_out, port_fr_in = self._getLinkPorts(optics[-1], optics[0])

        else:
            port_fr_out, _ = self._getLinkPorts(optics[0], optics[1])
            _, port_fr_in = self._getLinkPorts(optics[-2], optics[-1])
            port_bk_out, _ = self._getLinkPorts(optics[-1], optics[-2])
            _, port_bk_in = self._getLinkPorts(optics[1], optics[0])

        self.ABCD_0 = self.opt.getABCD(optics[0], port_fr_in, port_fr_out)
        self.ABCD_n = self.opt.getABCD(optics[-1], port_bk_in, port_bk_out)

        # Compute the round trip ABCD matrix
        print(M_bk, M_fr)
        self.rt_ABCD = np.einsum(
            'ij,jk,kl,lm->im', self.ABCD_0, M_bk, self.ABCD_n, M_fr)

    def propagateABCD(self, *optics):
        nopt = len(optics)  # optics not necessarily self.optics
        tot_ABCD= np.identity(2)
        path_ABCD = np.zeros((nopt - 2, 2, 2))
        link_lens = np.zeros(nopt - 1)
        for ii, (opt0, opt1, opt2) in enumerate(triples(list(optics))):
            _, port_in = self._getLinkPorts(opt0, opt1)
            port_out, _ = self._getLinkPorts(opt1, opt2)
            opt_ABCD = self.opt.getABCD(opt1, port_in, port_out)
            path_ABCD[ii] = opt_ABCD
            link_lens[ii] = self.opt.getLinkLength(opt0, opt1)
            len_ABCD = np.array([[1, link_lens[ii]],
                                 [0, 1]])
            tot_ABCD = np.einsum(
                'ij,jk,kl->il', opt_ABCD, len_ABCD, tot_ABCD)

        # add the length of the last link not accounted for with triples
        if nopt == 2:
            link_lens[0] = self.opt.getLinkLength(*optics)
        else:
            link_lens[-1] = self.opt.getLinkLength(*optics[-2:])
        len_ABCD = np.array([[1, link_lens[-1]],
                             [0, 1]])
        tot_ABCD = np.einsum('ij,jk->ik', len_ABCD, tot_ABCD)
        print(tot_ABCD)

        return link_lens, path_ABCD, tot_ABCD

    def traceBeam(self, *args, q=None):
        pass

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