import numpy as np
from itertools import tee
from .utils import mat2py, py2mat, str2mat


def pairwise(iterable):
    """ Iterator for s -> (s0,s1), (s1,s2), (s2, s3), ...

    From https://docs.python.org/3.7/library/itertools.html#itertools-recipes
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


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
        if len(optics) < 2:
            raise ValueError('Need at least two optics to trace a beam')

        self.opt = opt
        self.optics = list(optics)


    def propagateABCD(self, *optics, q=1):
        abcd_path = np.identity(2)
        abcd_start = np.identity(2)
        for optStart, optEnd in pairwise(list(optics)):
            outPort, inPort = self._getLinkPorts(optStart, optEnd)
            abcd_path = np.einsum(
                'ij,jk,kl->il', self.getABCD(optStart, optEnd),
                abcd_start, abcd_path)
            abcd_start = self.getABCD(optEnd, inPort, outPort)

        return abcd_path

    def traceBeam(self, *args, q=None):
        pass

    def _eval(self, cmd, nargout=0):
        """Evaluate a matlab command using the pytickle model's engine

        Inputs:
          cmd: the matlab command string
          nargout: the number of outputs to be returned (Defualt: 0)

        Returns:
          The outputs from matlab
        """
        return self.opt.eng.eval(cmd, nargout=nargout)

    def _getLinkPorts(self, linkStart, linkEnd):
        """Get the ports of the start and end of a link
        """
        self._eval("nlink = {:s}.getLinkNum({:s}, {:s});".format(
            self.opt.optName, str2mat(linkStart), str2mat(linkEnd)))
        self._eval("link = opt.link(nlink);")
        self._eval(("optStart = {opt}.getOptic({opt}".format(opt=self.opt.optName)
                    + ".getOpticName(link.snSource));"))
        self._eval(("optEnd = {opt}.getOptic({opt}".format(opt=self.opt.optName)
                    + ".getOpticName(link.snSink));"))
        startInfo = self._eval("optStart.getOutputName(link.portSource)", 1)
        endInfo = self._eval("optEnd.getInputName(link.portSink)", 1)
        startName, startPort = startInfo.split('->')
        endName, endPort = endInfo.split('<-')

        if startName != linkStart:
            raise ValueError(
                'linkStart {:s} != startName {:s}'.format(linkStart, startName))
        if endName != linkEnd:
            raise ValueError(
                'linkEnd {:s} != endName {:s}'.format(linkEnd, endName))

        return startPort, endPort
