import numpy as np
from .utils import mat2py, py2mat, str2mat


def applyABCD(abcd, qi, n=1):
    """Transform a complex q parameter through an ABCD matrix

    Inputs:
      abcd: the ABCD matrix
      qi: the initial q parameter
      n: index of refraction (Default: 1)

    Returns:
      qf: the final q parameter
    """
    qi = qi / n
    qf = (abcd[0, 0]*qi + abcd[0, 1]) / (abcd[1, 0]*qi + abcd[1, 1])
    return qf * n


class GaussianPropagation:
    def __init__(self, opt):
        self.opt = opt

    def getABCD(self, name, inPort, outPort):
        """Get the ABCD matrix of an optic

        Inputs:
          name: name of the optic
          inPort: input port of the transformation
          outPort: output port of the transformation

        Returns:
          abcd: the ABCD matrix

        Example:
          To compute the ABCD matrix for reflection from the front of EX:
            gp.getABCD('EX', 'fr', 'fr')
        """
        self._eval("obj = {:s}.getOptic({:s})".format(
            self.opt.optName, str2mat(name)))
        self._eval("nIn = obj.getInputPortNum({:s})".format(
            str2mat(inPort)))
        self._eval("nOut = obj.getOutputPortNum({:s})".format(
            str2mat(outPort)))
        self._eval("qm = obj.getBasisMatrix()")
        abcd = mat2py(self._eval("qm(nOut, nIn).y", 1))
        return abcd

    def _eval(self, cmd, nargout=0):
        """Evaluate a matlab command using the pytickle model's engine

        Inputs:
          cmd: the matlab command string
          nargout: the number of outputs to be returned (Defualt: 0)

        Returns:
          The outputs from matlab
        """
        return self.opt.eng.eval(cmd, nargout=nargout)
