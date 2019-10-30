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

    def getABCD(self, *args, dof='pitch'):
        """Get the ABCD matrix of an optic or path

        Returns the ABCD matrix of an optic if three arguments are supplied
        and the ABCD matrix of a path if two arguments are supplied

        Inputs (3 for optic):
          name: name of the optic
          inPort: input port of the transformation
          outPort: output port of the transformation
          dof: degree of freedom 'pitch' or 'yaw' (Default: 'pitch')

        Inputs (2 for path):
          linkStart: name of the start of the link
          linkEnd: name of the end of the link
          dof: degree of freedom 'pitch' or 'yaw' (Default: 'pitch')

        Returns:
          abcd: the ABCD matrix

        Examples:
          To compute the ABCD matrix for reflection from the front of EX:
            gp.getABCD('EX', 'fr', 'fr')
          To compute the ABCD matrix for propagation from IX to EX:
            gp.getABCD('IX', 'EX')
        """
        if len(args) == 3:
            name, inPort, outPort = args

            if dof == 'pitch':
                ax = "y"
            elif dof == 'yaw':
                ax = "x"

            self._eval("obj = {:s}.getOptic({:s})".format(
            self.opt.optName, str2mat(name)))
            self._eval("nIn = obj.getInputPortNum({:s})".format(
                str2mat(inPort)))
            self._eval("nOut = obj.getOutputPortNum({:s})".format(
                str2mat(outPort)))
            self._eval("qm = obj.getBasisMatrix()")
            abcd = mat2py(self._eval("qm(nOut, nIn).{:s}".format(ax), 1))

        elif len(args) == 2:
            linkStart, linkEnd = args
            linkLen = self.opt.getLinkLength(linkStart, linkEnd)
            abcd = np.array([[1, linkLen],
                             [0, 1]])

        else:
            raise ValueError('Incorrect number of arguments')

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
