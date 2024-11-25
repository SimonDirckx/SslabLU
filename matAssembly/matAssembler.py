


from scipy.sparse.linalg   import LinearOperator
import numpy as np



class matAssemblerOptions:
    """
    Options for matrix constuction
    """
    def __init__(self,method:str='dense',tol:np.double=1e-5,maxRank:int=20,tree=None):
        #todo: add checks of str!='dense'
        self.method     = method
        self.tol        = tol
        self.maxRank    = maxRank
        self.tree       = tree


class matAssembler:
    """
    Wrapper class for the construction of matrices
    """
    def __init__(self,linOp:LinearOperator,matOpts:matAssemblerOptions):
        self.linOp      = linOp
        self.matOpts    = matOpts
    def assemble(self):
        return 0