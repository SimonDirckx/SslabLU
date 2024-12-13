


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
    def __init__(self,matOpts:matAssemblerOptions=matAssemblerOptions()):
        self.matOpts    = matOpts
    def assemble(self,linOp:LinearOperator):
        if self.matOpts.method == 'dense':
            return linOp.matmat(np.identity(linOp.shape[1]))


'''
DEFAULT ASSEMBLERS
'''

class denseMatAssembler(matAssembler):
    def __init__(self):
        super(denseMatAssembler,self).__init__(matAssemblerOptions())

class rkHMatAssembler(matAssembler):
    def __init__(self,rk,tree):
        super(denseMatAssembler,self).__init__(matAssemblerOptions('HBF',1e-5,rk,tree))

class tolHMatAssembler(matAssembler):
    def __init__(self,tol,tree):
        super(denseMatAssembler,self).__init__(matAssemblerOptions('HBF',tol,20,tree))