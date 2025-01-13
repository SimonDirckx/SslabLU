


from scipy.sparse.linalg   import LinearOperator
import numpy as np
import HBS.HBSTree as HBS


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
        if self.matOpts.method == 'HBS':
            T = HBS.copy_tree_to_HBS(self.matOpts.tree)
            m=linOp.shape[0]
            n=linOp.shape[1]
            Om  = np.random.standard_normal(size=(n,self.matOpts.maxRank))
            Psi = np.random.standard_normal(size=(m,self.matOpts.maxRank))
            Y = linOp.matmat(Om)
            Z = linOp.rmatmat(Psi)
            HBS.compress_HBS_eps(T,Om,Psi,Y,Z,self.matOpts.maxRank,self.matOpts.tol)
            def matmat(v,transpose=False):

                if (v.ndim == 1):
                    v_tmp = v[:,np.newaxis]
                else:
                    v_tmp = v

                if (not transpose):
                    result = HBS.apply_HBS(T,v)
                else:
                    result = AIJ.T @ v_tmp
                    result -= AiJ.T @ (LUii.T@(AIi.T @ v_tmp))

                if (v.ndim == 1):
                    result = result.flatten()
                return result
            return LinearOperator(shape=(m,n),\
                matvec = matmat, rmatvec = lambda v: matmat(v,transpose=True),\
                matmat = matmat, rmatmat = lambda v: matmat(v,transpose=True))


'''
DEFAULT ASSEMBLERS
'''

class denseMatAssembler(matAssembler):
    def __init__(self):
        super(denseMatAssembler,self).__init__(matAssemblerOptions())

class rkHMatAssembler(matAssembler):
    def __init__(self,rk,tree):
        super(denseMatAssembler,self).__init__(matAssemblerOptions('HBS',1e-5,rk,tree))

class tolHMatAssembler(matAssembler):
    def __init__(self,tol,tree):
        super(denseMatAssembler,self).__init__(matAssemblerOptions('HBS',tol,20,tree))