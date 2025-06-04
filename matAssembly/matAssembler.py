


from scipy.sparse.linalg   import LinearOperator
import numpy as np
import matAssembly.HBS.HBSTree as HBS
import solver.solver as solver
import time
import matplotlib.pyplot as plt
class matAssemblerOptions:
    """
    Options for matrix constuction
    """
    def __init__(self,method:str='dense',tol:np.double=1e-5,leaf_size:int = 0,maxRank:int=8,tree=None):
        #todo: add checks of str!='dense'
        self.method     = method
        self.tol        = tol
        self.leaf_size  = leaf_size
        self.maxRank    = maxRank
        self.tree       = tree
        


class matAssembler:
    """
    Wrapper class for the construction of matrices
    """
    def __init__(self,matOpts:matAssemblerOptions=matAssemblerOptions()):
        self.matOpts    = matOpts
    def assemble(self,stMap:solver.stMap,tree=None):
        linOp = stMap.A
        
        if self.matOpts.method == 'epsHBS' or self.matOpts.method == 'rkHBS':
            treeI = HBS.HBS_tree_from_points(stMap.XXI,self.matOpts.leaf_size)
        if self.matOpts.method == 'dense':
            return linOp.matmat(np.identity(linOp.shape[1]))
        
        if self.matOpts.method == 'epsHBS':
            
            m=linOp.shape[0]
            n=linOp.shape[1]
            s=(self.matOpts.maxRank+10)
            Om  = np.random.standard_normal(size=(n,s))
            Psi = np.random.standard_normal(size=(m,s))
            Y = linOp.matmat(Om)
            Z = linOp.rmatmat(Psi)
            HBS.compress_HBS_eps(treeI,Om,Psi,Y,Z,self.matOpts.maxRank,s,self.matOpts.tol)
            self.tree = treeI
            def matmat(v,transpose=False):
                return HBS.apply_HBS(treeI,v,transpose)
            return LinearOperator(shape=(m,n),\
                matvec = matmat, rmatvec = lambda v: matmat(v,transpose=True),\
                matmat = matmat, rmatmat = lambda v: matmat(v,transpose=True))
        if self.matOpts.method == 'rkHBS':
            m=linOp.shape[0]
            n=linOp.shape[1]
            s=5*self.matOpts.maxRank
            Om  = np.random.standard_normal(size=(n,s))
            Psi = np.random.standard_normal(size=(m,s))
            #print("sizes = ",n,"//",m)
            start = time.time()
            Y = linOp.matmat(Om)
            Z = linOp.rmatmat(Psi)
            stop = time.time()
            #print("YZ time = ",stop-start)
            #print("sizes Y,Z = ",Y.shape,"//",Z.shape)
            #start=time.time()
            HBS.compress_HBS(treeI,Om,Psi,Y,Z,self.matOpts.maxRank,s)
            self.tree = treeI
            #stop=time.time()
            #print("time compress = ",stop-start)
            def matmat(v,transpose=False):
                return HBS.apply_HBS(treeI,v,transpose)
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
    def __init__(self,leaf_size,rk):
        super(rkHMatAssembler,self).__init__(matAssemblerOptions('rkHBS',0,leaf_size,rk))

class tolHMatAssembler(matAssembler):
    def __init__(self,tol,rk=4):
        super(tolHMatAssembler,self).__init__(matAssemblerOptions('epsHBS',tol,rk))