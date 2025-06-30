


from scipy.sparse.linalg   import LinearOperator
import numpy as np
from matAssembly.HBS import HBSTreeNEW as HBS
import solver.solver as solver
import time
import matplotlib.pyplot as plt
from matAssembly.HBS.simpleoctree import simpletree as tree
import torch
def compute_c0_L0(XX):
    N,ndim = XX.shape
    c0 = np.sum(XX,axis=0)/N
    L0 = np.max(np.max(XX,axis=0)-np.min(XX,axis=0)) #too tight for some reason
    return c0,L0+1e-5
class matAssemblerOptions:
    """
    Options for matrix constuction
    """
    def __init__(self,method:str='dense',tol:np.double=1e-5,leaf_size:int=8,maxRank:int=8,tree=None):
        #todo: add checks of str!='dense'
        self.method     = method
        self.tol        = tol
        self.maxRank    = maxRank
        self.tree       = tree
        self.leaf_size  = leaf_size


class matAssembler:
    """
    Wrapper class for the construction of matrices
    """
    def __init__(self,matOpts:matAssemblerOptions=matAssemblerOptions()):
        self.matOpts    = matOpts
        self.nbytes       = 0
    def assemble(self,stMap:solver.stMap,reduced=False,dbg=0):
        linOp = stMap.A
        if self.matOpts.method == 'dense':
            M=linOp@np.identity(linOp.shape[1])
            self.nbytes = M.nbytes
            return M
        
        if self.matOpts.method == 'epsHBS':
            
            return TypeError('epsHBS currently not implemented')
        if self.matOpts.method == 'rkHBS':
            start = time.time()
            if self.matOpts.tree:
                tree0 = self.matOpts.tree
            else:
                c0,L0 = compute_c0_L0(stMap.XXI)
                tree0 =  tree.BalancedTree(stMap.XXI,self.matOpts.leaf_size,c0,L0)
            m=linOp.shape[0]
            n=linOp.shape[1]
            s=6*(self.matOpts.maxRank+10)
            s=max(s,self.matOpts.maxRank+10+self.matOpts.leaf_size)
            Om  = np.random.standard_normal(size=(n,s))
            Psi = np.random.standard_normal(size=(m,s))
            
            Y = linOp@Om
            Z = linOp.T@Psi
            Y = torch.from_numpy(Y)
            Z = torch.from_numpy(Z)
            Om = torch.from_numpy(Om)
            Psi = torch.from_numpy(Psi)
            timeTreeAndRand = time.time()-start
            if dbg>0:
                print("time tree & Om+Psi = ",timeTreeAndRand)
            start = time.time()
            hbsMat = HBS.HBSMAT(tree0,Om,Psi,Y,Z,self.matOpts.maxRank+10,reduced)
            timeHBS = time.time()-start
            if dbg>0:
                print("time HBS construction = ",timeHBS)
            self.nbytes = hbsMat.nbytes
            #stop=time.time()
            #print("time compress = ",stop-start)
            def matmat(v,transpose=False):
                if transpose:
                    return hbsMat.matvecT(v.astype('float64'))
                else:
                    return hbsMat.matvec(v.astype('float64'))
            
            LinopTest = LinearOperator(shape=(m,n),\
                matvec = matmat, rmatvec = lambda v: matmat(v,transpose=True),\
                matmat = matmat, rmatmat = lambda v: matmat(v,transpose=True))
            return LinopTest

'''
DEFAULT ASSEMBLERS
'''

class denseMatAssembler(matAssembler):
    def __init__(self):
        super(denseMatAssembler,self).__init__(matAssemblerOptions())

class rkHMatAssembler(matAssembler):
    def __init__(self,leaf_size,rk,tree=None):
        super(rkHMatAssembler,self).__init__(matAssemblerOptions('rkHBS',0,leaf_size,rk,tree))

class tolHMatAssembler(matAssembler):
    def __init__(self,tol,leaf_size,rk):
        super(tolHMatAssembler,self).__init__(matAssemblerOptions('epsHBS',tol,leaf_size,rk))