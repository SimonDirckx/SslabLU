


from scipy.sparse.linalg   import LinearOperator
import numpy as np
from matAssembly.HBS import HBSTree as HBS
import solver.solver as solver
import time
import matplotlib.pyplot as plt
from matAssembly.HBS.simpleoctree import simpletree as tree
import torch
def compute_c0_L0(XX):
    N,ndim = XX.shape
    XX_np = XX
    if torch.is_tensor(XX):
        XX_np = XX_np.detach().cpu().numpy()
    c0 = np.sum(XX_np,axis=0)/N
    L0 = np.max(np.max(XX_np,axis=0)-np.min(XX_np,axis=0)) #too tight for some reason
    return c0,L0+1e-5
class matAssemblerOptions:
    """
    Options for matrix constuction
    """
    def __init__(self,method:str='dense',tol:np.double=1e-5,leaf_size:int=8,maxRank:int=8,tree=None,reduced=False):
        #todo: add checks of str!='dense'
        self.method     = method
        self.tol        = tol
        self.maxRank    = maxRank
        self.tree       = tree
        self.leaf_size  = leaf_size
        self.reduced    = reduced
class matAssemblerStats:
    def __init__(self):
        self.timeMatvecs    = 0
        self.timeHBS        = 0
        self.nbytes         = 0


class matAssembler:
    """
    Wrapper class for the construction of matrices
    """
    def __init__(self,matOpts:matAssemblerOptions=matAssemblerOptions()):
        self.matOpts    = matOpts
        self.nbytes       = 0
        self.stats = matAssemblerStats()
    def assemble(self,stMap:solver.stMap,dbg=0):
        reduced = self.matOpts.reduced
        linOp = stMap.A

        print("MAT ASSEMBLER METHOD=%s" % self.matOpts.method) if dbg > 0 else None
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
            toc_tree  = time.time() - start

            start = time.time()
            m=linOp.shape[0]
            n=linOp.shape[1]
            s=5*(self.matOpts.maxRank+10)
            s=max(s,self.matOpts.maxRank+10+self.matOpts.leaf_size)
            Om  = np.random.standard_normal(size=(n,s))
            Psi = np.random.standard_normal(size=(m,s))
            start = time.time()
            Y = linOp@Om
            Z = linOp.T@Psi
            timeRand = time.time()-start
            self.stats.timeMatvecs=timeRand

            Y = torch.from_numpy(Y)
            Z = torch.from_numpy(Z)
            Om = torch.from_numpy(Om)
            Psi = torch.from_numpy(Psi)
            
            if dbg>0:
                print("\t Toc tree %5.2f s, toc solve %d random pdes %5.2f s" %(toc_tree, s, self.stats.timeMatvecs))
            start = time.time()
            hbsMat = HBS.HBSMAT(tree0,Om,Psi,Y,Z,self.matOpts.maxRank,reduced)
            timeHBS = time.time()-start
            self.stats.timeHBS=timeHBS
            if dbg>0:
                print("\t Toc HBS construction %5.2f s"%self.stats.timeHBS)
            self.stats.nbytes = hbsMat.nbytes
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
