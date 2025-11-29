


from scipy.sparse.linalg   import LinearOperator
import numpy as np
from matAssembly.HBS import HBSTree as HBS
import solver.solver as solver
import time
import matplotlib.pyplot as plt
from matAssembly.HBS.simpleoctree import simpletree as tree
import torch


# Classes for matrix assembly



def compute_c0_L0(XX):
    """
    compute the center and charasteristic length of the domain to be clustered
    (this is necessary only becuase there is a bug in simpletree)
    
    """
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

    method: dense(default), epsHBS (deprecated), rkHPS (rank based HBS compression)
    tol:    tolerance for HBS construction (deprecated)
    leaf_size: #DOFs per leaf (default = 8)
    maxRank: maximal HBS rank
    tree:   if specified, use this cluster tree set-up
    ndim:   dimension of the interfaces (2 if problem is 3D, 1 if problem is 1D)
    reduced: (deprecated, default false)


    """
    def __init__(self,method:str='dense',tol:np.double=1e-5,leaf_size:int=8,maxRank:int=8,tree=None,ndim=3,reduced=False):
        #todo: add checks of str!='dense'
        self.method     = method
        self.tol        = tol
        self.maxRank    = maxRank
        self.tree       = tree
        self.leaf_size  = leaf_size
        self.ndim = 3
        self.reduced    = reduced
class matAssemblerStats:
    """
    Stats of matrix constuction (only for dbg putposes)
    
    """
    
    def __init__(self):
        self.timeSample     = 0
        self.timeCompress   = 0
        self.nbytes         = 0


class matAssembler:
    """
    Wrapper class for the construction of matrices

    input
        matOpts:matAssemblerOptions, specifies the assembly options
    
    output
        either dense or HBS approx of operator, in linOp form (scipy compat)


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
            tic = time.time()
            M=linOp@np.identity(linOp.shape[1])
            self.stats.timeSample = time.time()-tic
            self.stats.nbytes = M.nbytes
            return M #linOp
        
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
            if self.matOpts.ndim==3:
                s=5*(self.matOpts.maxRank+10)
            if self.matOpts.ndim==2:
                s=4*(self.matOpts.maxRank+10)
            #s=max(s,self.matOpts.maxRank+10+self.matOpts.leaf_size)
            Om  = np.random.standard_normal(size=(n,s))
            Psi = np.random.standard_normal(size=(m,s))
            Y = linOp@Om
            Z = linOp.T@Psi
            self.stats.timeSample=time.time()-start

            Y = torch.from_numpy(Y)
            Z = torch.from_numpy(Z)
            Om = torch.from_numpy(Om)
            Psi = torch.from_numpy(Psi)
            
            if dbg>0:
                print("\t Toc tree %5.2f s, toc solve %d random pdes %5.2f s" %(toc_tree, s, self.stats.timeSample))
            start = time.time()
            hbsMat = HBS.HBSMAT(tree0,Om,Psi,Y,Z,self.matOpts.maxRank,reduced)
            timeHBS = time.time()-start
            self.stats.timeCompress=timeHBS
            if dbg>0:
                print("\t Toc HBS construction %5.2f s"%self.stats.timeCompress)
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
    def __init__(self,leaf_size,rk,tree=None,ndim=3):
        super(rkHMatAssembler,self).__init__(matAssemblerOptions('rkHBS',0,leaf_size,rk,tree,ndim))

class tolHMatAssembler(matAssembler):
    def __init__(self,tol,leaf_size,rk,ndim=3):
        super(tolHMatAssembler,self).__init__(matAssemblerOptions('epsHBS',tol,leaf_size,rk,ndim))
