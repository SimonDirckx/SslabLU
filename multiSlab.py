import numpy as np
import scipy as sc

import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import pdo.pdo
from geometry.slabGeometry import slabGeometry
from solver.solver import solverWrapper
from scipy.sparse.linalg   import LinearOperator
import itertools
import pdo.pdo as pdo
from matAssembly.matAssembler import matAssembler

class slabMap:
    def __init__(self,A,I,J):
        assert(A.shape[0]==len(J) and A.shape[1]==len(I))
        self.A=A
        self.globIdxs=[I,J]



class Slab:

    """
    Class representing slab geometry

    @param:
    geom        : geometry (l2g maps,bounds,...)
    PDE         : PDE to be solved
    solver      : the solver to be used (HPS/Spectral/Stencil/...)
    lrmaps      : left-right maps (encode type of slab decomp)
    matMethod   : method for the construction of the system blocks (dense/HBS/...)    
    """

    
    def __init__(   self,
                    geom:slabGeometry,
                    PDE:pdo.PDO2d,
                    solverWrap:solverWrapper,
                    mapType:str='DtD'
                ):
        
        self.solverWrap         =   solverWrap
        #
        self.solverWrap.construct(geom,PDE)
        self.Ii = self.solverWrap.Ii
        self.Ib = self.solverWrap.Ib
        self.XX             =   solverWrap.XX
        self.ndofs          =   self.XX.shape[0]
        self.N              =   self.ndofs
        self.geom           =   geom
        self.mapsComputed = False
        self.mapType=mapType
        self.computeLocalIdxs()
        if mapType=='DtD':
            self.globIdxs=[[],[],[]]
        if mapType=='DtN':
            self.globIdxs=[[],[]]
        
    #########################
    #   geometric methods   #
    #########################
    
    ######################################################################

    def l2g(self,i):
        """maps the ith local dof to the corresponding global coordinates"""
        p=self.XX[i,:]
        return self.geom.l2g(p)
    
    def getGlobalXX(self):
        """return global coordinates of vectorized dofs"""
        return np.array([self.l2g(i) for i in range(self.ndofs)])
    
    def eval_global_func(self,f,i):
        """helper function to evaluate globally defined f @ dof i"""
        p=self.l2g(i)
        #print(p)
        return f(p)
    ######################################################################

    #########################
    #   function methods    #
    #########################

    ######################################################################
    
    def computeRHS(self,f):
        """helper function to compute local RHS"""
        fGb=np.array([f(self.geom.l2g(self.solverWrap.XXb[i,:])) for i in self.idxsGB])
        rhs=[-self.solverWrap.stMap(self.idxsGB,idxs,self.mapType)@fGb for idxs in self.targetIdxs if idxs]
        rhsIdxs = [idxs for idxs in self.globTargetIdxs if idxs]
        self.rhs    =   rhs
        self.rhsIdxs=   rhsIdxs
        return rhs,rhsIdxs

    def computeLocalIdxs(self):
        
        
        leftIdxs    = self.solverWrap.leftIdxs
        rightIdxs   = self.solverWrap.rightIdxs
        middleIdxs  = self.solverWrap.middleIdxs
        idxsGB      = self.solverWrap.IGB
        
        if self.mapType=='DtD':
            
            sourceIdxs=[leftIdxs,rightIdxs]
            targetIdxs=[middleIdxs]
            
        elif self.mapType=='DtN':
            sourceIdxs=[leftIdxs,rightIdxs]
            targetIdxs=[rightIdxs,leftIdxs]
        else:
            TypeError("Slab: Invalid mapping type")
        self.sourceIdxs = [Idx for Idx in sourceIdxs if Idx]
        self.targetIdxs = [Idx for Idx in targetIdxs if Idx]
        self.idxsGB     = idxsGB

    ######################################################################

    #########################
    #      Idx methods      #
    #########################
        
    def setGlobalIdxs(self,globIdxs:range,loc:int):
        """
        Set global indices
        if 'DtD':   0=left
                    1=middle
                    2=right
        if 'DtN':   0=left
                    1=right
        """
        self.globIdxs[loc]=globIdxs
        self.computeGlobSourceTarget()

    def computeGlobSourceTarget(self):
        """
        From the global indices, compute the mapping indices in global coordinates
        We need to be careful here: global idxs should pair one-to-one to localIdxs
        """
        if self.mapType=='DtD':
            globSourceIdxs=[self.globIdxs[0],self.globIdxs[2]]
            globTargetIdxs=[self.globIdxs[1]]
            
        elif self.mapType=='DtN':
            globSourceIdxs=[self.globIdxs[0],self.globIdxs[1]]
            globTargetIdxs=[self.globIdxs[1],self.globIdxs[0]]
        else:
            TypeError("Slab: Invalid mapping type")
        self.globSourceIdxs=globSourceIdxs
        self.globTargetIdxs=globTargetIdxs

    
    ######################################################################
    

    
    #########################
    #   Block constructors  #
    #########################
    
    ######################################################################

    def constructMats(self,matAssembler:matAssembler):
        
        globIdxs=[x for x in itertools.product(self.globSourceIdxs,self.globTargetIdxs) if x[0] and x[1]]#cart. prod.
        [globIdxs.append(x) for x in itertools.product(self.globTargetIdxs,self.globTargetIdxs) if x not in globIdxs and x[0] and x[1]]

        locIdxs=[x for x in itertools.product(self.sourceIdxs,self.targetIdxs) if x[0] and x[1]]
        [locIdxs.append(x) for x in itertools.product(self.targetIdxs,self.targetIdxs) if x not in locIdxs and x[0] and x[1]]
        
        self.locIdxs=locIdxs
        self.globIdxs=globIdxs
        B                   = [matAssembler.assemble(self.solverWrap.stMap(Idxs[0],Idxs[1],self.mapType)) for Idxs in locIdxs]
        maps                = [slabMap(b,Idxs[0],Idxs[1]) for b,Idxs in zip(B,globIdxs)]
        self.maps           = maps
        self.mapsComputed   = True
    

class multiSlab:
    """
    class for multislab
    assumption: slabs are ordered left-to-right
    """
    def __init__(self,slabList:list[Slab],assemblerList:list[matAssembler]):
        self.slabList   = slabList
        self.assemblerList = assemblerList
        self.N   =   0
        self.nSlabs=len(slabList)
        for slab in slabList:
            mxi=max([a.stop for a in slab.globTargetIdxs if a])
            self.N=max(self.N,mxi)
        self.constructed = False
    def constructMats(self):
        for assembler,slab in zip(self.assemblerList,self.slabList):
            slab.constructMats(assembler)
        self.constructed = True

    def RHS(self,f):
        rhs=np.zeros(shape=(self.N,))
        for slab in self.slabList:
            rhsloc,idxsloc=slab.computeRHS(f)
            for xi,idxs in zip(rhsloc,idxsloc):
                rhs[idxs]+=xi
        return rhs

        
    def apply(self,x):
        if not self.constructed:
            self.constructMats()
        
        if (x.ndim == 1):
            x0 = x[:,np.newaxis]
        else:
            x0 = x
        y = np.zeros(shape=x0.shape)
        for slab in self.slabList:
            for map in slab.maps:
                y[map.globIdxs[1],:]+=map.A@x0[map.globIdxs[0],:]

        if (x.ndim == 1):
            y = y.flatten()
        return y
    

    def applyT(self,x):
        if not self.constructed:
            self.constructMats()
        if (x.ndim == 1):
            x0 = x[:,np.newaxis]
        else:
            x0 = x
        y=np.zeros(shape=x0.shape)
        for slab in self.slabList:
            for map in slab.maps:
                y[map.globIdxs[0],:]+=map.A.T@x0[map.globIdxs[1],:]

        if (x.ndim == 1):
            y = y.flatten()
        return y

    
    def getLinOp(self):
        if not self.constructed:
            self.constructMats()
        return LinearOperator(shape=(self.N,self.N),\
            matvec = self.apply, rmatvec = self.applyT,\
            matmat = self.apply, rmatmat = self.applyT)


"""
    evalSolInterior
    method to evaluate the solution u (given as a vector) in the interior of a slab
    @param:
    slab        : slab object
    b           : vector of values on the boundary (source Idxs set!)
"""
def evalSolInterior(slab:Slab,b):
    locIdxs = slab.sourceIdxs
    I   = [Idx for Idx in locIdxs]
    A   = slab.solverWrap.solver
    u   = A.solver_ii@A.Aib[:,I]@b
    return u
    









            


         

