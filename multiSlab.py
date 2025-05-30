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
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import griddata

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
        # important: these are ALL indices, not just the interior DOFs
        p=self.XX[i,:]
        return self.geom.l2g(p)
    
    def getGlobalXX(self):
        """return global coordinates of vectorized dofs"""
        return np.array([self.l2g(i) for i in range(self.ndofs)])
    
    def evalGlobalFuncAtIntDof(self,f,i):
        """helper function to evaluate globally defined f @ dof i"""
        p=self.l2g(i)
        return f(p)
    def evalGlobFuncAtLocPts(self,f,p):
        """helper function to evaluate globally defined f @ dof i"""
        p0=self.geom.l2g(p)
        return f(p0)
    ######################################################################

    #########################
    #   function methods    #
    #########################

    ######################################################################
    
    def computeRHS(self,f,globalLoad=lambda p : 0):
        """helper function to compute local RHS"""
        # unified framework for T and S map
        fGb=np.array([f(self.geom.l2g(self.solverWrap.XXb[i,:])) for i in self.idxsGB])
        # currently only for S-map!!!
        # TODO: similar for T-map
        load = np.array([self.evalGlobalFuncAtIntDof(globalLoad,i) for i in self.Ii])
        rr = self.solverWrap.solver_ii@load
        rhs=[rr[idxs]-self.solverWrap.compute_stMap(self.idxsGB,idxs,self.mapType).A@fGb for idxs in self.targetIdxs if idxs]
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
        B                   = [matAssembler.assemble(self.solverWrap.compute_stMap(Idxs[0],Idxs[1],self.mapType)) for Idxs in locIdxs]
        maps                = [slabMap(b,Idxs[0],Idxs[1]) for b,Idxs in zip(B,globIdxs)]
        self.maps           = maps
        self.mapsComputed   = True
    """
    evalSolInterior
    method to evaluate the solution u (given as a vector) in the interior of a slab
    @param:
    slab        : slab object
    b           : vector of values on the boundary (source Idxs set!)
    """
    def solveInterior(self,b,f,load = lambda p : 0):
        g   =   np.zeros(shape=(len(self.Ib),))
        for idxs,ind in zip(self.sourceIdxs,range(len(self.sourceIdxs))):
            if idxs:
                g[idxs] =   b[ind]
        g[self.Ib[self.idxsGB]]  =   np.array([f(self.geom.l2g(self.XX[self.Ib[i],:])) for i in self.idxsGB])
        return self.solverWrap.solveInterior(g,load)
        
    def evalSol(self,b,f,load = lambda p : 0):
        # important: assumes that ordering is consistent!
        Aib = self.solverWrap.Aib
        fGb=np.array([f(self.geom.l2g(self.solverWrap.XXb[i,:])) for i in self.idxsGB])
        rhs = Aib[:,self.idxsGB]@fGb
        ind = 0
        for idxs in self.sourceIdxs:
            rhs += Aib[:,idxs]@b[ind]
            ind += 1
        ld   = np.array([self.evalGlobalFuncAtIntDof(load,i) for i in self.Ii])
        solver   = self.solverWrap.solver_ii
        ui   = solver@(ld-rhs)
        u = np.zeros(shape=(self.XX.shape[0],))
        u[self.Ii] = ui
        #u[self.idxsGB] = fGb
        ind = 0
        #for idxs in self.sourceIdxs:
        #    u[self.Ib[idxs]] = b[ind]
        #    ind += 1
        resolution = 2000
        Ii = self.Ii
        min_x = np.min(self.XX[Ii,0])
        max_x = np.max(self.XX[Ii,0])
        min_y = np.min(self.XX[Ii,1])
        max_y = np.max(self.XX[Ii,1])
        grid_x, grid_y    = np.mgrid[min_x:max_x:resolution*1j, min_y:max_y:resolution*1j]
        grid_solution     = griddata(self.XX, u, (grid_x, grid_y), method='linear').T
        plot_pad=0.1
        max_sol = np.max(u)
        min_sol = np.min(u)
        plt.figure(1)
        plt.scatter(self.XX[:,0],self.XX[:,1])
        plt.figure(2)
        plt.imshow(grid_solution, extent=(min_x-plot_pad,max_x+plot_pad,\
                                    min_y-plot_pad,max_y+plot_pad),\
                                        #vmin=min_sol, vmax=max_sol,\
                origin='lower',cmap = 'jet')
        plt.show()
        return u
    

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

    def RHS(self,bdry,load=lambda p:0):
        rhs=np.zeros(shape=(self.N,))
        for slab in self.slabList:
            rhsloc,idxsloc=slab.computeRHS(bdry,load)
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
    
    def evalGlobalOnInterfaces(self,f):
        fvec = np.zeros(shape=(self.N,))
        for slab in self.slabList:
            globIdxs    = slab.globTargetIdxs
            locIdxs     = slab.targetIdxs
            for idxGlob,idxLoc in zip(globIdxs,locIdxs):
                floc = np.array([f(slab.geom.l2g(slab.XX[slab.Ii[i],:])) for i in idxLoc])
                fvec[idxGlob] = floc
        return fvec



    









            


         

