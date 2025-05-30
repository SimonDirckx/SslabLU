import geometry.slabGeometry as slabGeom
import pdo.pdo as pdo
import numpy as np
from scipy.sparse.linalg   import LinearOperator
from solver.stencil.stencilSolver import stencilSolver as stencil
from solver.spectral.spectralSolver import spectralSolver as spectral
from solver.spectralmultidomain.hps import hps_multidomain as hps
import solver.spectralmultidomain.hps.geom as hpsGeom
"""
    This header takes care of the Solver Wrapper class
    Recipe:
    - user has some external solver (e.g. 'mySolver') in folder 'mySolverFolder'
    - places mySolverFolder in folder 'solver'
    - add 'from solver.mySolverFolder.mySolver import mySolver' (or variant thereof)
    - add to class solverOptions: 'type==mySolver' and then set order//nyz//...
    - add geometry conversion if needed to 'convertGeom'
    - add class init ( if self.type=='mySolver'...self.solver=mySolver(...) )to solverWrapper
    REQUIREMENTS FOR SOLVER:
    Solver must inherit from AbstractPDESolver or be compatible with it
"""

class stMap:
    def __init__(self,A:LinearOperator,XXI,XXJ):
        self.XXI = XXI
        self.XXJ = XXJ
        self.A = A


class solverOptions:
    """
    Class that encodes the options for a local slab Solver
    @param:
    type:       type of discretization (HPS/cheb/stencil)
    ordx,ordy:  order in x and y directions
    a:          characteristic scale in case of HPS
    """
    def __init__(self,type:str,ord,a=1):
        self.type   =   type
        self.ord    =   ord
        self.a      =   a
        self.nyz    =   1
        if type=='stencil':
            for i in range(1,len(ord)):
                self.nyz *= (ord[i]-2)
        if type=='hps':
            for i in range(1,len(ord)):
                self.nyz *= (int)(np.round((ord[i]-2)*(.5/a)))
        if type=='spectral':
            for i in range(1,len(ord)):
                self.nyz *= (ord[i]-1)
        print("#dofs = ",self.nyz)
def convertGeom(opts,geom):
    if opts.type=='hps':
        return hpsGeom.BoxGeometry(np.array([geom.bounds[0],geom.bounds[1]]))


class solverWrapper:
    """
    Wrapper class for local Solver
    @param:
    opts:       slab options
    """
    def __init__(self,opts:solverOptions):
        self.ord   = opts.ord
        self.type   = opts.type
        self.a      = opts.a
        self.solver = None
        self.type = opts.type
        self.constructed = False
        self.opts=opts
    def construct(self,geom:slabGeom.slabGeometry,PDE:pdo):
        """
        Actual construction of the local solver
        """
        self.geom   = geom
        self.PDE    = PDE
        if self.type=='stencil':
            self.solver = stencil(PDE, geom, self.ord)
            self.constructed=True
            '''
            adapt these to fit the notation of custom solver
            '''
            self.XX = self.solver.XX
            self.Ii = self.solver._Ji
            self.Ib = self.solver._Jb
            
            self.Aib = self.solver.Aib
            self.Abi = self.solver.Abi
            self.Abb = self.solver.Abb
            self.solver_ii = self.solver.solver_ii
        if self.type=='hps':
            geomHPS = convertGeom(self.opts,geom)
            self.solver = hps.HPSMultidomain(PDE, geomHPS,self.a, self.ord[0])
            self.constructed=True
            '''
            adapt these to fit the notation of custom solver
            '''
            self.XX = self.solver.XX
            print("XX shape in wrap = ",self.XX.shape)
            self.Ii = self.solver._Ji
            print("Ii len in wrap = ",len(self.Ii))
            self.Ib = self.solver._Jx
            print("Ib len in wrap = ",len(self.Ib))
            self.Aib = self.solver.Aix
            self.Abi = self.solver.Axi
            self.Abb = self.solver.Axx
            self.solver_ii = self.solver.solver_Aii
            print("solver shape in wrap = ",self.solver_ii.shape)
        if self.type=='spectral':
            self.solver = spectral(PDE, geom, self.ord)
            self.constructed=True
            '''
            adapt these to fit the notation of custom solver
            '''
            self.XX = self.solver.XX
            self.Ii = self.solver._Ji
            self.Ib = self.solver._Jb
            
            self.Aib = self.solver.Aib
            self.Abi = self.solver.Abi
            self.Abb = self.solver.Abb
            self.solver_ii = self.solver.solver_ii
        
        self.XXi = self.solver.XX[self.Ii,:]
        self.XXb = self.solver.XX[self.Ib,:]
        self.ndofs = self.XX.shape[0]
        self.constructMapIdxs()
    
    def constructMapIdxs(self):
        Il,Ic,Ir,IGB=self.geom.getIlIcIr(self.XXi,self.XXb)
        self.leftIdxs=Il
        self.rightIdxs=Ir
        self.middleIdxs=Ic
        self.IGB=IGB
            
    def compute_stMap(self,J:list[int],I:list[int],stType):
        #for now:   in case of DtD, I should be subset indices in Ii, J should be subset indices in Ib
        #           in case of DtN, both should be subset indices in Ib
    
        if stType=='DtD':
            XXI = self.XXi[I,:]
            if not I==J:
                AiJ  = self.Aib[:,J]
                XXJ = self.XXb[J,:]
            else:
                XXJ = self.XXi[J,:]
            LUii  = self.solver_ii

            def matmat(v,transpose=False):
                if I==J:
                    return v 
                if (v.ndim == 1):
                    v_tmp = v[:,np.newaxis]
                else:
                    v_tmp = v

                if (not transpose):
                    result = (LUii@(AiJ@v_tmp))[I]
                else:
                    result      = np.zeros(shape=(len(self.solver._Ji),v.shape[1]))
                    result[I,:] = v_tmp
                    result      = AiJ.T @ (LUii.T@(result))
                if (v.ndim == 1):
                    result = result.flatten()
                return result

            A = LinearOperator(shape=(len(I),len(J)),\
                matvec = matmat, rmatvec = lambda v: matmat(v,transpose=True),\
                matmat = matmat, rmatmat = lambda v: matmat(v,transpose=True))
            
            return stMap(A,XXI,XXJ)
        if stType=='DtN':
            AIJ = self.Abb[I][:,J]
            AIi  = self.Abi[I]
            AiJ  = self.Aib[:,J]
            LUii  = self.solver_ii

            def matmat(v,transpose=False):

                if (v.ndim == 1):
                    v_tmp = v[:,np.newaxis]
                else:
                    v_tmp = v

                if (not transpose):

                    result = AIJ @ v_tmp
                    result -= AIi @ (LUii@(AiJ @ v_tmp))
                else:
                    result = AIJ.T @ v_tmp
                    result -= AiJ.T @ (LUii.T@(AIi.T @ v_tmp))

                if (v.ndim == 1):
                    result = result.flatten()
                return result

            return LinearOperator(shape=AIJ.shape,\
                matvec = matmat, rmatvec = lambda v: matmat(v,transpose=True),\
                matmat = matmat, rmatmat = lambda v: matmat(v,transpose=True))
    def solveInterior(self,g,load):
        
        return 0