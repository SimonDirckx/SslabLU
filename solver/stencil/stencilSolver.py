import numpy as np
from scipy.sparse.linalg   import LinearOperator

from scipy.sparse        import block_diag
from solver.pde_solver import AbstractPDESolver
import pdo.pdo as pdo
from geometry.slabGeometry import slabGeometry as slab

def stencilD(pts):
    h=pts[1]-pts[0]
    D=np.eye(len(pts))
    e=np.ones(shape=(len(pts)-1,))
    D=D-np.diag(e,-1)
    return D/h
def stencilD2(pts):
    D=stencilD(pts)
    D2 = D.T@D
    D2[0,0]=-D2[0,1]
    return D2
def stencilDxy(ptsx,ptsy):
    hx=ptsx[1]-ptsx[0]
    hy=ptsy[1]-ptsy[0]
    ex = np.ones(shape=(len(ptsx),))
    ey = np.ones(shape=(len(ptsy),))
    Dx = -np.diag(ex,-1)-np.diag(ex,1)
    Dy = -np.diag(ey,-1)-np.diag(ey,1)
    DxDy = np.kron(Dx,Dy)/(4.*hx*hy)
    return DxDy
def constructPDO2D(pdo,xpts,ypts,XX,geom):
    L=np.zeros(shape=(XX.shape[0],XX.shape[0]))
    L-=np.diag(pdo.c11(geom.l2g(XX)))@np.kron(stencilD2(xpts),np.identity(len(ypts)))
    L-=np.diag(pdo.c22(geom.l2g(XX)))@np.kron(np.identity(len(xpts)),stencilD2(ypts))
    if pdo.c12:
        L-=2*np.diag(pdo.c12(geom.l2g(XX)))@stencilDxy(xpts,ypts)
    return L

def constructPDO3D(pdo,XX):
    return


# stencil domain class for handling discretizations
class stencilSolver(AbstractPDESolver):
    
    def __init__(self, pdo, geom:slab, ord):
        """
        Initializes the stencil solver with domain 
        information and discretization parameters.
        
        Parameters:
        - pdo               : An object representing the partial differential operator.
        - geom              : The computational domain represented as an array.
        - ord (list[int])   : order in the x,y and (possibly) z-direction
        """

        self._box_geom = geom.bounds
        self._geom     = geom
        if  (self.ndim() == 2):
            xpts        = np.linspace(self._box_geom[0][0],self._box_geom[1][0],ord[0])
            ypts        = np.linspace(self._box_geom[0][1],self._box_geom[1][1],ord[1])
            self._XX    = np.vstack([np.concatenate((np.tile(x,ypts.shape)[:,np.newaxis],ypts[:,np.newaxis]),axis=1) for x in xpts])
            self._A      = constructPDO2D(pdo,xpts,ypts,self._XX,self.geom)
        elif (self.ndim() == 3):
            xpts        = np.linspace(self._box_geom[0][0],self._box_geom[1][0],ord[0])
            ypts        = np.linspace(self._box_geom[0][1],self._box_geom[1][1],ord[1])
            zpts        = np.linspace(self._box_geom[0][2],self._box_geom[1][2],ord[2])
            YZ          = np.vstack([np.concatenate((np.tile(y,zpts.shape)[:,np.newaxis],zpts[:,np.newaxis]),axis=1) for y in ypts])
            self._XX    = np.vstack([np.concatenate((np.tile(x,YZ.shape[0])[:,np.newaxis],YZ),axis=1) for x in xpts])
            self._A      = constructPDO3D(pdo,xpts,ypts,self._XX,self.geom)
        else:
            raise ValueError
        self._Ji=[i for i in range(self._XX.shape[0]) if not self.geom.isLocalBoundary(self._XX[i,:])]
        self._Jb=[i for i in range(self._XX.shape[0]) if self.geom.isLocalBoundary(self._XX[i,:])]
        self._XXi=self._XX[self._Ji,:]
        self._XXb=self._XX[self._Jb,:]
        self._Aii = self._A[self._Ji][:,self._Ji]
        self._Aib = self._A[self._Ji][:,self._Jb]
        self._Abi = self._A[self._Jb][:,self._Ji]
        self._Abb = self._A[self._Jb][:,self._Jb]
        self.constructSolverii()
    @property
    def npoints_dim(self):
        return self.npan_dim * self.p

    @property
    def geom(self):
        return self._geom

    @property
    def XX(self):
        return self._XX
    @property
    def XXi(self):
        return self._XXi
    @property
    def XXb(self):
        return self._XXb

    @property
    def Ji(self):
        return self._Ji

    @property
    def Jb(self):
        return self._Jb

    @property
    def Aii(self):
        return self._Aii
    
    @property
    def Aib(self):
        return self._Aib

    @property
    def Abi(self):
        return self._Abi
    
    @property
    def Abb(self):
        return self._Abb

    @property
    def p(self):
        return self._p
    def constructSolverii(self):
        N=len(self._Ji)
        def matmat(v,transpose=False):
            if transpose:
                return np.linalg.solve(self._Aii.T,v)
            else:
                return np.linalg.solve(self._Aii,v)
        self.solver_ii = LinearOperator(shape=(N,N),\
            matvec = matmat, rmatvec = lambda v: matmat(v,transpose=True),\
            matmat = matmat, rmatmat = lambda v: matmat(v,transpose=True))
    
