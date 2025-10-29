import numpy as np
from scipy.sparse.linalg   import LinearOperator
import scipy.sparse as sparse
from scipy.sparse        import block_diag
from solver.pde_solver import AbstractPDESolver
import solver.hpsmultidomain.hpsmultidomain.pdo as pdo

from solver import sparse_utils
import matplotlib.pyplot as plt
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
    return -D2
def stencilDxy(ptsx,ptsy):
    hx=ptsx[1]-ptsx[0]
    hy=ptsy[1]-ptsy[0]
    ex = np.ones(shape=(len(ptsx),))
    ey = np.ones(shape=(len(ptsy),))
    Dx = -np.diag(ex,-1)-np.diag(ex,1)
    Dy = -np.diag(ey,-1)-np.diag(ey,1)
    DxDy = sparse.kron(Dx,Dy)/(4.*hx*hy)
    return DxDy

def constructPDO2D(pdo,xpts,ypts,XX,geom):
    N=XX.shape[0]
    C11 = -sparse.spdiags(pdo.c11(XX),[0],N,N)
    C22 = -sparse.spdiags(pdo.c22(XX),[0],N,N)
    L   =  C11@sparse.kron(stencilD2(xpts),np.identity(len(ypts)))
    L   += C22@sparse.kron(np.identity(len(xpts)),stencilD2(ypts))
    if pdo.c1:
        C1 = sparse.spdiags(pdo.c1(XX),[0],N,N)
        L   += C1@sparse.kron(np.identity(len(xpts)),stencilD(ypts))
    if pdo.c:
        C = sparse.spdiags(pdo.c(XX),[0],N,N)
        L   += C
    return L

def constructPDO3D(pdo,xpts,ypts,zpts,XX,geom):
    N=XX.shape[0]
    C11 = -sparse.spdiags(pdo.c11(XX),[0],N,N)
    C22 = -sparse.spdiags(pdo.c22(XX),[0],N,N)
    C33 = -sparse.spdiags(pdo.c33(XX),[0],N,N)
    L   =  C11@sparse.kron( sparse.kron( stencilD2(xpts) , np.identity(len(ypts))) , np.identity(len(zpts)) )
    L   += C22@sparse.kron( sparse.kron( np.identity(len(xpts)) , stencilD2(ypts) ) , np.identity(len(zpts)) )
    L   += C33@sparse.kron( sparse.kron( np.identity(len(xpts)) , np.identity(len(ypts)) ) , stencilD2(zpts) )
    #TODO: other terms
    return L


# stencil domain class for handling discretizations
class stencilSolver(AbstractPDESolver):
    
    def __init__(self, pdo, geom, ord):
        """
        Initializes the stencil solver with domain 
        information and discretization parameters.
        
        Parameters:
        - pdo               : An object representing the partial differential operator.
        - geom              : The computational domain represented as an array.
        - ord (list[int])   : order in the x,y and (possibly) z-direction
        """

        self._box_geom = geom.bounds
        ndim = self._box_geom.shape[-1]
        self._geom     = geom
        if  (ndim == 2):
            xpts        = np.linspace(self._box_geom[0][0],self._box_geom[1][0],ord[0])
            ypts        = np.linspace(self._box_geom[0][1],self._box_geom[1][1],ord[1])
            self._XX = np.zeros(shape=(ord[0]*ord[1],2))
            self._XX[:,0] = np.kron(xpts,np.ones_like(ypts))
            self._XX[:,1] = np.kron(np.ones_like(xpts),ypts)
            self._A      = constructPDO2D(pdo,xpts,ypts,self._XX,self.geom).tocsr()
            self._Ji=np.where( (self._box_geom[0][0]<self._XX[:,0])     & (self._box_geom[1][0]>self._XX[:,0]) & (self._box_geom[0][1]<self._XX[:,1]) & (self._box_geom[1][1]>self._XX[:,1]))[0]
            self._Jx=np.where( (self._box_geom[0][0]==self._XX[:,0])    | (self._box_geom[1][0]==self._XX[:,0]) | (self._box_geom[0][1]==self._XX[:,1]) | (self._box_geom[1][1]==self._XX[:,1]))[0] 
        elif (ndim == 3):
            xpts        = np.linspace(self._box_geom[0][0],self._box_geom[1][0],ord[0])
            ypts        = np.linspace(self._box_geom[0][1],self._box_geom[1][1],ord[1])
            zpts        = np.linspace(self._box_geom[0][2],self._box_geom[1][2],ord[2])
            self._XX = np.zeros(shape=(ord[0]*ord[1]*ord[2],3))
            self._XX[:,0] = np.kron(np.kron(xpts,np.ones_like(ypts)),np.ones_like(zpts))
            self._XX[:,1] = np.kron(np.kron(np.ones_like(xpts),ypts),np.ones_like(zpts))
            self._XX[:,2] = np.kron(np.kron(np.ones_like(xpts),np.ones_like(ypts)),zpts)
            self._A      = constructPDO3D(pdo,xpts,ypts,zpts,self._XX,self.geom).tocsr()
            self._Ji=np.where( (self._box_geom[0][0]<self._XX[:,0]) & (self._box_geom[1][0]>self._XX[:,0]) & (self._box_geom[0][1]<self._XX[:,1]) & (self._box_geom[1][1]>self._XX[:,1])& (self._box_geom[0][2]<self._XX[:,2]) & (self._box_geom[1][2]>self._XX[:,2]))[0]
            self._Jx=np.where( (self._box_geom[0][0]==self._XX[:,0])    | (self._box_geom[1][0]==self._XX[:,0]) | (self._box_geom[0][1]==self._XX[:,1]) | (self._box_geom[1][1]==self._XX[:,1]) | (self._box_geom[0][2]==self._XX[:,2]) | (self._box_geom[1][2]==self._XX[:,2]))[0] 
        else:
            raise ValueError
        
        self._XXi=self._XX[self._Ji,:]
        self._XXb=self._XX[self._Jx,:]
        self._Aii = self._A[self._Ji][:,self._Ji]
        self._Aix = self._A[self._Ji][:,self._Jx]
        self._Axi = self._A[self._Jx][:,self._Ji]
        self._Axx = self._A[self._Jx][:,self._Jx]
        #self.constructSolverii()
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
    def Jx(self):
        return self._Jx

    @property
    def Aii(self):
        return self._Aii
    
    @property
    def Aix(self):
        return self._Aix

    @property
    def Axi(self):
        return self._Axi
    
    @property
    def Axx(self):
        return self._Axx

    @property
    def p(self):
        return self._p
    
