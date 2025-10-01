import numpy as np
from scipy.sparse.linalg   import LinearOperator
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from scipy.sparse        import block_diag
from solver.pde_solver import AbstractPDESolver
import pdo.pdo as pdo
from solver import sparse_utils



def clenshaw_curtis_compute ( n ):

#*****************************************************************************80
#
## CLENSHAW_CURTIS_COMPUTE computes a Clenshaw Curtis quadrature rule.
#
#  Discussion:
#
#    This method uses a direct approach.  The paper by Waldvogel
#    exhibits a more efficient approach using Fourier transforms.
#
#    The integral:
#
#      integral ( -1 <= x <= 1 ) f(x) dx
#
#    The quadrature rule:
#
#      sum ( 1 <= i <= n ) w(i) * f ( x(i) )
#
#    The abscissas for the rule of order N can be regarded
#    as the cosines of equally spaced angles between 180 and 0 degrees:
#
#      X(I) = cos ( ( N - I ) * PI / ( N - 1 ) )
#
#    except for the basic case N = 1, when
#
#      X(1) = 0.
#
#    A Clenshaw-Curtis rule that uses N points will integrate
#    exactly all polynomials of degrees 0 through N-1.  If N
#    is odd, then by symmetry the polynomial of degree N will
#    also be integrated exactly.
#
#    If the value of N is increased in a sensible way, then
#    the new set of abscissas will include the old ones.  One such
#    sequence would be N(K) = 2*K+1 for K = 0, 1, 2, ...
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    02 April 2015
#
#  Author:
#
#    John Burkardt
#
#  Reference:
#
#    Charles Clenshaw, Alan Curtis,
#    A Method for Numerical Integration on an Automatic Computer,
#    Numerische Mathematik,
#    Volume 2, Number 1, December 1960, pages 197-205.
#
#    Philip Davis, Philip Rabinowitz,
#    Methods of Numerical Integration,
#    Second Edition,
#    Dover, 2007,
#    ISBN: 0486453391,
#    LC: QA299.3.D28.
#
#    Joerg Waldvogel,
#    Fast Construction of the Fejer and Clenshaw-Curtis Quadrature Rules,
#    BIT Numerical Mathematics,
#    Volume 43, Number 1, 2003, pages 1-18.
#
#  Parameters:
#
#    Input, integer N, the order.
#
#    Output, real X(N), the abscissas.
#
#    Output, real W(N), the weights.
#
  import numpy as np

  if ( n == 1 ):

    x = np.zeros ( n )
    w = np.zeros ( n )

    w[0] = 2.0

  else:

    theta = np.zeros ( n )

    for i in range ( 0, n ):
      theta[i] = float ( n - 1 - i ) * np.pi / float ( n - 1 )

    x = np.cos ( theta )
    w = np.zeros ( n )

    for i in range ( 0, n ):

      w[i] = 1.0

      jhi = ( ( n - 1 ) // 2 )

      for j in range ( 0, jhi ):

        if ( 2 * ( j + 1 ) == ( n - 1 ) ):
          b = 1.0
        else:
          b = 2.0

        w[i] = w[i] - b * np.cos ( 2.0 * float ( j + 1 ) * theta[i] ) \
             / float ( 4 * j * ( j + 2 ) + 3 )

    w[0] = w[0] / float ( n - 1 )
    for i in range ( 1, n - 1 ):
      w[i] = 2.0 * w[i] / float ( n - 1 )
    w[n-1] = w[n-1] / float ( n - 1 )

  return x, w



def cheb(p):
	"""
	Computes the Chebyshev differentiation matrix and Chebyshev points for a given degree p.
	
	Parameters:
	- p: The polynomial degree
	
	Returns:
	- D: The Chebyshev differentiation matrix
	- x: The Chebyshev points
	"""
	x = np.cos(np.pi * np.arange(p+1) / p)
	c = np.concatenate((np.array([2]), np.ones(p-1), np.array([2])))
	c = np.multiply(c,np.power(np.ones(p+1) * -1, np.arange(p+1)))
	X = x.repeat(p+1).reshape((-1,p+1))
	dX = X - X.T
	# create the off diagonal entries of D
	D = np.divide(np.outer(c,np.divide(np.ones(p+1),c)), dX + np.eye(p+1))
	D = D - np.diag(np.sum(D,axis=1))
	return D,x
def spectralD(xpts):
    p = len(xpts)-1
    D0,x0 = cheb(p)
    g = (xpts[len(xpts)-1]-xpts[0])/(x0[len(x0)-1]-x0[0])
    return D0/g
def spectralD2(xpts):
    D = spectralD(xpts)
    return D@D

def constructPDO2D(pdo,xpts,ypts,XX,geom):
    N=XX.shape[0]
    C11 = -sparse.spdiags(pdo.c11(XX),[0],N,N)
    C22 = -sparse.spdiags(pdo.c22(XX),[0],N,N)
    L   =  C11@sparse.kron(spectralD2(xpts),np.identity(len(ypts)))
    L   += C22@sparse.kron(np.identity(len(xpts)),spectralD2(ypts))
    if pdo.c1:
        C1 = sparse.spdiags(pdo.c1(XX),[0],N,N)
        L   += C1@sparse.kron(np.identity(len(xpts)),spectralD(ypts))
    if pdo.c:
        C = sparse.spdiags(pdo.c(XX),[0],N,N)
        L   += C
    return L

def constructPDO3D(pdo,xpts,ypts,zpts,XX,geom):
    N=XX.shape[0]
    Dxx = spectralD2(xpts)
    Dyy = spectralD2(ypts)
    Dzz = spectralD2(zpts)
    
    Ex  = np.identity(len(xpts))
    Ey  = np.identity(len(ypts))
    Ez  = np.identity(len(zpts))
    
    C11 = sparse.spdiags(pdo.c11(XX),[0],N,N)
    C22 = sparse.spdiags(pdo.c22(XX),[0],N,N)
    C33 = sparse.spdiags(pdo.c33(XX),[0],N,N)
    
    L   =-C11@sparse.kron(sparse.kron(Dxx,Ey),Ez)
    L   -=C22@sparse.kron(sparse.kron(Ex,Dyy),Ez)
    L   -=C33@sparse.kron(sparse.kron(Ex,Ey),Dzz)
    
    L+=sparse.spdiags(pdo.c(XX),[0],N,N)
    return L


# stencil domain class for handling discretizations
class spectralSolver(AbstractPDESolver):
    
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
        ndim = geom.bounds.shape[-1]
        self._geom     = geom
        if  (ndim == 2):
            xmin = self._box_geom[0][0]
            xmax = self._box_geom[1][0]
            ordx = ord[0]
            D0,cheb0 = cheb(ordx)
            xpts = ((cheb0-cheb0[0])*(xmax-xmin)/(cheb0[len(cheb0)-1]-cheb0[0])) + xmin
            ymin = self._box_geom[0][1]
            ymax = self._box_geom[1][1]
            ordy = ord[1]
            D0,cheb0 = cheb(ordy)
            ypts = ((cheb0-cheb0[0])*(ymax-ymin)/(cheb0[len(cheb0)-1]-cheb0[0])) + ymin
            self._XX    = np.vstack([np.concatenate((np.tile(x,ypts.shape)[:,np.newaxis],ypts[:,np.newaxis]),axis=1) for x in xpts])
            self._A      = constructPDO2D(pdo,xpts,ypts,self._XX,self.geom).tocsr()
            self._Ji=np.where( (self._box_geom[0][0]<self._XX[:,0])     & (self._box_geom[1][0]>self._XX[:,0]) & (self._box_geom[0][1]<self._XX[:,1]) & (self._box_geom[1][1]>self._XX[:,1]))[0]
            self._Jx=np.where( (self._box_geom[0][0]==self._XX[:,0])    | (self._box_geom[1][0]==self._XX[:,0]) | (self._box_geom[0][1]==self._XX[:,1]) | (self._box_geom[1][1]==self._XX[:,1]))[0] 
        elif (ndim == 3):
            xmin = self._box_geom[0][0]
            xmax = self._box_geom[1][0]
            ymin = self._box_geom[0][1]
            ymax = self._box_geom[1][1]
            zmin = self._box_geom[0][2]
            zmax = self._box_geom[1][2]
            ordx = ord[0]
            ordy = ord[1]
            ordz = ord[2]
            _,chebx = cheb(ordx)
            _,cheby = cheb(ordy)
            _,chebz = cheb(ordz)
            xpts = ((chebx-chebx[0])*(xmax-xmin)/(chebx[len(chebx)-1]-chebx[0])) + xmin
            ypts = ((cheby-cheby[0])*(ymax-zmin)/(cheby[len(cheby)-1]-cheby[0])) + ymin
            zpts = ((chebz-chebz[0])*(zmax-zmin)/(chebz[len(chebz)-1]-chebz[0])) + zmin
            XX = np.zeros(shape = (len(xpts)*len(ypts)*len(zpts),3))
            XX[:,0] = np.kron(np.kron(xpts,np.ones_like(ypts)),np.ones_like(zpts))
            XX[:,1] = np.kron(np.kron(np.ones_like(xpts),ypts),np.ones_like(zpts))
            XX[:,2] = np.kron(np.kron(np.ones_like(xpts),np.ones_like(ypts)),zpts)
            self._XX    = XX
            self._A      = constructPDO3D(pdo,xpts,ypts,zpts,self._XX,self.geom)
            self._Ji=np.where( (self._box_geom[0][0]<self._XX[:,0])     & (self._box_geom[1][0]>self._XX[:,0]) & (self._box_geom[0][1]<self._XX[:,1]) & (self._box_geom[1][1]>self._XX[:,1])& (self._box_geom[0][2]<self._XX[:,2]) & (self._box_geom[1][2]>self._XX[:,2]))[0]
            self._Jx=np.where( (self._box_geom[0][0]==self._XX[:,0])    | (self._box_geom[1][0]==self._XX[:,0]) | (self._box_geom[0][1]==self._XX[:,1]) | (self._box_geom[1][1]==self._XX[:,1])  | (self._box_geom[0][2]==self._XX[:,2]) | (self._box_geom[1][2]==self._XX[:,2]))[0] 
        else:
            raise ValueError
        self._XXi=self._XX[self._Ji,:]
        self._XXb=self._XX[self._Jx,:]
        self._Aii = self._A[self._Ji][:,self._Ji]
        self._Aix = self._A[self._Ji][:,self._Jx]
        self._Axi = self._A[self._Jx][:,self._Ji]
        self._Axx = self._A[self._Jx][:,self._Jx]
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
    
