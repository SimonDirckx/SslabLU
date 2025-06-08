import numpy as np
import pdo.pdo as pdo
import solver.solver as solverWrap
from solver.pde_solver import AbstractPDESolver
import multiSlab as MS
import matAssembly.matAssembler as mA
import matplotlib.pyplot as plt
import geometry.standardGeometries as stdGeom
import geometry.skeleton as skelTon
import itertools
import scipy.linalg as splinalg
from scipy.sparse.linalg import gmres
import time
import pandas as pd


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

try:
	from petsc4py import PETSc
	petsc_imported = True
except:
	petsc_imported = False

start = time.time()
kapp = 0.
laplace = True
ndim = 2

#set up pde
if laplace:
    def c11(p):
        return np.ones(shape=(p.shape[0],))
else:
    def c11(p):
        f = np.zeros(shape=(p.shape[0],))
        for i in range(p.shape[0]):
            f[i] = 1.+(np.sin(5*np.pi*p[i,0])*np.sin(5*np.pi*p[i,0]))
        return f

def c22(p):
    return np.ones(shape=(p.shape[0],))
if ndim==2:
    bnds = [[0.,0.],[1.,1.]]
    Om=stdGeom.Box(bnds)
elif ndim==3:
    def c33(p):
        return np.ones(shape=(p.shape[0],))
    bnds = [[0.,0.,0.],[1.,1.,1.]]
    Om=stdGeom.Box(bnds)
else:
    raise ValueError("ndim must be two or three!")
def c(p):
    return kapp*kapp*np.ones(shape=(p.shape[0],))

Lapl=pdo.PDO2d(c11,c22)#,None,None,None,c)


k       = 2
N       = (2**k)-1
H       = 1./(N+1)
skel    = skelTon.standardBoxSkeleton(Om,N)
method  = 'stencil'
if method == 'stencil':
    kmin=3
    kmax=12
elif method=='hps':
    kmin = 5
    kmax = 30
elif method=='spectral':
    kmin = 2
    kmax = 30
else:
    raise ValueError("method must be stencil or HPS")


for iter in range(kmin,kmax):

    if method=='stencil':
        ny = 2**iter
        nx = 101
        ord = [nx,ny]
        opts = solverWrap.solverOptions(method,ord)
    elif method=='hps':
        ord = [iter,iter]
        a   = H/4.
        opts = solverWrap.solverOptions(method,ord,a)
    elif method=='spectral':
        p = 2*iter
        ord = [p,p]
        opts = solverWrap.solverOptions(method,ord)
    else:
        raise ValueError("method must be stencil or HPS")
    
    overlapping = True


    skel.setGlobalIdxs(skelTon.computeUniformGlobalIdxs(skel,opts))
    slabList = skelTon.buildSlabs(skel,Lapl,opts,overlapping)
    assembler = mA.denseMatAssembler()
    assemblerList = [assembler for slab in slabList]
    MultiSlab = MS.multiSlab(slabList,assemblerList)
    MultiSlab.constructMats()
    Linop       = MultiSlab.getLinOp()
    E=np.identity(Linop.shape[0])
    Stot = Linop@E
    print('nrm sym',np.linalg.norm(Stot-Stot.T))
    W=np.zeros(shape=Stot.shape)
    if method=='spectral':
        x,w = clenshaw_curtis_compute(ord[1]+1)
        #print("x len = ",len(x))
        #print("ord = ",ord)
        #w[1]-=w[0]
        #w[len(w)-2]-=w[len(w)-1]
        scl = np.sqrt(w[1:len(w)-1])
        Wloc = np.diag(scl)
        nyz = Stot.shape[1]//N
        for jj in range(N):
            W[jj*nyz:(jj+1)*nyz,jj*nyz:(jj+1)*nyz]=Wloc
        Stot = W@Stot@np.linalg.inv(W)
    if method=='stencil':
        x=np.linspace(0,1,ord[1])
    
    [U,s,V]=np.linalg.svd(Stot)
    print("shape S = ",Stot.shape)
    print("smin//smax",np.min(s),"//",np.max(s))
    print("smax/smin",np.max(s)/np.min(s))
    if method=='spectral':
        x=(x+1)/2.
    u = np.zeros(shape=(len(x),))
    u[1:len(x)-1]=U[0:len(x)-2,0]
    u[0]=0.
    u[len(u)-1]=0.
    snx = np.sin(np.pi*x)
    fc=u[len(x)//2]/snx[len(x)//2]
    plt.figure(1)
    plt.plot(x,u)
    plt.plot(x,snx*fc)
    #plt.show()