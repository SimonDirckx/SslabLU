import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy.polynomial.chebyshev as chebpoly

from solver.hpsmultidomain.hpsmultidomain import domain_driver as hpsalt
import solver.hpsmultidomain.hpsmultidomain.pdo as pdoalt
import solver.hpsmultidomain.hpsmultidomain.geom as hpsaltGeom
import SOMS
import matAssembly.matAssembler as mA
import solver.solver as solver



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



"""
script that illustrates the 2-box problem

     ___________ __________
    |           |           |
    |   tau     |   sig     |
  ul|         uc|           |
    |___________|___________|

Solution map S: ul-> uc has known eigenvalues and eigenfunction, both for Laplace and Helmholtz
Constructed in two ways: overlapping and non-overlapping

"""

k = 2
Lx = 1/8
Ly = 1
kh = 20
def  c11(p):
    return torch.ones_like(p[:,0])
def  c22(p):
    return torch.ones_like(p[:,1])
def  c(p):
    return kh*torch.ones_like(p[:,1])
def  c_alt(p,kh):
    return kh*torch.ones_like(p[:,1])
def  bc(p):
    return torch.sin(np.pi*k*p[:,1])*torch.sinh(k*np.pi*(Lx-p[:,0]))/np.sinh(k*np.pi*Lx)
def  bc_np(p):
    return np.sin(np.pi*k*p[:,1])*np.sinh(k*np.pi*(Lx-p[:,0]))/np.sinh(k*np.pi*Lx)
HH = pdoalt.PDO_2d(c11=c11,c22=c22,c=c)

cx = Lx/2
bnds = np.array([[0,0],[Lx,Ly]])
Om = hpsaltGeom.BoxGeometry(bnds)
nby = 16
nbx = 2
ax = .5*(bnds[1,0]/nbx)
ay = .5*(bnds[1,1]/nby)
#isotropic disc
py=10
px=10
print("px,py = ",px," , ",py)

solver_hps = hpsalt.Domain_Driver(Om, HH, kh, np.array([ax,ay]), [px+1,py+1], 2)
solver_hps.build("reduced_cpu", "MUMPS",verbose=False)

XX = solver_hps.XX
XXfull = solver_hps.XXfull

Jb = solver_hps._Jx
Ji = solver_hps.Ji

print("Ji size = ",len(Ji))
print("Jb size = ",len(Jb))

XXi = XX[Ji,:]
XXb = XX[Jb,:]
Jc = np.where(XXi[:,0]==cx)[0]
Jl = np.where((XXb[:,0]==0) & (XXb[:,1]>0) & (XXb[:,1]<Ly))[0]

Aii = np.array(solver_hps.Aii.todense())
Aib = np.array(solver_hps.Aix.todense())

rhsT = bc(XXb).cpu().detach().numpy()
uT = bc(XXi[Jc,:]).cpu().detach().numpy()

ST = -np.linalg.solve(Aii,Aib[:,Jl])[Jc,:]
uhat_T = ST@rhsT[Jl]

print("err1 = ",np.linalg.norm(uhat_T-uT,ord=2)/np.linalg.norm(uT,ord=2))



Stot,XYtot,Ii,Ib = SOMS.SOMS_solver(px,py,nbx,nby,Lx,Ly,kh)


XXi = XYtot[Ii,:]
XXb = XYtot[Ib,:]

plt.figure(1)
plt.scatter(XYtot[:,0],XYtot[:,1])
plt.scatter(XXi[:,0],XXi[:,1])
plt.scatter(XXb[:,0],XXb[:,1])
plt.legend(['xy','i','b'])
plt.show()

Jc = np.where(XXi[:,0]==cx)[0]
Jl = np.where((XXb[:,0]==0))[0]

AiiS = Stot[Ii,:][:,Ii]
AibS = Stot[Ii,:][:,Ib]

rhsS = bc_np(XXb)
uS = bc_np(XXi[Jc,:])
SS = -np.linalg.solve(AiiS,AibS[:,Jl])[Jc,:]
uhat_S = SS@rhsS[Jl]


print("err2 = ",np.linalg.norm(uhat_S-uS,ord=2)/np.linalg.norm(uS,ord=2))


condSS = np.linalg.cond(SS)
condST = np.linalg.cond(ST)

print("condSS = ",condSS)
print("condST = ",condST)


Jcsub = np.where((XXi[Jc,1]>1./8.) & (XXi[Jc,1]<1./4.))[0]
Jlsub = np.where((XXb[Jl,1]>0) & (XXi[Jc,1]<1./8.))[0]

SSsub = SS[Jcsub,:][:,Jlsub]
STsub = ST[Jcsub,:][:,Jlsub]


[_,sS,_] = np.linalg.svd(SS)
[_,sT,_] = np.linalg.svd(ST)



plt.figure(1)
plt.semilogy(sS/sS[0])
plt.semilogy(sT/sT[0])
plt.legend(['sS','sT'])
plt.show()
##

assemblerS = mA.rkHMatAssembler(py-1,10,tree=None,ndim=2)
assemblerT = mA.rkHMatAssembler(py-1,10,tree=None,ndim=2)

STmap = solver.stMap(ST,XXb[Jl,:],XXi[Jc,:])
SSmap = solver.stMap(SS,XXb[Jl,:],XXi[Jc,:])


STlinop = assemblerT.assemble(STmap)
SSlinop = assemblerS.assemble(SSmap)


ES = np.identity(SS.shape[1])
ET = np.identity(ST.shape[1])
STHdense = STlinop@ET
SSHdense = SSlinop@ES


print("Hmat err ST = ",np.linalg.norm(ST-STHdense,ord=2)/np.linalg.norm(ST,ord=2))
print("Hmat err SS = ",np.linalg.norm(SS-SSHdense,ord=2)/np.linalg.norm(SS,ord=2))










