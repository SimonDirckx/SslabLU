import numpy as np
import hps.pdo as pdo
import matplotlib.pyplot as plt
import multislab.oms as oms
import matAssembly.matAssembler as mA
import solver.solver as solverWrap
import hps.pdo as pdo
from solver.stencil.stencilSolver import stencilSolver as stencil
import solver.stencil.geom as stencilGeom
from matplotlib import cm
from multislab.oms import slab
from scipy.sparse.linalg import LinearOperator
from solver.spectral import spectralSolver as spectral
import geometry.geom_2D.square as square

kh = 9.80177
laplace = False

if laplace:
    def c11(p):
        return np.ones_like(p[:,0])
    def c22(p):
        return np.ones_like(p[:,0])
else:
    def c11(p):
        return (1.+.5*np.cos(2*np.pi*p[:,0]))
    def c22(p):
        return (1.+.5*p[:,0]*p[:,0]*np.sin(3*np.pi*p[:,1]))

diff_op=pdo.PDO2d(c11,c22)
#def c(p):
#    return -kh*kh*np.ones_like(p[:,0])
#Lapl = pdo.PDO2d(c11=c11,c22=c22,c=c)#,c12=c12)

def bc(p):
    return np.sin(np.pi*p[:,0])*np.sinh(np.pi*p[:,1])


Om = square.box_geom()

method = 'spectral'

k = 3
H = 1./(2**k)
print("H = ",H)
if method =='spectral':
    ordy = 50
if method =='stencil':
    ordy = 200
ordx = int(np.round(2*ordy*H))
if method == 'spectral':
    if ordx%2:
        ordx += 1
if method == 'stencil':
    if not ordx%2:
        ordx += 1
ord = [ordx,ordy]

N=(int)(1./H)
dSlabs,connectivity,H = square.dSlabs(N)
assembler = mA.denseMatAssembler()
opts = solverWrap.solverOptions(method,ord)
OMS = oms.oms(dSlabs,diff_op,lambda p :square.gb(p,True),opts,connectivity)
Stot,rhstot = OMS.construct_Stot_and_rhstot(bc,assembler,2)
print("S_op done")
E = np.identity(Stot.shape[0])
S = Stot@E
print("S done")
if method=='spectral':
    yy= spectral.clenshaw_curtis_compute(ordy+1)[0]
    yy = (yy+1)/2.
    w = np.sqrt(np.pi*np.sqrt((yy[1:ordy]-yy[1:ordy]**2)/(2*(ordy+1))))
    W = np.diag(w)
    S = np.kron(np.identity(N-1),W)@S@np.kron(np.identity(N-1),np.linalg.inv(W))

e = np.linalg.eigvals(S)
print(np.max(np.imag(e)))

d=np.inf
for i in range(len(e)):
    di=np.inf
    for j in range(len(e)):
        dij = np.abs(np.real(e[i])+np.real(e[j])-2)
        di=min([di,dij])
    d=min([d,di])

print("sym e = ",d)



I = np.where(np.abs(np.imag(e))>0)[0]

#fileName = 'eig_Helm'+method+'.csv'
#eMat = np.zeros(shape=(len(e),2))
#eMat[:,0] = np.real(e)
#eMat[:,1] = np.imag(e)
#with open(fileName,'w') as f:
#    f.write('real,imag\n')
#    np.savetxt(f,eMat,fmt='%.16e',delimiter=',')


#fileName = 'eig_Helm'+method+'_im.csv'
#eMat = np.zeros(shape=(len(I),2))
#eMat[:,0] = np.real(e[I])
#eMat[:,1] = np.imag(e[I])
#with open(fileName,'w') as f:
#    f.write('real,imag\n')
#    np.savetxt(f,eMat,fmt='%.16e',delimiter=',')

plt.figure(1)
plt.scatter(np.real(e[I]),np.imag(e[I]))
plt.show()
plt.figure(2)
plt.scatter(np.real(e),np.imag(e))
plt.show()