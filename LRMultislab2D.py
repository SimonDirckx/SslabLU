import numpy as np
import multiSlab2D as mS
import matplotlib.pyplot as plt
import spectralDisc as spectral
from simpleoctree import simpletree
import HBSTree as HBS



'''
script for Hmat approx. of 2D stencil S-operator
'''

#############################
#   PART ONE: LR from DENSE
#############################


H   =   .5
L   =   1.

k   =   7 #depending on sys specs you may have to switch to a sparse version of multiSlab
N   =   2**k+2

def l2g(x,y):
    return x,y
part    =   mS.partition(L,H)
disc    =   mS.discretization(N,N,'stencil')

dS      =   mS.dSlab(disc,part,l2g,H,L)
dS.setAsEdge(1)
dS.setAsEdge(2)
dS.computeL()
MS=mS.multiSlab([dS])
np.set_printoptions(linewidth=200)
S=dS.getSR()
XX=dS.dofs[dS.Ibl,:]
tree = simpletree.BalancedTree(XX,4)
leaves = tree.get_leaves()
print('number of levels in t:' , tree.nlevels)





N0      = 2**k
rnk     = 4
s=10*rnk
OMEGA   = np.random.standard_normal(size=(N0,s))
PSI     = np.random.standard_normal(size=(N0,s))

Y=S@OMEGA
Z=S.T@PSI

eps = 1e-3
T=HBS.random_compression_HBS(tree,OMEGA,PSI,Y,Z,rnk,s)
T_eps=HBS.random_compression_HBS_eps(tree,OMEGA,PSI,Y,Z,rnk,s,eps)

q=np.random.standard_normal(size=(N0,))
q=q/np.linalg.norm(q)
u=HBS.apply_HBS(T,q)
u_eps=HBS.apply_HBS(T_eps,q)

print('#####################')
nB = T.total_bytes()
nB_eps = T_eps.total_bytes()
print('#DOFs : ',N0)
print('compression factor rk: ',nB/(N0*N0*8))
print('compression factor eps: ',nB_eps/(N0*N0*8))
print('#####################')

def func(x,y):
    return np.sinh(3*np.pi*x)*np.sin(3*np.pi*y)/12000.

f=np.zeros(shape=(N,))
g=np.zeros(shape=(N,))
ypts=np.linspace(0,L,N)
for i in range(N):
    f[i] = func(2.*H,ypts[i])
    g[i] = func(H,ypts[i])
f0 = f[1:N-1]
g0 = g[1:N-1]
u=-S@f0
uH=-HBS.apply_HBS(T,f0)
uHeps=-HBS.apply_HBS(T_eps,f0)
print('S solve err. = ',np.linalg.norm(u-g0,np.Inf))
print('S_H solve err. = ',np.linalg.norm(uH-g0,np.Inf))
print('S_eps solve err. = ',np.linalg.norm(uHeps-g0,np.Inf))