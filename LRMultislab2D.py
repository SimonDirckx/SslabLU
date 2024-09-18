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


H=.25
L=1.

k=9 #depending on sys specs you may have to switch to a sparse version of multiSlab
N=2**k+2

def l2g(x,y):
    return x,y
part = mS.partition(L,H)
disc = mS.discretization(N,N,'stencil')

dS=mS.dSlab(disc,part,l2g,H,L)
dS.computeL()
S=dS.getSL()
XX=dS.dofs[dS.Ibl,:]
tree = simpletree.BalancedTree(XX,8)
leaves = tree.get_leaves()
print('number of levels in t:' , tree.nlevels)





N0      = 2**k
rnk     = 8 
s=3*8
OMEGA   = np.random.standard_normal(size=(N0,s))
PSI     = np.random.standard_normal(size=(N0,s))

Y=S@OMEGA
Z=S.T@PSI


T=HBS.random_compression_HBS(tree,OMEGA,PSI,Y,Z,rnk,s)

q=np.random.standard_normal(size=(N0,))
q=q/np.linalg.norm(q)
q0 = q
u=HBS.apply_HBS(T,q,rnk)

print('#####################')
nB = T.total_bytes()
print('#DOFs : ',N0)
print('compression factor: ',nB/(N0*N0*8))
print('S err. : ',np.linalg.norm(S@q0-u)/np.linalg.norm(S@q0))
print('#####################')

#print('q size : ',q.shape)
#print('q0 size : ',q0.shape)
#print(tree.get_box_children(root[0]))
#print(tree.get_boxes_level(1))
#print(tree.get_boxes_level(2))



