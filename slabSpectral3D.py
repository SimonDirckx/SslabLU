import numpy as np
import multiSlab3D as mS
import matplotlib.pyplot as plt
import spectralDisc as spectral

###########################################
#   THIS SCRIPT PROVIDES A SIMPLE USE GUIDE
#   FOR THE SQUARE MULTISLAB SET-UP IN 3D
###########################################


# Here, equispaced slabs are assumed
# We specify the slab spacing H and
# the slab height L
# we also pick the slab order, here p
# Note: currently, all slabs should be
# discretized uniformly, but this
# restriction will be dropped

ndSlabs = 10
H = 1./(1.*ndSlabs+1.)

Ly = 1.
Lz = 1.
px = 10
py = 20
pz = 20


def f_factory(i):
    def f(x,y,z):
        return x+i*H,y,z  # i is now a *local* variable of f_factory and can't ever change
    return f
l2glist=[]
for ind in range(ndSlabs):
    l2glist+=[f_factory(ind)]

part = mS.partition(Ly,Lz,H)
disc = mS.discretization(px,py,pz,'cheb')
doubleSlabList=[]
for ind in range(ndSlabs):
    dS = mS.dSlab(disc,part,l2glist[ind],H,Ly,Lz)
    doubleSlabList+=[dS]
doubleSlabList[0].setAsEdge(1)
doubleSlabList[ndSlabs-1].setAsEdge(2)
totalDofs = 0
for dS in doubleSlabList:
    totalDofs+=dS.ndofs
    dS.computeL()

MS=mS.multiSlab(doubleSlabList)

print(doubleSlabList[ndSlabs-1].l2g(doubleSlabList[ndSlabs-1].Ibr[0]))


#############################################

# That' sit!

# solving a system is done as follows:

# 1: specify a function on the boundary

def func(x,y,z):
    return np.sin(3*np.pi*x)*np.sinh(3*np.pi*y/np.sqrt(2.))*np.sinh(3*np.pi*z/np.sqrt(2.))/np.sinh(3*np.pi)

# 2: create total S-system and RHS
# for now this is dense, but 
# in the future a sparse solver
# and a specialized tridiag solver
# will be available

Stot = MS.form_total_S()
#print('Stot = ',Stot)

btot = MS.form_total_SRHS(func)
condS = np.linalg.cond(Stot)

# then just solve

u = np.linalg.solve(Stot,btot)

# we can test using the 
# 'eval_at_interfaces' function
# that evaluates a given 
# function at the interface dofs

u0 = MS.eval_at_interfaces(func)
print('u err. @ ',totalDofs,' dofs = ',np.linalg.norm(u-u0,np.inf))
print('cond. @ H = ',H,' is ',condS)
k2 = (2./(np.pi*np.pi))/(H*H)
C = condS/k2
print('C = ',C)
