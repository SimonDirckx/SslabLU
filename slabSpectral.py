import numpy as np
import multiSlab as mS
import matplotlib.pyplot as plt
import spectralDisc as spectral

###########################################
#   THIS SCRIPT PROVIDES A SIMPLE USE GUIDE
#   FOR THE SQUARE MULTISLAB SET-UP
###########################################


# Here, equispaced slabs are assumed
# We specify the slab spacing H and
# the slab height L
# we also pick the slab order, here p
# Note: currently, all slabs should be
# discretized uniformly, but this
# restriction will be dropped


H = 1./3.
L = 1.
p = 10


# Per slab, we specify a
# local-to-global map

def l2g1(x,y):
    return x,y
def l2g2(x,y):
    return x+H,y
def l2g3(x,y):
    return x+2.*H,y


# We collect the dofs and their partitions
# note: because the discr. are all the same
# we can get away here with only doing this once

XY,xpts,ypts = spectral.XYpoints(H,L,p)
Iobl,Ioblc,Icbl,Icblc,Iobr,Iobrc,Icbr,Icbrc,Ii  = spectral.partition_single_slab(H,L,XY)


# Next, we set-up the individual
# discrete laplacians
# Again, here we can recycle

N = spectral.discrN(xpts)
Lp = spectral.discrLaplace(xpts,ypts)


# single slabs created

slab1 = mS.Slab(Iobl,Ioblc,Icbl,Icblc,Iobr,Iobrc,Icbr,Icbrc,Ii,Lp,N,N,XY,l2g1)
slab2 = mS.Slab(Iobl,Ioblc,Icbl,Icblc,Iobr,Iobrc,Icbr,Icbrc,Ii,Lp,N,N,XY,l2g2)
slab3 = mS.Slab(Iobl,Ioblc,Icbl,Icblc,Iobr,Iobrc,Icbr,Icbrc,Ii,Lp,N,N,XY,l2g3)

# we indicate to the program
# that slab 1 & 3 live at the edge of the domain

slab1.set_edge()
slab3.set_edge()


# we create the two corresponding double slabs
# and collect them
dSlab1 = mS.dSlab(slab1,slab2)
dSlab2 = mS.dSlab(slab2,slab3)
slabList = [dSlab1,dSlab2]

# until we command it to,
# the double slab object does not compute its
# local Laplacian (this way we could delay it
# to some distributed parallel loop, or whatever)
# Note: the local laplacian contains the normal
# comtinuity conditions
# In future versions, this could be adapted
# such that one discr. per double slab is
# utilized

dSlab1.computeL()
dSlab2.computeL()


# multislab is now a rather simple wrapper
# with internal functions that create the balanced S-system

MS=mS.multiSlab(slabList)


#############################################

# That' sit!

# solving a system is done as follows:

# 1: specify a function on the boundary

def func(x,y):
    return np.sin(3*np.pi*x)*np.sinh(3*np.pi*y)/np.sinh(3*np.pi)

# 2: create total S-system and RHS
# for now this is dense, but 
# in the future a sparse solver
# and a specialized tridiag solver
# will be available

Stot = MS.form_total_S()
btot = MS.form_total_SRHS(func)

# then just solve

u = np.linalg.solve(Stot,btot)

# we can test using the 
# 'eval_at_interfaces' function
# that evaluates a given 
# function at the interface dofs

u0 = MS.eval_at_interfaces(func)
print('u err. = ',np.linalg.norm(u-u0,np.inf))