import numpy as np
import jax.numpy as jnp
import solver.spectralmultidomain.hps.pdo as pdo
from packaging.version import Version
import scipy
import matplotlib.pyplot as plt
from scipy.sparse.linalg   import LinearOperator
from solver.solver import stMap
import sys
# oms packages
import solver.solver as solverWrap
import matAssembly.matAssembler as mA
import multislab.oms as oms
from scipy.sparse.linalg import gmres
from matAssembly.HBS.simpleoctree import simpletree as tree
from time import time
import gc

import argparse
import pickle

def compute_c0_L0(XX):
    N,ndim = XX.shape
    c0 = np.sum(XX,axis=0)/N
    L0 = np.max(np.max(XX,axis=0)-np.min(XX,axis=0)) #too tight for some reason
    return c0,L0+1e-5

def compute_stmaps(Il,Ic,Ir,XXi,XXb,solver):
        A_solver = solver.solver_ii    
        def smatmat(v,I,J,transpose=False):
            if (v.ndim == 1):
                v_tmp = v[:,np.newaxis]
            else:
                v_tmp = v

            if (not transpose):
                result = (A_solver@(solver.Aib[:,J]@v_tmp))[I,:]
            else:
                result      = np.zeros(shape=(len(solver.Ii),v_tmp.shape[1]))
                result[I,:] = v_tmp
                result      = solver.Aib[:,J].T @ (A_solver.T@(result))
            if (v.ndim == 1):
                result = result.flatten()
            return result

        Linop_r = LinearOperator(shape=(len(Ic),len(Ir)),\
            matvec = lambda v:smatmat(v,Ic,Ir), rmatvec = lambda v:smatmat(v,Ic,Ir,transpose=True),\
            matmat = lambda v:smatmat(v,Ic,Ir), rmatmat = lambda v:smatmat(v,Ic,Ir,transpose=True))
        Linop_l = LinearOperator(shape=(len(Ic),len(Il)),\
            matvec = lambda v:smatmat(v,Ic,Il), rmatvec = lambda v:smatmat(v,Ic,Il,transpose=True),\
            matmat = lambda v:smatmat(v,Ic,Il), rmatmat = lambda v:smatmat(v,Ic,Il,transpose=True))
        
        st_r = stMap(Linop_r,XXb[Ir,:],XXi[Ic,:])
        st_l = stMap(Linop_l,XXb[Il,:],XXi[Ic,:])
        return st_l,st_r
##################################################################################################################

parser = argparse.ArgumentParser(description="A simple script that takes in 'p' and 'pickle_loc'.")
parser.add_argument('--p', type=int, required=False, default = 6, help="The value for parameter p.")
parser.add_argument('--Horder', type=int, required=False, default = 2, help="The value for parameter Horder.")
parser.add_argument('--pickle_loc', type=str, required=True, help="Path to the pickle file location.")

args = parser.parse_args()

kh = 9.80177
print("kappa = %5.10f, p = %d, Horder=%d" % (kh,args.p,args.Horder))
def c11(p):
    return jnp.ones_like(p[...,0])
def c22(p):
    return jnp.ones_like(p[...,0])
def c33(p):
    return jnp.ones_like(p[...,0])
def bfield(p):
    return -kh*kh*jnp.ones_like(p[...,0])
helmholtz = pdo.PDO3d(c11=c11,c22=c22,c33=c33,c=pdo.const(-kh*kh))

bnds = [[0.,0.,0.],[1.,1.,1.]]
box_geom   = jnp.array(bnds)
def gb_vec(P):
    # P is (N, 2)
    return (
        (np.abs(P[:, 0] - bnds[0][0]) < 1e-14) |
        (np.abs(P[:, 0] - bnds[1][0]) < 1e-14) |
        (np.abs(P[:, 1] - bnds[0][1]) < 1e-14) |
        (np.abs(P[:, 1] - bnds[1][1]) < 1e-14) |
        (np.abs(P[:, 2] - bnds[0][2]) < 1e-14) |
        (np.abs(P[:, 2] - bnds[1][2]) < 1e-14)
    )

def bc(p):
    return jnp.ones_like(p[...,0])


##################################################################################################################

H = 1./(2.0**args.Horder)
N = (int)(1./H)
a = [H/4.,1/32,1/32]

p = args.p

opts    = solverWrap.solverOptions('hps',[p,p,p],a)
geom    = np.array([[0.,0.,0.],[2*H,1.,1.]])
solver  = solverWrap.solverWrapper(opts)
solver.construct(geom,helmholtz,verbose=True)

XX = solver.XX
XXb = XX[solver.Ib,:]
XXi = XX[solver.Ii,:]
xl = geom[0][0]
xr = geom[1][0]
xc=(xl+xr)/2.
print("\t SLAB BOUNDS xl,xc,xr=",xl,",",xc,",",xr)

Il = np.where(np.abs(XXb[:, 0] - xl) < 1e-14)[0]
Ir = np.where(np.abs(XXb[:, 0] - xr) < 1e-14)[0]
Ic = np.where(np.abs(XXi[:, 0] - xc) < 1e-14)[0]
Igb = np.where(gb_vec(XXb))[0]

print("\t SLAB dofs = ",len(Ic))
st,_ = compute_stmaps(Il,Ic,Ir,XXi,XXb,solver)

n=len(Il)

E = np.identity(n)

chunk_size = 10
chunks = []

for i in range(0, n, chunk_size):
    E_chunk = E[:, i:i+chunk_size]  # Select 10 identity columns
    A_chunk = st.A @ E_chunk                     # Get corresponding columns of A
    chunks.append(A_chunk)                       # Store the chunk

# Concatenate all the chunks horizontally to rebuild st.A
A_full = np.hstack(chunks)

# Now do the full SVD
s = np.linalg.svd(A_full, compute_uv=False)

d = {'svd':s,'p':p,'kh':kh,'Horder':args.Horder}
with open(args.pickle_loc, 'wb') as f:
    pickle.dump(d, f)

print("Saved spectrum to pickle %s" % args.pickle_loc)