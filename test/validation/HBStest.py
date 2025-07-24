import numpy as np
import jax.numpy as jnp
import hps.pdo as pdo
import sys

# oms packages
import solver.solver as solverWrap
import matAssembly.matAssembler as mA
import multislab.oms as oms
from solver.solver import stMap
import scipy.sparse.linalg   as splinalg
from scipy.sparse.linalg   import LinearOperator
import torch
import matAssembly.HBS.HBSTree as HBS
import matplotlib.pyplot as plt
def compute_stmaps(Il,Ic,Ir,XXi,XXb,solver):    
    def smatmat(v,I,J,transpose=False):
        if (v.ndim == 1):
            v_tmp = v[:,np.newaxis]
        else:
            v_tmp = v

        if (not transpose):
            result = (solver.solver_ii@(solver.Aib[:,J]@v_tmp))[I,:]
        else:
            result      = np.zeros(shape=(len(solver.Ii),v_tmp.shape[1]))
            result[I,:] = v_tmp
            result      = solver.Aib[:,J].T @ (solver.solver_ii.T@(result))
        if (v.ndim == 1):
            result = result.flatten()
        return result

    Linop_r = LinearOperator(shape=(len(Ic),len(Ir)),\
        matvec = lambda v:smatmat(v,Ic,Ir), rmatvec = lambda v:smatmat(v,Ic,Ir,transpose=True),\
        matmat = lambda v:smatmat(v,Ic,Ir), rmatmat = lambda v:smatmat(v,Ic,Ir,transpose=True))
    Linop_l = LinearOperator(shape=(len(Ic),len(Il)),\
        matvec = lambda v:smatmat(v,Ic,Il), rmatvec = lambda v:smatmat(v,Ic,Il,transpose=True),\
        matmat = lambda v:smatmat(v,Ic,Il), rmatmat = lambda v:smatmat(v,Ic,Il,transpose=True))
    
    st_r = stMap(Linop_r,XXb[Ir,...],XXi[Ic,...])
    st_l = stMap(Linop_l,XXb[Il,...],XXi[Ic,...])
    return st_l,st_r


kh = 9.80177

def c11(p):
    return jnp.ones_like(p[...,0])
def c22(p):
    return jnp.ones_like(p[...,0])
def c33(p):
    return jnp.ones_like(p[...,0])
H=.125
bnds=[[0,0,0],[2*H,1.,1.]]

def gb_vec(P):
    # P is (N, 3)
    return (
        (np.abs(P[:, 0] - bnds[0][0]) < 1e-14) |
        (np.abs(P[:, 0] - bnds[1][0]) < 1e-14) |
        (np.abs(P[:, 1] - bnds[0][1]) < 1e-14) |
        (np.abs(P[:, 1] - bnds[1][1]) < 1e-14) | 
        (np.abs(P[:, 2] - bnds[0][2]) < 1e-14) | 
        (np.abs(P[:, 2] - bnds[1][2]) < 1e-14)
    )

helmholtz = pdo.PDO3d(c11=c11,c22=c22,c33=c33,c=pdo.const(-kh*kh))
pvec = [6,8,10,12]
rkvec = [50,75,100,125,150,175]
errMat = np.zeros(shape=(len(pvec),len(rkvec)))
for indp in range(len(pvec)):
    p=pvec[indp]
    a = [H/8.,1/32,1/32]
    opts = solverWrap.solverOptions('hps',[p,p,p],a)
    geom = np.array(bnds)
    slab_i = oms.slab(geom,gb_vec)
            
    solver = solverWrap.solverWrapper(opts)
    solver.construct(geom,helmholtz)
    print("solver done")
    Il,Ir,Ic,Igb,XXi,XXb = slab_i.compute_idxs_and_pts(solver)
    tree0 = mA.HBS.tree.BalancedTree(XXi[Ic,1:3],p*p,np.array([.5,.5]),np.array([1.+1e-5,1.+1e-5]))            
    st_l,st_r = compute_stmaps(Il,Ic,Ir,XXi,XXb,solver)
    m =st_l.A.shape[0]
    n =st_l.A.shape[1]

    for indrk in range(len(rkvec)):
        rk = rkvec[indrk]
        print("compression at rank ",rk)
        s=5*(rk+10)
        s=max(s,p*p+10)
        err = 0.
        nexp=5
        for j in range(nexp):
            Om  = np.random.standard_normal(size=(n,s))
            Psi = np.random.standard_normal(size=(m,s))
            Y = st_l.A@Om
            Z = st_l.A.T@Psi
            print("random sample done")
            mat = HBS.HBSMAT(tree0,torch.from_numpy(Om),torch.from_numpy(Psi),torch.from_numpy(Y),torch.from_numpy(Z),rk)

            print("compression rate = ",mat.nbytes/(8*np.prod(st_l.A.shape)))
            v=np.random.standard_normal(size=(st_l.A.shape[1],))
            w=v.copy()
     
            for i in range(20):
                v=v/np.linalg.norm(v)
                w=w/np.linalg.norm(w)
                v=mat.matvec(v)-st_l.A@v
                v=mat.matvecT(v)-st_l.A.T@v
                w=st_l.A.T@(st_l.A@w)
            err += np.sqrt(np.linalg.norm(v)/np.linalg.norm(w))
        err/=nexp
        print("avg. err at (p,rk)=(",p,",",rk,") = ",err)
        errMat[indp,indrk]=err
print("errMat = ",errMat)

