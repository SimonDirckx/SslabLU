# basic packages
import numpy as np
import jax.numpy as jnp
import scipy
from packaging.version import Version


# oms packages
import solver.solver as solverWrap
import matAssembly.matAssembler as mA
import multislab.oms as oms
import multislab.omsdirectsolve as omsdirectsolve

# validation&testing
import time
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt
import geometry.geom_3D.twistedTorus as twisted

import torch

class gmres_info(object):
    def __init__(self, disp=False):
        self._disp = disp
        self.niter = 0
        self.resList=[]
    def __call__(self, rk=None):
        self.niter += 1
        self.resList+=[rk]
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))


jax_avail    = False
torch_avail  = True
hpsalt       = True
direct_solve = True





bnds = twisted.bnds

################################################################
#
#   SET-UP BVP:         Helmholtz on 3D Annulus
#   - wave number       (kh)
#   - bfield            (= kh*ones)
#   - pdo_mod           (pdo transformed to square)
#   - BC
#   - known exact sol.  (u_exact)
#
################################################################

nwaves = 1.24
wavelength = 4/nwaves
kh = (nwaves/4)*2.*np.pi

if jax_avail:
    def bfield(p,kh):
        return -kh*kh*jnp.ones_like(p[...,0])
elif torch_avail:
    def bfield(p,kh):
        return -kh*kh*torch.ones(p.shape[0])
else:
    def bfield(p,kh):
        return -kh*kh*np.ones(shape=(p.shape[0],))
param_geom=twisted.param_geom(jax_avail=jax_avail, torch_avail=torch_avail, hpsalt=hpsalt)
pdo_mod = param_geom.transform_helmholtz_pdo(bfield,kh)


def bc(p):
    z1=twisted.z1(p,jax_avail=jax_avail,torch_avail=torch_avail)
    z2=twisted.z2(p,jax_avail=jax_avail,torch_avail=torch_avail)
    z3=twisted.z3(p,jax_avail=jax_avail,torch_avail=torch_avail)
    rr = np.sqrt(z1**2+z2**2+z3**2)
    return np.cos(kh*rr)/(4*np.pi*rr)


N = 8
dSlabs,connectivity,H = twisted.dSlabs(N)


formulation = "hps"
p = 4
p_disc = p
if hpsalt:
    formulation = "hpsalt"
    p_disc = p_disc + 2 # To handle different conventions between hps and hpsalt
a = np.array([H/8.,1/8,1/8])
assembler = mA.rkHMatAssembler(p*p,50)
opts = solverWrap.solverOptions(formulation,[p_disc,p_disc,p_disc],a)

for slabInd in range(len(dSlabs)):
    geom    = np.array(dSlabs[slabInd])
    slab_i  = oms.slab(geom,lambda p : twisted.gb(p,jax_avail=jax_avail,torch_avail=torch_avail))
    solver  = oms.solverWrap.solverWrapper(opts)
    solver.construct(geom,pdo_mod)
    Il,Ir,Ic,Igb,XXi,XXb = slab_i.compute_idxs_and_pts(solver)
    g1 = np.zeros(shape=(XXb.shape[0],))
    g2 = np.zeros(shape=(XXb.shape[0],))
    g3 = np.zeros(shape=(XXb.shape[0],))
    gtot = np.zeros(shape=(XXb.shape[0],))
    gl=bc(XXb[Il,:])
    gr=bc(XXb[Ir,:])
    gb = bc(XXb[Igb,:])

    gl = np.array(gl)
    gr = np.array(gr)
    gb = np.array(gb)


    g1[Il] = gl
    g2[Ir] = gr
    g3[Igb] = gb

    gtot[Il] = gl
    gtot[Ir] = gr
    gtot[Igb] = gb

    Aib = np.array(solver.Aib.todense())

    u1 = -solver.solver_ii@(Aib@g1)
    u2 = -solver.solver_ii@(Aib@g2)
    u3 = -solver.solver_ii@(Aib@g3)
    utot = -solver.solver_ii@(Aib@gtot)


    Sl = -(solver.solver_ii@(Aib[:,Il]))[Ic,:]
    Sr = -(solver.solver_ii@(Aib[:,Ir]))[Ic,:]

    rhs = u3[Ic]
    uc = utot[Ic]

    print("additivity = ",np.linalg.norm(-Sl@gl-Sr@gr+uc-rhs)/np.linalg.norm(utot))

