import numpy as np
import direct_solve.omsdirectsolveHBS_torch as omsdirectHBS
import direct_solve.omsdirectsolveHBS_torch_new2 as omsdirectHBS_new
from direct_solve.omsdirectsolveHBS_torch import rkStrat
from direct_solve.omsdirectsolveHBS_torch_new2 import rkStrat as rkStrat_new
import torch
import scipy
# oms packages
import solver.solver as solverWrap
import matAssembly.matAssembler as mA
import multislab.oms as oms
import solver.hpsmultidomain.hpsmultidomain.pdo as pdoalt
import time
import geometry.geom_3D.cube as cube
import matAssembly.HBS.HBStorch as HBStorch
import scipy.special as special
from packaging.version import Version
from scipy.sparse.linalg import gmres


def dense_to_linop(A):
    A = np.array(A)
    n = A.shape[0]
    lo = LinearOperator(
        shape=(n, n), dtype=A.dtype,
        matvec  = lambda v: A @ v,
        rmatvec = lambda v: A.T @ v,
        matmat  = lambda V: A @ V,
        rmatmat = lambda V: A.T @ V,
    )
    lo.solve = lambda v, mode='N': (
        np.linalg.solve(A, v) if mode == 'N' else np.linalg.solve(A.T, v)
    )
    lo.tree = lo.quad = None
    return lo

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

kh = 40*np.pi

def c11(p):
    return torch.ones_like(p[:,0])
def c22(p):
    return torch.ones_like(p[:,1])
def c33(p):
    return torch.ones_like(p[:,2])
def c(p):
    return -kh*kh*torch.ones_like(p[:,0])
Helm=pdoalt.PDO_3d(c11=c11,c22=c22,c33=c33,c=c)

def bc(p):
    c0 = -1
    c1 = -1
    c2 = -1
    r = np.sqrt((p[:,0]-c0)**2+(p[:,1]-c1)**2+(p[:,2]-c2)**2)
    return np.cos(kh*r)/(4*np.pi*r)


N = 33
dSlabs,connectivity,H = cube.dSlabs(N)
p = 8
p_disc = p + 2 # To handle different conventions between hps and hpsalt
a = np.array([H/4,1/64,1/64])
opts = solverWrap.solverOptions('hpsalt',[p_disc,p_disc,p_disc],a,reduced_gpu=True)
rk0 = 200
assembler = mA.rkHMatAssembler(8*p*p,rk0,ndim=3)
opts = solverWrap.solverOptions('hpsalt',[p_disc,p_disc,p_disc],a,reduced_gpu=True)
tic_sys = time.time()
OMS = oms.oms(dSlabs,Helm,lambda p :cube.gb(p,jax_avail=False,torch_avail=True),opts,connectivity,stiff_mat_const=True,constructHBS=True,keepLU=True,keepDense=False)
print("oms built")
Stot_lu,  rhstot_lu  = OMS.construct_Stot_and_rhstot(bc, assembler,dbg = 0)               # LU operator + LU rhs
print("sys mats done in ",time.time()-tic_sys,"s")
Stot_HBS, rhstot_HBS = OMS.construct_Stot_and_rhstot(bc, assembler, rhsHBS=True,dbg = 0)  # HBS operator + HBS rhs
print("sys mats done in ",time.time()-tic_sys,"s")
S_rk_list = OMS.hbs_blocks
Ntot, nc = OMS.Ntot,OMS.nc
step = 40
strat = rkStrat_new.linear(200+step,step,skip_first_level=False)
#rb_solver = omsdirectHBS.RedBlackSolverHBS(nc,strat,tree = S_rk_list[0][0].tree,quad = False,compress_diag=True,fast=True,identity_diag=True,compute_device='cpu',seed=None)

def set_solve_device(rb, dev):
    """Point the RB solver and every HBS block it uses at `dev`.
    Moves no data: the factors already live in the host masters."""
    dev = torch.device(dev)
    rb._drop_slot()
    rb.release_solve_cache()
    blocks = {id(b): b for b in rb._blocks}
    for SiM, T, T_hbs, SiP in rb.RB:                 # catch anything not registered
        for op in (*SiM, *T, *T_hbs, *SiP):
            blocks.setdefault(id(op), op)
    for b in blocks.values():
        if isinstance(b, HBStorch.HBSMAT):
            b.to(dev)                                # evicts any device copy
    rb.compute_device = dev

#set_solve_device(rb_solver, 'cpu')

# ---------------------------------------------------------------------------
# Solver variants: identical setup, only the `normalized` flag differs.
# Add an entry here to test another variant.
# ---------------------------------------------------------------------------
variants = {"unnormalized": False, "normalized": True}

def make_inv_op(rb):
    """Wrap one RB solver as a LinearOperator (bound to that solver, not a loop variable)."""
    def matvec(v):
        return rb.solve(v)
    return scipy.sparse.linalg.LinearOperator(shape=(Ntot,Ntot),matvec=matvec,dtype=np.float64)

solvers, Sinv_ops, factor_times = {}, {}, {}
for label, normalized in variants.items():
    tic = time.time()
    rb = omsdirectHBS_new.RedBlackSolverHBS(nc,strat,tree = S_rk_list[0][0].tree,quad = False,compute_device='cuda',seed=None,dbg=1,normalized=normalized)
    rb.factorize(S_rk_list)
    factor_times[label] = time.time()-tic
    print(f"construct + factor time {label} = ",factor_times[label])
    solvers[label]  = rb
    Sinv_ops[label] = make_inv_op(rb)

gInfo = gmres_info()
stol = 1e-7

# ---------------------------------------------------------------------------
# Apply timings (each solver warmed up before its timed apply)
# ---------------------------------------------------------------------------
v = np.random.standard_normal(size=(Ntot,))

apply_times = {}
for label, Sinv in Sinv_ops.items():
    solvers[label].solve(v) #warmup
    tic = time.time()
    w = Sinv@v
    apply_times[label] = time.time()-tic
    print(f"precond {label} apply time = ",apply_times[label])

tic = time.time()
w = Stot_lu@v
print("LU apply time = ",time.time()-tic)

# ---------------------------------------------------------------------------
# Accuracy: same test vector and same operator products for every solver
# ---------------------------------------------------------------------------
v = np.random.default_rng(0).standard_normal(Ntot)
y_hbs = Stot_HBS@v
y_lu  = Stot_lu@v
err_hbs_vs_lu = np.linalg.norm(y_lu-y_hbs)/np.linalg.norm(y_lu)   # solver-independent
print("HBS vs LU operator",err_hbs_vs_lu)

def accuracy_tests(Sinv, v, y_hbs, y_lu):
    """Solver-dependent accuracy checks; y_hbs = Stot_HBS@v, y_lu = Stot_lu@v."""
    nv = np.linalg.norm(v)
    y1 = Sinv@v; y2 = Sinv@v
    return {
        "repeatability":         np.linalg.norm(y1-y2)/np.linalg.norm(y1),
        "precond vs HBS":        np.linalg.norm(v-Sinv@y_hbs)/nv,
        "precond vs LU (bwd)":   np.linalg.norm(v-Sinv@y_lu)/nv,
    }

accuracy = {}
for label, Sinv in Sinv_ops.items():
    accuracy[label] = accuracy_tests(Sinv, v, y_hbs, y_lu)
    for name, val in accuracy[label].items():
        print(f"[{label}] {name}",val)

# ---------------------------------------------------------------------------
# Side-by-side summary
# ---------------------------------------------------------------------------
labels = list(variants)
rows = [("construct + factor [s]", factor_times), ("apply time [s]", apply_times)]
rows += [(name, {l: accuracy[l][name] for l in labels}) for name in accuracy[labels[0]]]
name_w = max(len(name) for name, _ in rows)
print()
print(f"{'':<{name_w}}" + "".join(f"{l:>16}" for l in labels))
for name, vals in rows:
    print(f"{name:<{name_w}}" + "".join(f"{vals[l]:>16.3e}" for l in labels))
print(f"(HBS vs LU operator, solver-independent: {err_hbs_vs_lu:.3e})")

#for slabInd in range(len(dSlabs)):
#    gc_HBS = uhat[slabInd*nc:(slabInd+1)*nc]
#    geom    = np.array(dSlabs[slabInd])
#    slab_i  = oms.slab(geom,lambda p : cube.gb(p,False,True))
#    solver  = oms.solverWrap.solverWrapper(opts)
#    solver.construct(geom,Helm,False,True)
#    Il,Ir,Ic,Igb,XXi,XXb = slab_i.compute_idxs_and_pts(solver)
#    XXc = XXi[Ic,:]
#    gc = bc(XXc).detach().cpu().numpy()
#    g_from_solver = -solver.solver_ii@(solver.Aib@(bc(XXb).detach().cpu().numpy()))
#    err = np.linalg.norm(gc_HBS-gc,ord=np.inf)
#    err_ii = np.linalg.norm(gc-g_from_solver[Ic],ord=np.inf)
#    err_HBS_vs_ii = np.linalg.norm(gc_HBS-g_from_solver[Ic],ord=np.inf)
#    print("===================LOCAL ERR===================")
#    print("err = ",err)
#    print("err_ii = ",err_ii)
#    print("err_HBS_vs_ii = ",err_HBS_vs_ii)
#    print("===============================================")
