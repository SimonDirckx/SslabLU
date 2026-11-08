import numpy as np
#import direct_solve.omsdirectsolveHBS_torch as omsdirectHBS
import direct_solve.omsdirectsolveHBS_torch_new2 as omsdirectHBS_new
import direct_solve.omsnormalized as omsnormalized
#from direct_solve.omsdirectsolveHBS_torch import rkStrat
from direct_solve.omsdirectsolveHBS_torch_new2 import rkStrat as rkStrat_new
from direct_solve.omsnormalized import rkStrat as rkStrat_normalized
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
import matAssembly.HBS.HBStorch_new as HBStorch_new
import scipy.special as special
from packaging.version import Version
from scipy.sparse.linalg import LinearOperator, gmres


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

kh = 40.3*np.pi

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
p = 10
p_disc = p + 2 # To handle different conventions between hps and hpsalt
a = np.array([H/2,1/64,1/64])
opts = solverWrap.solverOptions('hpsalt',[p_disc,p_disc,p_disc],a,reduced_gpu=True)
rk0 = 300
leaf_size = 8*p*p
assembler = mA.rkHMatAssembler(leaf_size,rk0,ndim=3)
opts = solverWrap.solverOptions('hpsalt',[p_disc,p_disc,p_disc],a,reduced_gpu=True)
tic_sys = time.time()
compare_input_hbs = True  # Optional old input-compression accuracy comparison.
OMS = oms.oms(dSlabs,Helm,lambda p :cube.gb(p,jax_avail=False,torch_avail=True),opts,connectivity,stiff_mat_const=True,constructHBS=compare_input_hbs,keepLU=True,keepDense=False)
print("oms built")
Stot_lu,  rhstot_lu  = OMS.construct_Stot_and_rhstot(bc, assembler,dbg = 0)               # LU operator + LU rhs
print("sys mats done in ",time.time()-tic_sys,"s")
Stot_HBS = OMS.global_operator('hbs') if compare_input_hbs else None
Ntot, nc = OMS.Ntot,OMS.nc
step = 50
strat = rkStrat_new.linear(rk0+step,step,skip_first_level=False)

# All three preconditioners sample the same original OMS operators.
compute_device = 'cuda'
constant_blocks = True  # Constant interiors; non-cyclic boundary rows remain distinct.
strat_normalized = rkStrat_normalized.linear(rk0+step, step, skip_first_level=False)


def benchmark_clock():
    """Include completed GPU work in wall-clock measurements."""
    if torch.device(compute_device).type == 'cuda':
        torch.cuda.synchronize(torch.device(compute_device))
    return time.perf_counter()


# operator=None follows OMS: LU when retained, otherwise HBS.
# Set operator='hbs' explicitly to compare HBS-backed input (requires it built).
variants = {
    'unnormalized': lambda: omsdirectHBS_new.RedBlackSolverHBS(
        strat, OMS, leaf_size=leaf_size,compute_device=compute_device, seed=None, dbg=1,
        normalized=False, constant_blocks=constant_blocks,operator="hbs")#,
    #'normalized': lambda: omsdirectHBS_new.RedBlackSolverHBS(
        #strat, OMS,leaf_size=leaf_size, compute_device=compute_device, seed=None, dbg=1,
        #normalized=True, constant_blocks=constant_blocks,operator="hbs"),
    #'new_normalized': lambda: omsnormalized.RedBlackSolverHBS(
        #strat_normalized, OMS,leaf_size=leaf_size, compute_device=compute_device,
        #work_device='cpu', seed=None, dbg=1, block_diagnostics=False,
        #sample_panel_cols=64, random_buffer_bytes=32 << 20,
        #constant_blocks=constant_blocks,operator="hbs"),
}


def make_inv_op(rb):
    """Bind each preconditioner to its own solver."""
    return LinearOperator(shape=(Ntot, Ntot), matvec=rb.solve, dtype=np.float64)


solvers, Sinv_ops, factor_times, setup_times = {}, {}, {}, {}
for label, make_solver in variants.items():
    tic = benchmark_clock()
    rb = make_solver()
    setup_times[label] = benchmark_clock()-tic
    tic = benchmark_clock()
    rb.factorize()
    factor_times[label] = benchmark_clock()-tic
    print(f"input preparation time {label} = ", setup_times[label])
    print(f"construct + factor time {label} = ", factor_times[label])
    solvers[label] = rb
    Sinv_ops[label] = make_inv_op(rb)

gInfo = gmres_info()
stol = 1e-9

# ---------------------------------------------------------------------------
# Apply timings (each solver warmed up before its timed apply)
# ---------------------------------------------------------------------------
v = np.random.standard_normal(size=(Ntot,))

apply_times = {}
for label, Sinv in Sinv_ops.items():
    solvers[label].solve(v) #warmup
    tic = benchmark_clock()
    w = Sinv@v
    apply_times[label] = benchmark_clock()-tic
    print(f"precond {label} apply time = ",apply_times[label])

tic = benchmark_clock()
w = Stot_lu@v
print("LU apply time = ", benchmark_clock()-tic)

# ---------------------------------------------------------------------------
# Accuracy: same test vector and same operator products for every solver
# ---------------------------------------------------------------------------
v = np.random.default_rng(0).standard_normal(Ntot)
y_hbs = Stot_HBS@v if Stot_HBS is not None else None
y_lu  = Stot_lu@v
err_hbs_vs_lu = None if y_hbs is None else np.linalg.norm(y_lu-y_hbs)/np.linalg.norm(y_lu)
if err_hbs_vs_lu is not None:
    print("HBS vs LU operator", err_hbs_vs_lu)

def accuracy_tests(Sinv, v, y_hbs, y_lu):
    """Compare against the original LU operator and optional input HBS operator."""
    nv = np.linalg.norm(v)
    y1 = Sinv@v; y2 = Sinv@v
    result = {
        "repeatability":         np.linalg.norm(y1-y2)/np.linalg.norm(y1),
        "precond vs LU (bwd)":   np.linalg.norm(v-Sinv@y_lu)/nv,
    }
    if y_hbs is not None:
        result["precond vs HBS"] = np.linalg.norm(v-Sinv@y_hbs)/nv
    return result

accuracy = {}
for label, Sinv in Sinv_ops.items():
    accuracy[label] = accuracy_tests(Sinv, v, y_hbs, y_lu)
    for name, val in accuracy[label].items():
        print(f"[{label}] {name}",val)


# Exact manufactured trace in original OMS ordering, computed once outside timings.
u_exact = OMS.interface_values(bc)
u_exact_norm = np.linalg.norm(u_exact)

# GMRES iteration count with each solver as preconditioner
tol_kw = {"rtol": 1e-9} if Version(scipy.__version__) >= Version("1.12") else {"tol": 1e-9}
for label, Sinv in Sinv_ops.items():
    info_cb = gmres_info()
    tic = benchmark_clock()
    x, info = gmres(Stot_lu, rhstot_lu, M=Sinv, restart=100,maxiter=1,
                    callback=info_cb,callback_type='pr_norm', **tol_kw)
    gmres_time = benchmark_clock()-tic
    interface_error = np.linalg.norm(x - u_exact)
    relative_error = (interface_error / u_exact_norm if u_exact_norm else
                      (0.0 if interface_error == 0.0 else float("inf")))
    accuracy[label]["gmres iterations"] = info_cb.niter
    accuracy[label]["gmres time [s]"] = gmres_time
    accuracy[label]["gmres info"] = info
    accuracy[label]["gmres relative interface error"] = relative_error
    print(f"[{label}] gmres iterations = {info_cb.niter} (info = {info}), "
          f"time = {gmres_time:.3e} s, "
          f"true interface error = {relative_error:.3e}")

# ---------------------------------------------------------------------------
# Side-by-side summary
# ---------------------------------------------------------------------------
labels = list(variants)
rows = [("input preparation [s]", setup_times),
        ("construct + factor [s]", factor_times),
        ("total setup [s]", {l: setup_times[l]+factor_times[l] for l in labels}),
        ("apply time [s]", apply_times)]
rows += [(name, {l: accuracy[l][name] for l in labels}) for name in accuracy[labels[0]]]
name_w = max(len(name) for name, _ in rows)
print()
print(f"{'':<{name_w}}" + "".join(f"{l:>16}" for l in labels))
for name, vals in rows:
    print(f"{name:<{name_w}}" + "".join(f"{vals[l]:>16.3e}" for l in labels))
if err_hbs_vs_lu is not None:
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
