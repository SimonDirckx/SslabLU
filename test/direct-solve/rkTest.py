import numpy as np
import direct_solve.omsdirectsolveHBS as omsdirectHBS
from direct_solve.omsdirectsolveHBS import rkStrat
import torch

# oms packages
import solver.solver as solverWrap
import matAssembly.matAssembler as mA
import multislab.oms as oms
import solver.hpsmultidomain.hpsmultidomain.pdo as pdoalt
import time
import geometry.geom_2D.square as square
import matAssembly.HBS.HBStorch as HBStorch
import scipy.special as special


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

kh = 157.02
def bfield(xx):            
    mag   = 0.930655
    width = 2500; 
    
    b = torch.zeros_like(xx[:,0])
    
    dist = 0.04
    x0=0.1+0.5*dist; x1 = 0.50; x2 = x1+2.5*dist; x3= 0.9
    y0=0.1+0.5*dist; y1 = 0.50; y2 = y1+2.5*dist; y3= 0.9
    
    # box of points [x0,x1] x [y0,y1]
    for x in np.arange(x0,x1,dist):
        for y in np.arange(y0,y1,dist):
            xx_sq_c = (xx[:,0] - x)**2 + (xx[:,1] - y)**2
            b += mag * torch.exp(-width * xx_sq_c)

    # box of points [x0,x1] x [y0,y2]
    for x in np.arange(x2,x3,dist):
        for y in np.arange(y0,y2-0.5*dist,dist):
            xx_sq_c = (xx[:,0] - x)**2 + (xx[:,1] - y)**2
            b += mag * torch.exp(-width * xx_sq_c)
            
    # box of points [x0,x3] x [y2,y3]
    for x in np.arange(x0,x3,dist):
        for y in np.arange(y2,y3,dist):
            xx_sq_c = (xx[:,0] - x)**2 + (xx[:,1] - y)**2
            b += mag * torch.exp(-width * xx_sq_c)    
    
    kh_fun = -kh**2 * (1 - b)
    return kh_fun
def c(p):
    return bfield(p)



def c11(p):
    return torch.ones_like(p[:,0])
def c22(p):
    return torch.ones_like(p[:,1])
Helm=pdoalt.PDO_2d(c11=c11,c22=c22,c=c)

def bc(p):
    return np.ones_like(p[:,0])


N = 33
dSlabs,connectivity,H = square.dSlabs(N)
pvec = np.array([20],dtype = np.int64)
for indp in range(len(pvec)):
    p = pvec[indp]
    p_disc = p
    formulation = "hpsalt"
    p_disc = p_disc + 2 # To handle different conventions between hps and hpsalt
    a = np.array([H/4,1/64])
    rk0 = 50
    assembler = mA.rkHMatAssembler(4*p,rk0,ndim=2)
    opts = solverWrap.solverOptions(formulation,[p_disc,p_disc],a,reduced_gpu=False)
    tic_sys = time.time()
    OMS = oms.oms(dSlabs,Helm,lambda p :square.gb(p,jax_avail=False,torch_avail=True),opts,connectivity,stiff_mat_const=False,constructHBS=True,keepLU=True,keepDense=True)
    print("oms built")
    Stot_lu,  rhstot_lu  = OMS.construct_Stot_and_rhstot(bc, assembler)               # LU operator + LU rhs
    print("sys mats done in ",time.time()-tic_sys,"s")
    Stot_HBS, rhstot_HBS = OMS.construct_Stot_and_rhstot(bc, assembler, rhsHBS=True)  # HBS operator + HBS rhs
    print("sys mats done in ",time.time()-tic_sys,"s")
    nc = OMS.nc
    S_rk_list = OMS.hbs_blocks
    S_dense_list = OMS.S_dense_list
    step = 10
    #strat = rkStrat.linear(rk0+step,step,skip_first_level=False)
    strat = rkStrat.logarithmic(rk0+step,80,skip_first_level=False)
    #strat = rkStrat.constant(rk0)
    #rb_solver = omsdirectHBS.RedBlackSolverHBS(nc,strat,tree = S_rk_list[0][0].tree,quad = False,compress_diag=True,diagnostics=True,fast=True,identity_diag=True,seed=None)
    diagnostics = True
    #rb_solver = omsdirectHBS.RedBlackSolverHBS(nc,strat,tree = S_rk_list[0][0].tree,quad = False,compress_diag=True,diagnostics=diagnostics,fast=True,identity_diag=True,seed=None)
    rb_solver = omsdirectHBS.ThomasSolverHBS(nc,strat,diagnostics=diagnostics)
    
    if diagnostics:
        rb_solver.factorize(S_rk_list,S_exact=S_dense_list)
        print(rb_solver.report.table("comb",   rows="stages"))  # one level isolated
        print(rb_solver.report.table("ladder", rows="stages"))  # the solver's own path
        for r in rb_solver.report.comb():
            if r["block"] is not None and r["stage"] >= 3:
                norm = r["dense_abs"] / r["dense_rel"]
                print(f'{r["stage"]}  {r["block"]:22s}  norm {norm:.2e}  rel.err {r["dense_rel"]:.2e}')
                Sdense_lu = np.identity(nc*(N-1))
        uhat =rb_solver.solve(rhstot_lu)
        for i in range(len(S_dense_list)):
            if i>0:
                Sdense_lu[:,(i-1)*nc:i*nc][i*nc:(i+1)*nc,:] = S_dense_list[i][0]
            if i<len(S_dense_list)-1:
                Sdense_lu[:,(i+1)*nc:(i+2)*nc][i*nc:(i+1)*nc,:] = S_dense_list[i][-1]
        udense = np.linalg.solve(Sdense_lu,rhstot_lu)
        
        for slabInd in range(len(dSlabs)):
            gc_HBS = uhat[slabInd*nc:(slabInd+1)*nc]
            gc_dense = udense[slabInd*nc:(slabInd+1)*nc]
            err = np.linalg.norm(gc_HBS-gc_dense,ord=np.inf)
            print("===================LOCAL ERR===================")
            print("err = ",err)
            print("===============================================")
    else:
        tic = time.time()
        rb_solver.factorize(S_rk_list)
        print("factorization done in ",time.time()-tic)
        uhat =rb_solver.solve(rhstot_lu)

    
    

    # ---- volumetric solution -------------------------------------------- #
    # One local solve per slab against the kept LU factors (keepLU=True),
    # then piecewise spectral interpolation onto a plotting grid.
    import matplotlib.pyplot as plt
    boxes = np.array([np.asarray(s, dtype=float) for s in dSlabs])
    lo, hi = boxes.min(axis=(0, 1)), boxes.max(axis=(0, 1))
    nplot = 500
    X, Y = np.meshgrid(np.linspace(lo[0], hi[0], nplot),
                       np.linspace(lo[1], hi[1], nplot))
    pts = np.column_stack([X.ravel(), Y.ravel()])

    tic = time.time()
    sol = OMS.reconstruct(uhat, bc)       # expensive part, done once
    print("reconstruction done in ", time.time() - tic, "s")
    U = sol(pts).reshape(X.shape)         # cheap, repeatable

    plt.figure(figsize=(6, 5))
    plt.pcolormesh(X, Y, U, shading="auto", cmap="RdBu_r")
    plt.colorbar()
    plt.gca().set_aspect("equal")
    plt.title("OMS solution, p = %d" % p)
    plt.savefig("oms_solution_p%d.png" % p, dpi=200, bbox_inches="tight")
    plt.show()