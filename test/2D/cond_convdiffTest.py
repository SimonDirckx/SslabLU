import numpy as np
import jax.numpy as jnp
import torch
import scipy.linalg as sclinalg
from packaging.version import Version

# oms packages
import solver.solver as solverWrap
import multislab.oms as oms
import solver.hpsmultidomain.hpsmultidomain.pdo as pdo
# validation & testing
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt
import geometry.geom_2D.square as square
import globalSolution as G
import matAssembly.matAssembler as mA
import time
import solver.hpsmultidomain.hpsmultidomain.hps_leaf_disc as leaf_disc

CPU = torch.device("cpu")
def _cc_weights(pp):
        """Clenshaw-Curtis weights on [-1,1] for pp Chebyshev-Lobatto nodes (ascending)."""
        n = pp - 1
        if n == 0:
            return np.array([0.0]), np.array([2.0])
        th = np.pi*np.arange(n+1)/n
        x  = np.cos(th)
        w  = np.zeros(n+1); ii = np.arange(1, n); v = np.ones(n-1)
        if n % 2 == 0:
            w[0] = w[n] = 1.0/(n*n - 1)
            for k in range(1, n//2):
                v -= 2*np.cos(2*k*th[ii])/(4*k*k - 1)
            v -= np.cos(n*th[ii])/(n*n - 1)
        else:
            w[0] = w[n] = 1.0/(n*n)
            for k in range(1, (n-1)//2 + 1):
                v -= 2*np.cos(2*k*th[ii])/(4*k*k - 1)
        w[ii] = 2*v/n
        return x[::-1], w[::-1]                       # ascending nodes & matching weights


def _legendre_1d(p, scl):
    """p open Gauss-Legendre nodes and weights on (0, scl), increasing order."""
    pts, wts = np.polynomial.legendre.leggauss(p)
    pts = (pts + 1) / 2 * scl
    wts = wts * scl / 2
    return pts, wts

# ---------------------------------------------------------------------------
# Conservative (divergence-form) operator  (1/dt) - div(D grad)
#   c11=c22=D,  c1=-Dx,  c2=-Dy,  c=1/dt
# ---------------------------------------------------------------------------

#formulation = "hpsalt"
formulation = "stencil"


def make_pdo_div_Dxy(Dbar, eps, sy, sx, dt,formulation):
    if formulation == 'hpsalt':
        tp = 2*np.pi
        Dfun  = lambda xx: Dbar*(1.0 + eps*(sy*(2.0*xx[:,1]-1.0) + sx*torch.cos(tp*xx[:,0]))) # -op, because of HPS impl.
        c1fun = lambda xx: tp*Dbar*eps*sx*torch.sin(tp*xx[:,0])      # = -Dx
        c2fun = lambda xx: -(2.0*Dbar*eps*sy) + 0.0*xx[:,0]          # = -Dy (const)
    else:
        tp = 2*np.pi
        Dfun  = lambda xx: -Dbar*(1.0 + eps*(sy*(2.0*xx[:,1]-1.0) + sx*np.cos(tp*xx[:,0])))
        c1fun = lambda xx: tp*Dbar*eps*sx*np.sin(tp*xx[:,0])            # = Dx
        c2fun = lambda xx: -(2.0*Dbar*eps*sy) + 0.0*xx[:,0]             # = Dy (const)
    return pdo.PDO_2d(Dfun, Dfun, c1=c1fun, c2=c2fun, c=pdo.const(1))


# ===========================================================================
# Problem constants
# ===========================================================================

hpsalt   = True
jax_avail = False
torch_avail = True

p                   = 64
nn = 2**8+1
use_lu              = False
p_disc              = p + 2
cx, cy              = 0.5, 0.5
Dbar, eps, sy, sx   = 1, 0.3, 1.0, 0.5
Nvec        = np.array([4,8,16,32,64],dtype = np.int32)


nrm_diff_vec_stencil    =   np.zeros((len(Nvec),))
nrm_diff_vec_HPS        =   np.zeros((len(Nvec),))
lamdiff_vec_stencil     =   np.zeros((len(Nvec),))
lamdiff_vec_HPS         =   np.zeros((len(Nvec),))
condstat_vec_stencil    =   np.zeros((len(Nvec),))
condstat_vec_HPS        =   np.zeros((len(Nvec),))

for indN in range(len(Nvec)):
    N                       = Nvec[indN]
    dSlabs, connectivity, H = square.dSlabs(N,periodic=False)
    print("number of dSlabs = ",len(dSlabs))
    ntile_y = 4
    a                       = np.array([1/64, 1/(2*ntile_y)])            # leaf box scale (x, y)
    ax  = 2*a[0]                          # leaf x-extent (a is half-width in HPS convention)
    op_HPS  = make_pdo_div_Dxy(Dbar, eps, sy, sx, 0.,'spectral')
    op_stencil  = make_pdo_div_Dxy(Dbar, eps, sy, sx, 0.,'stencil')
    
    
    opts_HPS = solverWrap.solverOptions('spectral', [2*p//N+2, p_disc], a, 'Dirichlet')
    opts_stencil = solverWrap.solverOptions('stencil', [2*nn//N+1, nn], 'Dirichlet')

    if use_lu:
        OMS_HPS  = oms.oms_lu(dSlabs, op_HPS,
                    lambda P: square.gb(P, jax_avail=jax_avail, torch_avail=torch_avail),
                    opts_HPS, connectivity)
        OMS_stencil = oms.oms_lu(dSlabs, op_stencil,
                    lambda P: square.gb(P, jax_avail=jax_avail, torch_avail=torch_avail),
                    opts_stencil, connectivity)
    else:
        OMS_HPS  = oms.oms(dSlabs, op_HPS,
                    lambda P: square.gb(P, jax_avail=jax_avail, torch_avail=torch_avail),
                    opts_HPS, connectivity)
        OMS_stencil  = oms.oms(dSlabs, op_stencil,
                    lambda P: square.gb(P, jax_avail=jax_avail, torch_avail=torch_avail),
                    opts_stencil, connectivity)
    def bc(p):
        return np.ones_like(p[:,0])
    if use_lu:
        S_rk_list_HPS, rhs_list, Ntot, nc_hps = OMS_HPS.construct_Stot_helper(bc)
        Stot_HPS, rhstot = OMS_HPS.construct_Stot_and_rhstot_linearOperator(S_rk_list_HPS, rhs_list, Ntot, nc_hps, dbg=0)
        S_rk_list_stencil, rhs_list, Ntot, nc_stencil = OMS_stencil.construct_Stot_helper(bc)
        Stot_stencil, rhstot = OMS_stencil.construct_Stot_and_rhstot_linearOperator(S_rk_list_stencil, rhs_list, Ntot, nc_stencil, dbg=0)
    else:
        assembler = mA.denseMatAssembler()
        S_rk_list_HPS, rhs_list, Ntot, nc_hps = OMS_HPS.construct_Stot_helper(bc,assembler)
        Stot_HPS, rhstot = OMS_HPS.construct_Stot_and_rhstot_linearOperator(S_rk_list_HPS, rhs_list, Ntot, nc_hps, dbg=0)
        S_rk_list_stencil, rhs_list, Ntot, nc_stencil = OMS_stencil.construct_Stot_helper(bc,assembler)
        Stot_stencil, rhstot = OMS_stencil.construct_Stot_and_rhstot_linearOperator(S_rk_list_stencil, rhs_list, Ntot, nc_stencil, dbg=0)
    
    
    _, w = _cc_weights(p+3)
    w[1]+=w[0]/2
    w[-2]+=w[-1]/2
    w = w[1:-1]
    w = np.sqrt(w).T
    W_HPS = np.kron(np.identity(len(dSlabs)),np.diag(w))

    S_HPS = Stot_HPS@np.identity(Stot_HPS.shape[0])
    S_HPS = W_HPS@S_HPS@(np.linalg.inv(W_HPS))

    

    S_stencil = Stot_stencil@np.identity(Stot_stencil.shape[0])
    
    middle_ind = (N-1)//2
    nc_hps = S_HPS.shape[0]//(N-1)
    nc_stencil = S_stencil.shape[0]//(N-1)
    Block1_HPS = S_HPS[:,middle_ind*nc_hps:(middle_ind+1)*nc_hps][(middle_ind-1)*nc_hps:middle_ind*nc_hps,:]
    Block2_HPS = S_HPS[:,(middle_ind-1)*nc_hps:middle_ind*nc_hps][middle_ind*nc_hps:(middle_ind+1)*nc_hps,:]
    Block1_stencil = S_stencil[:,middle_ind*nc_stencil:(middle_ind+1)*nc_stencil][(middle_ind-1)*nc_stencil:middle_ind*nc_stencil,:]
    Block2_stencil = S_stencil[:,(middle_ind-1)*nc_stencil:middle_ind*nc_stencil][middle_ind*nc_stencil:(middle_ind+1)*nc_stencil,:]
    print("nc_hps = ",nc_hps)
    print("nc_stencil = ",nc_stencil)
    
    nrm_diff_HPS = np.linalg.norm(Block1_HPS-Block2_HPS.T,ord=2)/np.linalg.norm(Block1_HPS,ord=2)
    nrm_diff_stencil = np.linalg.norm(Block1_stencil-Block2_stencil.T,ord=2)/np.linalg.norm(Block1_stencil,ord=2)
    print("nrm diff HPS = ",nrm_diff_HPS)
    print("nrm diff stencil = ",nrm_diff_stencil)

    e_HPS = np.linalg.eigvals(S_HPS)
    e_stencil = np.linalg.eigvals(S_stencil)

    s_HPS = sclinalg.svdvals(S_HPS)
    s_stencil = sclinalg.svdvals(S_stencil)


    s_stencil_copy = s_stencil.copy()
    s_HPS_copy = s_HPS.copy()
    e_HPS = np.abs(e_HPS)
    e_stencil = np.abs(e_stencil)
    
    lamdiff_HPS = 0
    lamdiff_stencil = 0

    e_HPS = np.sort(e_HPS)[::-1]
    e_stencil = np.sort(e_stencil)[::-1]
    s_HPS = s_HPS
    s_stencil = s_stencil
    lamdiff_HPS = np.linalg.norm(e_HPS-s_HPS,ord = np.inf)
    lamdiff_stencil = np.linalg.norm(e_stencil-s_stencil,ord = np.inf)
    mx_ind_HPS = np.argmax(e_HPS-s_HPS)
    mx_ind_stencil = np.argmax(e_stencil-s_stencil)
    
    print("lamdiff HPS = ",lamdiff_HPS," at ", mx_ind_HPS, " = ",e_HPS[mx_ind_HPS])
    print("lamdiff stencil",lamdiff_stencil," at ", mx_ind_stencil," = ",e_stencil[mx_ind_stencil])

    cond_stencil  = max(s_stencil)/min(s_stencil)
    cond_est_stencil  = max(e_stencil)/min(e_stencil)

    cond_HPS        = max(s_HPS)/min(s_HPS)
    cond_est_HPS    = max(e_HPS)/min(e_HPS)
    print("cond stat HPS = ",cond_HPS/cond_est_HPS - 1.)
    print("cond stat stencil = ",cond_stencil/cond_est_stencil - 1.)

    plt.figure(1)
    plt.scatter(np.real(e_HPS),np.imag(e_HPS))
    plt.scatter(np.real(s_HPS),np.imag(s_HPS))
    plt.legend(['e_HPS','s_HPS'])
    plt.figure(2)
    plt.scatter(np.real(e_stencil),np.imag(e_stencil))
    plt.scatter(np.real(s_stencil),np.imag(s_stencil))
    plt.legend(['e_stencil','s_stencil'])
    #plt.show()
    nrm_diff_vec_stencil[indN]      =  nrm_diff_stencil
    nrm_diff_vec_HPS[indN]          =  nrm_diff_HPS
    lamdiff_vec_stencil[indN]       =  lamdiff_stencil
    lamdiff_vec_HPS[indN]           =  lamdiff_HPS 
    condstat_vec_stencil[indN]      =  cond_stencil/cond_est_stencil - 1.
    condstat_vec_HPS[indN]          =  cond_HPS/cond_est_HPS - 1.


fileName = 'cond_stats_conv_diff.csv'
ARR = np.zeros(shape=(len(Nvec),7))
ARR[:,0] = 1./Nvec
ARR[:,1] = nrm_diff_vec_stencil
ARR[:,2] = nrm_diff_vec_HPS
ARR[:,3] = lamdiff_vec_stencil
ARR[:,4] = lamdiff_vec_HPS
ARR[:,5] = condstat_vec_stencil
ARR[:,6] = condstat_vec_HPS
try:
    with open(fileName,'w') as f:
        f.write('H,nrm_diff_stencil,nrm_diff_HPS,lamdiff_stencil,lamdiff_HPS,condstat_stencil,condstat_HPS\n')
        np.savetxt(f,ARR,fmt='%.16e',delimiter=',')
except PermissionError as exc:
    print("Skipping CSV write: %s" % exc)
    
