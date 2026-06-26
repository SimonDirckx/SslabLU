"""
One backward-Euler step of advection-diffusion with constant D, on the periodic ring of
overlapping double-wide slabs.  Operator (coercive, no nullspace):
        (1/dt) U^{n+1} - D Δ U^{n+1} = f,   f = (1/dt) U^n - v·∇U^n  (J=0),
periodic in x, homogeneous Neumann on y=0,1.
Velocity = plane Poiseuille (exact Stokes, div-free, no-slip walls), flowing right->left.

Validation:
  (A) self/independent convergence vs a spectral reference (Fourier-x × cosine-y),
  (B) discrete mass conservation  ∫U1 = ∫U0  (advection is incompressible -> zero net flux).
"""
import numpy as np, torch
from scipy.integrate import trapezoid
torch.set_default_dtype(torch.double)
import modhelm_const as MH

# ---- Poiseuille channel flow, right -> left (exact Stokes, div-free, no-slip walls) ----
def vfield(V):
    return (lambda x,y: -V*4.0*y*(1.0-y)), (lambda x,y: 0.0*y)

# ---- independent spectral reference for (1/dt - D Δ)U1 = f, periodic-x, Neumann-y ----
class SpectralRef:
    def __init__(self,D,dt,Nx=160,Ny=257,Ncut=60):
        self.D=D; self.dt=dt; self.Nx=Nx; self.Ny=Ny; self.Ncut=Ncut
        self.xg=np.arange(Nx)/Nx; self.yg=np.linspace(0,1,Ny)
        self.kx=np.fft.fftfreq(Nx,d=1.0/Nx)
    def solve(self,f_np):
        X,Y=np.meshgrid(self.xg,self.yg,indexing='ij'); F=f_np(X,Y)
        Fx=np.fft.fft(F,axis=0)/self.Nx
        n=np.arange(self.Ncut+1); cosny=np.cos(np.pi*np.outer(self.yg,n))
        cn=np.ones(self.Ncut+1); cn[1:]=2.0
        Fhat=cn[None,:]*trapezoid(Fx[:,:,None]*cosny[None,:,:],self.yg,axis=1)
        sigma=1.0/self.dt+self.D*((2*np.pi*self.kx[:,None])**2+(np.pi*n[None,:])**2)
        self.U=Fhat/sigma; return self
    def eval(self,P):
        ex=np.exp(2j*np.pi*np.outer(P[:,0],self.kx))
        cs=np.cos(np.pi*np.outer(P[:,1],np.arange(self.Ncut+1)))
        return np.real(np.einsum('pm,mn,pn->p',ex,self.U,cs))
    def mass(self): return np.real(self.U[0,0])

def validate_spectral(D=0.1,dt=0.1):
    uex=lambda P: np.cos(2*np.pi*P[:,0])*np.cos(np.pi*P[:,1])
    f_np=lambda X,Y:(1.0/dt+5*np.pi**2*D)*np.cos(2*np.pi*X)*np.cos(np.pi*Y)
    S=SpectralRef(D,dt).solve(f_np); P=np.random.rand(2000,2)
    print("spectral reference vs manufactured cos*cos: rel-l2 = %.2e"
          %(np.linalg.norm(S.eval(P)-uex(P))/np.linalg.norm(uex(P))))

# ---- one backward-Euler step loads: U0 bump, Poiseuille advection ----
def make_loads(D,dt,V,a=100.0,cx=0.5,cy=0.5):
    def f_torch(P,shift=0.0):
        x=P[:,0]+shift; y=P[:,1]; U0=torch.exp(-a*((x-cx)**2+(y-cy)**2))
        return (U0*((1.0/dt)-8*a*V*(x-cx)*y*(1.0-y))).unsqueeze(-1)
    def f_np(X,Y):
        U0=np.exp(-a*((X-cx)**2+(Y-cy)**2))
        return U0*((1.0/dt)-8*a*V*(X-cx)*Y*(1.0-Y))
    return f_torch,f_np,(lambda X,Y: np.exp(-a*((X-cx)**2+(Y-cy)**2)))

def advection_step(D=0.1,dt=0.1,V=1.0,a=100.0):
    f_torch,f_np,_=make_loads(D,dt,V,a); ref=SpectralRef(D,dt).solve(f_np)
    print("\nOne backward-Euler step, Poiseuille advection (V=%.1f, right->left), bump (a=%.0f)."%(V,a))
    print("D=%.2f dt=%.3f.  Ring solution vs INDEPENDENT spectral reference:\n"%(D,dt))
    print("   interface rel-l2 vs spectral      GMRES iters")
    print("        nby=4   nby=6   nby=8       nby=4 nby=6 nby=8")
    for p in (6,8,10):
        erow="  p=%2d "%p; irow=""
        for nb in (4,6,8):
            R=MH.Ring(nb,p,D,dt); res=R.solve(f_torch)
            nodes=np.vstack([np.column_stack([np.full(R.m,k*R.w),R.yv]) for k in range(R.S)])
            erow+="  %.2e"%(np.linalg.norm(res['v']-ref.eval(nodes))/np.linalg.norm(ref.eval(nodes)))
            irow+="   %2d "%res['iters']
        print(erow+"      "+irow)

# ---- Clenshaw-Curtis quadrature on Chebyshev-Lobatto leaf nodes ----
def clencurt(n):
    if n==0: return np.array([0.0]),np.array([2.0])
    th=np.pi*np.arange(n+1)/n; x=np.cos(th); w=np.zeros(n+1); ii=np.arange(1,n); v=np.ones(n-1)
    if n%2==0:
        w[0]=w[n]=1.0/(n*n-1)
        for k in range(1,n//2): v-=2*np.cos(2*k*th[ii])/(4*k*k-1)
        v-=np.cos(n*th[ii])/(n*n-1)
    else:
        w[0]=w[n]=1.0/(n*n)
        for k in range(1,(n-1)//2+1): v-=2*np.cos(2*k*th[ii])/(4*k*k-1)
    w[ii]=2*v/n; o=np.argsort(x); return x[o],w[o]

def box_integral(coords,vals,p):
    ux=np.unique(np.round(coords[:,0],10)); uy=np.unique(np.round(coords[:,1],10))
    ix=np.searchsorted(ux,np.round(coords[:,0],10)); iy=np.searchsorted(uy,np.round(coords[:,1],10))
    U=np.zeros((p,p)); U[ix,iy]=vals; _,wc=clencurt(p-1)
    return (wc*(ux[-1]-ux[0])/2)@U@(wc*(uy[-1]-uy[0])/2)

def reconstruct_slab(R,uD,f_i):
    uC,uX=R.solve_mixed(uD,np.zeros(R.nX),ffunc=f_i)
    ntot=R.dom.XX_active.shape[0]; uu=torch.zeros(ntot,1)
    uu[R.dom.I_Ctot]=torch.tensor(uC).unsqueeze(-1); uu[R.dom.I_Xtot]=torch.tensor(uX).unsqueeze(-1)
    flat,_=R.hps.solve(MH.CPU,uu,ff_body_func=f_i); p=int(R.hps.p[0]); nb=int(R.hps.nboxes)
    return R.hps.grid_xx.numpy().reshape(nb,p*p,2), flat.numpy().reshape(nb,p*p)

def mass_conservation(D=0.1,dt=0.1,V=1.0,a=100.0,nby=8,p=10):
    f_torch,_,U0_np=make_loads(D,dt,V,a)
    R=MH.Ring(nby,p,D,dt); v=R.solve(f_torch)['v'].reshape(R.S,R.m)
    Rs=R.R; w=R.w; pp=int(Rs.hps.p[0]); mU1=0.0; mU0=0.0
    for i in range(R.S):
        uD=np.concatenate([v[i],v[(i+2)%R.S]]); fi=lambda P,sh=i*w: f_torch(P,shift=sh)
        coords,vals=reconstruct_slab(Rs,uD,fi)
        for b in range(coords.shape[0]):
            if coords[b,:,0].mean()<w-1e-9:                 # left column = strip i (no overlap)
                gx=coords[b].copy(); gx[:,0]+=i*w
                mU1+=box_integral(coords[b],vals[b],pp)
                mU0+=box_integral(coords[b],U0_np(gx[:,0],gx[:,1]),pp)
    print("\nMASS CONSERVATION (ring solver, non-overlapping strip partition), nby=%d p=%d:"%(nby,p))
    print("  ∫U1 (reconstructed)  = %.10f"%mU1)
    print("  ∫U0 (same quadrature)= %.10f"%mU0)
    print("  |∫U1 - ∫U0| = %.2e   (analytic π/a = %.10f)"%(abs(mU1-mU0),np.pi/a))

if __name__=="__main__":
    validate_spectral(); advection_step(); mass_conservation()