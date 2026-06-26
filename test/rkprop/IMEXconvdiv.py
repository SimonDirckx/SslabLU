"""
Multi-step backward-Euler IMEX loop, FULL variable diffusivity D(x,y), with Poiseuille
advection (V>0, right->left).  Conservative split: keep D(x,y)Δu IMPLICIT, lag the
first-order grad(D).grad u^n and advection v.grad u^n into the RHS.

  (1/dt)U^{n+1} - D(x,y) Δ U^{n+1} = f^n,
  f^n = (1/dt)U^n + (Dx(x,y) - v_x(y)) ∂x U^n + Dy(x,y) ∂y U^n        [conservative-IMEX]
        (1/dt)U^n -  v_x(y) ∂x U^n                                    [non-conservative]

D(x,y) = Dbar[1 + eps(sy(2y-1) + sx cos 2πx)]  : cross-channel (temperature) + streamwise (solubility).
D(x,y) breaks x-translation invariance -> each slab is factored separately (per-slab RefSlab list).
No closed-form / Fourier reference survives; verification = manufactured gate + mass telescoping
(+ self-convergence under p,nby externally).  ∇U^n from the NUMERICAL field via leaf Cheb matrices Ds.
U0 = tight Gaussian bump (value & gradient ~0 at straight walls -> homogeneous Neumann compatible).
"""
import numpy as np, torch, scipy.linalg as sla
from scipy.sparse.linalg import gmres, LinearOperator
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
torch.set_default_dtype(torch.double)
import modhelm_const as MH, advection_const as AD
import hpsmultidomain.pdo as pdo
CPU=MH.CPU

# ---------- D(x,y) operator and its analytic gradient ----------
def make_pdo_Dxy(Dbar,eps,sy,sx,dt,shift):
    Dfun=lambda xx: Dbar*(1.0+eps*(sy*(2.0*xx[:,1]-1.0)+sx*torch.cos(2*np.pi*(xx[:,0]+shift))))
    return pdo.PDO_2d(Dfun,Dfun,c=pdo.const(1.0/dt))
def D_grad(Dbar,eps,sy,sx):
    Dx=lambda x,y: Dbar*eps*sx*(-2*np.pi*np.sin(2*np.pi*x))
    Dy=lambda x,y: Dbar*eps*sy*2.0+0.0*x
    D =lambda x,y: Dbar*(1.0+eps*(sy*(2*y-1)+sx*np.cos(2*np.pi*x)))
    return D,Dx,Dy

# ---------- body-vector plumbing (numerical-field load) ----------
def body_terms_vec(R,fvec):
    rb=R.hps.get_DtNs(CPU,mode='reduce_body',ff_body_vec=fvec).flatten(0,-2).numpy().real.ravel()
    return rb[R.Ic1]+rb[R.Ic2], rb[R.Iext]
def solve_mixed_vec(R,uD,fvec):
    bC,bX=body_terms_vec(R,fvec)
    top=-R.ACX[:,R.Didx]@uD+bC
    bot=-R.AXX[np.ix_(R.N,R.Didx)]@uD+bX[R.N]                    # homogeneous Neumann
    w=sla.lu_solve(R.lu,np.concatenate([top,bot]))
    uX=np.zeros(R.nX); uX[R.Didx]=uD; uX[R.N]=w[R.nC:]
    return w[:R.nC],uX
def reconstruct_vec(R,uD,fvec):
    uC,uX=solve_mixed_vec(R,uD,fvec)
    ntot=R.dom.XX_active.shape[0]; uu=torch.zeros(ntot,1)
    uu[R.dom.I_Ctot]=torch.tensor(uC).unsqueeze(-1); uu[R.dom.I_Xtot]=torch.tensor(uX).unsqueeze(-1)
    flat,_=R.hps.solve(CPU,uu,ff_body_vec=fvec)
    return flat.numpy().reshape(int(R.hps.nboxes),int(R.hps.p[0])**2)

# ---------- manufactured gate: single slab, D(x,y), cubic (collocation-exact) ----------
def gate(Dbar=0.1,eps=0.3,sy=1.0,sx=0.5,dt=0.1,w=0.25,nby=4,p=8):
    op=make_pdo_Dxy(Dbar,eps,sy,sx,dt,shift=0.0); R=MH.RefSlab(w,nby,p,Dbar,dt,pdo_op=op)
    D,_,_=D_grad(Dbar,eps,sy,sx); u=lambda P:P[:,0]**3+P[:,1]**3
    def fbody(P): return ((1.0/dt)*(P[:,0]**3+P[:,1]**3)-D(P[:,0],P[:,1])*(6*P[:,0]+6*P[:,1])).unsqueeze(-1)
    gradu=np.stack([3*R.xx[:,0]**2,3*R.xx[:,1]**2],1); gN=(gradu*R.nrm).sum(1)
    uC,uX=R.solve_mixed(u(R.xx[R.Didx]),gN,ffunc=fbody)
    return (np.linalg.norm(uC-u(R.cc))/np.linalg.norm(u(R.cc)),
            np.linalg.norm(uX[R.N]-u(R.xx[R.N]))/np.linalg.norm(u(R.xx[R.N])))

# ---------- the time loop ----------
class TimeLoopXY:
    def __init__(self,Dbar,eps,sy,sx,dt,V,a,cx,cy,nby,p,conservative,slabs=None):
        self.dt=dt; self.cons=conservative; self.S=nby; self.w=1.0/nby
        if slabs is None:
            slabs=[MH.RefSlab(self.w,nby,p,Dbar,dt,pdo_op=make_pdo_Dxy(Dbar,eps,sy,sx,dt,shift=i*self.w))
                   for i in range(self.S)]
        self.slabs=slabs; R0=slabs[0]; self.m=R0.m
        self.D1=R0.hps.H.Ds[3].numpy(); self.D2=R0.hps.H.Ds[4].numpy()
        self.gx=R0.hps.grid_xx.numpy(); self.nb=self.gx.shape[0]; self.pp2=self.gx.shape[1]; self.pp=int(R0.hps.p[0])
        _,self.Dxf,_=D_grad(Dbar,eps,sy,sx); self.Dyc=Dbar*eps*sy*2.0
        self.vx=lambda y:-4*V*y*(1.0-y)
        self.leftcols=sorted([b for b in range(self.nb) if self.gx[b,:,0].mean()<self.w-1e-9],
                             key=lambda b:self.gx[b,:,1].mean())
        self.U=[np.exp(-a*((self.gx[:,:,0]+i*self.w-cx)**2+(self.gx[:,:,1]-cy)**2)) for i in range(self.S)]
    def body_fvec(self,i,Ui):
        xg=self.gx[:,:,0]+i*self.w; y=self.gx[:,:,1]
        ux=np.einsum('ij,bj->bi',self.D1,Ui); uy=np.einsum('ij,bj->bi',self.D2,Ui)
        if self.cons: f=(1.0/self.dt)*Ui+(self.Dxf(xg,y)-self.vx(y))*ux+self.Dyc*uy
        else:         f=(1.0/self.dt)*Ui-self.vx(y)*ux
        return torch.tensor(f.reshape(self.nb*self.pp2,1))
    def _matvec(self,v):
        Vv=np.asarray(v,float).reshape(self.S,self.m); o=Vv.copy()
        for i in range(self.S):
            o[(i+1)%self.S]-=self.slabs[i].Cop@np.concatenate([Vv[i],Vv[(i+2)%self.S]])
        return o.ravel()
    def mass(self,Ul):
        return sum(AD.box_integral(self.gx[b],Ul[i][b],self.pp) for i in range(self.S) for b in self.leftcols)
    def step(self):
        fvecs=[self.body_fvec(i,self.U[i]) for i in range(self.S)]
        g=np.zeros((self.S,self.m))
        for i in range(self.S):
            uC,_=solve_mixed_vec(self.slabs[i],np.zeros(2*self.m),fvecs[i]); g[(i+1)%self.S]=uC[self.slabs[i].Cx]
        K=LinearOperator((self.S*self.m,)*2,matvec=self._matvec)
        cnt=[0]
        v,info=gmres(K,g.ravel(),rtol=1e-10,atol=1e-12,restart=min(200,self.S*self.m),maxiter=4000,callback=lambda *a: cnt.__setitem__(0,cnt[0]+1))
        self.iters=cnt[0]
        v=v.reshape(self.S,self.m)
        self.U=[reconstruct_vec(self.slabs[i],np.concatenate([v[i],v[(i+2)%self.S]]),fvecs[i]) for i in range(self.S)]
        return self.mass(self.U),max(np.abs(u).max() for u in self.U),info

# ---------- smooth per-leaf-interpolated field plot ----------
def bary_mat(nodes,targets):
    n=len(nodes); wt=(-1.0)**np.arange(n); wt[0]*=0.5; wt[-1]*=0.5; M=np.zeros((len(targets),n))
    for k,t in enumerate(targets):
        d=t-nodes
        if np.any(np.abs(d)<1e-13): M[k,np.argmin(np.abs(d))]=1.0
        else: wv=wt/d; M[k]=wv/wv.sum()
    return M
def plot_field(loop,Ul,fname,title,nx=18,ny=18):
    S=loop.S; w=loop.w; gx=loop.gx; pp=loop.pp; lc=loop.leftcols; nrow=len(lc)
    img=np.zeros((S*nx,nrow*ny)); Xc=np.zeros(S*nx); Yc=np.zeros(nrow*ny)
    for i in range(S):
        for jr,b in enumerate(lc):
            uxn=np.unique(np.round(gx[b,:,0],10)); uyn=np.unique(np.round(gx[b,:,1],10))
            ix=np.searchsorted(uxn,np.round(gx[b,:,0],10)); iy=np.searchsorted(uyn,np.round(gx[b,:,1],10))
            U2=np.zeros((pp,pp)); U2[ix,iy]=Ul[i][b]
            xf=np.linspace(uxn[0],uxn[-1],nx,endpoint=False); yf=np.linspace(uyn[0],uyn[-1],ny,endpoint=False)
            img[i*nx:(i+1)*nx,jr*ny:(jr+1)*ny]=bary_mat(uxn,xf)@U2@bary_mat(uyn,yf).T
            Xc[i*nx:(i+1)*nx]=xf+i*w; Yc[jr*ny:(jr+1)*ny]=yf
    plt.figure(figsize=(6.2,3.4)); plt.pcolormesh(Xc,Yc,img.T,shading='auto',cmap='viridis')
    plt.colorbar(label='u'); plt.xlabel('x'); plt.ylabel('y'); plt.title(title)
    plt.tight_layout(); plt.savefig(fname,dpi=130); plt.close(); return fname

# ---------- driver ----------
def demo(Dbar=0.1,eps=0.3,sy=1.0,sx=0.5,V=0.5,a=120.0,cx=0.5,cy=0.5,nby=6,p=8,N=10,CFL=0.5):
    eC,eN=gate(Dbar,eps,sy,sx); print("GATE D(x,y) manufactured: interior %.2e  neumann %.2e"%(eC,eN))
    # build slabs once (operator identical for cons/noncons), pick CFL-safe dt
    probe=MH.RefSlab(1.0/nby,nby,p,Dbar,0.1,pdo_op=make_pdo_Dxy(Dbar,eps,sy,sx,0.1,0.0))
    uxn=np.unique(np.round(probe.hps.grid_xx.numpy()[0,:,0],10)); dxmin=np.diff(uxn).min()
    dt=CFL*dxmin/max(V,1e-12); dt=min(dt,0.02)
    print("dx_min=%.4f  V=%.2f  -> CFL-safe dt=%.4f  (T=%.3f over %d steps)"%(dxmin,V,dt,N*dt,N))
    slabs=[MH.RefSlab(1.0/nby,nby,p,Dbar,dt,pdo_op=make_pdo_Dxy(Dbar,eps,sy,sx,dt,shift=i/nby))
           for i in range(nby)]
    cons=TimeLoopXY(Dbar,eps,sy,sx,dt,V,a,cx,cy,nby,p,True ,slabs=slabs)
    nonc=TimeLoopXY(Dbar,eps,sy,sx,dt,V,a,cx,cy,nby,p,False,slabs=slabs)
    M0=cons.mass(cons.U); Mc=[M0]; Mn=[M0]; files=[plot_field(cons,cons.U,"u_t0.png","u(x,y), step 0")]
    maxc=0.0
    for n in range(1,N+1):
        mc,xc,ic=cons.step(); mn,xn,ino=nonc.step(); Mc.append(mc); Mn.append(mn); maxc=max(maxc,xc)
        if n in (5,N): files.append(plot_field(cons,cons.U,"u_t%d.png"%n,"u(x,y), step %d (t=%.3f)"%(n,n*dt)))
    print("\n max|u| over run = %.3f -> %s"%(maxc,"stable" if maxc<5 else "CHECK STABILITY"))
    print("\n step   non-conservative |M-M0|    conservative-IMEX |M-M0|")
    for n in range(N+1):
        print("  %3d        %.3e                %.3e"%(n,abs(Mn[n]-M0),abs(Mc[n]-M0)))
    print("\n M0=%.8f"%M0)
    print(" non-conservative final drift = %.3e   conservative-IMEX final drift = %.3e"%(abs(Mn[-1]-M0),abs(Mc[-1]-M0)))
    return files

if __name__=="__main__":
    demo()

# ---------- pointwise evaluation (per-leaf barycentric) for cross-p comparison ----------
def _box_meta(loop):
    gx=loop.gx; meta=[]
    for b in loop.leftcols:
        uxn=np.unique(np.round(gx[b,:,0],10)); uyn=np.unique(np.round(gx[b,:,1],10))
        ixn=np.searchsorted(uxn,np.round(gx[b,:,0],10)); iyn=np.searchsorted(uyn,np.round(gx[b,:,1],10))
        meta.append((b,uxn,uyn,ixn,iyn))
    return meta
def evaluate(loop,Ul,pts):
    meta=_box_meta(loop); S=loop.S; w=loop.w; pp=loop.pp
    yr=[(m[2][0],m[2][-1]) for m in meta]; vals=np.zeros(len(pts))
    for k in range(len(pts)):
        x,y=pts[k]; i=min(int(x/w),S-1); xr=x-i*w; r=0
        for ri,(y0,y1) in enumerate(yr):
            if y0-1e-9<=y<=y1+1e-9: r=ri; break
        b,uxn,uyn,ixn,iyn=meta[r]; U2=np.zeros((pp,pp)); U2[ixn,iyn]=Ul[i][b]
        vals[k]=(bary_mat(uxn,np.array([xr]))@U2@bary_mat(uyn,np.array([y])).T)[0,0]
    return vals

# ---------- self-convergence over p at fixed nby, fixed dt, conservative ----------
def selfconv(Dbar=0.1,eps=0.3,sy=1.0,sx=0.5,V=0.5,a=120.0,cx=0.5,cy=0.5,nby=8,ps=(6,8,10),pref=16,N=20,CFL=0.5,form='imex'):
    print("D(x,y) = %.2f*(1 + %.1f*(%.1f*(2y-1) + %.1f*cos(2*pi*x)))  in [%.3f, %.3f]"
          %(Dbar,eps,sy,sx,Dbar*(1-eps*(sy+sx)),Dbar*(1+eps*(sy+sx))))
    probe=MH.RefSlab(1.0/nby,nby,pref,Dbar,0.1,pdo_op=make_pdo_Dxy(Dbar,eps,sy,sx,0.1,0.0))
    uxn=np.unique(np.round(probe.hps.grid_xx.numpy()[0,:,0],10)); dxmin=np.diff(uxn).min()
    dt=min(CFL*dxmin/max(V,1e-12),0.02)
    print("fixed dt = %.5f (CFL-safe at p=%d, dx_min=%.5f),  %d steps -> T=%.4f,  nby=%d"%(dt,pref,dxmin,N,N*dt,nby))
    Ng=50; g=(np.arange(Ng)+0.5)/Ng; X,Y=np.meshgrid(g,g,indexing='ij'); pts=np.column_stack([X.ravel(),Y.ravel()])
    runs={}
    for p in list(ps)+[pref]:
        mkop=make_pdo_div_Dxy if form=='div' else make_pdo_Dxy
        slabs=[MH.RefSlab(1.0/nby,nby,p,Dbar,dt,pdo_op=mkop(Dbar,eps,sy,sx,dt,shift=i/nby)) for i in range(nby)]
        L=TimeLoopXY(Dbar,eps,sy,sx,dt,V,a,cx,cy,nby,p,(form!='div'),slabs=slabs)
        mx=0.0
        for _ in range(N): _,xi,_=L.step(); mx=max(mx,xi)
        runs[p]=evaluate(L,L.U,pts); print("  p=%2d done (max|u| over run %.3f)"%(p,mx))
    uref=runs[pref]
    print("\n self-convergence [form=%s] at t=%.4f (%d steps), nby=%d, vs p=%d reference:"%(form,N*dt,N,nby,pref))
    for p in ps:
        print("   p=%2d :  rel-l2 = %.3e   max-abs = %.3e"
              %(p,np.linalg.norm(runs[p]-uref)/np.linalg.norm(uref),np.max(np.abs(runs[p]-uref))))


# ---------- temporal (dt) convergence at fixed grid; same nodes -> compare directly ----------
def field_vec(loop,Ul):
    return np.concatenate([Ul[i][b] for i in range(loop.S) for b in loop.leftcols])
def dt_convergence(Dbar=0.1,eps=0.3,sy=1.0,sx=0.5,V=0.5,a=120.0,cx=0.5,cy=0.5,
                   nby=8,p=10,T=0.01,dts=(0.0025,0.00125,0.000625),dt_ref=0.00015625):
    print("D(x,y) = %.2f*(1 + %.1f*(%.1f*(2y-1) + %.1f*cos(2*pi*x)))  in [%.3f, %.3f]"
          %(Dbar,eps,sy,sx,Dbar*(1-eps*(sy+sx)),Dbar*(1+eps*(sy+sx))))
    probe=MH.RefSlab(1.0/nby,nby,p,Dbar,0.1,pdo_op=make_pdo_Dxy(Dbar,eps,sy,sx,0.1,0.0))
    uxn=np.unique(np.round(probe.hps.grid_xx.numpy()[0,:,0],10)); dxmin=np.diff(uxn).min()
    print("dt-convergence: fixed p=%d nby=%d, T=%.4f, conservative.  CFL dt_max~%.4f (V=%.2f, dx_min=%.5f)"
          %(p,nby,T,0.5*dxmin/V,V,dxmin))
    def run(dt):
        N=int(round(T/dt))
        mkop=make_pdo_div_Dxy if form=='div' else make_pdo_Dxy
        slabs=[MH.RefSlab(1.0/nby,nby,p,Dbar,dt,pdo_op=mkop(Dbar,eps,sy,sx,dt,shift=i/nby)) for i in range(nby)]
        L=TimeLoopXY(Dbar,eps,sy,sx,dt,V,a,cx,cy,nby,p,(form!='div'),slabs=slabs); mx=0.0
        for _ in range(N): _,xi,_=L.step(); mx=max(mx,xi)
        return field_vec(L,L.U),mx,N
    fref,mxr,Nr=run(dt_ref); print("  reference dt=%.6f (N=%d steps, max|u|=%.3f)"%(dt_ref,Nr,mxr))
    print("\n     dt        N     rel-l2 vs ref     ratio   (O(dt) -> ~2.0)")
    prev=None
    for dt in dts:
        f,mx,N=run(dt); err=np.linalg.norm(f-fref)/np.linalg.norm(fref)
        r=("%5.2f"%(prev/err)) if prev else "  -  "
        print("  %.6f   %3d     %.4e        %s"%(dt,N,err,r)); prev=err


# ---------- Form 2: genuine divergence operator (1/dt) - div(D grad) ----------
def make_pdo_div_Dxy(Dbar,eps,sy,sx,dt,shift):
    tp=2*np.pi
    Dfun =lambda xx: Dbar*(1.0+eps*(sy*(2.0*xx[:,1]-1.0)+sx*torch.cos(tp*(xx[:,0]+shift))))
    c1fun=lambda xx: tp*Dbar*eps*sx*torch.sin(tp*(xx[:,0]+shift))      # = -Dx
    c2fun=lambda xx: -(2.0*Dbar*eps*sy)+0.0*xx[:,0]                    # = -Dy (const)
    return pdo.PDO_2d(Dfun,Dfun,c1=c1fun,c2=c2fun,c=pdo.const(1.0/dt))

def gate_div(Dbar=0.1,eps=0.3,sy=1.0,sx=0.5,dt=0.1,w=0.25,nby=4,p=8):
    op=make_pdo_div_Dxy(Dbar,eps,sy,sx,dt,0.0); R=MH.RefSlab(w,nby,p,Dbar,dt,pdo_op=op)
    D,Dx,Dy=D_grad(Dbar,eps,sy,sx); u=lambda P:P[:,0]**3+P[:,1]**3
    def fbody(P):  # (1/dt)u - D Δu - (Dx ux + Dy uy) ;  Δu=6x+6y, ux=3x^2, uy=3y^2
        x=P[:,0]; y=P[:,1]
        return ((1.0/dt)*(x**3+y**3)-D(x,y)*(6*x+6*y)-(Dx(x,y)*3*x**2+Dy(x,y)*3*y**2)).unsqueeze(-1)
    gradu=np.stack([3*R.xx[:,0]**2,3*R.xx[:,1]**2],1); gN=(gradu*R.nrm).sum(1)
    uC,uX=R.solve_mixed(u(R.xx[R.Didx]),gN,ffunc=fbody)
    return (np.linalg.norm(uC-u(R.cc))/np.linalg.norm(u(R.cc)),
            np.linalg.norm(uX[R.N]-u(R.xx[R.N]))/np.linalg.norm(u(R.xx[R.N])))

def compare_forms(Dbar=0.1,eps=0.3,sy=1.0,sx=0.5,V=0.5,a=120.0,cx=0.5,cy=0.5,nby=6,p=8,N=10,CFL=0.5):
    print("D(x,y)=%.2f(1+%.1f(%.1f(2y-1)+%.1f cos2pi x)) in [%.3f,%.3f]"%(Dbar,eps,sy,sx,Dbar*(1-eps*(sy+sx)),Dbar*(1+eps*(sy+sx))))
    eC,eN=gate_div(Dbar,eps,sy,sx); print("GATE Form2 div(D grad) manufactured: interior %.2e neumann %.2e"%(eC,eN))
    probe=MH.RefSlab(1.0/nby,nby,p,Dbar,0.1,pdo_op=make_pdo_Dxy(Dbar,eps,sy,sx,0.1,0.0))
    uxn=np.unique(np.round(probe.hps.grid_xx.numpy()[0,:,0],10)); dt=min(CFL*np.diff(uxn).min()/V,0.02)
    print("dt=%.5f  N=%d  T=%.4f  nby=%d p=%d\n"%(dt,N,N*dt,nby,p))
    lap=[MH.RefSlab(1.0/nby,nby,p,Dbar,dt,pdo_op=make_pdo_Dxy(Dbar,eps,sy,sx,dt,shift=i/nby)) for i in range(nby)]
    div=[MH.RefSlab(1.0/nby,nby,p,Dbar,dt,pdo_op=make_pdo_div_Dxy(Dbar,eps,sy,sx,dt,shift=i/nby)) for i in range(nby)]
    runs={"non-conservative (D Δ, drop gradD)":TimeLoopXY(Dbar,eps,sy,sx,dt,V,a,cx,cy,nby,p,False,slabs=lap),
          "Form1 IMEX (D Δ impl, gradD lagged)":TimeLoopXY(Dbar,eps,sy,sx,dt,V,a,cx,cy,nby,p,True ,slabs=lap),
          "Form2 divergence (div(D grad) impl)":TimeLoopXY(Dbar,eps,sy,sx,dt,V,a,cx,cy,nby,p,False,slabs=div)}
    M0=runs["Form1 IMEX (D Δ impl, gradD lagged)"].mass; M0=list(runs.values())[0].mass(list(runs.values())[0].U)
    hist={k:[v.mass(v.U)] for k,v in runs.items()}; iters={k:[] for k in runs}; maxu=0.0
    for n in range(N):
        for k,v in runs.items(): m,xu,_=v.step(); hist[k].append(m); iters[k].append(v.iters); maxu=max(maxu,xu)
    k0,k1,k2=list(runs)
    print(" max|u| over run = %.3f -> %s\n"%(maxu,"stable" if maxu<5 else "CHECK STABILITY"))
    print(" step   non-conservative |M-M0|   Form1 IMEX |M-M0|   Form2 divergence |M-M0|")
    for n in range(N+1):
        print("  %3d        %.3e               %.3e            %.3e"
              %(n,abs(hist[k0][n]-M0),abs(hist[k1][n]-M0),abs(hist[k2][n]-M0)))
    print("\n final |M-M0|:")
    for k in runs: print("   %-38s %.3e"%(k,abs(hist[k][-1]-M0)))
    print(" mean GMRES iters/step:  non-cons=%.1f  Form1=%.1f  Form2=%.1f"
          %(np.mean(iters[k0]),np.mean(iters[k1]),np.mean(iters[k2])))


# ---------- unified test: conservation + self-convergence for all three formulations ----------
def full_comparison(Dbar=0.1,eps=0.3,sy=1.0,sx=0.5,V=0.5,a=120.0,cx=0.5,cy=0.5,
                    nbyA=6,pA=8,NA=10, nbyB=8,ps=(6,8,10),pref=16,NB=10, CFL=0.5):
    print("D(x,y)=%.2f(1+%.1f(%.1f(2y-1)+%.1f cos2pi x))  in [%.3f,%.3f]"
          %(Dbar,eps,sy,sx,Dbar*(1-eps*(sy+sx)),Dbar*(1+eps*(sy+sx))))
    eC,eN=gate_div(Dbar,eps,sy,sx); print("gate div(D grad): interior %.1e neumann %.1e"%(eC,eN))
    forms=[("non-conservative","lap",False),("Form1 IMEX      ","lap",True),("Form2 divergence","div",False)]
    def build(nby,p,dt,kind):
        mk=make_pdo_div_Dxy if kind=="div" else make_pdo_Dxy
        return [MH.RefSlab(1.0/nby,nby,p,Dbar,dt,pdo_op=mk(Dbar,eps,sy,sx,dt,shift=i/nby)) for i in range(nby)]
    def cfl_dt(nby,p):
        pr=MH.RefSlab(1.0/nby,nby,p,Dbar,0.1,pdo_op=make_pdo_Dxy(Dbar,eps,sy,sx,0.1,0.0))
        return min(CFL*np.diff(np.unique(np.round(pr.hps.grid_xx.numpy()[0,:,0],10))).min()/V,0.02)

    # ---- Part A: conservation (per-step, three formulations) ----
    dtA=cfl_dt(nbyA,pA); slA={"lap":build(nbyA,pA,dtA,"lap"),"div":build(nbyA,pA,dtA,"div")}
    LA={nm:TimeLoopXY(Dbar,eps,sy,sx,dtA,V,a,cx,cy,nbyA,pA,c,slabs=slA[k]) for nm,k,c in forms}
    nm=list(LA); M0=LA[nm[0]].mass(LA[nm[0]].U); H={n:[LA[n].mass(LA[n].U)] for n in LA}; mx=0.0
    for _ in range(NA):
        for n in LA: m,xu,_=LA[n].step(); H[n].append(m); mx=max(mx,xu)
    print("\n=== CONSERVATION  (nby=%d p=%d dt=%.4f, %d steps, max|u|=%.3f %s) ==="
          %(nbyA,pA,dtA,NA,mx,"stable" if mx<5 else "UNSTABLE"))
    print(" step   non-conservative      Form1 IMEX            Form2 divergence")
    for n in range(NA+1):
        print("  %3d      %.3e            %.3e            %.3e"
              %(n,abs(H[nm[0]][n]-M0),abs(H[nm[1]][n]-M0),abs(H[nm[2]][n]-M0)))

    # ---- Part B: spatial self-convergence (p-sweep vs p=pref, three formulations) ----
    dtB=cfl_dt(nbyB,pref)
    Ng=50; gg=(np.arange(Ng)+0.5)/Ng; XX,YY=np.meshgrid(gg,gg,indexing='ij'); pts=np.column_stack([XX.ravel(),YY.ravel()])
    fld={nm0:{} for nm0,_,_ in forms}
    for p in list(ps)+[pref]:
        sl={"lap":build(nbyB,p,dtB,"lap"),"div":build(nbyB,p,dtB,"div")}
        for nm0,k,c in forms:
            L=TimeLoopXY(Dbar,eps,sy,sx,dtB,V,a,cx,cy,nbyB,p,c,slabs=sl[k])
            for _ in range(NB): L.step()
            fld[nm0][p]=evaluate(L,L.U,pts)
        print("  self-conv p=%2d done"%p)
    print("\n=== SELF-CONVERGENCE  (nby=%d, dt=%.5f fixed, %d steps, rel-l2 vs p=%d) ==="%(nbyB,dtB,NB,pref))
    print("   p     non-conservative     Form1 IMEX          Form2 divergence")
    for p in ps:
        r="  %2d   "%p
        for nm0,_,_ in forms:
            r+="    %.3e    "%(np.linalg.norm(fld[nm0][p]-fld[nm0][pref])/np.linalg.norm(fld[nm0][pref]))
        print(r)

if __name__=="__main__":
    full_comparison()