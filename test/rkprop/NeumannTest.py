"""
Overlapping double-wide slab solver for the homogeneous-Neumann Laplace problem,
chain of slabs, dense block-tridiagonal reduced (S-) system.

Square boxes: global mesh = nby x nby boxes (x-strips = nby), each double-wide slab
spans 2 adjacent x-strips x nby y-boxes; there are S-1 = nby-1 overlapping slabs.

Interior vertical lines x = k*w (w=1/nby), k=1..S-1, carry unknown traces u_k.
Slab i (i=0..S-2) maps its edge data to its interior interface:
        u_{i+1} = S^L_i u_i + S^R_i u_{i+2} + g_i
Reduced system (block tridiagonal, dense):  row k:  u_k - S^L_{k-1} u_{k-1} - S^R_{k-1} u_{k+1} = g_{k-1}
(end terms with u_0 / u_S dropped: x=0 and x=1 are physical Neumann edges, folded into g.)

Each slab solve is a MIXED (Zaremba) BVP assembled from the driver's merged blocks:
   [A_CC  A_CX_N][u_i ]   [ -A_CX_D uD          ]
   [A_XC_N A_XX_NN][u_xN] = [ gN - A_XX_ND uD    ]
where [A_XC|A_XX] is the spectral outward-flux (Neumann) operator at exterior nodes.
"""
import numpy as np, torch
torch.set_default_dtype(torch.double)
from hpsmultidomain.geom import BoxGeometry
from hpsmultidomain.domain_driver import Domain_Driver
import hpsmultidomain.pdo as pdo
from scipy.sparse.linalg import gmres 

TOL=1e-9
SRC=np.array([-0.1,-0.15])             # Laplace source OUTSIDE Omega
def uex(P):  return np.log(np.linalg.norm(P-SRC,axis=1))
def gradu(P):
    d=P-SRC; r2=(d**2).sum(1,keepdims=True); return d/r2
def fluxu(P,n): return (gradu(P)*n).sum(1)

class Slab:
    def __init__(self,x0,x1,nby,p):
        box=torch.tensor([[x0,0.0],[x1,1.0]])
        dom=Domain_Driver(BoxGeometry(box),pdo.PDO_2d(pdo.ones,pdo.ones),0.0,
                          np.array([(x1-x0)/(2*2),1.0/(2*nby)]),p=p,d=2,periodic_bc=False)
        dom.build(sparse_assembly='reduced_cpu',solver_type='MUMPS',verbose=False)
        self.x0,self.x1=x0,x1
        self.cc=dom.hps.xx_ext.numpy()[dom.hps.I_copy1.numpy()]
        self.xx=dom.XX_active[dom.I_Xtot].numpy()
        self.ACC=dom.A_CC.toarray(); self.ACX=dom.A_CX.toarray()
        self.AXC=dom.A_XC.toarray(); self.AXX=dom.A_XX.toarray()
        self.nC=self.cc.shape[0]; self.nX=self.xx.shape[0]
        n=np.zeros_like(self.xx)
        n[np.abs(self.xx[:,0]-x0)<TOL]=[-1,0]; n[np.abs(self.xx[:,0]-x1)<TOL]=[1,0]
        n[np.abs(self.xx[:,1]-0.)<TOL]=[0,-1]; n[np.abs(self.xx[:,1]-1.)<TOL]=[0,1]
        self.nrm=n
    def Xline(self,xv):  # X-space indices on vertical line x=xv, sorted by y
        idx=np.where(np.abs(self.xx[:,0]-xv)<TOL)[0]; return idx[np.argsort(self.xx[idx,1])]
    def Cline(self,xv):  # C-space indices on vertical interior line, sorted by y
        idx=np.where(np.abs(self.cc[:,0]-xv)<TOL)[0]; return idx[np.argsort(self.cc[idx,1])]
    def mixed(self, Didx, uD, gN_full):
        PN=np.setdiff1d(np.arange(self.nX),Didx)
        M=np.block([[self.ACC,       self.ACX[:,PN]],
                    [self.AXC[PN],    self.AXX[np.ix_(PN,PN)]]])
        rhs=np.concatenate([-self.ACX[:,Didx]@uD, gN_full[PN]-self.AXX[np.ix_(PN,Didx)]@uD])
        w=np.linalg.solve(M,rhs)
        return w[:self.nC]

def build_reduced(nby,p,verbose=False):
    S=nby; w=1.0/S; m=nby*(p-2)
    slabs=[Slab(i*w,(i+2)*w,nby,p) for i in range(S-1)]
    nlines=S-1                                   # interior lines k=1..S-1
    N=nlines*m
    K=np.zeros((N,N)); g=np.zeros(N)
    rowsum_ok=[]
    for i,sl in enumerate(slabs):               # slab i -> equation for line k=i+1
        xl,xr,xc=i*w,(i+2)*w,(i+1)*w
        leftA  = i>0                             # left edge artificial?
        rightA = i<S-2                           # right edge artificial?
        Lx=sl.Xline(xl); Rx=sl.Xline(xr); Cx=sl.Cline(xc)
        assert len(Cx)==m and len(Lx)==m and len(Rx)==m
        Didx=np.concatenate(([Lx] if leftA else [])+([Rx] if rightA else [])).astype(int) if (leftA or rightA) else np.array([],int)
        krow=i                                   # block-row index (line k=i+1 -> 0-based i)
        # S^L_i : data on left edge -> interior trace (right edge Dirichlet 0)
        if leftA:
            SL=np.zeros((m,m))
            for c in range(m):
                uD=np.zeros(len(Didx)); uD[:m if leftA else 0][c]=1.0  # left block is first m of Didx
                SL[:,c]=sl.mixed(Didx,uD,np.zeros(sl.nX))[Cx]
            K[krow*m:(krow+1)*m,(krow-1)*m:krow*m]=-SL
        # S^R_i : data on right edge -> interior trace (left edge Dirichlet 0)
        if rightA:
            SR=np.zeros((m,m)); off=m if leftA else 0
            for c in range(m):
                uD=np.zeros(len(Didx)); uD[off+c]=1.0
                SR[:,c]=sl.mixed(Didx,uD,np.zeros(sl.nX))[Cx]
            K[krow*m:(krow+1)*m,(krow+1)*m:(krow+2)*m]=-SR
        # diagonal I
        K[krow*m:(krow+1)*m,krow*m:(krow+1)*m]+=np.eye(m)
        # g_i : physical Neumann data -> interior trace (all artificial Dirichlet = 0)
        gN=fluxu(sl.xx,sl.nrm)
        if Didx.size: gN[Didx]=0.0
        g[krow*m:(krow+1)*m]=sl.mixed(Didx,np.zeros(len(Didx)),gN)[Cx]
        # constant->constant row-sum check (S^L+S^R applied to 1)
        const_trace=sl.mixed(Didx,np.ones(len(Didx)),np.zeros(sl.nX))[Cx] if Didx.size else None
        if const_trace is not None: rowsum_ok.append((const_trace.min(),const_trace.max()))
    # canonical exact interface values
    yidx=slabs[0].Cline(1*w)                    # y-coords of an interior line (same for all)
    yv=slabs[0].cc[yidx,1]
    xtrue=np.concatenate([uex(np.column_stack([np.full(m,(k+1)*w),yv])) for k in range(nlines)])
    # solve (constant nullspace -> lstsq), compare mod constant

    v,res  = gmres(K,g,maxiter=100,restart=100,rtol=1e-13,x0=np.zeros((K.shape[0],)))


    sv=np.linalg.svd(K,compute_uv=False)
    #v=np.linalg.solve(K,g)
    diff=v-xtrue; diff-=diff.mean()
    relerr=np.linalg.norm(diff)/np.linalg.norm(xtrue-xtrue.mean())
    kappa_eff=sv[0]/sv[-2]                       # exclude the single ~0 (constant) mode
    if verbose:
        print("S=%d strips, %d slabs, m=%d/line, reduced %dx%d"%(S,S-1,m,N,N))
        print("  const->const trace range:",["[%.4f,%.4f]"%(a,b) for a,b in rowsum_ok][:3],"...")
        print("  K sv: max %.3f  min %.2e  2nd-min %.3f  ->  kappa_eff=%.2f"%(sv[0],sv[-1],sv[-2],kappa_eff))
        print("  reduced residual ||Kv-g|| = %.2e"%np.linalg.norm(K@v-g))
        print("  interface rel-l2 (mod const) = %.3e"%relerr)
    return relerr,kappa_eff,N

def sweep():
    ps=[6,8,10,12,14,16,18,20]; nbys=[4,6,8,10,12]
    print("\nLaplace source at (%.2f,%.2f), dist to Omega ~ %.2f\n"%(SRC[0],SRC[1],np.hypot(max(0-SRC[0],0),max(0-SRC[1],0))))
    print("interface rel-l2 error (mod constant):")
    print("            "+"".join("   nby=%-2d  "%nb for nb in nbys))
    for p in ps:
        row="  p=%2d :  "%p
        kap=[]
        for nb in nbys:
            e,k,N=build_reduced(nb,p); row+="  %.2e"%e; kap.append(k)
        print(row)
    print("\neffective condition number kappa_eff = sigma_max/sigma_2nd-min:")
    print("            "+"".join("   nby=%-2d  "%nb for nb in nbys))
    for p in ps:
        row="  p=%2d :  "%p
        for nb in nbys:
            e,k,N=build_reduced(nb,p); row+="   %5.2f "%k
        print(row)

if __name__=="__main__":
    print("validate one config (nby=4,p=6):")
    sweep()
