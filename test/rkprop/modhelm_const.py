import numpy as np, torch, scipy.linalg as sla
torch.set_default_dtype(torch.double)
from solver.hpsmultidomain.hpsmultidomain.geom import BoxGeometry
from solver.hpsmultidomain.hpsmultidomain.domain_driver import Domain_Driver
import solver.hpsmultidomain.hpsmultidomain.pdo as pdo
TOL=1e-6; CPU=torch.device('cpu')

def make_pdo(D,dt):
    # operator  L u = (1/dt) u - D Δu   ->  assembled as -D∂xx -D∂yy + (1/dt) I  (coercive)
    return pdo.PDO_2d(pdo.const(D), pdo.const(D), c=pdo.const(1.0/dt))

class RefSlab:
    """Double-wide reference slab [0,2w]x[0,1], 2 x nby boxes, order p, operator (1/dt) - D Δ."""
    def __init__(self,w,nby,p,D,dt,pdo_op=None):
        self.w=w; self.D=D; self.dt=dt
        box=torch.tensor([[0.0,0.0],[2*w,1.0]])
        op=pdo_op if pdo_op is not None else make_pdo(D,dt)
        dom=Domain_Driver(BoxGeometry(box),op,0.0,
                          np.array([w/2,1.0/(2*nby)]),p=p,d=2,periodic_bc=False)
        dom.build(sparse_assembly='reduced_cpu',solver_type='MUMPS',verbose=False)
        self.dom=dom; self.hps=dom.hps
        self.cc=self.hps.xx_ext.numpy()[self.hps.I_copy1.numpy()]
        self.xx=dom.XX_active[dom.I_Xtot].numpy()
        self.ACC=dom.A_CC.toarray(); self.ACX=dom.A_CX.toarray()
        self.AXC=dom.A_XC.toarray(); self.AXX=dom.A_XX.toarray()
        self.nC=self.cc.shape[0]; self.nX=self.xx.shape[0]
        self.Ic1=self.hps.I_copy1.numpy(); self.Ic2=self.hps.I_copy2.numpy()
        self.Iext=dom.I_Xtot_in_unique.numpy()
        n=np.zeros_like(self.xx)
        n[np.abs(self.xx[:,1]-0.)<TOL]=[0,-1]; n[np.abs(self.xx[:,1]-1.)<TOL]=[0,1]
        n[np.abs(self.xx[:,0]-0.)<TOL]=[-1,0]; n[np.abs(self.xx[:,0]-2*w)<TOL]=[1,0]
        self.nrm=n
        Xl=lambda xv:(lambda i:i[np.argsort(self.xx[i,1])])(np.where(np.abs(self.xx[:,0]-xv)<TOL)[0])
        Cl=lambda xv:(lambda i:i[np.argsort(self.cc[i,1])])(np.where(np.abs(self.cc[:,0]-xv)<TOL)[0])
        self.Lx=Xl(0.0); self.Rx=Xl(2*w); self.Cx=Cl(w)
        self.N =np.where((np.abs(self.xx[:,1])<TOL)|(np.abs(self.xx[:,1]-1)<TOL))[0]
        self.Didx=np.concatenate([self.Lx,self.Rx]); self.m=len(self.Lx)
        self.M=np.block([[self.ACC,self.ACX[:,self.N]],
                         [self.AXC[self.N],self.AXX[np.ix_(self.N,self.N)]]])
        self.lu=sla.lu_factor(self.M)
        RHS=np.vstack([-self.ACX[:,self.Didx], -self.AXX[np.ix_(self.N,self.Didx)]])
        self.Cop=sla.lu_solve(self.lu,RHS)[:self.nC][self.Cx]      # [S^L | S^R], m x 2m
    def body_terms(self,ffunc):
        rb=self.hps.get_DtNs(CPU,mode='reduce_body',ff_body_func=ffunc).flatten(0,-2).numpy().real.ravel()
        return rb[self.Ic1]+rb[self.Ic2], rb[self.Iext]
    def solve_mixed(self,uD,gN_full,ffunc=None):
        bC=np.zeros(self.nC); bXN=np.zeros(len(self.N))
        if ffunc is not None:
            bC,bX=self.body_terms(ffunc); bXN=bX[self.N]
        top=-self.ACX[:,self.Didx]@uD + bC
        bot=gN_full[self.N]-self.AXX[np.ix_(self.N,self.Didx)]@uD + bXN
        w=sla.lu_solve(self.lu,np.concatenate([top,bot]))
        uX=np.zeros(self.nX); uX[self.Didx]=uD; uX[self.N]=w[self.nC:]
        return w[:self.nC],uX

def gate(D=0.1,dt=0.1,w=0.25,nby=4,p=6):
    """single-slab mixed BVP with reaction+body; u=x^3+y^3 (deg<=p exact)."""
    R=RefSlab(w,nby,p,D,dt)
    u=lambda P: P[:,0]**3+P[:,1]**3
    def fbody(P):  # = L u = (1/dt)u - D*Laplacian(u),  torch (N,1)
        val=(1.0/dt)*(P[:,0]**3+P[:,1]**3) - D*(6*P[:,0]+6*P[:,1])
        return val.unsqueeze(-1)
    gradu=np.stack([3*R.xx[:,0]**2,3*R.xx[:,1]**2],1); gN=(gradu*R.nrm).sum(1)
    uC,uX=R.solve_mixed(u(R.xx[R.Didx]),gN,ffunc=fbody)
    eC=np.linalg.norm(uC-u(R.cc))/np.linalg.norm(u(R.cc))
    eN=np.linalg.norm(uX[R.N]-u(R.xx[R.N]))/np.linalg.norm(u(R.xx[R.N]))
    return eC,eN

if __name__=="__main__":
    for (D,dt) in [(0.1,0.1),(0.1,0.01),(1.0,1.0)]:
        eC,eN=gate(D,dt)
        print("GATE D=%.2f dt=%.3f (alpha^2=1/(D dt)=%6.1f): interior %.2e  neumann-edge %.2e"
              %(D,dt,1.0/(D*dt),eC,eN))

from scipy.sparse.linalg import LinearOperator, gmres

class Ring:
    """Ring of S=nby double-wide slabs, operator (1/dt) - D Δ, periodic in x,
       homogeneous Neumann on y=0,1. Coercive -> nonsingular -> plain GMRES."""
    def __init__(self,nby,p,D,dt,pdo_op=None):
        self.S=nby; self.w=1.0/nby; self.D=D; self.dt=dt
        self.R=RefSlab(self.w,nby,p,D,dt,pdo_op=pdo_op); self.m=self.R.m
        self.yv=self.R.cc[self.R.Cx,1]
    # manufactured exact solution (periodic in x, homog Neumann at y=0,1)
    def uex(self,P):  return np.cos(2*np.pi*P[:,0])*np.cos(np.pi*P[:,1])
    def fman(self,P,shift=0.0):              # = L uex,  torch (N,1)
        x=P[:,0]+shift
        val=(1.0/self.dt + 5*np.pi**2*self.D)*torch.cos(2*np.pi*x)*torch.cos(np.pi*P[:,1])
        return val.unsqueeze(-1)
    def _matvec(self,v):
        S,m,Cop=self.S,self.m,self.R.Cop
        V=np.asarray(v,float).reshape(S,m); o=V.copy()
        for i in range(S):
            o[(i+1)%S]-=Cop@np.concatenate([V[i],V[(i+2)%S]])
        return o.ravel()
    def Kop(self): return LinearOperator((self.S*self.m,)*2,matvec=self._matvec,dtype=float)
    def rhs(self,ffunc):
        S,m=self.S,self.m; g=np.zeros((S,m))
        for i in range(S):
            fi=lambda P,sh=i*self.w: ffunc(P,shift=sh)
            g[(i+1)%S]=self.R.solve_mixed(np.zeros(2*m),np.zeros(self.R.nX),ffunc=fi)[0][self.R.Cx]
        return g.ravel()
    def solve(self,ffunc,rtol=1e-11):
        K=self.Kop(); g=self.rhs(ffunc)
        K1=np.linalg.norm(self._matvec(np.ones(self.S*self.m)))/np.sqrt(self.S*self.m)
        it=[0]
        v,info=gmres(K,g,rtol=rtol,atol=1e-13,restart=min(200,self.S*self.m),
                     maxiter=2000,callback=lambda pr: it.__setitem__(0,it[0]+1),callback_type='pr_norm')
        return dict(v=v,iters=it[0],info=info,K1=K1,N=self.S*self.m)
    def manufactured(self,rtol=1e-11):
        r=self.solve(self.fman,rtol)
        xt=np.concatenate([self.uex(np.column_stack([np.full(self.m,k*self.w),self.yv]))
                           for k in range(self.S)])
        r['relerr']=np.linalg.norm(r['v']-xt)/np.linalg.norm(xt)   # NO mod-constant: unique soln
        return r

def sweep(D=0.1,dt=0.1):
    ps=[6,8,10]; nbys=[4,6,8]
    print("\nConstant-D modified Helmholtz on the ring:  (1/dt) u - D Δu = f,")
    print("periodic in x, homog Neumann y=0,1.  D=%.2f, dt=%.3f (alpha^2=%.0f). Coercive: no nullspace.\n"%(D,dt,1/(D*dt)))
    print("manufactured interface rel-l2 error  (||K*1|| in [] confirms nonsingular):")
    print("           "+"".join("   nby=%-2d  "%nb for nb in nbys))
    for p in ps:
        row="  p=%2d :  "%p
        for nb in nbys:
            r=Ring(nb,p,D,dt).manufactured(); row+="  %.2e"%r['relerr']
        print(row)
    print("\nGMRES iterations to rtol=1e-11  (||K*1|| shown once, nby=8,p=8):")
    r=Ring(8,8,D,dt).manufactured(); print("   sample ||K*1|| = %.3f  (>>0 -> constant is NOT a nullvector)"%r['K1'])
    print("           "+"".join("   nby=%-2d  "%nb for nb in nbys))
    for p in ps:
        row="  p=%2d :  "%p
        for nb in nbys:
            r=Ring(nb,p,D,dt).manufactured(); row+="   %2d(%d)"%(r['iters'],r['N'])
        print(row)

if __name__!="__main__":
    pass