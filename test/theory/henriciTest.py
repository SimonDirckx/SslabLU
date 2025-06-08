import numpy as np
import pdo.pdo as pdo
import solver.solver as solverWrap
from solver.pde_solver import AbstractPDESolver
import multiSlab as MS
import matAssembly.matAssembler as mA
import matplotlib.pyplot as plt
import geometry.standardGeometries as stdGeom
import geometry.skeleton as skelTon
import itertools
import scipy.linalg as splinalg
from scipy.sparse.linalg import gmres
import time
import pandas as pd

try:
	from petsc4py import PETSc
	petsc_imported = True
except:
	petsc_imported = False

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

# set-up global geometry
start = time.time()
bnds = [[0.,0.],[1.,1.]]
Om=stdGeom.Box(bnds)
kapp = 10.
#set up pde
def c11(p):
    f = np.zeros(shape=(p.shape[0],))
    for i in range(p.shape[0]):
        f[i] = 1.+(np.sin(5*np.pi*p[i,0])*np.sin(5*np.pi*p[i,0]))
    return f
#def c11(p):
#    return np.ones(shape=(p.shape[0],))
def c22(p):
    return np.ones(shape=(p.shape[0],))
def c33(p):
    return np.ones(shape=(p.shape[0],))
def c(p):
    return -kapp*kapp*np.ones(shape=(p.shape[0],))
Lapl=pdo.PDO2d(c11,c22)#,None,None,None,c)



########################
#   Set up skeleton
########################

#   Explanation:
#   this implementation is meant to allow for the general case of:
#           - unordered interfaces (e.g. hierarchical domain splitting)
#           - nonuniform interface discretizations (e.g. disc. galerkin)
#   (non-uniform here means that the discretization varies from one slab to the next)
#
#   in these cases, the interface connectivity ('skel.C')
#   and the global Idxs ('skel.globIdxs') are deferred to seperate methods.
#   The uniform, standardly ordered case is provided below

# set-up constants
# N             : number of interfaces in skeleton
# [ordx,ordy]   : stencil order in x-and-y direction
#with open("nGMRES_HPS103D.dat", "w") as f:
#    f.write("Nslabs H ITERS err\n")
kmax = 5
Nmax = 2**kmax-1
Hmin = Om.bnds[1][0]/(Nmax+1) 
kvec = [3,4,5]
iter = 0
henr = np.zeros(shape=(len(kvec),))
henrm1 = np.zeros(shape=(len(kvec),))
Hvec = np.zeros(shape=(len(kvec),))
Nvec = np.zeros(shape=(len(kvec),))
for k in kvec:
    N=2**k-1
    H=Om.bnds[1][0]/(N+1)
    Hvec[iter] = H
    Nvec[iter] = N
    skel = skelTon.standardBoxSkeleton(Om,N)
    ord=[10,10]
    a=Hmin/2.
    hx = H/(ord[0]-1)
    overlapping = True
    opts = solverWrap.solverOptions('hps',ord,a)

    # a skeleton is a collection of interfaces each carrying their own indices
    # referred to as 'global idxs'
    # uniformGlobalIdxs is a special method that assumes each will be discretized
    # in the same way
    # a slablist is a list of overlapping/non-overlapping slabs

    skel.setGlobalIdxs(skelTon.computeUniformGlobalIdxs(skel,opts))
    slabList = skelTon.buildSlabs(skel,Lapl,opts,overlapping)


    ########################
    #   Test assembly
    ########################


    #default dense assembler
    assembler = mA.denseMatAssembler()
    assemblerList = [assembler for slab in slabList]
    MultiSlab = MS.multiSlab(slabList,assemblerList)
    MultiSlab.constructMats()
    hy=1./(ord[1]-1)

    u = np.random.normal(0.,1./np.sqrt(N),size=(MultiSlab.N,))
    Linop       = MultiSlab.getLinOp()
    rhs = Linop@u
    gInfo = gmres_info()
    stol = (1e-5)*H*H
    if petsc_imported == True:
        uhat,info   = gmres(Linop,rhs,rtol=stol,callback=gInfo,maxiter=25000,restart=25000)
    else:
        uhat,info   = gmres(Linop,rhs,tol=stol,callback=gInfo,maxiter=25000,restart=25000)
    res = MultiSlab.apply(uhat)-rhs
    stop = time.time()
    print("=============SUMMARY==============")
    print("H            = ",'%10.3E'%H)
    print("(hx//hy)     = ("'%10.3E'%hx,"//",'%10.3E'%hy,")")
    print("L2 rel. res  = ", np.linalg.norm(res)/np.linalg.norm(rhs))
    print("L2 rel. err  = ", np.linalg.norm(u-uhat)/np.linalg.norm(u))
    print("GMRES iters  = ", gInfo.niter)
    print("elapsed time = ",stop-start)
    print("==================================")

    #with open("nGMRES_HPS103D.dat", "a") as f:
    #    s=[str(N)," ",str(H)," ",str(gInfo.niter)," ",str(np.linalg.norm(u-uhat)/np.linalg.norm(u)),"\n"]
    #    f.writelines(s)
    E=np.identity(Linop.shape[0])
    Stot = Linop@E
    [T,V]=splinalg.schur(E-Stot)
    e=np.diag(T)
    dF = np.linalg.norm(T-np.diag(e),ord=2)
    #SST = .5*(Stot+Stot.T)
    #[e,V]=np.linalg.eig(E-Stot)
    #print("smax//emax",np.max(s),"//",np.max(abs(e)))
    #print("smin//emin",np.min(s),"//",np.min(abs(e)))
    #print("ks//ke",np.max(np.abs(s))/np.min(np.abs(s)),"//",np.max(abs(e))/np.min(abs(e)))
    #print("ks/ke",(np.max(np.abs(s))/np.min(np.abs(s)))/(np.max(abs(e))/np.min(abs(e))))
    #print("dF = ",dF)
    henr[iter] = dF
    print("dF//emin",henr[iter],"//",np.min(np.abs(e)))
    #henrm1[iter] = (np.max(np.abs(s))/np.min(np.abs(s)))-(np.max(abs(e))/np.min(abs(e)))-1
    iter+=1

N2 = [x*x for x in Nvec]
H2=[1./(x*x) for x in Hvec]
H1=[1./x for x in Hvec]
f2 = henr[len(Nvec)-1]/H2[len(N2)-1]
f1 = henr[len(Nvec)-1]/H1[len(N2)-1]
H1=[1.5*x*f1 for x in H1]
H2=[1.5*x*f2 for x in H2]
plt.figure(1)
plt.loglog(Hvec,henr)
plt.loglog(Hvec,H2)
plt.loglog(Hvec,H1)
plt.legend(["henr","-2","-1"])
plt.show()
#E=np.identity(Linop.shape[0])
#Stot = Linop@E
#nyy = ny-2
#ndofs = nyy*N

#print("normality of S: ",np.linalg.norm(Stot-Stot.T)/np.linalg.norm(Stot))