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
kapp = 0.
#set up pde
#def c11(p):
#    f = np.zeros(shape=(p.shape[0],))
#    for i in range(p.shape[0]):
#        f[i] = 1.+(np.sin(5*np.pi*p[i,0])*np.sin(5*np.pi*p[i,0]))
#    return f
def c11(p):
    return np.ones(shape=(p.shape[0],))
def c22(p):
    return np.ones(shape=(p.shape[0],))
def c33(p):
    return np.ones(shape=(p.shape[0],))
def c(p):
    return kapp*kapp*np.ones(shape=(p.shape[0],))
Lapl=pdo.PDO2d(c11,c22,None,None,None,c)

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
kvec = [2,3,4,5]
for k in kvec:
    N=2**k-1
    H=Om.bnds[1][0]/(N+1)
    skel = skelTon.standardBoxSkeleton(Om,N)
    ny = 100
    nx = (int)(np.ceil(ny*H))
    nx = 2*nx+1
    print("nx = ",nx)
    ord=[6,6]
    a=Hmin/4.
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
    Stot = E-Linop@E
    [T,V] = splinalg.schur(Stot)
    e = np.diag(T)
    [U,s,V] = np.linalg.svd(Stot)
    emin = np.min(np.abs(e))
    smin = np.min(s)
    print("dep = ",np.linalg.norm(T-np.diag(e),ord=2))
    print("emin = ",emin)
    print("smin = ",smin)
    print("diff = ",smin-emin)
    w=np.zeros(shape=(100,))
    for i in range(len(w)):
         x=np.array(np.random.standard_normal(size=(Stot.shape[1])))
         x=x/np.linalg.norm(x)
         w[i] = x.T@Stot@x
    plt.figure(1)
    plt.scatter(np.real(w),np.imag(w))
    plt.show()
#E=np.identity(Linop.shape[0])
#Stot = Linop@E
#nyy = ny-2
#ndofs = nyy*N

#print("normality of S: ",np.linalg.norm(Stot-Stot.T)/np.linalg.norm(Stot))