import numpy as np
import jax.numpy as jnp
import torch
import scipy.special as special
import time
import SOMS3D as SOMS
import matplotlib.pyplot as plt

from solver.hpsmultidomain.hpsmultidomain import domain_driver as hpsalt
import solver.hpsmultidomain.hpsmultidomain.geom as hpsaltGeom
from solver.hpsmultidomain.hpsmultidomain import pdo as pdoalt

#######################################
#             TESTS OF SOMS
#######################################


def my_condest(A):
    #estim norm A:
    v = np.random.standard_normal(size=(A.shape[1],))
    v = v/np.linalg.norm(v)
    Niter = 50#(int)(np.sqrt(A.shape[0]))
    for i in range(Niter):
        w = A.T@v
        w = w/np.linalg.norm(w)
        v = A@w
        v = v/np.linalg.norm(v)
    nrmA = np.abs(v.T@(A@w))
    #estim norm for Ai
    v = np.random.standard_normal(size=(A.shape[1],))
    v = v/np.linalg.norm(v)
    Niter = 50#(int)(np.sqrt(A.shape[0]))
    for i in range(Niter):
        w = np.linalg.solve(A.T,v)
        w = w/np.linalg.norm(w)
        v = np.linalg.solve(A,w)
        v = v/np.linalg.norm(v)
    
    nrmAi = np.abs(v.T@(np.linalg.solve(A,w)))
    return nrmA*nrmAi

def bc_laplace(p):
    r = np.sqrt(((p[:,0]+.1)**2)+((p[:,1]+.1)**2)+((p[:,2]+.1)**2))
    
    return 1./(4*np.pi*r)

def bc_helmholtz(p,kh):
    r = np.sqrt(((p[:,0]+.1)**2)+((p[:,1]+.1)**2))
    
    return np.sin(kh*r)/(4*np.pi*r)


pvec = np.array([4,6,8],dtype=np.int64)
condvecS_L = np.zeros(shape=(len(pvec),))
condvecT_L = np.zeros(shape=(len(pvec),))
condvecS_H = np.zeros(shape=(len(pvec),))
condvecT_H = np.zeros(shape=(len(pvec),))

errvecS_L = np.zeros(shape=(len(pvec),))
errvecT_L = np.zeros(shape=(len(pvec),))
errvecS_H = np.zeros(shape=(len(pvec),))
errvecT_H = np.zeros(shape=(len(pvec),))

Nvec = np.zeros(shape=(len(pvec),))

for indp in range(len(pvec)):
    px = pvec[indp]
    py = pvec[indp]
    pz = pvec[indp]
    nbx = 4
    nby = 1
    nbz = 1
    kh = 0.

    ####################################################
    # compare laplace problem: accuracy and conditioning

    print("########### LAPLACE PROBLEM #############")

    print("#DOFS = ",px*py*pz*nbx*nby*nbz)
    Nvec[indp] = px*py*pz*nbx*nby*nbz


    tic = time.time()
    Sii,Sib,XX,Ii,Ib = SOMS.SOMS_solver(px,py,pz,nbx,nby,nbz,1.,1.,1.,0.)
    toc = time.time()-tic
    print("elapsed time S = ",toc)
    print("Sii shape = ",Sii.shape)
    tic = time.time()
    condS = np.linalg.cond(Sii)
    toc_cond = time.time()-tic
    tic = time.time()
    condestS = my_condest(Sii)
    toc_cond_est = time.time()-tic
    condvecS_L[indp] = condS
    print("condS,condestS = ",condS,',',condestS)
    print("t,t_est = ",toc_cond,',',toc_cond_est)
    u = bc_laplace(XX)
    uhat = -np.linalg.solve(Sii,Sib@u[Ib])
    err = np.linalg.norm(u[Ii]-uhat)/np.linalg.norm(u[Ii])
    print("S sol err. = ",err)
    errvecS_L[indp] = err

    bnds = np.array([[0,0,0],[1,1,1]])
    geom = hpsaltGeom.BoxGeometry(bnds)
    xl = geom.bounds[0,0]
    xr = geom.bounds[1,0]

    def c11(p):
        return torch.ones_like(p[:,0])
    def c22(p):
        return torch.ones_like(p[:,0])
    def c33(p):
        return torch.ones_like(p[:,0])

    diff_op = pdoalt.PDO_3d(c11=c11,c22=c22,c33=c33)

    ax = .5/nbx
    ay = .5/nby
    az = .5/nbz
    tic = time.time()
    solver = hpsalt.Domain_Driver(geom, diff_op, 0, np.array([ax,ay,az]), [px+1,px+1,px+1], 3)
    solver.build("reduced_cpu", "MUMPS",verbose=False)
    toc = time.time()-tic
    Aii = np.array(solver.Aii.todense())
    Aib = np.array(solver.Aix.todense())
    print("elapsed time T = ",toc)
    print("Tii shape = ",Sii.shape)
    condT = np.linalg.cond(Aii)
    condestT = my_condest(Aii)
    condvecT_L[indp] = condT
    print("condT,condestT = ",condT,',',condestT)
    XX = solver.XX
    Ii = solver._Ji
    Ib = solver._Jx
    u = bc_laplace(XX).numpy()
    uhat = -np.linalg.solve(Aii,Aib@u[Ib])
    err = np.linalg.norm(u[Ii]-uhat)/np.linalg.norm(u[Ii])
    print("T sol err. = ",err)
    errvecT_L[indp] = err

    '''

    ####################################################
    # compare Helmholtz problem: accuracy and conditioning

    print("########### HELMHOLTZ PROBLEM #############")
    print("#DOFS = ",px*py*nbx*nby)


    tic = time.time()
    Sii,Sib,XX,Ii,Ib = SOMS.SOMS_solver(px,py,nbx,nby,kh,2,1)
    toc = time.time()-tic
    print("elapsed time S = ",toc)
    print("Sii shape = ",Sii.shape)
    condS = 1.#np.linalg.cond(Sii)
    condvecS_H[indp] = condS
    print("condS = ",condS)
    u = bc_helmholtz(XX,kh)
    uhat = -np.linalg.solve(Sii,Sib@u[Ib])
    err = np.linalg.norm(u[Ii]-uhat)/np.linalg.norm(u[Ii])
    print("S sol err. = ",err)
    errvecS_H[indp] = err


    bnds = np.array([[0,0],[2,1]])
    geom = hpsaltGeom.BoxGeometry(bnds)
    xl = geom.bounds[0,0]
    xr = geom.bounds[1,0]

    def c11(p):
        return torch.ones_like(p[:,0])
    def c22(p):
        return torch.ones_like(p[:,0])
    def c(p):
        return -kh*kh*torch.ones_like(p[:,0])
    
    diff_op = pdoalt.PDO_2d(c11=c11,c22=c22,c=c)

    ax = .5/nbx
    ay = .5/nby
    tic = time.time()
    solver = hpsalt.Domain_Driver(geom, diff_op, kh, np.array([ax,ay]), px+1, 2)
    solver.build("reduced_cpu", "MUMPS")
    toc = time.time()-tic
    Aii = np.array(solver.Aii.todense())
    print("Aii shape = ",Aii.shape)
    Aib = np.array(solver.Aix.todense())
    print("elapsed time T = ",toc)
    print("DOFS T = ",Aii.shape[0])
    condT = 1.#np.linalg.cond(Aii)
    condvecT_H[indp] = condT
    print("condT = ",condT)
    XX = solver.XX
    Ii = solver._Ji
    Ib = solver._Jx
    u = bc_helmholtz(XX,kh).numpy()
    uhat = -np.linalg.solve(Aii,Aib@u[Ib])
    err = np.linalg.norm(u[Ii]-uhat)/np.linalg.norm(u[Ii])
    print("T sol err. = ",err)
    errvecT_H[indp] = err
    '''


Tfit_L = 1.*Nvec**(2/3)
Tfit_L *= (condvecT_L[-1]/Tfit_L[-1])*1.1
#Tfit_H = 1.*Nvec*np.log2(Nvec)
#Tfit_H *= (condvecT_H[-1]/Tfit_H[-1])*1.1

Sfit_L = (1.+np.log2(Nvec))**3
Sfit_L *= (condvecS_L[-1]/Sfit_L[-1])*1.1
#Sfit_H = (1.+np.log2(Nvec))**2
#Sfit_H *= (condvecS_H[-1]/Sfit_H[-1])*1.1


plt.figure(1)
plt.loglog(Nvec,condvecS_L)
plt.loglog(Nvec,Sfit_L,linestyle='dashed')
plt.loglog(Nvec,condvecT_L)
plt.loglog(Nvec,Tfit_L,linestyle='dashed')
plt.legend(['condS','fitS','condT','fitT'])

#plt.figure(2)
#plt.loglog(Nvec,condvecS_H)
#plt.loglog(Nvec,Sfit_H,linestyle='dashed')
#plt.loglog(Nvec,condvecT_H)
#plt.loglog(Nvec,Tfit_H,linestyle='dashed')
#plt.legend(['condS','fitS','condT','fitT'])


plt.figure(3)
plt.loglog(Nvec,errvecS_L)
plt.loglog(Nvec,errvecT_L)
plt.legend(['errS','errT'])
plt.show()

#plt.figure(4)
#plt.loglog(Nvec,errvecS_H)
#plt.loglog(Nvec,errvecT_H)
#plt.legend(['errS','errT'])
#plt.show()