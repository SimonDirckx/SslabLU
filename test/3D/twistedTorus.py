# basic packages
import numpy as np
import jax.numpy as jnp
import scipy
from packaging.version import Version
import matplotlib.tri as tri

# oms packages
import solver.solver as solverWrap
import matAssembly.matAssembler as mA
import multislab.oms as oms
import multislab.omsdirectsolve as omsdirect
import torch
# validation&testing
import time
from scipy.sparse.linalg import gmres
#import solver.HPSInterp3D as interp
import matplotlib.pyplot as plt

import geometry.geom_3D.squareTorus as domain


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
from petsc4py import PETSc
#try:
#    vec = PETSc.Vec().createSeq(10**9)
#    print("vec created successfully")
#except Exception as e:
#    print(e)

bnds = domain.bnds

################################################################
#
#   SET-UP BVP:         Helmholtz on 3D Annulus
#   - wave number       (kh)
#   - bfield            (= kh*ones)
#   - pdo_mod           (pdo transformed to square)
#   - BC
#   - known exact sol.  (u_exact)
#
################################################################

nwaves = 2.24
wavelength = 4/nwaves
kh = (nwaves/4)*2.*np.pi
kh = 10.6
# What to modify to use the Jax-based hps ("hps") or Torch-based ("hpsalt")
jax_avail   = False
torch_avail = not jax_avail
hpsalt      = not jax_avail


if jax_avail:
    def bfield(p,kh):
        return -kh*kh*jnp.ones_like(p[...,0])
elif torch_avail:
    def bfield(p,kh):
        return -kh*kh*torch.ones(p.shape[0])
else:
    def bfield(p,kh):
        return -kh*kh*np.ones(shape=(p.shape[0],))
param_geom=domain.param_geom(jax_avail=jax_avail, torch_avail=torch_avail, hpsalt=hpsalt)
pdo_mod = param_geom.transform_helmholtz_pdo(bfield,kh)

def bc(p):
    z1=domain.z1(p,jax_avail=jax_avail,torch_avail=torch_avail)
    z2=domain.z2(p,jax_avail=jax_avail,torch_avail=torch_avail)
    z3=domain.z3(p,jax_avail=jax_avail,torch_avail=torch_avail)
    rr = np.sqrt((z1-5.)**2+z2**2+z3**2)
    return np.ones_like(z1)#np.sin(kh*z1)#np.cos(kh*rr)/(4*np.pi*rr)


N = 8
dSlabs,connectivity,H = domain.dSlabs(N)
formulation = "hps"
solve_method = 'iterative'
#solve_method = 'direct'
HBS = True

#pvec = np.array([4,6,8,10],dtype = np.int32)
pvec = np.array([6],dtype = np.int64)
err=np.zeros(shape = (len(pvec),))
discr_time=np.zeros(shape = (len(pvec),))
sample_time = np.zeros(shape=(len(pvec),))
compr_time=np.zeros(shape = (len(pvec),))
for indp in range(len(pvec)):
    p = pvec[indp]
    p_disc = p
    if hpsalt:
        formulation = "hpsalt"
        p_disc = p_disc + 2 # To handle different conventions between hps and hpsalt

    a = np.array([H/6.,1./16,1./16])
    if HBS:
        assembler = mA.rkHMatAssembler(p*p,125)
    else:
        assembler = mA.denseMatAssembler()
    opts = solverWrap.solverOptions(formulation,[p_disc,p_disc,p_disc],a)
    
    OMS = oms.oms(dSlabs,pdo_mod,lambda p :domain.gb(p,jax_avail=jax_avail,torch_avail=torch_avail),opts,connectivity)
    S_rk_list, rhs_list, Ntot, nc = OMS.construct_Stot_helper(bc, assembler, dbg=2)
    niter = 0


    if solve_method == 'iterative':
        Stot,rhstot  = OMS.construct_Stot_and_rhstot_linearOperator(S_rk_list,rhs_list,Ntot,nc,dbg=2)
        gInfo = gmres_info()
        stol = 1e-7*H*H
        if Version(scipy.__version__)>=Version("1.14"):
            uhat,info   = gmres(Stot,rhstot,rtol=stol,callback=gInfo,maxiter=1000,restart=1000)
        else:
            uhat,info   = gmres(Stot,rhstot,tol=stol,callback=gInfo,maxiter=1000,restart=1000)
        niter = gInfo.niter
    elif solve_method == 'direct':
        Stot,rhstot  = OMS.construct_Stot_and_rhstot_linearOperator(S_rk_list,rhs_list,Ntot,nc,dbg=2)
        T,block = omsdirect.build_block_cyclic_tridiagonal_solver(OMS,S_rk_list,rhs_list,Ntot,nc)
        uhat  = omsdirect.block_cyclic_tridiagonal_solve(OMS, T, block,rhstot)

    res = Stot@uhat-rhstot


    print("=============SUMMARY==============")
    print("H                        = ",'%10.3E'%H)
    print("ord                      = ",p)
    print("L2 rel. res              = ", np.linalg.norm(res)/np.linalg.norm(rhstot))
    print("GMRES iters              = ", niter)
    print("==================================")

    errInf = 0.
    nc = OMS.nc
    
    
    #test domain bounds correct

    xpts = np.linspace(domain.bnds[0][0],domain.bnds[1][0],20)
    ypts = np.linspace(domain.bnds[0][1],domain.bnds[1][1],20)
    zpts = np.linspace(domain.bnds[0][2],domain.bnds[1][2],20)

    YY = np.zeros(shape = (20*20*20,3))
    YY[:,0] = np.kron(np.kron(xpts,np.ones_like(ypts)),np.ones_like(zpts))
    YY[:,1] = np.kron(np.kron(np.ones_like(xpts),ypts),np.ones_like(zpts))
    YY[:,2] = np.kron(np.kron(np.ones_like(xpts),np.ones_like(ypts)),zpts)
    ZZ = np.zeros(shape = YY.shape)
    ZZ[:,0] = domain.z1(YY,False,False)
    ZZ[:,1] = domain.z2(YY,False,False)
    ZZ[:,2] = domain.z3(YY,False,False)

    fig = plt.figure(2)
    ax = fig.add_subplot(projection='3d')

    ax.scatter(ZZ[:,0], ZZ[:,1], ZZ[:,2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


    #test compat y and z

    YY0 = np.zeros(shape = (20*20*20,3))
    YY0[:,0] = domain.y1(ZZ,False,False)
    YY0[:,1] = domain.y2(ZZ,False,False)
    YY0[:,2] = domain.y3(ZZ,False,False)

    '''
    ytest =np.zeros(shape = (1,3))
    ytest[0,0] = 1
    ytest[0,1] = 1
    ytest[0,2] = 1
    
    ztest =np.zeros(shape = (1,3))
    z1 = domain.z1(ytest,False,False)
    z2 = domain.z2(ytest,False,False)
    z3 = domain.z3(ytest,False,False)
    print("z1,z2,z3 = ",z1,",",z2,",",z3)
    ztest[0,0] = z1[0]
    ztest[0,1] = z2[0]
    ztest[0,2] = z3[0]
    print("ztest = ",ztest)
    y1 = domain.y1(ztest,False,False)
    y2 = domain.z2(ztest,False,False)
    y3 = domain.z3(ztest,False,False)
    print("y1,y2,y3 = ",y1,",",y2,",",y3)
    '''
    print("YY0 err = ",np.linalg.norm(YY-YY0)/np.linalg.norm(YY))
    fig = plt.figure(2)
    ax = fig.add_subplot(projection='3d')

    ax.scatter(YY0[:,0], YY0[:,1], YY0[:,2])
    ax.scatter(YY[:,0], YY[:,1], YY[:,2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(['YY0','YY'])
    plt.show()



    nx=200
    ny=200




    xpts = np.linspace(-2,2,nx)
    ypts = np.linspace(-2,2,ny)

    ZZ = np.zeros(shape=(nx*ny,3))
    ZZ[:,0] = np.kron(xpts,np.ones_like(ypts))
    ZZ[:,1] = np.kron(np.ones_like(xpts),ypts)

    sliceYY = np.zeros(shape=ZZ.shape)
    sliceYY[:,0] = domain.y1(ZZ,False,False)
    sliceYY[:,1] = domain.y2(ZZ,False,False)
    sliceYY[:,2] = domain.y3(ZZ,False,False)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(sliceYY[:,0], sliceYY[:,1], sliceYY[:,2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

    I = np.where( (sliceYY[:,0]>=domain.bnds[0][0]) & (sliceYY[:,0]<=domain.bnds[1][0]) & (sliceYY[:,1]>=domain.bnds[0][1]) & (sliceYY[:,1]<=domain.bnds[1][1]) & (sliceYY[:,2]>=domain.bnds[0][2]) & (sliceYY[:,2]<=domain.bnds[1][2]) )[0]


    YY = sliceYY[I,:]




    gYY = np.zeros(shape=(YY.shape[0],))


    sliceZZ = np.zeros(shape=(YY.shape[0],3))
    sliceZZ[:,0] = domain.z1(YY,False,False)
    sliceZZ[:,1] = domain.z2(YY,False,False)
    sliceZZ[:,2] = domain.z3(YY,False,False)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(sliceZZ[:,0], sliceZZ[:,1], sliceZZ[:,2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


    ucheck = np.zeros(shape=(Stot.shape[0],))
    rhscheck = np.zeros(shape=(Stot.shape[1],))

    XXcprev = np.zeros(shape=(nc,3))

    scatterGlob = np.zeros(shape=(0,3))

    for slabInd in range(len(dSlabs)):
        geom    = np.array(dSlabs[slabInd])
        I0 = np.where(  (YY[:,0]>=geom[0,0]) & (YY[:,0]<=geom[1,0]) & (YY[:,1]>=geom[0,1]) & (YY[:,1]<=geom[1,1]) & (YY[:,2]>=geom[0,2]) & (YY[:,2]<=geom[1,2]) )[0]
        YY0 = YY[I0,:]
        slab_i  = oms.slab(geom,lambda p : domain.gb(p,jax_avail=jax_avail,torch_avail=torch_avail))
        solver  = oms.solverWrap.solverWrapper(opts)
        solver.construct(geom,pdo_mod)
        Il,Ir,Ic,Igb,XXi,XXb = slab_i.compute_idxs_and_pts(solver)
        startL = ((slabInd-1)%N)
        startR = ((slabInd+1)%N)
        ul = uhat[startL*nc:(startL+1)*nc]
        ur = uhat[startR*nc:(startR+1)*nc]
        print("ul err = ",np.linalg.norm(ul - np.array(bc(XXb[Il,:])))/np.linalg.norm(np.array(bc(XXb[Il,:]))))
        print("ur err = ",np.linalg.norm(ur - np.array(bc(XXb[Ir,:])))/np.linalg.norm(np.array(bc(XXb[Ir,:]))))
        
        g = np.zeros(shape=(XXb.shape[0],1))
        g[Il,0]=ul
        g[Ir,0]=ur
        g[Igb,0] = bc(XXb[Igb,:])
        if hpsalt:
            g=torch.from_numpy(g)
        uu = solver.solver.solve_dir_full(g)
        uu=uu.flatten()
        ghat = solver.interp(YY0,np.array(uu))
        gYY[I0] = ghat
        print("slab ",slabInd," done")


    #g_ref = np.load('ref_sol.npy')
    #print("err_I = ",np.linalg.norm(g_ref-gYY,ord=np.inf)/np.linalg.norm(g_ref,ord=np.inf))
    #err[indp] = np.linalg.norm(g_ref-gYY,ord=np.inf)/np.linalg.norm(g_ref,ord=np.inf)
    #sample_time[indp] = OMS.stats.sampl_timing
    #compr_time[indp] = OMS.stats.compr_timing
    #discr_time[indp] = OMS.stats.discr_timing

#fileName = 'domainTorus.csv'
#errMat = np.zeros(shape=(len(pvec),5))
#errMat[:,0] = pvec
#errMat[:,1] = err
#errMat[:,2] = sample_time
#errMat[:,3] = compr_time
#errMat[:,4] = discr_time
#with open(fileName,'w') as f:
#    f.write('p,err,sample,compr,discr\n')
#    np.savetxt(f,errMat,fmt='%.16e',delimiter=',')
   

triang = tri.Triangulation(sliceZZ[:,0],sliceZZ[:,1])
tri0 = triang.triangles

q1 = (sliceZZ[tri0[:,0],:]+sliceZZ[tri0[:,1],:])/2.
q2 = (sliceZZ[tri0[:,1],:]+sliceZZ[tri0[:,2],:])/2.
q3 = (sliceZZ[tri0[:,2],:]+sliceZZ[tri0[:,0],:])/2.


yy1 = np.zeros(shape = q1.shape)
yy2 = np.zeros(shape = q2.shape)
yy3 = np.zeros(shape = q3.shape)

yy1[:,0] = domain.y1(q1,False)
yy1[:,1] = domain.y2(q1,False)
yy1[:,2] = domain.y3(q1,False)

yy2[:,0] = domain.y1(q2,False)
yy2[:,1] = domain.y2(q2,False)
yy2[:,2] = domain.y3(q2,False)

yy3[:,0] = domain.y1(q3,False)
yy3[:,1] = domain.y2(q3,False)
yy3[:,2] = domain.y3(q3,False)


b1 = (yy1[:,0]<domain.bnds[0][0]) | (yy1[:,0]>domain.bnds[1][0]) | (yy1[:,1]<domain.bnds[0][1]) | (yy1[:,1]>domain.bnds[1][1]) | (yy1[:,2]<domain.bnds[0][2]) | (yy1[:,2]>domain.bnds[1][2])
b2 = (yy2[:,0]<domain.bnds[0][0]) | (yy2[:,0]>domain.bnds[1][0]) | (yy2[:,1]<domain.bnds[0][1]) | (yy2[:,1]>domain.bnds[1][1]) | (yy2[:,2]<domain.bnds[0][2]) | (yy2[:,2]>domain.bnds[1][2])
b3 = (yy3[:,0]<domain.bnds[0][0]) | (yy3[:,0]>domain.bnds[1][0]) | (yy3[:,1]<domain.bnds[0][1]) | (yy3[:,1]>domain.bnds[1][1]) | (yy3[:,2]<domain.bnds[0][2]) | (yy3[:,2]>domain.bnds[1][2])


mask = (b1&b2)|(b1&b3)|(b2&b3)
triang.set_mask(mask)
#np.save('ref_sol.npy',gYY)
plt.figure(5)
plt.tripcolor(triang, gYY, shading='gouraud',cmap='jet')
plt.axis('equal')
plt.colorbar()
plt.savefig('domainTorus.png',dpi=1000)
plt.show()
