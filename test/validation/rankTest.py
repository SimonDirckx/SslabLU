import numpy as np
import jax.numpy as jnp
import solver.spectralmultidomain.hps.pdo as pdo
from packaging.version import Version
import scipy
import matplotlib.pyplot as plt
from scipy.sparse.linalg   import LinearOperator
from solver.solver import stMap
import sys
# oms packages
import solver.solver as solverWrap
import matAssembly.matAssembler as mA
import multislab.oms as oms
from scipy.sparse.linalg import gmres
from matAssembly.HBS.simpleoctree import simpletree as tree
from time import time
import gc


def compute_c0_L0(XX):
    N,ndim = XX.shape
    c0 = np.sum(XX,axis=0)/N
    L0 = np.max(np.max(XX,axis=0)-np.min(XX,axis=0)) #too tight for some reason
    return c0,L0+1e-5

def compute_stmaps(Il,Ic,Ir,XXi,XXb,solver):
        A_solver = solver.solver_ii    
        def smatmat(v,I,J,transpose=False):
            if (v.ndim == 1):
                v_tmp = v[:,np.newaxis]
            else:
                v_tmp = v

            if (not transpose):
                result = (A_solver@(solver.Aib[:,J]@v_tmp))[I,:]
            else:
                result      = np.zeros(shape=(len(solver.Ii),v_tmp.shape[1]))
                result[I,:] = v_tmp
                result      = solver.Aib[:,J].T @ (A_solver.T@(result))
            if (v.ndim == 1):
                result = result.flatten()
            return result

        Linop_r = LinearOperator(shape=(len(Ic),len(Ir)),\
            matvec = lambda v:smatmat(v,Ic,Ir), rmatvec = lambda v:smatmat(v,Ic,Ir,transpose=True),\
            matmat = lambda v:smatmat(v,Ic,Ir), rmatmat = lambda v:smatmat(v,Ic,Ir,transpose=True))
        Linop_l = LinearOperator(shape=(len(Ic),len(Il)),\
            matvec = lambda v:smatmat(v,Ic,Il), rmatvec = lambda v:smatmat(v,Ic,Il,transpose=True),\
            matmat = lambda v:smatmat(v,Ic,Il), rmatmat = lambda v:smatmat(v,Ic,Il,transpose=True))
        
        st_r = stMap(Linop_r,XXb[Ir,:],XXi[Ic,:])
        st_l = stMap(Linop_l,XXb[Il,:],XXi[Ic,:])
        return st_l,st_r


def join_geom(slab1,slab2,period=None):
    ndim = len(slab1[0])
    if ndim==2:
        xl1 = slab1[0][0]
        xr1 = slab1[1][0]
        yl1 = slab1[0][1]
        yr1 = slab1[1][1]
        
        xl2 = slab2[0][0]
        xr2 = slab2[1][0]
        yl2 = slab2[0][1]
        yr2 = slab2[1][1]
        if(np.abs(xr1-xl2)>1e-10):
            if period:
                xl1 -= period
                xr1 -= period
                return join_geom([[xl1,yl1],[xr1,yr1]],slab2)
            else:
                ValueError("slab shift did not work (is your period correct?)")
        else:
            totalSlab = [[xl1, yl1],[xr2,yr2]]
        return totalSlab
    elif ndim==3:
        xl1 = slab1[0][0]
        xr1 = slab1[1][0]
        yl1 = slab1[0][1]
        yr1 = slab1[1][1]
        zl1 = slab1[0][2]
        zr1 = slab1[1][2]

        xl2 = slab2[0][0]
        xr2 = slab2[1][0]
        yl2 = slab2[0][1]
        yr2 = slab2[1][1]
        zl2 = slab2[0][2]
        zr2 = slab2[1][2]
        if(np.abs(xr1-xl2)>1e-10):
            if period:
                xl1 -= period
                xr1 -= period
                return join_geom([[xl1,yl1,zl1],[xr1,yr1,zr1]],slab2)
            else:
                ValueError("slab shift did not work (is your period correct?)")
        else:
            totalSlab = [[xl1, yl1,zl1],[xr2,yr2,zr2]]
        return totalSlab
    else:
        raise ValueError("ndim incorrect")

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

nwaves = 2.24
kh = (nwaves/4)*2.*np.pi
print("kappa = ",kh)
def c11(p):
    return jnp.ones_like(p[...,0])
def c22(p):
    return jnp.ones_like(p[...,0])
def c33(p):
    return jnp.ones_like(p[...,0])
def bfield(p):
    return -kh*kh*jnp.ones_like(p[...,0])
helmholtz = pdo.PDO3d(c11=c11,c22=c22,c33=c33,c=pdo.const(-kh*kh))

bnds = [[0.,0.,0.],[1.,1.,1.]]
box_geom   = jnp.array(bnds)

def gb(p):
    return np.abs(p[0]-bnds[0][0])<1e-14 or np.abs(p[0]-bnds[1][0])<1e-14 or np.abs(p[1]-bnds[0][1])<1e-14 or np.abs(p[1]-bnds[1][1])<1e-14

def gb_vec(P):
    # P is (N, 2)
    return (
        (np.abs(P[:, 0] - bnds[0][0]) < 1e-14) |
        (np.abs(P[:, 0] - bnds[1][0]) < 1e-14) |
        (np.abs(P[:, 1] - bnds[0][1]) < 1e-14) |
        (np.abs(P[:, 1] - bnds[1][1]) < 1e-14) |
        (np.abs(P[:, 2] - bnds[0][2]) < 1e-14) |
        (np.abs(P[:, 2] - bnds[1][2]) < 1e-14)
    )

def bc(p):
    return jnp.ones_like(p[...,0])

H = 1./8.
N = (int)(1./H)

slabs = []
for n in range(N):
    bnds_n = [[n*H,0.,0.],[(n+1)*H,1.,1.]]
    slabs+=[bnds_n]

connectivity = []
for i in range(N-1): 
    connectivity+=[[i,i+1]]

if_connectivity = []
for i in range(N-1):
    if i==0:
        if_connectivity+=[[-1,(i+1)]]
    elif i==N-2:
        if_connectivity+=[[(i-1),-1]]
    else:
        if_connectivity+=[[(i-1),(i+1)]]
period = 0.
binary = False
pvec = [4,5,6,7,8]
a = [H/2.,1/32,1/32]
nlvl = int(np.log2(1/(2*a[1])))
if binary:
    nlvl *=2
nlvl+=1

rkWeak=np.zeros(shape=(len(pvec),nlvl-2),dtype = np.int64)
rkStrong=np.zeros(shape=(len(pvec),nlvl-2),dtype=np.int64)

for indp in range(len(pvec)):
    p = pvec[indp]
    
    print("ppw = ",np.array( [ p/H , p*(2/a[1]) , p*(2/a[2]) ] )/nwaves)
    print("ppw = ",min([p/H,p*(2/a[1]),p*(2/a[2])])/nwaves)
    opts = solverWrap.solverOptions('hps',[p,p,p],a)

    slabInd = 0
    geom    = np.array(join_geom(slabs[connectivity[slabInd][0]],slabs[connectivity[slabInd][1]],period))
    slab_i  = oms.slab(geom,gb_vec)
    solver  = solverWrap.solverWrapper(opts)
    solver.construct(geom,helmholtz,verbose=True)

    XX = solver.XX
    XXb = XX[solver.Ib,:]
    XXi = XX[solver.Ii,:]
    xl = geom[0][0]
    xr = geom[1][0]
    xc=(xl+xr)/2.
    print("\t SLAB BOUNDS xl,xc,xr=",xl,",",xc,",",xr)

    ####################### Fast vectorized operations ####################

    tic = time()
    Il = np.where(np.abs(XXb[:, 0] - xl) < 1e-14)[0]
    Ir = np.where(np.abs(XXb[:, 0] - xr) < 1e-14)[0]
    Ic = np.where(np.abs(XXi[:, 0] - xc) < 1e-14)[0]
    Igb = np.where(gb_vec(XXb))[0]
    toc = time() - tic
    print("\t Toc fast index computations %5.2f s" % toc)
    print("\t SLAB dofs = ",len(Ic))

    st_l,st_r = compute_stmaps(Il,Ic,Ir,XXi,XXb,solver)
    n=len(Ic)

    ndim = XX.shape[1]
    if ndim == 2:
        leaf_size = p
        XXI = XXi[Ic,:]
        XXB = XXb[Ir,:]
    elif ndim == 3:
        leaf_size = p*p
        XXI = XXi[Ic,1:3]
        XXB = XXb[Ir,1:3]
    else:
        ValueError("ndim must be 2 or 3")
 

    tic = time()
    def check_regularity(t,binary):
        n=1
        check=True
        balance = True
        start = 0
        if binary:
            n=2
            start = 1
        for lvl in range(start,t.nlevels):
            l=len(t.get_boxes_level(lvl))
            check = check and (l==n)
            nI = len(t.get_box_inds(t.get_boxes_level(lvl)[0]))
            for i in range(l):
                len(t.get_box_inds(t.get_boxes_level(lvl)[i]))
                balance = balance and (nI == len(t.get_box_inds(t.get_boxes_level(lvl)[i])))
            if binary:
                n*=2
            else:
                n*=4
        if not check:
            raise ValueError("tree structure incorrect")
        if not balance:
            raise ValueError("tree not balanced")

    if binary:
        tree0 = tree.BinaryTree(XXI,leaf_size,np.array([.5,.5]),np.array([1.+1e-5,1.+1e-5]))
    else:
        tree0 = tree.BalancedTree(XXI,leaf_size,np.array([.5,.5]),np.array([1.+1e-5,1.+1e-5]))

    check_regularity(tree0,binary)
    print("actual nlvl = ",tree0.nlevels)
    toc = time() - tic
    print("\t Toc tree construction %5.2f s" % toc)

    def compute_ancestor(box,lvl):
        parent = box
        lvl0 = lvl
        while lvl0>1:
            parent = tree0.get_box_parent(parent)
            lvl0-=1
        return parent

    def near(box0,box1):
        c0 = tree0.get_box_center(box0)
        L = tree0.get_box_length(box0)
        c1 = tree0.get_box_center(box1)
        return np.linalg.norm(c0-c1)<np.sqrt(L[0]**2+L[1]**2)+1e-5

    def far(box0,box1):
        return not near(box0,box1)

    tol_rk = 1e-6

    for lvl in range(tree0.nlevels-1,1,-1):
        print("=================lvl ",lvl,"=================")
        boxes = tree0.get_boxes_level(lvl)
        box0 = boxes[0]
        Ibox = tree0.get_box_inds(box0)
        print("\t box0=%d ancestor = %d" %(box0,compute_ancestor(box0,lvl)))

        boxesc = [box for box in boxes if box!=box0]
        Ic = np.zeros(shape=(0,),dtype = np.int64)
        for box in boxesc:
            Ic=np.append(Ic,tree0.get_box_inds(box))

        boxes_far = [box for box in boxes if compute_ancestor(box,lvl)!=compute_ancestor(box0,lvl)]
        Ifar = np.zeros(shape=(0,),dtype = np.int64)
        for box in boxes_far:
            Ifar=np.append(Ifar,tree0.get_box_inds(box))
        E = np.identity(n)

        tic = time()
        tmp = np.random.randn(n,100)
        st_l.A @ tmp
        toc = time() - tic
        print("\t Toc solve PDE on double slab for %d rhs %5.2f s" % (tmp.shape[-1],toc))

        #########################
        #       c ranks
        #########################
        print("RANK RESULTS at tol %.2e" %(tol_rk))
        Sl = E[:,Ic].T@((st_l.A)@E[:,Ibox])
        [_,s,_] = np.linalg.svd(Sl)
        rk1 = sum(s>s[0]*tol_rk)
        print("\t shape//rk c   = (",len(Ibox),",",len(Ic),")","//",rk1)
        rkWeak[indp,lvl-2] = rk1

        #########################
        #       far ranks
        #########################
        Sl = E[:,Ifar].T@((st_l.A)@E[:,Ibox])
        [_,s,_] = np.linalg.svd(Sl)
        rk1 = sum(s>s[0]*tol_rk)
        print("\t shape//rk far = (",len(Ibox),",",len(Ifar),")","//",rk1)
        rkStrong[indp,lvl-2] = rk1
print("rkWeak = ",rkWeak)
print("rkStrong = ",rkStrong)

np.savetxt('rkWeak.out',rkWeak,delimiter=',',fmt='%d')
np.savetxt('rkStrong.out',rkStrong,delimiter=',',fmt='%d')

plt.figure(1)
labelweak = []
labelstrong = []
for i in range(2,nlvl):
    labelweak+=["weak lvl "+str(i)]
    labelstrong+=["strong lvl "+str(i)]
plt.plot(pvec,rkWeak,label=labelweak)
plt.gca().set_prop_cycle(None)
plt.plot(pvec,rkStrong,label=labelstrong,linestyle='dashed')
plt.legend()
plt.xticks(pvec)
plt.show()


'''
plt.figure(0)
plt.scatter(XXI[Ic,0],XXI[Ic,1],label='c')
plt.scatter(XXI[Ibox,0],XXI[Ibox,1],label='box')
plt.legend()
plt.axis('equal')

plt.figure(1)
plt.scatter(XXI[Ifar,0],XXI[Ifar,1],label='far')
plt.scatter(XXI[Ibox,0],XXI[Ibox,1],label='box')
plt.legend()
plt.axis('equal')

plt.figure(2)
plt.scatter(XXI[I2,0],XXI[I2,1],label='2')
plt.scatter(XXI[Ibox,0],XXI[Ibox,1],label='box')
plt.legend()
plt.axis('equal')

plt.figure(3)
plt.scatter(XXI[I3,0],XXI[I3,1],label='3')
plt.scatter(XXI[Ibox,0],XXI[Ibox,1],label='box')
plt.legend()
plt.axis('equal')

plt.figure(4)
plt.scatter(XXI[I4,0],XXI[I4,1],label='4')
plt.scatter(XXI[Ibox,0],XXI[Ibox,1],label='box')
plt.legend()
plt.axis('equal')
plt.show()
'''
