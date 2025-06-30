import numpy as np
import jax.numpy as jnp
import solver.spectralmultidomain.hps.pdo as pdo
from packaging.version import Version
import scipy
import matplotlib.pyplot as plt
from scipy.sparse.linalg   import LinearOperator
from solver.solver import stMap

# oms packages
import solver.solver as solverWrap
import matAssembly.matAssembler as mA
import multislab.oms as oms
from scipy.sparse.linalg import gmres
from matAssembly.HBS.simpleoctree import simpletree as tree
import time

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
                result = (A_solver@(solver.Aib[:,J]@v_tmp))[I]
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

nwaves = 5.24
kh = (nwaves/4)*2.*np.pi
def c11(p):
    return jnp.ones_like(p[...,0])
def c22(p):
    return jnp.ones_like(p[...,0])
def c33(p):
    return jnp.ones_like(p[...,0])
def bfield(p):
    return kh*kh*jnp.ones_like(p[...,0])
helmholtz = pdo.PDO3d(c11=c11,c22=c22,c33=c33,c=pdo.const(-kh*kh))

bnds = [[0.,0.,0.],[1.,1.,1.]]
box_geom   = jnp.array(bnds)

def gb(p):
    return np.abs(p[0]-bnds[0][0])<1e-14 or np.abs(p[0]-bnds[1][0])<1e-14 or np.abs(p[1]-bnds[0][1])<1e-14 or np.abs(p[1]-bnds[1][1])<1e-14
def bc(p):
    return jnp.ones_like(p[...,0])

H = 1./4.
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

p = 12
a = [H/2.,1/16,1/16]

opts = solverWrap.solverOptions('hps',[p,p,p],a)

slabInd = 0
geom    = np.array(join_geom(slabs[connectivity[slabInd][0]],slabs[connectivity[slabInd][1]],period))
slab_i  = oms.slab(geom,gb)
solver  = solverWrap.solverWrapper(opts)
solver.construct(geom,helmholtz)
print("solver done")

XX = solver.XX
XXb = XX[solver.Ib,:]
XXi = XX[solver.Ii,:]
xl = geom[0][0]
xr = geom[1][0]
xc=(xl+xr)/2.

Il = [i for i in range(len(solver.Ib)) if np.abs(XXb[i,0]-xl)<1e-14 ]
Ir = [i for i in range(len(solver.Ib)) if np.abs(XXb[i,0]-xr)<1e-14 ]
Ic = [i for i in range(len(solver.Ii)) if np.abs(XXi[i,0]-xc)<1e-14]
Igb = [i for i in range(len(solver.Ib)) if gb(XXb[i,:])]
print('ndofs = ',len(Ic))


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



c0,L0 = compute_c0_L0(XXI)
binary = False
if binary:
    tree0 = tree.BinaryTree(XXI,leaf_size,np.array([.5,.5]),np.array([1.,1.]))
else:
    tree0 = tree.BalancedTree(XXI,leaf_size,np.array([.5,.5]),np.array([1.,1.]))

reduced = False
print("nlevels = ",tree0.nlevels)
for lvl in range(tree0.nlevels):
    print(tree0.get_boxes_level(lvl))
print("nleaves = ",len(tree0.get_leaves()))

st_l,st_r = compute_stmaps(Il,Ic,Ir,XXi,XXb,solver)
assembler = mA.rkHMatAssembler(p*p,150,tree0)
tic = time.time()
Sr_rk = assembler.assemble(st_r,reduced,1)
tS = time.time()-tic
print("assembler time (s) = ",tS)
print("assembler data (GB) = ",assembler.nbytes/1e9)
print("assembler data per dof = ",assembler.nbytes/st_r.A.shape[0])
print("assembler compression = ",assembler.nbytes/(8*st_r.A.shape[0]*st_r.A.shape[1]))
print("Sr_rk shape = ",Sr_rk.shape)
print("Sr shape = ",st_r.A.shape)


'''
def sink(XY,k1,k2):
    return np.sin(np.pi*k1*XY[:,0])*np.sin(np.pi*k2*XY[:,1])


nc1=5
nc2=5
coeff1 = np.random.standard_normal(size=(nc1,))
coeff2 = np.random.standard_normal(size=(nc2,))

v=np.zeros(shape=(XXB.shape[0],))
for i1 in range(1,nc1+1):
    for i2 in range(1,nc2+1):
        v+=coeff1[i1-1]*coeff2[i2-1]*sink(XXB,i1,i2)

nc1=5
nc2=5
coeff1 = np.random.standard_normal(size=(nc1,))
coeff2 = np.random.standard_normal(size=(nc2,))

w=np.zeros(shape=(XXB.shape[0],))
for i1 in range(1,nc1+1):
    for i2 in range(1,nc2+1):
        w+=coeff1[i1-1]*coeff2[i2-1]*sink(XXB,i1,i2)

'''

v = np.random.standard_normal(size=(Sr_rk.shape[1],))
w = np.random.standard_normal(size=(Sr_rk.shape[1],))
for i in range(20):
    v/=np.linalg.norm(v)
    w/=np.linalg.norm(w)
    v=(st_r.A@v-Sr_rk@v)
    v=st_r.A.T@v-Sr_rk.T@v
    w=st_r.A@w
    w=st_r.A.T@w
err = np.sqrt(np.linalg.norm(v)/np.linalg.norm(w))
    

print("S err = ",err)
'''
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




boxes = tree0.get_leaves()
Itot = np.zeros(shape=(0,),dtype = np.int64)
for box in boxes:
    Itot=np.append(Itot,tree0.get_box_inds(box))

for box in boxes:
    Ibox = tree0.get_box_inds(box)
    boxes_near = [box0 for box0 in boxes if near(box,box0) and not box0==box]
    boxes_far = [box0 for box0 in boxes if  far(box,box0)]
    I_near = np.zeros(shape=(0,),dtype = Ibox[0].dtype)
    I_far = np.zeros(shape=(0,),dtype = Ibox[0].dtype)
    I_c = np.array([i for i in Itot if i not in Ibox])
    for box_near in boxes_near:
        I_near=np.append(I_near,tree0.get_box_inds(box_near))
    for box_far in boxes_far:
        I_far=np.append(I_far,tree0.get_box_inds(box_far))
    S_far = Sr[Ibox,:][:,I_far]
    S_near = Sr[Ibox,:][:,I_near]
    S_c = Sr[Ibox,:][:,I_c]
    print("far  shape//rk = ",S_far.shape,"//",np.linalg.matrix_rank(S_far,1e-5))
    print("near shape//rk = ",S_near.shape,"//",np.linalg.matrix_rank(S_near,1e-5))
    print("c    shape//rk = ",S_c.shape,"//",np.linalg.matrix_rank(S_c,1e-5))
'''