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
import time
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


p=6

a = [H/2.,1/32,1/32]
print("ppw = ",np.array( [ p/H , p*(2/a[1]) , p*(2/a[2]) ] )/nwaves)
print("ppw = ",min([p/H,p*(2/a[1]),p*(2/a[2])])/nwaves)
opts = solverWrap.solverOptions('hps',[p,p,p],a)

slabInd = 0
geom    = np.array(join_geom(slabs[connectivity[slabInd][0]],slabs[connectivity[slabInd][1]],period))
slab_i  = oms.slab(geom,gb)
solver  = solverWrap.solverWrapper(opts)
solver.construct(geom,helmholtz)

XX = solver.XX
XXb = XX[solver.Ib,:]
XXi = XX[solver.Ii,:]
xl = geom[0][0]
xr = geom[1][0]
xc=(xl+xr)/2.
print("xl,xc,xr=",xl,",",xc,",",xr)

Il = [i for i in range(len(solver.Ib)) if np.abs(XXb[i,0]-xl)<1e-14 ]
Ir = [i for i in range(len(solver.Ib)) if np.abs(XXb[i,0]-xr)<1e-14 ]
Ic = [i for i in range(len(solver.Ii)) if np.abs(XXi[i,0]-xc)<1e-14]
Igb = [i for i in range(len(solver.Ib)) if gb(XXb[i,:])]

print("#slab dofs = ",len(Ic))

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



c0,L0 = compute_c0_L0(XXI)
binary = False

if binary:
    tree0 = tree.BinaryTree(XXI,leaf_size,np.array([.5,.5]),np.array([1.,1.]))
else:
    tree0 = tree.BalancedTree(XXI,leaf_size,np.array([.5,.5]),np.array([1.,1.]))

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



lvl = tree0.nlevels-2
boxes = tree0.get_boxes_level(lvl)

box0 = boxes[0]
Ibox = tree0.get_box_inds(box0)
print("ancestor = ",compute_ancestor(box0,lvl))

boxesc = [box for box in boxes if box!=box0]
Ic = np.zeros(shape=(0,),dtype = np.int64)
for box in boxesc:
    Ic=np.append(Ic,tree0.get_box_inds(box))

boxes_far = [box for box in boxes if compute_ancestor(box,lvl)!=compute_ancestor(box0,lvl)]
Ifar = np.zeros(shape=(0,),dtype = np.int64)
for box in boxes_far:
    Ifar=np.append(Ifar,tree0.get_box_inds(box))

boxes_2 = [box for box in boxes if compute_ancestor(box,lvl)==2]
boxes_3 = [box for box in boxes if compute_ancestor(box,lvl)==3]
boxes_4 = [box for box in boxes if compute_ancestor(box,lvl)==4]
I2 = np.zeros(shape=(0,),dtype = np.int64)
I3 = np.zeros(shape=(0,),dtype = np.int64)
I4 = np.zeros(shape=(0,),dtype = np.int64)

for box in boxes_2:
    I2=np.append(I2,tree0.get_box_inds(box))
for box in boxes_3:
    I3=np.append(I3,tree0.get_box_inds(box))
for box in boxes_4:
    I4=np.append(I4,tree0.get_box_inds(box))

E = np.identity(n)
print("E formed")

#########################
#       c ranks
#########################

Sl = E[:,Ic].T@((st_l.A)@E[:,Ibox])
[_,s,_] = np.linalg.svd(Sl)
rk1 = sum(s>s[0]*1e-8)
print("shape//rk c = (",len(Ibox),",",len(Ic),")","//",rk1)

#########################
#       far ranks
#########################
Sl = E[:,Ifar].T@((st_l.A)@E[:,Ibox])
[_,s,_] = np.linalg.svd(Sl)
rk1 = sum(s>s[0]*1e-8)
print("shape//rk far = (",len(Ibox),",",len(Ifar),")","//",rk1)


#########################
#       2 ranks
#########################
Sl = E[:,I2].T@((st_l.A)@E[:,Ibox])
[_,s,_] = np.linalg.svd(Sl)
rk1 = sum(s>s[0]*1e-8)
print("shape//rk 2 = (",len(Ibox),",",len(I2),")","//",rk1)


#########################
#       3 ranks
#########################
Sl = E[:,I3].T@((st_l.A)@E[:,Ibox])
[_,s,_] = np.linalg.svd(Sl)
rk1 = sum(s>s[0]*1e-8)
print("shape//rk 3 = (",len(Ibox),",",len(I3),")","//",rk1)


#########################
#       4 ranks
#########################
Sl = E[:,I4].T@((st_l.A)@E[:,Ibox])
[_,s,_] = np.linalg.svd(Sl)
rk1 = sum(s>s[0]*1e-8)
print("shape//rk 4 = (",len(Ibox),",",len(I4),")","//",rk1)


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