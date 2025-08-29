import numpy as np
import jax.numpy as jnp
import solver.hpsmultidomain.hpsmultidomain.pdo as pdo
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
import torch

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

def compute_T(I,J,XXb,solver):
        A_solver = solver.solver_ii    
        def smatmat(v,I,J,transpose=False):
            if (v.ndim == 1):
                v_tmp = v[:,np.newaxis]
            else:
                v_tmp = v

            if (not transpose):
                result = (solver.Abb[I,:][:,J]@v_tmp)-(solver.Abi[I,:]@(A_solver@(solver.Aib[:,J]@v_tmp)))
            else:
                result      = ((solver.Abb[I,:][:,J]).T@v_tmp)-((solver.Aib[:,J]).T@(A_solver.T@((solver.Abi[I,:]).T@v_tmp)))
            if (v.ndim == 1):
                result = result.flatten()
            return result

        Linop_l = LinearOperator(shape=(len(J),len(I)),\
            matvec = lambda v:smatmat(v,J,I), rmatvec = lambda v:smatmat(v,J,I,transpose=True),\
            matmat = lambda v:smatmat(v,J,I), rmatmat = lambda v:smatmat(v,J,I,transpose=True))
        
        T = stMap(Linop_l,XXb[I,:],XXb[J,:])
        return T



nwaves = 6.24
kh = (nwaves/4)*2.*np.pi
print("kappa = ",kh)
def c11(p):
    return torch.ones_like(p[:,0])
def c22(p):
    return torch.ones_like(p[:,0])
def c33(p):
    return torch.ones_like(p[:,0])
def bfield(p):
    return -kh*kh*jnp.ones_like(p[:,0])
helmholtz = pdo.PDO_3d(c11=c11,c22=c22,c33=c33,c=pdo.const(-kh*kh))

bnds = [[0.,0.,0.],[1.,1.,1.]]
box_geom   = np.array(bnds)

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
    return np.ones_like(p[:,0])

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
binary = False
pvec = [6,8,10]
a = [H/8.,1/32,1/32]
nlvl = int(np.log2(1/(2*a[1])))
if binary:
    nlvl *=2
nlvl+=1
Hvec = [.25,.125,.0625,.03125]
coarsestLevel=3
mode = 'p'
if mode == 'H':
    rkWeak=np.zeros(shape=(len(Hvec),nlvl-coarsestLevel),dtype = np.int64)
    rkStrong=np.zeros(shape=(len(Hvec),nlvl-coarsestLevel),dtype=np.int64)
    indvec = Hvec
else:
    rkWeak=np.zeros(shape=(len(pvec),nlvl-coarsestLevel),dtype = np.int64)
    rkStrong=np.zeros(shape=(len(pvec),nlvl-coarsestLevel),dtype=np.int64)
    indvec=pvec
form = 'S'
#form = 'T'
ll = False
lr = not ll
p=10
H=1/8 
for ind in range(len(indvec)):
    if mode=='p':
        p = pvec[ind]
    else:
        H=Hvec[ind]
    a= np.array([H/8.,1/32,1/32])
    print("ppw = ",np.array( [p/a[0] , p*(2/a[1]) , p*(2/a[2]) ] )/nwaves)
    
    opts = solverWrap.solverOptions('hpsalt',[p+2,p+2,p+2],a)
    if form == 'S':
        geom    = np.array([[0.,0.,0.],[2*H,1.,1.]])
        solver  = solverWrap.solverWrapper(opts)
        solver.construct(geom,helmholtz,verbose=True)
    elif form == 'T':
        geom    = np.array([[0.,0.,0.],[H,1.,1.]])
        solver  = solverWrap.solverWrapper(opts)
        solver.construct(geom,helmholtz,verbose=True)
    else:
        raise ValueError('form must be S or T')
    XX = solver.XX
    XXb = XX[solver.Ib,:]
    XXi = XX[solver.Ii,:]
    delta = np.abs(XX[0,0]-XX[2*p*p,0])
    xl = geom[0][0]
    xr = geom[1][0]
    xc=(xl+xr)/2.
    print("\t SLAB BOUNDS xl,xc,xr=",xl,",",xc,",",xr)
    print("\t delta = ",delta)
    delta = 2e-14
    Il = np.where(np.abs(XXb[:, 0] - xl) < delta/2)[0]
    Ir = np.where(np.abs(XXb[:, 0] - xr) < delta/2)[0]
    Ic = np.where(np.abs(XXi[:, 0] - xc) < delta/2)[0]
    Igb = np.where(gb_vec(XXb))[0]
    
    print("\t SLAB dofs = ",len(Ic))
    if form=='S':
        st,_ = compute_stmaps(Il,Ic,Ir,XXi,XXb,solver)
    else:
        if ll:
            st = compute_T(Il,Il,XXb,solver)
        if lr:
            st = compute_T(Il,Ir,XXb,solver)
    n=len(Il)

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
        tree0 = tree.BinaryTree(XXB,leaf_size,np.array([.5,.5]),np.array([1.+1e-5,1.+1e-5]))
    else:
        tree0 = tree.BalancedTree(XXB,leaf_size,np.array([.5,.5]),np.array([1.+1e-5,1.+1e-5]))

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

    tol_rk = 1e-5

    for lvl in range(tree0.nlevels-1,coarsestLevel-1,-1):
        print("=================lvl ",lvl,"=================")
        boxes = tree0.get_boxes_level(lvl)
        indBox=0
        while compute_ancestor(boxes[indBox],lvl)==1:
            indBox+=1
        indBox -=1
        box0 = boxes[indBox]
        Ibox = tree0.get_box_inds(box0)
        print("\t box0=%d ancestor = %d" %(box0,compute_ancestor(box0,lvl)))

        boxesc = [box for box in boxes if box!=box0]
        Ic = np.zeros(shape=(0,),dtype = np.int64)
        for box in boxesc:
            Ic=np.append(Ic,tree0.get_box_inds(box))

        boxes_far = [box for box in boxes if far(box0,box)]
        Ifar = np.zeros(shape=(0,),dtype = np.int64)
        for box in boxes_far:
            Ifar=np.append(Ifar,tree0.get_box_inds(box))
        E = np.identity(n)
        tic = time()
        tmp = np.random.randn(n,100)
        st.A @ tmp
        toc = time() - tic
        print("\t Toc solve PDE on double slab for %d rhs %5.2f s" % (tmp.shape[-1],toc))
        #########################
        #       c ranks
        #########################
        print("RANK RESULTS at tol %.2e" %(tol_rk))
        Sl = ((st.A)@E[:,Ibox])[Ic,:]
        #[_,s,_] = np.linalg.svd(Sl)
        #rk1 = sum(s>s[0]*tol_rk)
        rk1 = np.linalg.matrix_rank(Sl,1e-5)
        print("\t shape//rk c   = ",Sl.shape,"//",rk1)
        rkWeak[ind,lvl-coarsestLevel] = rk1

        #########################
        #       far ranks
        #########################
        Sl = ((st.A)@E[:,Ibox])[Ifar,:]
        #[_,s,_] = np.linalg.svd(Sl)
        #rk1 = sum(s>s[0]*tol_rk)
        rk1 = np.linalg.matrix_rank(Sl,1e-5)
        print("\t shape//rk far = ",Sl.shape,"//",rk1)
        rkStrong[ind,lvl-coarsestLevel] = rk1
print("rkWeak = ",rkWeak)
print("rkStrong = ",rkStrong)

if mode == 'H':
    rkWeak = np.append(np.reshape(Hvec,shape=(len(Hvec),1)),rkWeak,axis=1)
    rkStrong=np.append(np.reshape(Hvec,shape=(len(Hvec),1)),rkStrong,axis=1)
else:
    rkWeak=np.append(np.reshape(pvec,shape=(len(pvec),1)),rkWeak,axis=1)
    rkStrong=np.append(np.reshape(pvec,shape=(len(pvec),1)),rkStrong,axis=1)
fileStrWeak = 'rkWeak'+form
fileStrStrong='rkStrong'+form
if ll and form=='T':
    fileStrWeak+='ll'
    fileStrStrong+='ll'
if lr and form=='T':
    fileStrWeak+='lr'
    fileStrStrong+='lr'
fileStrWeak+=mode+'.csv'
fileStrStrong+=mode+'.csv'
header = mode
for i in range(coarsestLevel,tree0.nlevels):
    header+=',lvl'+str(i)
with open(fileStrWeak,'w') as fweak:
    fweak.write(header+'\n')
    np.savetxt(fweak,rkWeak,delimiter=',',fmt='%d')
with open(fileStrStrong,'w') as fstrong:
    fstrong.write(header+'\n')
    np.savetxt(fstrong,rkStrong,delimiter=',',fmt='%d')
'''
plt.figure(1)
labelweak = []
labelstrong = []
for i in range(2,nlvl):
    labelweak+=["weak lvl "+str(i)]
    labelstrong+=["strong lvl "+str(i)]
plt.plot(Hvec,rkWeak,label=labelweak)
plt.gca().set_prop_cycle(None)
plt.plot(Hvec,rkStrong,label=labelstrong,linestyle='dashed')
plt.legend()
plt.xticks(pvec)
plt.show()
'''

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
