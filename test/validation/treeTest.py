import numpy as np
import matAssembly.HBS.simpleoctree.simpletree as tree
import matplotlib.pyplot as plt
from scipy.linalg import qr
from scipy.linalg import block_diag
from scipy.spatial import distance_matrix
from scipy.linalg import lstsq
import scipy.linalg
import time
import torch
import torch.linalg as tla



def qr_torch(x):
    q,r = tla.qr(x)
    return q , r


def qr_col_torch(x,k):
    q,r= tla.qr(x)
    return q[:,0:k]


x=np.linspace(0,1,64)
y=np.linspace(0,1,64)



nx = len(x)
ny = len(y)
XX = np.zeros(shape=(nx*ny,3))
XX[:,0] = np.kron(x,np.ones_like(y))
XX[:,1] = np.kron(np.ones_like(x),y)
YY = XX
YY[:,2] = 10



D = np.array(distance_matrix(XX,YY))
A=D

A = torch.from_numpy(A)

#np.ones(shape = (nx*ny,nx*ny))
#for ij in range(nx*ny):
#    A[ij,ij]=1.

print("shape A = ",A.shape)
nl = 4*4
t =  tree.BalancedTree(XX,nl)

print("NoL = ",t.nlevels)

rk  =  2*2
s   =   5*(rk+10)



startcompress = time.time()
startmul = time.time()

Om = torch.randn((A.shape[1],s),dtype=torch.float64)
Psi = torch.randn((A.shape[0],s),dtype=torch.float64)

Y = A@Om
Z = A.T@Psi

stopmul = time.time()
tMul = stopmul-startmul
print("multime = ",tMul,"s")
nc = 0

Omtot   = []
Psitot  = []
Ztot    = []
Ytot    = []

Om_list = []
Psi_list = []

Y_list = []
Z_list = []

U_list  = [[] for _ in range(t.nlevels)]
V_list  = [[] for _ in range(t.nlevels)]
D_list  = [[] for _ in range(t.nlevels)]

Om_list_new=[]
Om_list=[]
Psi_list_new=[]
Psi_list=[]
Y_list_new=[]
Y_list=[]
Z_list_new=[]
Z_list=[]
tQR=0
tnull=0
tD=0
for level in range(t.nlevels-1,-1,-1):
    boxes = t.get_boxes_level(level)
    shift = boxes[-1]+1
    for box in boxes:
        if level ==t.nlevels-1:
            idxs    = t.get_box_inds(box)
            Omtau   = Om[idxs,:]
            Psitau  = Psi[idxs,:]
            Ytau    = Y[idxs,:]
            Ztau    = Z[idxs,:]
        else:
            startaccum = time.time()
            children= t.get_box_children(box)
            Omtau   = torch.zeros(0,s)
            Psitau  = torch.zeros(0,s)
            Ytau    = torch.zeros(0,s)
            Ztau    = torch.zeros(0,s)
            for child in children:
                Omchild = Om_list[child-shift]
                Psichild = Psi_list[child-shift]
                Ychild = Y_list[child-shift]
                Zchild = Z_list[child-shift]
                Uchild = U_list[level+1][child-shift]
                Vchild = V_list[level+1][child-shift]
                Dchild = D_list[level+1][child-shift]

                Omtau=torch.cat((Omtau,Vchild.T@Omchild),dim=0)
                Psitau=torch.cat((Psitau,Uchild.T@Psichild),dim=0)

                Ytau=torch.cat((Ytau,Uchild.T@(Ychild-Dchild@Omchild)),dim=0)
                Ztau=torch.cat((Ztau,Vchild.T@(Zchild-Dchild.T@Psichild)),dim=0)
        Om_list_new+=[Omtau]
        Psi_list_new+=[Psitau]
        Y_list_new+=[Ytau]
        Z_list_new+=[Ztau]
        if level>0:
            if level==t.nlevels-1: 
                rk0 = t.leaf_size
            else:
                rk0 = rk
            start = time.time()
            [Qom,Rom] = qr_torch(Omtau.T)
            [Qpsi,Rpsi] = qr_torch(Psitau.T)
            stop=time.time()
            tnull+=stop-start
            start = time.time()
            YQ=(Ytau@Qom)
            ZQ=(Ztau@Qpsi)
            Utau = qr_col_torch(Ytau-YQ@Qom.T,rk0)
            Vtau = qr_col_torch(Ztau-ZQ@Qpsi.T,rk0)
            stop = time.time()
            tQR+=stop-start
            startD = time.time()

            YO=tla.solve(Rom,YQ.T).T
            ZP = tla.solve(Rpsi,ZQ.T).T
            Dtau = (YO-Utau@(Utau.T@YO))\
                +Utau@(Utau.T@(ZP-Vtau@(Vtau.T@ZP)).T)
            tD+=time.time()-startD
            U_list[level]+=[Utau]
            V_list[level]+=[Vtau]
            D_list[level]+=[Dtau]
        else:
            [Qom,Rom] = qr_torch(Omtau.T)
            YQ=(Ytau@Qom)
            Dtau=tla.solve(Rom,YQ.T).T
            D_list[level]+=[Dtau]
    Om_list=Om_list_new
    Om_list_new=[]
    Psi_list=Psi_list_new
    Psi_list_new=[]
    Y_list=Y_list_new
    Y_list_new=[]
    Z_list=Z_list_new
    Z_list_new=[]
    print("tQR at level",level," is ",tQR,"s")
    print("tnull at level",level," is ",tnull,"s")
stopcompress = time.time()
tCompress= stopcompress-startcompress
print("compression done in ",tCompress,"s")
print("tQr = ",tQR,"s")
print("tnull = ",tnull,"s")
print("tD = ",tD,"s")
sumT = tQR+tnull+tD+tMul
print("sum of times = ",sumT)
print("time in other stuff = ",tCompress-sumT)


D0 = block_diag(*D_list[0])
D1 = block_diag(*D_list[1])
D2 = block_diag(*D_list[2])
D3 = block_diag(*D_list[3])
D4 = block_diag(*D_list[4])

print("norm D0 = ",scipy.linalg.norm(D0))
print(tla.norm(D_list[0][0]))



V1 = block_diag(*V_list[1])
V2 = block_diag(*V_list[2])
V3 = block_diag(*V_list[3])
V4 = block_diag(*V_list[4])


U1 = block_diag(*U_list[1])
U2 = block_diag(*U_list[2])
U3 = block_diag(*U_list[3])
U4 = block_diag(*U_list[4])



Atest = U4@(U3@(U2@(U1@(D0@V1.T) + D1)@V2.T + D2)@V3.T+D3)@V4.T+D4

v=np.random.standard_normal(size=(A.shape[1],))
vperm=np.zeros_like(v)
Av = A@v
indtot = []
for leaf in t.get_leaves():
    indtot += t.get_box_inds(leaf).tolist()
vperm = v[indtot]
#E = np.identity(A.shape[0])
#Eperm = E[indtot,:]
#u=np.zeros_like(v)
#A0E = Atest@Eperm
#A0E[indtot,:] = A0E

vperm = Atest@vperm
vperm[indtot] = vperm

print("err = ",np.linalg.norm(Av-vperm)/np.linalg.norm(Av))

tree0 = t
qhat_list = [[] for _ in range(t.nlevels)]
uhat_list = [[] for _ in range(t.nlevels)]

def matvec(Ulist,Vlist,Dlist,v,tree):
    print("D0 norm in loop = ",tla.norm(Dlist[0][0]))
    # permute
    indtot = []
    for leaf in t.get_leaves():
        indtot += t.get_box_inds(leaf).tolist()
    vperm = v[indtot,:]
    
    n0 = 0
    qhat = torch.zeros(0,vperm.shape[1])
    #########################
    #       UPWARD PASS
    #########################
    for V in Vlist[tree.nlevels-1]:
        step = V.shape[0]
        qhat=torch.cat((qhat,V.T@vperm[n0:n0+step,:]),dim=0)
        n0+=step
    qhat_list[tree.nlevels-1]=qhat

    for level in range(tree.nlevels-2,0,-1):
        n0 = 0
        qhat = torch.zeros(0,vperm.shape[1])
        for V in Vlist[level]:
            step = V.shape[0]
            qhat=torch.cat((qhat,V.T@qhat_list[level+1][n0:n0+step,:]),dim=0)
            n0+=step
        qhat_list[level]=qhat

        
    #########################
    #       DOWNWARD PASS
    #########################

    u = torch.zeros(vperm.shape[0],vperm.shape[1])

    for level in range(0,tree.nlevels):
        n0 = 0
        if level==0:
            uhat=Dlist[0][0]@qhat_list[1]
            uhat_list[0]=uhat
            print("norm uhat = ",tla.norm(uhat))
        elif level<t.nlevels-1:
            print("level = ",level)
            uhat = torch.zeros(0,vperm.shape[1])
            n0 = 0
            for U in Ulist[level]:
                step = U.shape[1]
                uhat=torch.cat((uhat,U@uhat_list[level-1][n0:n0+step,:]),dim=0)#is this right?
                n0+=step
            n0l = 0
            stepl = 0
            n0r = 0
            stepr = 0
            for D in Dlist[level]:
                stepl = D.shape[0]
                stepr = D.shape[1]
                uhat[n0l:n0l+stepl,:]+=D@qhat_list[level+1][n0r:n0r+stepr,:]
                n0r+=stepr
                n0l+=stepl
            uhat_list[level]=uhat
            print("norm uhat = ",tla.norm(uhat))
        else:
            n0l=0
            n0r=0
            for U in Ulist[level]:
                stepl=U.shape[0]
                stepr=U.shape[1]
                u[n0l:n0l+stepl,:] = U@uhat_list[level-1][n0r:n0r+stepr,:]
                n0l+=stepl
                n0r+=stepr
            print("norm u = ",tla.norm(u))
            n0l=0
            n0r=0
            for D in Dlist[level]:
                stepl = D.shape[0]
                stepr = D.shape[1]
                u[n0l:n0l+stepl,:]+=D@vperm[n0r:n0r+stepr,:]
                n0r+=stepr
                n0l+=stepl
    return u
v=np.random.standard_normal(size=(Atest.shape[1],1))
v1 = torch.from_numpy(v)
u0 = Atest@v
print("norm D0 = ",tla.norm(D_list[0][0]))
u1 = matvec(U_list,V_list,D_list,v1,t)
u1 = u1.detach().cpu().numpy()
u1[indtot,:]=u1
print('matvec err = ',np.linalg.norm(u0-u1)/np.linalg.norm(u0))
print('u1 norm = ',np.linalg.norm(u1))