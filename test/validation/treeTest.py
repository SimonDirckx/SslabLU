import sys
sys.path.append('/home/simond/SslabLU')
import numpy as np
import matAssembly.HBS.simpleoctree.simpletree as tree
import matplotlib.pyplot as plt
from scipy.linalg import qr
from scipy.linalg import block_diag
from scipy.spatial import distance_matrix
from scipy.linalg import lstsq
import time

def null(B,k):
    [Q,R] = np.linalg.qr(B.T, mode='complete')
    n=Q.shape[1]
    return Q[:,n-k:n]
def qr_col(B,k):
    Q,R= np.linalg.qr(B, mode='reduced')
    return Q[:,0:k]


x=np.linspace(0,1,128)
y=np.linspace(0,1,128)



nx = len(x)
ny = len(y)
XX = np.zeros(shape=(nx*ny,3))
XX[:,0] = np.kron(x,np.ones_like(y))
XX[:,1] = np.kron(np.ones_like(x),y)
YY = XX
YY[:,2] = 1



D = np.array(distance_matrix(XX,YY))
A=D#np.ones(shape = (nx*ny,nx*ny))
#for ij in range(nx*ny):
#    A[ij,ij]=1.

print("shape A = ",A.shape)
nl = 16*16
t =  tree.BalancedTree(XX,nl)

print("NoL = ",t.nlevels)

rk  =  20*20
s   =   4*(rk+10)



startcompress = time.time()
startmul = time.time()
Om = np.random.standard_normal(size=(A.shape[1],s))
Psi = np.random.standard_normal(size=(A.shape[0],s))
stopmul = time.time()
print("multime = ",stopmul-startmul,"s")
Y = A@Om
Z = A.T@Psi
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
            Omtau   = np.zeros(shape=(0,s))
            Psitau  = np.zeros(shape=(0,s))
            Ytau    = np.zeros(shape=(0,s))
            Ztau    = np.zeros(shape=(0,s))
            for child in children:
                Omchild = Om_list[child-shift]
                Psichild = Psi_list[child-shift]
                Ychild = Y_list[child-shift]
                Zchild = Z_list[child-shift]
                Uchild = U_list[level+1][child-shift]
                Vchild = V_list[level+1][child-shift]
                Dchild = D_list[level+1][child-shift]

                Omtau=np.append(Omtau,Vchild.T@Omchild,axis=0)
                Psitau=np.append(Psitau,Uchild.T@Psichild,axis=0)

                Ytau=np.append(Ytau,Uchild.T@(Ychild-Dchild@Omchild),axis=0)
                Ztau=np.append(Ztau,Vchild.T@(Zchild-Dchild.T@Psichild),axis=0)
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
            #Ptau = null(Omtau,rk)
            [Qom,Rom] = np.linalg.qr(Omtau.T, mode='reduced')
            [Qpsi,Rpsi] = np.linalg.qr(Psitau.T, mode='reduced')
            stop=time.time()
            tnull+=stop-start
            start = time.time()
            YQ=(Ytau@Qom)
            ZQ=(Ztau@Qpsi)
            Utau = qr_col(Ytau-YQ@Qom.T,rk0)
            Vtau = qr_col(Ztau-ZQ@Qpsi.T,rk0)
            stop = time.time()
            tQR+=stop-start
            startD = time.time()

            YO=np.linalg.solve(Rom,YQ.T).T
            ZP = np.linalg.solve(Rpsi,ZQ.T).T
            Dtau = (YO-Utau@(Utau.T@YO))\
                +Utau@(Utau.T@(ZP-Vtau@(Vtau.T@ZP)).T)
            tD+=time.time()-startD
            U_list[level]+=[Utau]
            V_list[level]+=[Vtau]
            D_list[level]+=[Dtau]
        else:
            Dtau = Ytau@np.linalg.pinv(Omtau)
            D_list[level]+=[Dtau]
    Om_list=Om_list_new
    Om_list_new=[]
    Psi_list=Psi_list_new
    Psi_list_new=[]
    Y_list=Y_list_new
    Y_list_new=[]
    Z_list=Z_list_new
    Z_list_new=[]
stopcompress = time.time()
print("compression done in ",stopcompress-startcompress,"s")
print("tQr = ",tQR,"s")
print("tnull = ",tnull,"s")
print("tD = ",tD,"s")


D0 = block_diag(*D_list[0])
D1 = block_diag(*D_list[1])
D2 = block_diag(*D_list[2])
D3 = block_diag(*D_list[3])
#D4 = block_diag(*D_list[4])


V1 = block_diag(*V_list[1])
V2 = block_diag(*V_list[2])
V3 = block_diag(*V_list[3])
#V4 = block_diag(*V_list[4])


U1 = block_diag(*U_list[1])
U2 = block_diag(*U_list[2])
U3 = block_diag(*U_list[3])
#U4 = block_diag(*U_list[4])



Atest = U3@(U2@(U1@(D0@V1.T) + D1)@V2.T + D2)@V3.T+D3

v=np.random.standard_normal(size=(A.shape[1],))
vperm=np.zeros_like(v)
indtot = []
for leaf in t.get_leaves():
    indtot += t.get_box_inds(leaf).tolist()
vperm = v[indtot]
E = np.identity(A.shape[0])
Eperm = E[indtot,:]
u=np.zeros_like(v)
A0E = Atest@Eperm
A0E[indtot,:] = A0E
print("err = ",np.linalg.norm(A-A0E)/np.linalg.norm(A))


'''


tree0 = t
qhat_list = [[] for _ in range(t.nlevels)]
uhat_list = [[] for _ in range(t.nlevels)]

def matvec(Ulist,Vlist,Dlist,v):
    u = np.zeros_like(v)
    #upward pass

    for level in range(tree0.nlevels-1,0,-1):
        boxes = tree0.get_boxes_level(level)
        shift = tree0.get_boxes_level(level-1)[-1]+1
        for box in boxes:
            if level==tree0.nlevels-1:
                V = Vlist[level][box-shift]
                qhat = V.T@v[tree0.get_box_inds(box)]
            else:
                children = t.get_box_children(box)
                print("children = ",children)
                shift_child = tree0.get_boxes_level(level)[-1]+1
                print("shift_children = ",shift_child)
                qhat = np.zeros(shape=(0,))
                for child in children:
                    qhat = np.append(qhat,qhat_list[level+1][child-shift_child],axis=0)
                print("qhat shape",qhat.shape)
                qhat = Vlist[level][box-shift].T@qhat
            qhat_list[level]+=[qhat]
    #downward pass
    for level in range(tree0.nlevels):
        boxes = tree0.get_boxes_level(level)
        shift = tree0.get_boxes_level(level)[-1]+1

        for box in boxes:
            children = t.get_box_children(box)
            if level==0:
                uhat = np.zeros(shape=(0,))
                for child in children:
                    uhat = np.append(uhat,qhat_list[level+1][child-shift],axis=0)
                uhat = Dlist[level][box]@uhat
                n0=0
                for child in children:
                    step = qhat_list[level+1][child-shift].shape[0]
                    uhat_list[level+1] += [uhat[n0:n0+step]]
                    n0 +=step
                
            elif level<tree0.nlevels-1:
                boxshift = tree0.get_boxes_level(level-1)[-1]+1
                uhat = np.zeros(shape=(0,))
                for child in children:
                    uhat = np.append(uhat,qhat_list[level+1][child-shift],axis=0)
                uhat = Ulist[level][box-boxshift]@uhat_list[level][box-boxshift]+Dlist[level][box-boxshift]@uhat
                n0=0
                for child in children:
                    step = qhat_list[level+1][child-shift].shape[0]
                    uhat_list[level+1] += [uhat[n0:n0+step]]
                    n0 +=step
            else:
                inds = tree0.get_box_inds(box)
                boxshift = tree0.get_boxes_level(level-1)[-1]+1
                u[inds] = Ulist[level][box-boxshift]@uhat_list[level][box-boxshift]+Dlist[level][box-boxshift]@v[inds]
    return u
v=np.random.standard_normal(size=(Atest.shape[1],))
u0 = Atest@v
u1 = matvec(U_list,V_list,D_list,v)
print('matvec err = ',np.linalg.norm(u0-u1)/np.linalg.norm(u0))
'''