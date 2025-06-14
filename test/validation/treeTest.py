import numpy as np
import matAssembly.HBS.simpleoctree.simpletree as tree
import matAssembly.HBS.HBSTreeNEW as HBS
import matplotlib.pyplot as plt
from scipy.linalg import qr
from scipy.linalg import block_diag
from scipy.spatial import distance_matrix
from scipy.linalg import lstsq
import scipy.linalg
import time
import torch
import torch.linalg as tla


x=np.linspace(0,1,128)
#y=np.linspace(0,1,128)



nx = len(x)
#ny = len(y)
XX = np.zeros(shape=(nx,2))
XX[:,0] = x
YY = XX
YY[:,1] = .125



D = np.array(distance_matrix(XX,YY))
A=1./D

#A = torch.from_numpy(A)

for ij in range(nx):
    A[ij,ij]=1.

print("shape A = ",A.shape)
nl = 4
t =  tree.BalancedTree(XX,nl)
print("tree stats:")
print("nlevels = :",t.nlevels)
for i in range(t.nlevels):
    boxes = t.get_boxes_level(i)
    print("level ",i," = ",boxes)

print("NoL = ",t.nlevels)

rk  =  12
s   =   3*(rk+10)



startcompress = time.time()
startmul = time.time()

Om = torch.randn((A.shape[1],s),dtype=torch.float64)
Psi = torch.randn((A.shape[0],s),dtype=torch.float64)

Y = torch.from_numpy(A)@Om
Z = torch.from_numpy(A).T@Psi

stopmul = time.time()
tMul = stopmul-startmul

hbs_mat = HBS.HBSMAT(t,Om,Psi,Y,Z,rk)


v=np.random.standard_normal(size = (A.shape[1],1))
start = time.time()
u = hbs_mat.matvec(v)
stop = time.time()
tMatVec = stop-start
start = time.time()
Av = A@v
stop = time.time()
torig = stop-start

start = time.time()
uT = hbs_mat.matvecT(v)
stop = time.time()
tMatVecT = stop-start
start = time.time()
ATv = A.T@v
stop = time.time()
torigT = stop-start

print("err = ",np.linalg.norm(Av-u)/np.linalg.norm(Av))
print("tMatVec = ",tMatVec)
print("torig = ",torig)

print("err = ",np.linalg.norm(ATv-uT)/np.linalg.norm(ATv))
print("tMatVec = ",tMatVecT)
print("torig = ",torigT)


print(Av)
print(u)