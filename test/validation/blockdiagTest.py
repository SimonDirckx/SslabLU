import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import numpy as np
import matplotlib.pyplot as plt


Slvec = []
Srvec = []
n = 10
Nds = 7
for i in range(Nds):
    Sl = np.random.standard_normal(size=(n,n))
    Sr = np.random.standard_normal(size=(n,n))
    if i>0:
        Slvec +=[Sl]
    if i<Nds-1:
        Srvec +=[Sr]
#SL = tuple(Slvec)
#SR = tuple(Slvec)


Stot = sparse.bmat([[Slvec[i] if i == j-1 else np.eye(n) if i==j else Srvec[i-1] if i==j+1
                else None for i in range(Nds)]
                for j in range(Nds)], format='bsr')
Stot = Stot.todense()

Sl0 = Stot[10:20,:][:,0:10]
Sr0 = Stot[:10,:][:,10:20]
print("block err = ",np.linalg.norm(Sl0-Slvec[0]))
print("==============================================")
print("block err = ",np.linalg.norm(Sr0-Srvec[0]))


plt.figure(1)
plt.spy(Stot)
plt.show()