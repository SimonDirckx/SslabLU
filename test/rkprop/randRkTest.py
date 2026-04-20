import numpy as np
import numpy.linalg as nla

n = 1000
k = 20
A = np.random.standard_normal(size = (n,k))
B = np.random.standard_normal(size = (n,k))
M = A@B.T+1e-12*np.random.standard_normal(size = (n,n))

Om = np.random.standard_normal(size=(n,k))

q,r = nla.qr(M@Om,mode='reduced')
C = q.T@M
[U,s,Vh] = nla.svd(C,full_matrices=False)
U = q@U
Mapprox = (U*s)@Vh
print("AtA err = ",nla.norm(Mapprox-M)/nla.norm(M))