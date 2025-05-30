import numpy as np
import scipy.linalg as splinalg
import scipy as sc

A=np.random.standard_normal(size=(10,10))
B=np.random.standard_normal(size=(10,10))

AB1 = np.kron(A,B)
AB2 = splinalg.kron(A,B)
sub1 = AB1[range(5)][:,range(5)]
sub2 = AB2[range(5)][:,range(5)]

print("diff = ",np.linalg.norm(sub1-sub2))