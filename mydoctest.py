
import numpy as np

def l2g(i):
    return 2*i,i

test = np.array([l2g(i) for i in range(10)])
print(test.shape)