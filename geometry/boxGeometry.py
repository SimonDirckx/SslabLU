import numpy as np
class boxGeometry:
    def __init__(self,bounds):
        self.bounds=bounds
    def isBoundary(self,p,tol = 1e-13):
        return any(np.abs(p-self.bounds[0])<tol) or any(np.abs(p-self.bounds[1])<tol)