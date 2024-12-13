import numpy as np

class Box:
    def __init__(self,bnds):
        self.bnds=bnds
        self.ndim = len(bnds[0])
    def isBoundary(self,p):
        p0  = p-np.array(self.bnds[0])
        p1  = np.array(self.bnds[1])-np.array(self.bnds[0])
        b1  =   any(np.abs(p0)<1e-13) or any(np.abs(p0-p1)<1e-13)
        b2  =   any(p0<-1e13) or any(p0>p1+1e-13) 
        return b1 and not b2
    
class unitSquare(Box):
    def __init__(self):
        super(unitSquare,self).__init__([[0,0],[1,1]])

class unitCube:
    def __init__(self):
        super(unitCube,self).__init__([[0,0,0],[1,1,1]])

class Circle:
    def __init__(self,r):
        self.r=r
        self.ndim=2
    def isBoundary(self,p):
        return np.abs(np.linalg.norm(p)-self.r)<1e-14

class unitCircle(Circle):
    def __init__(self):
        super(unitCircle,self).__init__(1.)

class Sphere:
    def __init__(self,r):
        self.r=r
        self.ndim=2
    def isBoundary(self,p):
        return np.abs(np.linalg.norm(p)-self.r)<1e-14

class unitSphere(Sphere):
    def __init__(self):
        super(unitSphere,self).__init__(1.)
    