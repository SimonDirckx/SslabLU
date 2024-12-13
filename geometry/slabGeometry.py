import numpy as np
from   abc   import ABCMeta, abstractmethod, abstractproperty

class slabGeometry(metaclass=ABCMeta):

    """
    
    A abstract class used to represent the geometry of a local slab
    
    """
    
    @abstractmethod
    def isLocalBoundary(self,p,tol=1e-13):
        pass
    @abstractmethod
    def l2g(self,p):
        pass
    @abstractmethod
    def isGlobalBoundary(self,p):
        pass
    @abstractmethod
    def getIlIcIr(self,XX:np.array):
        pass
    @abstractmethod
    def isLocalBoundary(self,p):
        pass
    @abstractproperty
    def ndim(self):
        pass
    
    
    
class boxSlab(slabGeometry):

    def __init__(self,l2g,bounds,globalGeometry):
         self._l2g              = l2g
         self.bounds            = bounds
         self.globalGeometry    = globalGeometry
         self._ndim = np.array(self.bounds).shape[-1]
    def ndim(self):
        return self._ndim
    def l2g(self,p):
        return self._l2g(p)
    
    def isLocalBoundary(self,p,tol=1e-13):
        return any(np.abs(p-self.bounds[0])<tol) or any(np.abs(p-self.bounds[1])<tol)
    
    def isGlobalBoundary(self,p):
        return self.globalGeometry.isBoundary((self.l2g(p)))
    
    def getIlIcIr(self,XXi,XXb,tol = 1e-13):
        xl = self.bounds[0][0]
        xr = self.bounds[1][0]
        Il=[i for i in range(XXb.shape[0]) if not self.isGlobalBoundary(XXb[i,:]) and np.abs(XXb[i,0]-xl)<tol]
        Ir=[i for i in range(XXb.shape[0]) if not self.isGlobalBoundary(XXb[i,:]) and np.abs(XXb[i,0]-xr)<tol]
        IGB=[i for i in range(XXb.shape[0]) if self.isGlobalBoundary(XXb[i,:])]
        Ic=[i for i in range(XXi.shape[0]) if np.abs(XXi[i,0]-.5*(xl+xr))<tol]
        if len(Il)==0:
            Il=[]
        if len(Ir)==0:
            Ir=[]
        if len(Ic)==0:
            Ic=[]
        return Il,Ic,Ir,IGB