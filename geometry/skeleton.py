import numpy as np
import abc
from   abc              import ABCMeta, abstractmethod, abstractproperty
from geometry.standardGeometries import unitSquare
from geometry.standardGeometries import Box
import geometry.slabGeometry as slab
from solver.solver import solverWrapper as solverWrap
import pdo.pdo as pdo
import multiSlab as MS
class abstractSkeleton(ABCMeta):
    @abstractproperty
    def paramList(self):
        pass
    @abstractproperty
    def C(self):
        pass
    @abstractproperty
    def Om(self):
        pass
    @abstractproperty
    def globIdxs(self):
        pass
    @abstractmethod
    def localGeom(self,i,j):
        """method that constructs local parametrization from global, in between slabs"""
        pass
    @abstractmethod
    def localdGeom(self,i,j,k):
        """method that constructs local parametrization from global, in between slabs, centered on j"""
        pass

    @abstractmethod
    def setGlobalIdxs(self,globIdxs:list[range]):
        pass

class standardBoxSkeleton:

    def __init__(self,Om:Box,N):
        self.H=(Om.bnds[1][0]-Om.bnds[0][0])/(1.*N+1.)
        self.d0=np.array(Om.bnds[0])
        self.Om=Om
        self.ndim = Om.ndim
        C=[]
        paramList = []
        j=0
        for i in range(N):
            C+=[[j-1,j+1]]
            j+=1
            paramList+=[j*self.H]
            
        C[N-1][1]=-1
        self.C=C
        self.paramList=paramList

    def l2g_factory(self,delta):
        def l2g(p):
            if p.ndim>1:
                return np.array(p+np.tile(delta,(p.shape[0],1)))
            else:
                return np.array(p+delta)
        return l2g
    
    def localGeom(self,i,j):
        if i==-1:
            H = self.paramList[j]-self.Om.bnds[0][0]
            d = 0
        elif j == -1:
            H = self.Om.bnds[1][0]-self.paramList[i]
            d=self.paramList[i]
        else:
            H = self.paramList[j]-self.paramList[i]
            d=self.paramList[i]
        delta=[x for x in self.d0]#deep copy needed for some
        
        delta[0]=delta[0]+d
        if self.ndim==2:
            bounds=[[0,0],[H,self.Om.bnds[1][1]-self.Om.bnds[0][1]]]
        elif self.ndim==3:
            bounds=[[0,0,0],[H,self.Om.bnds[1][1]-self.Om.bnds[0][1],self.Om.bnds[1][2]-self.Om.bnds[0][2]]]
        else:
            raise(ValueError("ndim must be 2 or 3"))
        return slab.boxSlab(self.l2g_factory(delta),bounds,self.Om)
    def localdGeom(self,i,j,k):
        if i==-1:
            Hl = self.paramList[j]-self.Om.bnds[0][0]
        else:
            Hl = self.paramList[j]-self.paramList[i]
        if k == -1:
            Hr = self.Om.bnds[1][0]-self.paramList[j]
        else:    
            Hr = self.paramList[k]-self.paramList[j]
        d=self.paramList[j]
        delta=[x for x in self.d0]#deep copy needed for some
        
        delta[0]=delta[0]+d
        if self.ndim==2:
            bounds=[[-Hl,0],[Hr,self.Om.bnds[1][1]-self.Om.bnds[0][1]]]
        elif self.ndim==3:
            bounds=[[-Hl,0,0],[Hr,self.Om.bnds[1][1]-self.Om.bnds[0][1],self.Om.bnds[1][2]-self.Om.bnds[0][2]]]
        else:
            raise(ValueError("ndim must be 2 or 3"))
        return slab.boxSlab(self.l2g_factory(delta),bounds,self.Om)
    
    def setGlobalIdxs(self,globIdxs:list[range]):
        assert(len(globIdxs)==len(self.paramList))
        self.globIdxs = globIdxs

def computeUniformGlobalIdxs(sk:abstractSkeleton,opts):
    globIdxs=[]
    ctr = 0
    for i in range(len(sk.paramList)):
        globIdxs+=[range(ctr,ctr+opts.nyz)]
        ctr+=opts.nyz
    return globIdxs

def buildSlabs(sk:abstractSkeleton,pde:pdo,opts,overlapping):
    slablist=[]
    C=sk.C
    prm=sk.paramList
    for i in range(len(prm)):
        cL,cR=C[i]
        if not overlapping:
            if cL>-2:
                gloc = sk.localGeom(cL,i)
                solverS = solverWrap(opts)
                solverS.construct(gloc,pde)
                slabi = MS.Slab(gloc,pde,solverS,'DtN')
                if not cL==-1:
                    slabi.setGlobalIdxs(sk.globIdxs[i],1)
                    slabi.setGlobalIdxs(sk.globIdxs[cL],0)
                    C[cL][1]=-2
                else:
                    slabi.setGlobalIdxs(sk.globIdxs[i],1)
                slablist+=[slabi]

            if cR>-2:
                gloc = sk.localGeom(i,cR)
                solverS = solverWrap(opts)
                solverS.construct(gloc,pde)
                slabi = MS.Slab(gloc,pde,solverS,'DtN')
                if not cR==-1:
                    slabi.setGlobalIdxs(sk.globIdxs[i],0)
                    slabi.setGlobalIdxs(sk.globIdxs[cR],1)
                    C[cR][0]=-2
                else:
                    slabi.setGlobalIdxs(sk.globIdxs[i],0)
                slablist+=[slabi]
        else:
            gloc = sk.localdGeom(cL,i,cR)
            solverS = solverWrap(opts)
            solverS.construct(gloc,pde)
            slabi = MS.Slab(gloc,pde,solverS,'DtD')
            if cL>-1:
                slabi.setGlobalIdxs(sk.globIdxs[cL],0)
            if cR>-1:
                slabi.setGlobalIdxs(sk.globIdxs[cR],2)
            slabi.setGlobalIdxs(sk.globIdxs[i],1)
            slablist+=[slabi]
    return slablist
        