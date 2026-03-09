import numpy as np

# XX assumed 2D
def construct_perm(XXJ,J,nl,dir=0):
    if XXJ.shape[0]<=nl:
        return J
    else:
        c = np.mean(XXJ,axis=0)
        J1 = np.where(XXJ[:,dir]<c[dir])[0]
        J2 = np.where(XXJ[:,dir]>=c[dir])[0]
        return np.append( construct_perm(XXJ[J1,:],J[J1],nl,(1+dir)%2) , construct_perm(XXJ[J2,:],J[J2],nl,(1+dir)%2) )
class stree:
    def __init__(self,XX,nl):
        if XX.shape[1]==3:
            self.XX = XX[:,1:3]
        else:
            self.XX = XX
        self.perm_leaf = construct_perm(self.XX,np.arange(self.XX.shape[0]),nl)
        self.nleaves = XX.shape[0]//nl
        self.nlevels = int(np.log2(self.nleaves))+1
        self.Nbvec = []
        Nb = self.nleaves
        for i in range(self.nlevels):
            self.Nbvec+=[Nb]
            Nb=Nb//2
        print("Nbvec = ", self.Nbvec)
        
