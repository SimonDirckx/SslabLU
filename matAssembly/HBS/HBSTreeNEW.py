import numpy as np
import matAssembly.HBS.simpleoctree.simpletree as tree
import matplotlib.pyplot as plt
from scipy.linalg import qr
from scipy.linalg import block_diag
from scipy.spatial import distance_matrix
from scipy.linalg import lstsq
import scipy.linalg
import time
import torch
import torch.linalg as tla



def qr_torch(x):
    q,r = tla.qr(x)
    return q , r


def qr_col_torch(x,k):
    q,r= tla.qr(x)
    return q[:,0:k]

def null_torch(x,k):
    q,r= tla.qr(x.T,mode='complete')
    n = q.shape[1]
    return q[:,n-k:n]



class HBSMAT:

    def __init__(self,tree,Om,Psi,Y,Z,rk):
        self.U_list  = [[] for _ in range(tree.nlevels)]
        self.V_list  = [[] for _ in range(tree.nlevels)]
        self.D_list  = [[] for _ in range(tree.nlevels)]
        self.perm = []
        for leaf in tree.get_leaves():
            self.perm += tree.get_box_inds(leaf).tolist()
        self.construct(tree,Om,Psi,Y,Z,rk)
        self.nbytes=sum([sum([U.nbytes for U in self.U_list[i]]) for i in range(len(self.U_list))])+\
                    sum([sum([V.nbytes for V in self.V_list[i]]) for i in range(len(self.V_list))])+\
                    sum([sum([D.nbytes for D in self.D_list[i]]) for i in range(len(self.D_list))])       
    
    def construct(self,t,Om,Psi,Y,Z,rk):
        s = Om.shape[1]
        Om_list_new=[]
        Psi_list_new=[]
        Y_list_new=[]
        Z_list_new=[]
        data_type = Om.dtype
        for level in range(t.nlevels-1,-1,-1):
            boxes = t.get_boxes_level(level)
            shift = boxes[-1]+1
            for box in boxes:
                if level ==t.nlevels-1:
                    idxs    = t.get_box_inds(box)
                    Omtau   = Om[idxs,:]
                    Psitau  = Psi[idxs,:]
                    Ytau    = Y[idxs,:]
                    Ztau    = Z[idxs,:]
                else:
                    children= t.get_box_children(box)
                    Omtau   = torch.zeros((0,s),dtype=data_type)
                    Psitau  = torch.zeros((0,s),dtype=data_type)
                    Ytau    = torch.zeros((0,s),dtype=data_type)
                    Ztau    = torch.zeros((0,s),dtype=data_type)
                    for child in children:
                        Omchild = Om_list[child-shift]
                        Psichild = Psi_list[child-shift]
                        Ychild = Y_list[child-shift]
                        Zchild = Z_list[child-shift]
                        Uchild = self.U_list[level+1][child-shift]
                        Vchild = self.V_list[level+1][child-shift]
                        Dchild = self.D_list[level+1][child-shift]

                        Omtau=torch.cat((Omtau,Vchild.T@Omchild),dim=0)
                        Psitau=torch.cat((Psitau,Uchild.T@Psichild),dim=0)

                        Ytau=torch.cat((Ytau,Uchild.T@(Ychild-Dchild@Omchild)),dim=0)
                        Ztau=torch.cat((Ztau,Vchild.T@(Zchild-Dchild.T@Psichild)),dim=0)
                Om_list_new+=[Omtau]
                Psi_list_new+=[Psitau]
                Y_list_new+=[Ytau]
                Z_list_new+=[Ztau]
                if level>0:
                    rk0 = rk
                    
                    Ptau = null_torch(Omtau,rk0)
                    Qtau = null_torch(Psitau,rk0)
                    Utau = qr_col_torch(Ytau@Ptau,rk0)
                    Vtau = qr_col_torch(Ztau@Qtau,rk0)
                    
                    YO=Ytau@tla.pinv(Omtau)
                    ZP = Ztau@tla.pinv(Psitau)
                    Dtau = (YO-Utau@(Utau.T@YO))\
                        +Utau@(Utau.T@((ZP-Vtau@(Vtau.T@ZP)).T))
                    self.U_list[level]+=[Utau]
                    self.V_list[level]+=[Vtau]
                    self.D_list[level]+=[Dtau]
                else:
                    Dtau=Ytau@tla.pinv(Omtau)
                    self.D_list[level]+=[Dtau]
            Om_list=Om_list_new
            Om_list_new=[]
            Psi_list=Psi_list_new
            Psi_list_new=[]
            Y_list=Y_list_new
            Y_list_new=[]
            Z_list=Z_list_new
            Z_list_new=[]
        del Om_list,Om_list_new,Psi_list,Psi_list_new,Y_list,Y_list_new,Z_list,Z_list_new

    def matvec(self,v):
        nlevels = len(self.D_list)
        qhat_list = [[] for _ in range(nlevels)]
        uhat_list = [[] for _ in range(nlevels)]
        if v.ndim == 1:
            vperm = v[:,np.newaxis]
        else:
            vperm = v[self.perm,:]
        vperm = torch.from_numpy(vperm)
        n0 = 0
        qhat = torch.zeros((0,vperm.shape[1]),dtype = vperm.dtype)
        
        #########################
        #       UPWARD PASS
        #########################
        for V in self.V_list[nlevels-1]:
            step = V.shape[0]
            qhat=torch.cat((qhat,V.T@vperm[n0:n0+step,:]),dim=0)
            n0+=step
        qhat_list[nlevels-1]=qhat

        for level in range(nlevels-2,0,-1):
            n0 = 0
            qhat = torch.zeros(0,vperm.shape[1])
            for V in self.V_list[level]:
                step = V.shape[0]
                qhat=torch.cat((qhat,V.T@qhat_list[level+1][n0:n0+step,:]),dim=0)
                n0+=step
            qhat_list[level]=qhat

            
        #########################
        #       DOWNWARD PASS
        #########################

        uperm = torch.zeros((vperm.shape[0],vperm.shape[1]),dtype=vperm.dtype)

        for level in range(0,nlevels):
            n0 = 0
            if level==0:
                uhat=self.D_list[0][0]@qhat_list[1]
                uhat_list[0]=uhat
            elif level<nlevels-1:
                uhat = torch.zeros(0,vperm.shape[1])
                n0 = 0
                for U in self.U_list[level]:
                    step = U.shape[1]
                    uhat=torch.cat((uhat,U@uhat_list[level-1][n0:n0+step,:]),dim=0)
                    n0+=step
                n0l = 0
                stepl = 0
                n0r = 0
                stepr = 0
                for D in self.D_list[level]:
                    stepl = D.shape[0]
                    stepr = D.shape[1]
                    uhat[n0l:n0l+stepl,:]+=D@qhat_list[level+1][n0r:n0r+stepr,:]
                    n0r+=stepr
                    n0l+=stepl
                uhat_list[level]=uhat
            else:
                n0l=0
                n0r=0
                for U in self.U_list[level]:
                    stepl=U.shape[0]
                    stepr=U.shape[1]
                    uperm[n0l:n0l+stepl,:] = U@uhat_list[level-1][n0r:n0r+stepr,:]
                    n0l+=stepl
                    n0r+=stepr
                n0l=0
                n0r=0
                for D in self.D_list[level]:
                    stepl = D.shape[0]
                    stepr = D.shape[1]
                    uperm[n0l:n0l+stepl,:]+=D@vperm[n0r:n0r+stepr,:]
                    n0r+=stepr
                    n0l+=stepl
        u = uperm.clone()
        u[self.perm,:] = uperm
        u = u.detach().cpu().numpy()
        return u
    def matvecT(self,v):
        nlevels = len(self.D_list)
        qhat_list = [[] for _ in range(nlevels)]
        uhat_list = [[] for _ in range(nlevels)]
        if v.ndim == 1:
            vperm = v[:,np.newaxis]
        else:
            vperm = v[self.perm,:]
        vperm = torch.from_numpy(vperm)
        
        n0 = 0
        qhat = torch.zeros((0,vperm.shape[1]),dtype=vperm.dtype)
        #########################
        #       UPWARD PASS
        #########################
        for U in self.U_list[nlevels-1]:
            step = U.shape[0]
            qhat=torch.cat((qhat,U.T@vperm[n0:n0+step,:]),dim=0)
            n0+=step
        qhat_list[nlevels-1]=qhat

        for level in range(nlevels-2,0,-1):
            n0 = 0
            qhat = torch.zeros(0,vperm.shape[1])
            for U in self.U_list[level]:
                step = U.shape[0]
                qhat=torch.cat((qhat,U.T@qhat_list[level+1][n0:n0+step,:]),dim=0)
                n0+=step
            qhat_list[level]=qhat

            
        #########################
        #       DOWNWARD PASS
        #########################

        uperm = torch.zeros((vperm.shape[0],vperm.shape[1]),dtype=vperm.dtype)

        for level in range(0,nlevels):
            n0 = 0
            if level==0:
                uhat=self.D_list[0][0].T@qhat_list[1]
                uhat_list[0]=uhat
            elif level<nlevels-1:
                uhat = torch.zeros(0,vperm.shape[1])
                n0 = 0
                for V in self.V_list[level]:
                    step = V.shape[1]
                    uhat=torch.cat((uhat,V@uhat_list[level-1][n0:n0+step,:]),dim=0)
                    n0+=step
                n0l = 0
                stepl = 0
                n0r = 0
                stepr = 0
                for D in self.D_list[level]:
                    stepl = D.shape[1]
                    stepr = D.shape[0]
                    uhat[n0l:n0l+stepl,:]+=D.T@qhat_list[level+1][n0r:n0r+stepr,:]
                    n0r+=stepr
                    n0l+=stepl
                uhat_list[level]=uhat
            else:
                n0l=0
                n0r=0
                for V in self.V_list[level]:
                    stepl=V.shape[0]
                    stepr=V.shape[1]
                    uperm[n0l:n0l+stepl,:] = V@uhat_list[level-1][n0r:n0r+stepr,:]
                    n0l+=stepl
                    n0r+=stepr
                n0l=0
                n0r=0
                for D in self.D_list[level]:
                    stepl = D.shape[1]
                    stepr = D.shape[0]
                    uperm[n0l:n0l+stepl,:]+=D.T@vperm[n0r:n0r+stepr,:]
                    n0r+=stepr
                    n0l+=stepl
        u = uperm.clone()
        u[self.perm,:] = uperm
        u = u.detach().cpu().numpy()
        return u
