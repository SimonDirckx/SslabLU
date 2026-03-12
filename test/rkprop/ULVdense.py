
import numpy as np
import matplotlib.pyplot as plt

def compute_QRW(Dtot,Vtot,Nb):
    
    if Nb==1:
        D = Dtot
        [Q,R] = np.linalg.qr(D)
        Qtot = Q
        
        R11 = R
        
        R12=0
        R22=0
        Wtot = np.identity(R.shape[1])
        NN = R.shape[0]
    else:
        n = Vtot.shape[0]//Nb
        k = Vtot.shape[1]//Nb
        NN = (n-k)*Nb
        Qtot = np.zeros(shape = (Nb*n,Nb*n))
        Wtot = np.zeros(shape = (Nb*n,Nb*n))
        R11 = np.zeros(shape = (NN,NN))
        R12 = np.zeros(shape = (NN,Nb*k))
        R22 = np.zeros(shape = (Nb*k,Nb*k))

        for box_ind in range(Nb):
            V = Vtot[box_ind*n:(box_ind+1)*n,:][:,box_ind*k:(box_ind+1)*k]
            if n-k>0:
                [Vr,_] = np.linalg.qr(V@V.T,mode='reduced')
                Vr = Vr[:,k:]
                W = np.append(Vr,V,axis=1)
            else:
                W = V
            
            D = Dtot[box_ind*n:(box_ind+1)*n,:][:,box_ind*n:(box_ind+1)*n]
            [Q,R] = np.linalg.qr(D@W)
            Qtot[box_ind*n:(box_ind+1)*n,:][:,box_ind*(n-k):(box_ind+1)*(n-k)] = Q[:,:n-k]
            Qtot[box_ind*n:(box_ind+1)*n,:][:,NN+box_ind*k:NN+(box_ind+1)*k] = Q[:,n-k:]
            Wtot[box_ind*n:(box_ind+1)*n,:][:,box_ind*(n-k):(box_ind+1)*(n-k)] = W[:,:n-k]
            Wtot[box_ind*n:(box_ind+1)*n,:][:,NN+box_ind*k:NN+(box_ind+1)*k] = W[:,n-k:]
            R11[box_ind*(n-k):(box_ind+1)*(n-k),:][:,box_ind*(n-k):(box_ind+1)*(n-k)] = R[:,:n-k][:n-k,:]
            R12[box_ind*(n-k):(box_ind+1)*(n-k),:][:,box_ind*k:(box_ind+1)*k] = R[:,n-k:][:n-k,:]
            R22[box_ind*k:(box_ind+1)*k,:][:,box_ind*k:(box_ind+1)*k] = R[:,n-k:][n-k:,:]
    return Qtot,Wtot,R11,R12,R22,NN



def compute_ULV(Umats,Dmats,Vmats,Nbvec):
    
    NNvec = np.zeros(shape=(0,),dtype=np.int64)
    NNvec = np.append(NNvec,0)
    Qlist = []
    Wlist = []
    Ulist  = []
    Rlist = []
    R_off_list = []
    for i in range(len(Dmats)):
        print("lvl = ",i)
        
        if i==0:
            Rprime = Dmats[0]
            Q,W,R_11,R_12,R_22,NN = compute_QRW(Rprime,Vmats[0],Nbvec[0])
        else:
            
            Rhat = (Ulist[-1]@Dmats[i]+R_22)
            if i<len(Vmats):
                Q,W,R_11,R_12,R_22,NN = compute_QRW(Rhat,Vmats[i],Nbvec[i])
            else:
                Q,W,R_11,R_12,R_22,NN = compute_QRW(Rhat,None,Nbvec[i])
        
        NNvec = np.append(NNvec,NNvec[-1]+NN)
        if i<len(Umats):
            if i == 0:
                Ulist+=[Q[:,NNvec[-1]:].T@Umats[0]]
            else:
                Ulist+=[Q[:,NNvec[-1]-NNvec[-2]:].T@Ulist[-1]@Umats[i]]
        Qlist+=[Q]
        Wlist+=[W]
        Rlist+=[R_11]
        R_off_list+=[R_12]
    return Qlist,Wlist,Ulist,Rlist,R_off_list,NNvec

