import numpy as np
from collections import deque
from scipy.linalg import qr
from matAssembly.HBS.simpleoctree import simpletree as tree

def qr_null(A, rnk0:int):
    Q, R, P = qr(A.T, mode='full', pivoting=True)
    tol = 1e-8#np.finfo(R.dtype).eps if tol is None else tol
    tol*=np.linalg.norm(np.diag(R))
    rnk = min(A.shape) - np.abs(np.diag(R))[::-1].searchsorted(tol)
    return Q[:, rnk:(rnk+rnk0)]
def qr_col(A, rnk:int):
    Q, R, P = qr(A, mode='full', pivoting=True)
    return Q[:, 0:rnk]
def qr_col_eps(A,B, eps):
    QA, RA, PA = qr(A, mode='full', pivoting=True)
    QB, RB, PB = qr(B, mode='full', pivoting=True)
    tol=.1*eps*min(np.linalg.norm(np.diag(RA)),np.linalg.norm(np.diag(RB)))
    rnkA = min(A.shape) - np.abs(np.diag(RA))[::-1].searchsorted(tol)
    rnkB = min(B.shape) - np.abs(np.diag(RB))[::-1].searchsorted(tol)
    rnk=max(rnkA,rnkB)
    return QA[:, 0:rnk],QB[:, 0:rnk]


'''
implements HBSTree
'''

class HBSTree:
    
    def __init__(self, idxs=0, children=None):
        self.children   =   []
        self.idxs       =   idxs
        self.local_idxs =   np.zeros(shape=idxs.shape,dtype=int)
        self.level      =   0
        self.is_leaf    =   True
        self.depth      =   0
        self.number_of_children = 0
        self.child_index = 0
        self.local_data = 0.
        self.qhat = np.zeros(shape=(0,0))
        if children is not None:
            for child in children:
                self.add_child(child)
    
    def add_child(self, node):
        assert isinstance(node, HBSTree)
        node.set_level(self.level+1)
        node.child_index = self.number_of_children
        self.children.append(node)
        self.number_of_children += 1
        self.is_leaf=False
        node_idxs = node.idxs
        
        for i in range(len(node_idxs)):
            a=(np.where(self.idxs==node_idxs[i]))[0]
            node.set_local_idx(i,a)

    def set_local_idx(self,i,a):
        self.local_idxs[i] = int(a)

    def set_idxs(self,idxs):
        self.idxs = idxs

    def set_qhat(self,q):
        self.qhat = q

    def set_uhat(self,u):
        self.uhat = u

    def set_UDV(self,U,D,V):
        assert U.shape[1]==V.shape[1]
        assert D.shape[0]==D.shape[1]
        assert D.shape[0]==V.shape[0]
        self.U          = U
        self.V          = V
        self.D          = D
        self.rnk        = U.shape[1]
        self.local_data += U.data.nbytes+V.data.nbytes+D.data.nbytes
        #self.local_data += U.shape[1]*U.shape[0]+V.shape[1]*V.shape[0]+D.shape[1]*D.shape[0]#+D.data.nbytes
    
    def set_D(self,D):
        self.D = D
        self.local_data += D.data.nbytes
        #self.local_data += D.shape[1]*D.shape[0]
    
    def set_OmPsiYZ(self,Om,Psi,Y,Z):
        self.Om=Om
        self.Psi=Psi
        self.Y=Y
        self.Z=Z

    def print(self):
        print(len(self.idxs),',',self.level,',',self.is_leaf)
        for child in self.children:
            child.print()
    
    def set_level(self,i):
        self.level = i
    
    
    def breadth_first_iter(self):
        
        queue = deque([self])

        while queue:
            node = queue.popleft()
            print(node.level,',',node.local_data,',',node.qhat.shape)
            for child in node.children:
                queue.append(child)
    
    def total_bytes(self):
        nB = self.local_data
        for child in self.children:
            nB+=child.total_bytes()
        return nB

    
    def get_level_nodes(self,l):
        #if l>L:
        #    raise ValueError('l exceeds tree depth')
        if self.level == l:
            print(self.idxs)
        else:
            for child in self.children:
                child.get_level_nodes(l)


###############################
#           METHODS
###############################


def copy_tree_to_HBS(tree,m=0,T=None):
    print("copying")
    if m==0:
        T=HBSTree(np.sort(tree.get_box_inds(0)))
    for child in tree.get_box_children(m):
        node = HBSTree(np.sort(tree.get_box_inds(child)))
        T.add_child(node)
        copy_tree_to_HBS(tree,child,node)
    return T
def HBS_tree_from_points(XX,nl=8):
    t =  tree.BalancedTree(XX,nl)
    return copy_tree_to_HBS(t)       



def compress_HBS(T:HBSTree,OMEGA,PSI,Y,Z,rnk,s):
    if T.is_leaf:
        idxs=T.idxs
        Om  = OMEGA[idxs,:]
        Psi = PSI[idxs,:]
        Yt  = Y[idxs,:]
        Zt  = Z[idxs,:]
        n=len(idxs)
        T.set_OmPsiYZ(Om,Psi,Yt,Zt)
        P = qr_null(Om,rnk)
        Q = qr_null(Psi,rnk)
        U = qr_col(Yt@P,min(rnk,n))
        V = qr_col(Zt@Q,min(rnk,n))
        D = (np.identity(n)-U@U.T)@Yt@np.linalg.pinv(Om)+U@U.T@(((np.identity(n)-V@V.T)@Zt@np.linalg.pinv(Psi)).T)
        T.set_UDV(U,D,V)
    else:
        n=0
        for child in T.children:
            compress_HBS(child,OMEGA,PSI,Y,Z,rnk,s)
            n+= child.rnk
        Om  = np.zeros(shape=(n,s))
        Psi = np.zeros(shape=(n,s))
        Yt  = np.zeros(shape=(n,s))
        Zt  = np.zeros(shape=(n,s))
        n0 = 0
        for child in T.children:
            Om[n0:n0+child.rnk,:]     = child.V.T@child.Om
            Psi[n0:n0+child.rnk,:]    = child.U.T@child.Psi
            Yt[n0:n0+child.rnk,:]     = child.U.T@(child.Y - child.D@child.Om)
            Zt[n0:n0+child.rnk,:]     = child.V.T@(child.Z - child.D.T@child.Psi)
            n0  +=   child.rnk
        T.set_OmPsiYZ(Om,Psi,Yt,Zt)
        if T.level>0:
            P = qr_null(Om,rnk)
            Q = qr_null(Psi,rnk)
            U = qr_col(Yt@P,rnk)
            V = qr_col(Zt@Q,rnk)
            D = (np.identity(n)-U@U.T)@Yt@np.linalg.pinv(Om)+U@U.T@(((np.identity(n)-V@V.T)@Zt@np.linalg.pinv(Psi)).T)
            T.set_UDV(U,D,V)
            
        else:
            D = Yt@np.linalg.pinv(Om)
            T.set_D(D)
def compress_HBS_eps(T:HBSTree,OMEGA,PSI,Y,Z,rnk,s,eps):
    if T.is_leaf:
        idxs=T.idxs
        Om  = OMEGA[idxs,:]
        Psi = PSI[idxs,:]
        Yt  = Y[idxs,:]
        Zt  = Z[idxs,:]
        n=len(idxs)
        T.set_OmPsiYZ(Om,Psi,Yt,Zt)
        P = qr_null(Om,rnk)
        Q = qr_null(Psi,rnk)
        #U = qr_col(Yt@P,min(rnk,n))
        #V = qr_col(Zt@Q,min(rnk,n))
        U,V = qr_col_eps(Yt@P,Zt@Q,eps)
        D = (np.identity(n)-U@U.T)@Yt@np.linalg.pinv(Om)+U@U.T@(((np.identity(n)-V@V.T)@Zt@np.linalg.pinv(Psi)).T)
        T.set_UDV(U,D,V)
    else:
        n=0
        for child in T.children:
            compress_HBS_eps(child,OMEGA,PSI,Y,Z,rnk,s,eps)
            n+= child.rnk
        Om  = np.zeros(shape=(n,s))
        Psi = np.zeros(shape=(n,s))
        Yt  = np.zeros(shape=(n,s))
        Zt  = np.zeros(shape=(n,s))
        n0 = 0
        for child in T.children:
            Om[n0:n0+child.rnk,:]     = child.V.T@child.Om
            Psi[n0:n0+child.rnk,:]    = child.U.T@child.Psi
            Yt[n0:n0+child.rnk,:]     = child.U.T@(child.Y - child.D@child.Om)
            Zt[n0:n0+child.rnk,:]     = child.V.T@(child.Z - child.D.T@child.Psi)
            n0  +=   child.rnk
        T.set_OmPsiYZ(Om,Psi,Yt,Zt)
        if T.level>0:
            P = qr_null(Om,rnk)
            Q = qr_null(Psi,rnk)
            U,V=qr_col_eps(Yt@P,Zt@Q,eps)
            D = (np.identity(n)-U@U.T)@Yt@np.linalg.pinv(Om)+U@U.T@(((np.identity(n)-V@V.T)@Zt@np.linalg.pinv(Psi)).T)
            T.set_UDV(U,D,V)
            
        else:
            D = Yt@np.linalg.pinv(Om)
            T.set_D(D)

def random_compression_HBS(tree,OMEGA,PSI,Y,Z,rnk,s):
    T=copy_tree_to_HBS(tree)
    compress_HBS(T,OMEGA,PSI,Y,Z,rnk,s)
    return T
def random_compression_HBS_eps(tree,OMEGA,PSI,Y,Z,rnk,s,eps):
    T=copy_tree_to_HBS(tree)
    compress_HBS_eps(T,OMEGA,PSI,Y,Z,rnk,s,eps)
    return T


def apply_HBS_upward(HBSMat:HBSTree,q,transpose=False):
    #upward pass
    m00 = q.shape[1]
    if not transpose:
        if HBSMat.is_leaf:
            qhat = HBSMat.V.T@q[HBSMat.idxs,:]
            HBSMat.set_qhat(qhat)
        elif HBSMat.level>0:
            n   = 0
            for child in HBSMat.children:
                n+=child.rnk
            qhat = np.zeros(shape=(n,m00))
            n0 = 0
            for child in HBSMat.children:
                apply_HBS_upward(child,q)
                qhat[n0:n0+child.rnk,:]=child.qhat
                n0+=child.rnk
            qhat = HBSMat.V.T@qhat
            HBSMat.set_qhat(qhat)
        else:
            for child in HBSMat.children:
                apply_HBS_upward(child,q)
    else:
        if HBSMat.is_leaf:
            qhat = HBSMat.U.T@q[HBSMat.idxs,:]
            HBSMat.set_qhat(qhat)
        elif HBSMat.level>0:
            n   = 0
            for child in HBSMat.children:
                n+=child.rnk
            qhat = np.zeros(shape=(n,m00))
            n0 = 0
            for child in HBSMat.children:
                apply_HBS_upward(child,q,transpose)
                qhat[n0:n0+child.rnk,:]=child.qhat
                n0+=child.rnk
            qhat = HBSMat.U.T@qhat
            HBSMat.set_qhat(qhat)
        else:
            for child in HBSMat.children:
                apply_HBS_upward(child,q,transpose)

def apply_HBS_downward(HBSMat:HBSTree,u,q,transpose=False):
    m00 = q.shape[1]
    if not transpose:
        if HBSMat.level==0:
            D=HBSMat.D
            n   = 0
            for child in HBSMat.children:
                n+=child.rnk
            qhat = np.zeros(shape=(n,m00))
            n0 = 0
            for child in HBSMat.children:
                qhat[n0:n0+child.rnk,:] = child.qhat
                n0+=child.rnk
            uhat = D@qhat
            n0 = 0
            for child in HBSMat.children:
                child.set_uhat(uhat[n0:n0+child.rnk])
                n0+=child.rnk
                apply_HBS_downward(child,u,q)
        elif not HBSMat.is_leaf:
            U = HBSMat.U
            D = HBSMat.D
            n   = 0
            for child in HBSMat.children:
                n+=child.rnk
            qhat = np.zeros(shape=(n,m00))
            n0 = 0
            for child in HBSMat.children:
                qhat[n0:n0+child.rnk,:] = child.qhat
                n0+=child.rnk
            uhat = U@HBSMat.uhat+D@qhat
            n0=0
            for child in HBSMat.children:
                child.set_uhat(uhat[n0:n0+child.rnk,:])
                n0+=child.rnk
                apply_HBS_downward(child,u,q)
        else:
            U=HBSMat.U
            D=HBSMat.D
            u[HBSMat.idxs,:]=U@HBSMat.uhat+D@q[HBSMat.idxs,:]
    else:
        if HBSMat.level==0:
            D=HBSMat.D
            n   = 0
            for child in HBSMat.children:
                n+=child.rnk
            qhat = np.zeros(shape=(n,m00))
            n0 = 0
            for child in HBSMat.children:
                qhat[n0:n0+child.rnk,:] = child.qhat
                n0+=child.rnk
            uhat = D.T@qhat
            n0 = 0
            for child in HBSMat.children:
                child.set_uhat(uhat[n0:n0+child.rnk])
                n0+=child.rnk
                apply_HBS_downward(child,u,q,transpose)
        elif not HBSMat.is_leaf:
            V = HBSMat.V
            D = HBSMat.D
            n   = 0
            for child in HBSMat.children:
                n+=child.rnk
            qhat = np.zeros(shape=(n,m00))
            n0 = 0
            for child in HBSMat.children:
                qhat[n0:n0+child.rnk,:] = child.qhat
                n0+=child.rnk
            uhat = V@HBSMat.uhat+D.T@qhat
            n0=0
            for child in HBSMat.children:
                child.set_uhat(uhat[n0:n0+child.rnk])
                n0+=child.rnk
                apply_HBS_downward(child,u,q,transpose)
        else:
            V=HBSMat.V
            D=HBSMat.D
            u[HBSMat.idxs,:]=V@HBSMat.uhat+D.T@q[HBSMat.idxs,:]


def apply_HBS(HBSMat:HBSTree,q0,transpose=False):
    if q0.ndim == 1:
        q = q0[:,np.newaxis]
    else:
        q = q0
    u=np.zeros(shape=q.shape)
    apply_HBS_upward(HBSMat,q,transpose)
    apply_HBS_downward(HBSMat,u,q,transpose)
    if q0.ndim == 1:
        u = u.flatten()
    return u



    