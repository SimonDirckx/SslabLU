import numpy as np







class Slab:

    """
    
    A class used to represent a single slab
    more doc to come

    """

    def __init__(self,Iobl,Ioblc,Icbl,Icblc,Iobr,Iobrc,Icbr,Icbrc,Ii,L,NrmlDL,NrmlDR,dofs,l2g):
        # open boundary left
        self.Iobl  =   Iobl
        # open boundary left complement
        self.Ioblc =   Ioblc
        # closed boundary left
        self.Icbl  =   Icbl
        # closed boundary left complement
        self.Icblc =   Icblc
        # open boundary right
        self.Iobr  =   Iobr
        # open boundary right complement
        self.Iobrc =   Iobrc
        # closed boundary right
        self.Icbr  =   Icbr
        # closed boundary right complement
        self.Icbrc =   Icbrc
        # internal
        self.Ii    =   Ii
        # local L-discretization
        self.L      =   L
        # local normal derivatives
        self.NrmlDL =   NrmlDL
        self.NrmlDR =   NrmlDR
        # local-to-global map
        self.l2g = l2g
        # bookkeeping
        self.dofs = dofs
        self.ndofs = len(dofs)
        self.edge = False
    
    def eval_global_func(self,f,i):
        x0,y0   =   self.dofs[i] 
        x,y     =   self.l2g(x0,y0)
        return f(x,y)
    def set_edge(self):
        self.edge = True


    
class dSlab:

    """
    
    A class used to represent a double slab
    with equilibrium conditions
    more doc to come

    """

    
    def __init__(self,SL:Slab,SR:Slab):
        self.SL = SL
        self.SR = SR
        
        if len(SL.Icbr)>=len(SR.Icbl):
            self.leftFiner = True
            self.ndofs = SL.ndofs+SR.ndofs-len(SL.Icbr)
            self.nshift = len(SL.Ii)+len(SL.Icbrc)+len(SL.Icbr)-len(SR.Icbl)
        else:
            self.leftFiner = False
            self.ndofs = SL.ndofs+SR.ndofs-len(SL.Icbl)
            self.nshift = len(SL.Ii)+len(SL.Icbrc)

        self.Ib = self.compute_Ib()
        self.Is = self.compute_Is()
        self.Ii = self.compute_Ii()
        self.compute_Il_Ir()
        self.left_edge = False
        self.right_edge = False
        if SL.edge:
             self.left_edge = True
        if SR.edge:
             self.right_edge = True
        self.edge = self.left_edge or self.right_edge

    def l2g(self,i):
         if self.leftFiner:
              if(i<self.SL.ndofs):
                   return self.SL.l2g(self.SL.dofs[i][0],self.SL.dofs[i][1])
              else:
                   return self.SR.l2g(self.SR.dofs[i-self.nshift][0],self.SR.dofs[i-self.nshift][1])
         else:
              if(i<self.SL.ndofs-self.nshift):
                   return self.SL.l2g(self.SL.dofs[i][0],self.SL.dofs[i][1])
              else:
                   return self.SR.l2g(self.SR.dofs[i-self.nshift][0],self.SR.dofs[i-self.nshift][1])

    def compute_Ib(self):
        if self.leftFiner:
            I=self.SL.Iobrc+[x+self.nshift for x in self.SR.Icblc]
        else:
            I=self.SL.Icbrc+[x+self.nshift for x in self.SR.Ioblc]
        return I
    
    def compute_Is(self):
        if self.leftFiner:
            I=self.SL.Iobr                          #open or closed?
        else:
            I=[x+self.nshift for x in self.SR.Iobl] #open or closed?
        return I
    def compute_Ii(self):
        Ii = self.SL.Ii+self.Is+[x+self.nshift for x in self.SR.Ii]
        return Ii
    def compute_Il_Ir(self):
        self.Ir = [x+self.nshift for x in self.SR.Iobr]
        self.Il = self.SL.Iobl
         
    
    def local_bc(self,bdryFunc):
        u = np.zeros(shape = (len(self.ndofs),))
        if self.leftFiner:
            u[self.SL.Iobrc] = self.SL.eval_global_func(bdryFunc,self.SL.Iobrc)
            Ibr = [x+self.nshift for x in self.SR.Icblc]
            u[Ibr] = self.SR.eval_global_func(bdryFunc,self.SL.Icblc)
        else:
            u[self.SL.Icbrc] = self.SL.eval_global_func(bdryFunc,self.SL.Icbrc)
            Ibr = [x+self.nshift for x in self.SR.Ioblc]
            u[Ibr] = self.SR.eval_global_func(bdryFunc,self.SL.Ioblc)
        return u
    def eval_global_func(self,f,i):
        x,y     =   self.l2g(i)
        return f(x,y)
    
    def computeL(self):
        
        # for now: different discretizations for L not supported!

        L = np.zeros(shape=(self.ndofs,self.ndofs))
        LL = self.SL.L
        LR = self.SR.L
        
        IL = self.SL.Ii+self.SL.Icbrc
        IR = self.SR.Icblc+self.SR.Ii

        J = IL+self.SL.Icbr
        for i in range(len(IL)):
                for j in range(len(J)):
                    L[IL[i],J[j]] = LL[IL[i],J[j]]
        
        J = IR+self.SR.Icbl
        for i in range(len(IL)):
                for j in range(len(J)):
                    L[IR[i]+self.nshift,J[j]+self.nshift] = LR[IR[i],J[j]]

        ILs = self.SL.Icbr
        IRs = self.SR.Icbl 
        
        NxL = self.SL.NrmlDR
        NxR = self.SR.NrmlDL
        for i in range(len(ILs)):
                for j in range(len(IL)):
                    L[ILs[i],IL[j]] = NxL[ILs[i],IL[j]]
            
        for i in range(len(IRs)):
                for j in range(len(IR)):
                    L[IRs[i]+self.nshift,IR[j]+self.nshift] = -NxR[IRs[i],IR[j]]#!!!!
            
        for i in range(len(ILs)):
                for j in range(len(ILs)):
                    L[ILs[i],ILs[j]] = NxL[ILs[i],ILs[j]]-NxR[IRs[i],IRs[j]]#!!!
        self.L = L
    def computeS_left(self):
        L=self.L
        Ii = self.Ii
        Il=self.SL.Iobl
        Lib0=L[Ii,:][:,Il]
        Lii = L[Ii,:][:,Ii]
        S0 =np.zeros(shape=(self.ndofs,len(Il)))
        S0[Ii,:] = np.linalg.solve(Lii,Lib0)
        S=S0[self.Is,:]
        return S
    
    def computeS_right(self):
        L=self.L
        Ii = self.Ii
        Ir = [x+self.nshift for x in self.SR.Iobr]
        Lib0=L[Ii,:][:,Ir]
        Lii = L[Ii,:][:,Ii]
        S0 =np.zeros(shape=(self.ndofs,len(Ir)))
        S0[Ii,:] = np.linalg.solve(Lii,Lib0)
        S=S0[self.Is,:]
        return S
    def compute_SRHS(self,f):
        f0 = np.zeros(shape = (self.ndofs,))
        for i in self.Ib:
            x,y=self.l2g(i)
            f0[i] = f(x,y)
        Il=self.SL.Iobl
        Ir = [x+self.nshift for x in self.SR.Iobr]
        Ii = self.Ii
        Ib = self.Ib
        L=self.L
        Lii=L[Ii,:][:,Ii]
        Lib=L[Ii,:][:,Ib]
        

        b0 = np.zeros(shape = (self.ndofs,))
        b00= np.zeros(shape = (len(self.Is),))
        if self.left_edge:
                SL = self.computeS_left()
                b00+=SL@f0[Il]
        if self.right_edge:
                SR = self.computeS_right()
                b00+=SR@f0[Ir]


        f0[Il] = 0.
        f0[Ir] = 0.
        
        
        b0[Ii] = -np.linalg.solve(Lii,Lib@f0[Ib])
        b=b0[self.Is]
        b-=b00
        return b

class multiSlab:

    """
    
    A class used to represent a multislab
    more doc to come

    """



    def __init__(self,dSlabs:list[dSlab]):
        self.dSlabs=dSlabs
        ndofs = 0
        for i in range(len(dSlabs)):
            Isi=dSlabs[i].Is
            ndofs+=len(Isi)
        self.ndofs = ndofs
    def form_total_S(self):
        #TODO: this breaks down when multislab is not ordered correcly
        #       -> make more robust
        Stot = np.zeros(shape=(self.ndofs,self.ndofs))
        k=0
        for iter in range(len(self.dSlabs)):
            dS = self.dSlabs[iter]
            nIsi=len(dS.Is)
            nIli=len(dS.Il)
            nIri=len(dS.Ir)
            SLi = dS.computeS_left()
            SRi = dS.computeS_right()
            if not dS.edge:
                for i in range(nIsi):
                    Stot[k+i,k+i]=1.
                    for j in range(nIli):
                        Stot[k+i,k+j-nIli]=SLi[i,j]
                    for j in range(nIri):
                        Stot[k+i,k+j+nIsi]=SRi[i,j]
                k+= nIsi
            elif dS.left_edge:
                for i in range(nIsi):
                    Stot[k+i,k+i]=1.
                    for j in range(nIri):
                        Stot[k+i,k+j+nIsi]=SRi[i,j]
                k+= nIsi
            elif dS.right_edge:
                for i in range(nIsi):
                    Stot[k+i,k+i]=1.
                    for j in range(nIli):
                        Stot[k+i,k+j-nIli]=SLi[i,j]
                k+= nIsi
        return Stot
    def form_total_SRHS(self,f):
        btot = np.zeros(shape=(self.ndofs,))
        k=0
        for iter in range(len(self.dSlabs)):
            dS = self.dSlabs[iter]
            nIsi=len(dS.Is)
            b0 = dS.compute_SRHS(f)
            btot[k:k+nIsi]=b0
            k+=nIsi
        return btot
    def eval_at_interfaces(self,f):
        u=np.zeros(shape=(self.ndofs,))
        k=0
        for iter in range(len(self.dSlabs)):
            dSi = self.dSlabs[iter]
            Isi=dSi.Is
            for i in range(len(Isi)):
                  u[k+i]=dSi.eval_global_func(f,Isi[i])
            k+=len(dSi.Is)
        return u
            


         

