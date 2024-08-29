import numpy as np
import spectralDisc as spectral

class partition:
     '''
     
     Class used to represent partition of a REGULAR double slab

     '''
     
    #    _____________
    #   |      |      |
    #   |      |      |
    #   |      |      |
    # L |  a   |  b   |
    #   |      c      |
    #   gl     |      gr
    #   |______|______|
    #       H     H
    #
    # Boundaries: ga,gb,g (of a, b and total resp.)t
     '''
     enum for point types:
     0: g-gl-gr
     1: gl
     2: gr
     3: I-c
     4: c

     '''

     def __init__(self,L,H):
          self.L = L
          self.H = H

     def classify(self,p):
          x=p[0]
          y=p[1]
          if np.abs(y)<1e-15 or np.abs(y-self.L)<1e-15:
               return 0
          elif np.abs(x)<1e-15:
               return 1
          elif np.abs(x-2*self.H)<1e-15:
               return 2
          elif np.abs(x-self.H)<1e-15:
               return 4
          else:
               return 3
               
          
     

class problem:
     '''
     Class used to represent BVP
     (for now not used)
     '''

class slabGeometry:

    """
    
    A class used to represent the geometry of a double slab
    for now: no restriction on map
    
    """
    def __init__(self,map):
         self.l2g = map
    

class discretization:

    '''
     
    Class used to represent discretization type of a double slab
    slab locally represented by L,H

    '''
    #    _____________
    #   |      |      |
    #   |      |      |
    #   |      |      |
    # L |      |      |
    #   |      |      |
    #   |      |      |
    #   |______|______|
    #       H     H
    #
    # y and x dir discretized using cheb or stencil
    # such that interface is in tensor product set

    def __init__(self,ordx:int,ordy:int,discType:str):
        self.discType = discType
        if self.discType=='cheb':
            self.ordx = 2*(ordx//2)
        self.ordy = ordy
    
    def discretize(self,H,L):
        self.H = H
        self.L = L
        if self.discType == 'cheb':    
            xpts = -spectral.cheb_pts(self.ordx)
            ypts = -spectral.cheb_pts(self.ordy)
            xpts = self.H*(xpts+1.)
            ypts = .5*self.L*(ypts+1.)
            self.xpts=xpts
            self.ypts=ypts
            XY=np.zeros(shape=((self.ordx+1)*(self.ordy+1),2))
            for j in range(self.ordy+1):
                y = ypts[j]
                for i in range(self.ordx+1):    
                    XY[j+i*(self.ordy+1),:] = [xpts[i],y]
            self.XY=XY
    
    def ndofs(self):
        if self.discType=='cheb':
            return (self.ordx+1)*(self.ordy+1)
        else:
            return 0
    
    def get_Dx(self):
        if self.discType=='cheb':
            Dx = spectral.Diffmat(self.xpts)
            return Dx
        else:
            return 0
    
    def get_Dy(self):
        if self.discType=='cheb':
            Dy = spectral.Diffmat(self.ypts)
            return Dy
        else:
            return 0
    
    def get_Ex(self):
        Ex = np.identity(len(self.xpts))
        return Ex

    def get_Ey(self):
        Ey = np.identity(len(self.ypts))
        return Ey
     
    
class dSlab:

    """
    
    A class used to represent a double slab
    
    """

    
    def __init__(self,disc:discretization,part:partition,localToGlobal,H,L): # future: add prblm:problem
        
        #self.prblm = prblm
        self.disc   =   disc
        disc.discretize(H,L)
        self.part   =   part
        self.localToGlobal=localToGlobal

        # ordinarily, dS has left and right edge interior to Omega
        # but edge double slabs exist
        
        self.leftEdge   = False
        self.rightEdge  = False
        self.edge       = False
        
        
        # compute dofs
        self.dofs = disc.XY
        self.ndofs = len(self.dofs[:,0])
        
        # partition
        self.compute_partition()

        # SL and SR only computed when needed
        self.SLComputed = False
        self.SRComputed = False
        

    def l2g(self,i):
         # observe difference between localToGlobal and l2g:
         # localToGlobal is map accepting any slab point
         # l2g assumes indexed dof from discretization is given
         x=self.dofs[i,0]
         y=self.dofs[i,1]
         return self.localToGlobal(x,y)
    def eval_global_func(self,f,i):
        x,y=self.l2g(i)
        return f(x,y)
    
    def setAsEdge(self,edgeType:int):
        if edgeType==1:
            self.leftEdge   = True
        if edgeType == 2:
            self.rightEdge  = True
        self.edge = self.leftEdge or self.rightEdge
        

    def compute_partition(self):

        Ibl =   []
        Ibr =   []
        Ii  =   []
        Ic  =   []
        Ib  =   []

        for i in range(len(self.dofs)):
            n=self.part.classify(self.dofs[i,:])
            if n<3:
                Ib+=[i]
                if n==1:
                    Ibl+=[i]
                if n==2:
                    Ibr+=[i]
            else:
                Ii+=[i]
                if n==4:
                    Ic+=[i]
        
        self.Ibl    =   Ibl
        self.Ibr    =   Ibr
        self.Ii     =   Ii
        self.Ic     =   Ic
        self.Ib     =   Ib
    
    
    def computeL(self):
        
        # future work: implement diff operator from "self.prblm"
        Dx = self.disc.get_Dx()
        Dy = self.disc.get_Dy()
        Dxx = Dx@Dx
        Dyy = Dy@Dy
        Ex  = self.disc.get_Ex()
        Ey  = self.disc.get_Ey()
        L = np.kron(Ex,Dyy)+np.kron(Dxx,Ey)
        self.L=L
    
    def computeS_left(self):
        L=self.L
        Ii = self.Ii
        Il=self.Ibl
        Libl=L[Ii,:][:,Il]
        Lii = L[Ii,:][:,Ii]
        S0 =np.zeros(shape=(self.ndofs,len(Il)))
        S0[Ii,:] = np.linalg.solve(Lii,Libl)
        S=S0[self.Ic,:]
        self.SL = S
        self.SLComputed = True
        
    
    def computeS_right(self):

        L=self.L
        Ii = self.Ii
        Ir = self.Ibr
        Libr=L[Ii,:][:,Ir]
        Lii = L[Ii,:][:,Ii]
        S0 =np.zeros(shape=(self.ndofs,len(Ir)))
        S0[Ii,:] = np.linalg.solve(Lii,Libr)
        S=S0[self.Ic,:]
        self.SR=S
        self.SRComputed = True
    
    def getSL(self):
        if self.SLComputed:
            return self.SL
        else:
            self.computeS_left()
            return self.SL
    
    def getSR(self):
        if self.SRComputed:
            return self.SR
        else:
            self.computeS_right()
            return self.SR
    
    def compute_SRHS(self,f):

        # future work: implement rhs operator from "self.prblm"
        f0 = np.zeros(shape = (self.ndofs,))
        for i in self.Ib:
            x,y=self.l2g(i)
            f0[i] = f(x,y)
        
        Il  =   self.Ibl
        Ir  =   self.Ibr
        Ii  =   self.Ii
        Ib  =   self.Ib
        L   =   self.L
        Lii =   L[Ii,:][:,Ii]
        Lib =   L[Ii,:][:,Ib]
        

        b0  =   np.zeros(shape = (self.ndofs,))
        b00 =   np.zeros(shape = (len(self.Ic),))
        
        if self.leftEdge:
                if not self.SLComputed:
                    self.computeS_left()
                b00+=self.SL@f0[Il]
        if self.rightEdge:
                if not self.SRComputed:
                    self.computeS_right()
                b00+=self.SR@f0[Ir]
        
        
        f0[Il] = 0.
        f0[Ir] = 0.
        
        b0[Ii] = -np.linalg.solve(Lii,Lib@f0[Ib])
        
        b   =   b0[self.Ic]
        b  -=   b00
        
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
            Isi=dSlabs[i].Ic
            ndofs+=len(Isi)
        self.ndofs = ndofs

    def form_total_S(self):
        #TODO: this breaks down when multislab is not ordered correcly
        #       -> make more robust
        
        Stot = np.zeros(shape=(self.ndofs,self.ndofs))
        k=0
        for iter in range(len(self.dSlabs)):
            
            dS      =   self.dSlabs[iter]
            nIsi    =   len(dS.Ic)
            nIli    =   len(dS.Ibl)
            nIri    =   len(dS.Ibr)
            SLi     =   dS.getSL()
            SRi     =   dS.getSR()

            if not dS.edge:
                
                for i in range(nIsi):
                    Stot[k+i,k+i]=1.
                    for j in range(nIli):
                        Stot[k+i,k+j-nIli]=SLi[i,j]
                    for j in range(nIri):
                        Stot[k+i,k+j+nIsi]=SRi[i,j]
                k+= nIsi
            
            elif dS.leftEdge:
                
                for i in range(nIsi):
                    Stot[k+i,k+i]=1.
                    for j in range(nIri):
                        Stot[k+i,k+j+nIsi]=SRi[i,j]
                k+= nIsi
            
            elif dS.rightEdge:
                
                for i in range(nIsi):
                    Stot[k+i,k+i]=1.
                    for j in range(nIli):
                        Stot[k+i,k+j-nIli]=SLi[i,j]
                k+= nIsi
        return Stot
    
    def form_total_SRHS(self,f):
        
        btot    =   np.zeros(shape=(self.ndofs,))
        k       =   0
        
        for iter in range(len(self.dSlabs)):
            
            dS = self.dSlabs[iter]
            nIsi=len(dS.Ic)
            b0 = dS.compute_SRHS(f)
            btot[k:k+nIsi]=b0
            k+=nIsi
        
        return btot
    
    def eval_at_interfaces(self,f):
        
        u   =   np.zeros(shape=(self.ndofs,))
        k   =   0
        
        for iter in range(len(self.dSlabs)):
            
            dSi = self.dSlabs[iter]
            Isi=dSi.Ic
            
            for i in range(len(Isi)):
                u[k+i]=dSi.eval_global_func(f,Isi[i])
            
            k+=len(dSi.Ic)
        
        return u
            


         

