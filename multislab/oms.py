import numpy as np
import solver.solver as solverWrap
from solver.solver import stMap

import numpy as np
import solver.solver as solverWrap
from scipy.sparse.linalg   import LinearOperator
from solver.solver import stMap
import time
import sys
import jax.numpy as jnp
#import gc


def join_geom(slab1,slab2,period=None):
    ndim = len(slab1[0])
    if ndim==2:
        xl1 = slab1[0][0]
        xr1 = slab1[1][0]
        yl1 = slab1[0][1]
        yr1 = slab1[1][1]
        
        xl2 = slab2[0][0]
        xr2 = slab2[1][0]
        yl2 = slab2[0][1]
        yr2 = slab2[1][1]
        if(np.abs(xr1-xl2)>1e-10):
            if period:
                xl1 -= period
                xr1 -= period
                return join_geom([[xl1,yl1],[xr1,yr1]],slab2)
            else:
                ValueError("slab shift did not work (is your period correct?)")
        else:
            totalSlab = [[xl1, yl1],[xr2,yr2]]
        return totalSlab
    elif ndim==3:
        xl1 = slab1[0][0]
        xr1 = slab1[1][0]
        yl1 = slab1[0][1]
        yr1 = slab1[1][1]
        zl1 = slab1[0][2]
        zr1 = slab1[1][2]

        xl2 = slab2[0][0]
        xr2 = slab2[1][0]
        yl2 = slab2[0][1]
        yr2 = slab2[1][1]
        zl2 = slab2[0][2]
        zr2 = slab2[1][2]
        if(np.abs(xr1-xl2)>1e-10):
            if period:
                xl1 -= period
                xr1 -= period
                return join_geom([[xl1,yl1,zl1],[xr1,yr1,zr1]],slab2)
            else:
                ValueError("slab shift did not work (is your period correct?)")
        else:
            totalSlab = [[xl1, yl1,zl1],[xr2,yr2,zr2]]
        return totalSlab
    else:
        raise ValueError("ndim incorrect")
    


class slab:
    def __init__(self,geom,gb,transform=None):
        self.geom       =   geom
        self.transform  =   transform
        self.gb         =   gb

    def compute_idxs_and_pts(self,solver):
        XX = solver.XX
        XXb = XX[solver.Ib,...]
        XXi = XX[solver.Ii,...]
        xl = self.geom[0][0]
        xr = self.geom[1][0]
        xc=(xl+xr)/2.
        gb = self.gb(XXb)
        
        Il = jnp.where( (jnp.abs(XXb[...,0]-xl)<1e-14)& ~gb)[0] #[i for i in range(len(solver.Ib)) if np.abs(XXb[i,0]-xl)<1e-14 and XXb[i,1]>1e-14 and XXb[i,1]<1-1e-14]
        Ir = jnp.where((jnp.abs(XXb[...,0]-xr)<1e-14) & ~gb)[0] #[i for i in range(len(solver.Ib)) if np.abs(XXb[i,0]-xr)<1e-14 and XXb[i,1]>1e-14 and XXb[i,1]<1-1e-14]
        Ic = jnp.where((jnp.abs(XXi[...,0]-xc)<1e-14))[0]       #[i for i in range(len(solver.Ii)) if np.abs(XXi[i,0]-xc)<1e-14]
        Igb = jnp.where(gb)[0]                                  #[i for i in range(len(solver.Ib)) if self.gb(XXb[i,:])]
        return Il,Ir,Ic,Igb,XXi,XXb



class oms:
    def __init__(self,slabList:list[slab],pdo,gb,solver_opts,connectivity,if_connectivity,period = 0.):
        self.slabList=slabList
        self.pdo = pdo
        self.connectivity = connectivity
        self.if_connectivity = if_connectivity
        self.opts = solver_opts
        self.gb = gb
        self.glob_target_dofs = []
        self.glob_source_dofs = []
        self.localSolver=None
        self.period = period
        self.nbytes = 0
        self.densebytes = 0 
    def compute_global_dofs(self):
        if not self.glob_source_dofs:
            glob_source_dofs=[]
            if self.glob_target_dofs:
                for slabInd in range(len(self.if_connectivity)):
                    IFLeft  = self.if_connectivity[slabInd][0]
                    IFRight = self.if_connectivity[slabInd][1]
                    if IFLeft<0:
                        glob_source_dofs+=[[self.glob_target_dofs[IFRight]]]
                    elif IFRight<0:
                        glob_source_dofs+=[[self.glob_target_dofs[IFLeft]]]
                    else:
                        glob_source_dofs+=[[self.glob_target_dofs[IFLeft],self.glob_target_dofs[IFRight]]]
        self.glob_source_dofs=glob_source_dofs

    def compute_stmaps(self,Il,Ic,Ir,XXi,XXb,solver):
        A_solver = solver.solver_ii    
        def smatmat(v,I,J,transpose=False):
            if (v.ndim == 1):
                v_tmp = v[...,np.newaxis]
            else:
                v_tmp = v

            if (not transpose):
                result = (A_solver@(solver.Aib[...,J]@v_tmp))[I,...]
            else:
                result      = np.zeros(shape=(len(solver.Ii),v.shape[1]))
                result[I,:] = v_tmp
                result      = solver.Aib[...,J].T @ (A_solver.T@(result))
            if (v.ndim == 1):
                result = result.flatten()
            return result

        Linop_r = LinearOperator(shape=(len(Ic),len(Ir)),\
            matvec = lambda v:smatmat(v,Ic,Ir), rmatvec = lambda v:smatmat(v,Ic,Ir,transpose=True),\
            matmat = lambda v:smatmat(v,Ic,Ir), rmatmat = lambda v:smatmat(v,Ic,Ir,transpose=True))
        Linop_l = LinearOperator(shape=(len(Ic),len(Il)),\
            matvec = lambda v:smatmat(v,Ic,Il), rmatvec = lambda v:smatmat(v,Ic,Il,transpose=True),\
            matmat = lambda v:smatmat(v,Ic,Il), rmatmat = lambda v:smatmat(v,Ic,Il,transpose=True))
        
        st_r = stMap(Linop_r,XXb[Ir,...],XXi[Ic,...])
        st_l = stMap(Linop_l,XXb[Il,...],XXi[Ic,...])
        return st_l,st_r

    def construct_Stot_and_rhstot(self,bc,assembler,dbg=0):
        '''
        construct S operator and total global rhs


        EXPLAINER OF CONVENTIONS:
            - global dof ordering is inferred from the supplied connectivity
            - joined slabs are contiguous (ficticious domain extension used for periodic domains)
            - no domain checks are done (garbage in, garbage out)
            - ranges are used for global dofs, to improve efficiency (global dofs of interfaces are assumed contiguous)
            - first INTERFACES (i.e. 'Ic') is assumed to be global dofs 0...len(Ic)-1
        '''
        connectivity    = self.connectivity
        slabs           = self.slabList
        Ntot = 0
        period = self.period
        S_rk_list = []
        
        rhs_list = []

        glob_target_dofs=[]
        startCentral = 0
        opts = self.opts
        pdo = self.pdo
        data = 0
        discrTime = 0
        compressTime=0
        shapeMatch = True
        relerrl=0
        relerrr=0
        for slabInd in range(len(connectivity)):
            geom = np.array(join_geom(slabs[connectivity[slabInd][0]],slabs[connectivity[slabInd][1]],period))
            slab_i = slab(geom,self.gb)
            start = time.time()
            solver = solverWrap.solverWrapper(opts)
            solver.construct(geom,pdo)
            tDisc = time.time()-start
            discrTime += tDisc
            if dbg>1:
                print("discretization time = ",tDisc)
            Il,Ir,Ic,Igb,XXi,XXb = slab_i.compute_idxs_and_pts(solver)
            nc = len(Ic)
            self.nc = nc
            Ntot += nc
            glob_target_dofs+=[range(startCentral,startCentral+nc)]
            startCentral += nc
            
            fgb = bc(XXb[Igb,...])
            
            st_l,st_r = self.compute_stmaps(Il,Ic,Ir,XXi,XXb,solver)
            rhs = solver.solver_ii@(solver.Aib[...,Igb]@fgb)
            rhs = rhs[Ic]
            rhs_list+=[rhs]
            start = time.time()
            
            rkMat_r = assembler.assemble(st_r,dbg)
            self.nbytes+=assembler.stats.nbytes
            rkMat_l = assembler.assemble(st_l,dbg)
            self.nbytes+=assembler.stats.nbytes
            
            self.densebytes+=np.prod(st_l.A.shape)*8
            self.densebytes+=np.prod(st_r.A.shape)*8
            tCompress=time.time()-start
            compressTime += tCompress
            shapeMatch = shapeMatch and (rkMat_l.shape==st_l.A.shape) and (rkMat_r.shape==st_r.A.shape)
            if dbg>0:
                Vl=np.random.standard_normal(size=(st_l.A.shape[1],assembler.matOpts.maxRank))
                Vr=np.random.standard_normal(size=(st_l.A.shape[1],assembler.matOpts.maxRank))
                Ul=st_l.A@Vl
                Ur=st_r.A@Vr
                Ulhat=rkMat_l@Vl
                Urhat=rkMat_r@Vr
                relerrl = max(relerrl,np.linalg.norm(Ul-Ulhat)/np.linalg.norm(Ul))
                relerrr = max(relerrr,np.linalg.norm(Ur-Urhat)/np.linalg.norm(Ur))
            if dbg>1:
                print("compression time = ",tCompress)
                print("compression rate = ",self.nbytes/self.densebytes)
                print("error = ",relerrl,"//",relerrr)
            del st_l,st_r,Il,Ir,Ic,XXi,XXb,solver
            
            
            if self.if_connectivity[slabInd][0]<0:
                S_rk_list += [[rkMat_r]]
            elif self.if_connectivity[slabInd][1]<0:
                S_rk_list += [[rkMat_l]]
            else:
                S_rk_list += [[rkMat_l,rkMat_r]]
            
            if dbg>0: print("overlapping slab ",slabInd+1," of ",len(connectivity)," done")
        if dbg>0:
            print('============================OMS SUMMARY============================')
            print('avg. discr. time             = ',discrTime/(len(connectivity)-1))
            print('avg. compr. time             = ',compressTime/(len(connectivity)-1))
            print('compression rate             = ',self.nbytes/self.densebytes)
            print('shapes match?                = ',shapeMatch)
            print('total dofs                   = ',sum([len(dof) for dof in glob_target_dofs]))
            print('estim. max. err. ( l // r )  = (',relerrl," // ", relerrr,")")
            print('===================================================================')
        self.glob_target_dofs = glob_target_dofs
        self.compute_global_dofs()
        rhstot = np.zeros(shape = (Ntot,))        
        for rhsInd in range(len(rhs_list)):
            rhstot[rhsInd*nc:(rhsInd+1)*nc]=-rhs_list[rhsInd]

        def smatmat(v,transpose=False):
            if (v.ndim == 1):
                v_tmp = v[...,jnp.newaxis].astype('float64')
            else:
                v_tmp = v.astype('float64')
            result  = v_tmp.copy()
            if (not transpose):
                for i in range(len(self.glob_target_dofs)):
                    for j in range(len(self.glob_source_dofs[i])):
                            result[glob_target_dofs[i]]+=S_rk_list[i][j]@v_tmp[self.glob_source_dofs[i][j]]
            else:
                for i in range(len(glob_target_dofs)):
                    for j in range(len(self.glob_source_dofs[i])):
                            result[self.glob_source_dofs[i][j]]+=S_rk_list[i][j].T@v_tmp[glob_target_dofs[i]]
            if (v.ndim == 1):
                result = result.flatten()
            return result
        
        Linop = LinearOperator(shape=(Ntot,Ntot),\
        matvec = smatmat, rmatvec = lambda v: smatmat(v,transpose=True),\
        matmat = smatmat, rmatmat = lambda v: smatmat(v,transpose=True))
        return Linop,rhstot
    

    def construct_Stot(self,assembler):
        '''
        construct only S operator


        EXPLAINER OF CONVENTIONS:
            - global dof ordering is inferred from the supplied connectivity
            - joined slabs are contiguous (ficticious domain extension used for periodic domains)
            - no domain checks are done (garbage in, garbage out)
            - ranges are used for global dofs, to improve efficiency (global dofs of interfaces are assumed contiguous)
            - first INTERFACES (i.e. 'Ic') is assumed to be global dofs 0...len(Ic)-1
        '''
        connectivity    = self.connectivity
        slabs           = self.slabList
        Ntot = 0
        period = 1.

        Sl_rk_list = []
        Sr_rk_list = []

        glob_target_dofs=[]
        startCentral = 0
        for slabInd in range(len(connectivity)):
            geom = np.array(join_geom(slabs[connectivity[slabInd][0]],slabs[connectivity[slabInd][1]],period))
            slab_i = slab(geom,self.gb)
            
            solver = solverWrap.solverWrapper(self.opts)
            solver.construct(geom,self.pdo)
            Il,Ir,Ic,Igb,XXi,XXb = slab_i.compute_idxs_and_pts(solver)
            
            nc = len(Ic)
            Ntot += nc
            glob_target_dofs+=[range(startCentral,startCentral+nc)]
            startCentral += nc
            
            st_l,st_r = self.compute_stmaps(Il,Ic,Ir,XXi,XXb,solver)
            rkMat_r = assembler.assemble(st_r)
            rkMat_l = assembler.assemble(st_l)
            Sl_rk_list += [rkMat_l]
            Sr_rk_list += [rkMat_r]
        
        self.glob_target_dofs = glob_target_dofs
        self.compute_global_dofs()
        def smatmat(v,transpose=False):
            if (v.ndim == 1):
                v_tmp = v[:,np.newaxis]
            else:
                v_tmp = v
            result  = v_tmp.copy().astype('float64')
            if (not transpose):
                for i in range(len(glob_target_dofs)):
                    result[glob_target_dofs[i]]+=Sl_rk_list[i]@v_tmp[self.glob_source_dofs[i][0]]
                    result[glob_target_dofs[i]]+=Sr_rk_list[i]@v_tmp[self.glob_source_dofs[i][1]]
            else:
                for i in range(len(glob_target_dofs)):
                    result[self.glob_source_dofs[i][0]]+=Sl_rk_list[i].T@v_tmp[glob_target_dofs[i]]
                    result[self.glob_source_dofs[i][1]]+=Sr_rk_list[i].T@v_tmp[glob_target_dofs[i]]
            if (v.ndim == 1):
                result = result.flatten()
            return result

        Linop = LinearOperator(shape=(Ntot,Ntot),\
        matvec = smatmat, rmatvec = lambda v: smatmat(v,transpose=True),\
        matmat = smatmat, rmatmat = lambda v: smatmat(v,transpose=True))
        return Linop
    
    def construct_rhstot(self,bc):
        '''
        TODO IMPLEMENT RHS
        '''
        return 0