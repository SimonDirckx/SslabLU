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
import matplotlib.pyplot as plt
#import gc
    


class slab:
    def __init__(self,geom,gb_vec,transform=None):
        self.geom       =   geom
        self.transform  =   transform
        self.gb_vec     =   gb_vec
        try:
            ndim = self.geom.shape[-1]
            res  = gb_vec(np.random.randn(5,ndim))
            assert res.ndim == 1 and res.shape[0] == 5
        except:
            raise ValueError ("Input gb needs to accept inputs of size numpoints x ndim") 

    def compute_idxs_and_pts(self,solver):
        XX = solver.XX
        XXb = XX[solver.Ib,...]
        XXi = XX[solver.Ii,...]
        xl = self.geom[0][0]
        xr = self.geom[1][0]
        xc=(xl+xr)/2.

        Il = np.where((np.abs(XXb[..., 0] - xl) < 1e-14) & ~(self.gb_vec(XXb)))[0]
        Ir = np.where((np.abs(XXb[..., 0] - xr) < 1e-14) & ~(self.gb_vec(XXb)))[0]
        Ic = np.where(np.abs(XXi[..., 0] - xc) < 1e-14)[0]
        Igb = np.where(self.gb_vec(XXb))[0]    

        return Il,Ir,Ic,Igb,XXi,XXb
class omsStats:
    def __init__(self):
        self.compression = None
        self.compr_timing = None
        self.discr_timing = None

class oms:
    def __init__(self,slabList:list[slab],pdo,gb,solver_opts,connectivity):
        self.slabList=slabList
        self.pdo = pdo
        self.connectivity = connectivity
        self.opts = solver_opts
        self.gb = gb
        self.glob_target_dofs = []
        self.glob_source_dofs = []
        self.localSolver=None
        self.nbytes = 0
        self.densebytes = 0 
        self.stats = omsStats()
    def compute_global_dofs(self):
        if not self.glob_source_dofs:
            glob_source_dofs=[]
            if self.glob_target_dofs:
                for slabInd in range(len(self.connectivity)):
                    IFLeft  = self.connectivity[slabInd][0]
                    IFRight = self.connectivity[slabInd][1]
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
        for slabInd in range(len(slabs)):
            geom = np.array(slabs[slabInd])
            slab_i = slab(geom,self.gb)
            start = time.time()
            solver = solverWrap.solverWrapper(opts)
            solver.construct(geom,pdo,verbose=dbg)
            tDisc = time.time()-start
            discrTime += tDisc
            if dbg>1:
                print("SLAB %2.0d discretization time = %5.2f s" % (slabInd,tDisc))
            Il,Ir,Ic,Igb,XXi,XXb = slab_i.compute_idxs_and_pts(solver)
            nc = len(Ic)
            self.nc = nc
            Ntot += nc
            glob_target_dofs+=[range(startCentral,startCentral+nc)]
            startCentral += nc
            
            fgb = bc(XXb[Igb,...])
            
            st_l,st_r = self.compute_stmaps(Il,Ic,Ir,XXi,XXb,solver)

            rhs = solver.solver_ii@(solver.Aib[...,Igb]@fgb)
            rhs = -rhs[Ic]
            rhs_list+=[rhs]
            bool_r = len(Ir)>0
            bool_l = len(Il)>0
            start = time.time()
            compression_l = 0 
            compression_r = 0
            if bool_r:
                rkMat_r = assembler.assemble(st_r,dbg=dbg)
                self.nbytes+=assembler.stats.nbytes
                compression_r = assembler.stats.nbytes
            if bool_l:
                rkMat_l = assembler.assemble(st_l,dbg=dbg)
                compression_l = assembler.stats.nbytes
                self.nbytes+=assembler.stats.nbytes
            
            self.densebytes+=np.prod(st_l.A.shape)*8
            compression_l/=np.prod(st_l.A.shape)*8
            self.densebytes+=np.prod(st_r.A.shape)*8
            compression_r/=np.prod(st_r.A.shape)*8
            tCompress=time.time()-start
            compressTime += tCompress
            #shapeMatch = shapeMatch and (rkMat_l.shape==st_l.A.shape) and (rkMat_r.shape==st_r.A.shape)
            if dbg>0:
                if bool_l:
                    Vl=np.random.standard_normal(size=(st_l.A.shape[1],assembler.matOpts.maxRank))
                    Ul=st_l.A@Vl
                    Ulhat=rkMat_l@Vl
                    relerrl = max(relerrl,np.linalg.norm(Ul-Ulhat)/np.linalg.norm(Ul))
                if bool_r:
                    Vr=np.random.standard_normal(size=(st_r.A.shape[1],assembler.matOpts.maxRank))
                    Ur=st_r.A@Vr
                    Urhat=rkMat_r@Vr
                    relerrr = max(relerrr,np.linalg.norm(Ur-Urhat)/np.linalg.norm(Ur))
            if dbg>1:
                print("SLAB %d compression time %5.2f s"% (slabInd,tCompress))
                if bool_l and bool_r:
                    print("SLAB %d error = %5.2e // %5.2e\n" % (slabInd,relerrl,relerrr))
                    print("SLAB %d compression = %5.3e // %5.3e\n" % (slabInd,compression_l,compression_r))
                elif not bool_l:
                    print("SLAB %d error = %5.2e\n" % (slabInd,relerrr))
                    print("SLAB %d compression = %5.3e\n" % (slabInd,compression_r))
                elif not bool_r:
                    print("SLAB %d error = %5.2e\n" % (slabInd,relerrl))
                    print("SLAB %d compression = %5.3e\n" % (slabInd,compression_l))
            del st_l,st_r,Il,Ir,Ic,XXi,XXb,solver
            
            
            if self.connectivity[slabInd][0]<0:
                S_rk_list += [[rkMat_r]]
            elif self.connectivity[slabInd][1]<0:
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
        self.stats.compression=self.nbytes/self.densebytes
        self.stats.compr_timing = compressTime/(len(connectivity)-1)
        self.stats.discr_timing = compressTime/(len(connectivity)-1)
        self.glob_target_dofs = glob_target_dofs
        self.compute_global_dofs()
        rhstot = np.zeros(shape = (Ntot,))        
        for rhsInd in range(len(rhs_list)):
            rhstot[rhsInd*nc:(rhsInd+1)*nc]=rhs_list[rhsInd]

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
    

    
    def construct_rhstot(self,bc):
        '''
        TODO IMPLEMENT RHS
        '''
        return 0
