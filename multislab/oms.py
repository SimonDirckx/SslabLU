import numpy as np
import solver.solver as solverWrap
from solver.solver import stMap
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

import numpy as np
import pdo.pdo as pdo
import solver.solver as solverWrap
import matAssembly.matAssembler as mA
import matplotlib.pyplot as plt
import geometry.standardGeometries as stdGeom
import geometry.skeleton as skelTon
import time
import hps.hps_multidomain as HPS
import hps.geom as hpsGeom
from scipy.sparse        import block_diag
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from scipy import interpolate
from scipy.interpolate import griddata
from scipy.sparse.linalg   import LinearOperator
from scipy.sparse.linalg import gmres
from solver.solver import stMap
import matAssembly.matAssembler as mA
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Delaunay 
from hps.geom              import BoxGeometry, ParametrizedGeometry2D,ParametrizedGeometry3D
import scipy.special as special
from matplotlib.patches import Polygon
from hps.geom              import BoxGeometry, ParametrizedGeometry2D,ParametrizedGeometry3D
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import solver.HPSInterp3D as interp
import multislab.oms as oms




def join_geom(slab1,slab2,period=None):
    
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
            return join([[xl1,yl1,zl1],[xr1,yr1,zr1]],slab2)
        else:
            ValueError("slab shift did not work (is your period correct?)")
    else:
        totalSlab = [[xl1, yl1,zl1],[xr2,yr2,zr2]]
    return totalSlab


class slab:
    def __init__(self,geom,gb,transform=None):
        self.geom       =   geom
        self.transform  =   transform
        self.gb         =   gb

    def compute_idxs_and_pts(self,solver):
        XX = solver.XX
        XXb = XX[solver.Ib,:]
        XXi = XX[solver.Ii,:]
        xl = self.geom[0][0]
        xr = self.geom[1][0]
        xc=(xl+xr)/2.
        Il = [i for i in range(len(solver.Ib)) if np.abs(XXb[i,0]-xl)<1e-14 and XXb[i,1]>1e-14 and XXb[i,1]<1-1e-14]
        Ir = [i for i in range(len(solver.Ib)) if np.abs(XXb[i,0]-xr)<1e-14 and XXb[i,1]>1e-14 and XXb[i,1]<1-1e-14]
        Ic = [i for i in range(len(solver.Ii)) if np.abs(XXi[i,0]-xc)<1e-14]
        Igb = [i for i in range(len(solver.Ib)) if self.gb(XXb[i,:])]
        return Il,Ir,Ic,Igb,XXi,XXb



class oms:
    def __init__(self,slabList:list[slab],pdo,gb,solver_opts,connectivity):
        self.slabList=slabList
        self.pdo = pdo
        self.connectivity = connectivity
        self.opts = solver_opts
        self.gb = gb


    def construct_Stot_and_rhstot(self,bc,assembler):
        connectivity    = self.connectivity
        slabs           = self.slabList
        Ntot = 0
        period = 1.

        Sl_rk_list = []
        Sr_rk_list = []

        rhs_list = []

        glob_source_dofs=[]
        glob_target_dofs=[]


        for slabInd in range(len(connectivity)):
            geom = np.array(join(slabs[connectivity[slabInd][0]],slabs[connectivity[slabInd][1]],period))
            hpsgeom = hpsGeom.BoxGeometry(geom)
            solver = solverWrap.solverWrapper(self.opts)
            solver.construct(hpsgeom,self.pdo)
            slabTest = slab(geom,self.gb)
            Il,Ir,Ic,Igb,XXi,XXb = slabTest.compute_idxs_and_pts(solver)
            
            nc = len(Ic)
            Ntot += nc
            
            IFLeft  = connectivity[slabInd][0]
            IFRight = connectivity[slabInd][1]
            
            startLeft = IFLeft*nc
            startRight = ((IFRight+1)%len(connectivity))*nc
            startCentral = (IFLeft+1)%len(connectivity)*nc
            
            glob_source_dofs+=[[range(startLeft,startLeft+nc),range(startRight,startRight+nc)]]
            glob_target_dofs+=[range(startCentral,startCentral+nc)]
            
            fgb = bc(XXb[Igb,:])
            disc = solver.solver
            
            A_solver = solver.solver_ii    
            def smatmat(v,I,J,transpose=False):
                if (v.ndim == 1):
                    v_tmp = v[:,np.newaxis]
                else:
                    v_tmp = v

                if (not transpose):
                    result = (A_solver@(disc.Aix[:,J]@v_tmp))[I]
                else:
                    result      = np.zeros(shape=(len(disc.Ji),v.shape[1]))
                    result[I,:] = v_tmp
                    result      = disc.Aix[:,J].T @ (A_solver.T@(result))
                if (v.ndim == 1):
                    result = result.flatten()
                return result

            Linop_r = LinearOperator(shape=(len(Ic),len(Ir)),\
                matvec = lambda v:smatmat(v,Ic,Ir), rmatvec = lambda v:smatmat(v,Ic,Ir,transpose=True),\
                matmat = lambda v:smatmat(v,Ic,Ir), rmatmat = lambda v:smatmat(v,Ic,Ir,transpose=True))
            Linop_l = LinearOperator(shape=(len(Ic),len(Il)),\
                matvec = lambda v:smatmat(v,Ic,Il), rmatvec = lambda v:smatmat(v,Ic,Il,transpose=True),\
                matmat = lambda v:smatmat(v,Ic,Il), rmatmat = lambda v:smatmat(v,Ic,Il,transpose=True))
            
            st_r = stMap(Linop_r,XXb[Ir,:],XXi[Ic,:])
            st_l = stMap(Linop_l,XXb[Il,:],XXi[Ic,:])
            rkMat_r = assembler.assemble(st_r)
            rkMat_l = assembler.assemble(st_l)
            Sl_rk_list += [rkMat_l]
            Sr_rk_list += [rkMat_r]
            rhs = splinalg.spsolve(disc.Aii,disc.Aix[:,Igb]@fgb)
            rhs = rhs[Ic]
            rhs_list+=[rhs]
        
        rhstot = np.zeros(shape = (Ntot,))

        for rhsInd in range(len(rhs_list)):
            rhstot[rhsInd*nc:(rhsInd+1)*nc]=-rhs_list[rhsInd]

        def smatmat(v,transpose=False):
            if (v.ndim == 1):
                v_tmp = v[:,np.newaxis]
            else:
                v_tmp = v
            result  = v_tmp.copy()
            if (not transpose):
                for i in range(len(glob_target_dofs)):
                    result[glob_target_dofs[i]]+=Sl_rk_list[i]@v_tmp[glob_source_dofs[i][0]]
                    result[glob_target_dofs[i]]+=Sr_rk_list[i]@v_tmp[glob_source_dofs[i][1]]
            else:
                for i in range(len(glob_target_dofs)):
                    result[glob_source_dofs[i][0]]+=Sl_rk_list[i].T@v_tmp[glob_target_dofs[i]]
                    result[glob_source_dofs[i][1]]+=Sr_rk_list[i].T@v_tmp[glob_target_dofs[i]]
            if (v.ndim == 1):
                result = result.flatten()
            return result

        Linop = LinearOperator(shape=(Ntot,Ntot),\
        matvec = smatmat, rmatvec = lambda v: smatmat(v,transpose=True),\
        matmat = smatmat, rmatmat = lambda v: smatmat(v,transpose=True))
        return Linop,rhstot