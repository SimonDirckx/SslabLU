import numpy as np

import torch
import jax.numpy as jnp
import geometry.geom_2D.square as square
import matplotlib.pyplot as plt

import solver.spectral.spectralSolver as spectral
import solver.spectral.geom as spectralGeom


import solver.stencil.stencilSolver as stencil
import solver.stencil.geom as stencilGeom

from solver.hpsmultidomain.hpsmultidomain import domain_driver as hpsalt
from solver.hpsmultidomain.hpsmultidomain import pdo as pdoalt
import solver.hpsmultidomain.hpsmultidomain.geom as hpsGeomalt

from solver.spectralmultidomain.hps import hps_multidomain as hps
from solver.spectralmultidomain.hps import pdo as pdo
import solver.hpsmultidomain.hpsmultidomain.geom as hpsGeom
import SOMS
import time
kh = 10.
def eigvals_compute(disc,ny,xr,n_boxes_y = 2):

    # set up problem
    bnds = np.array([[0,0],[xr,1]])
    match disc:
        case 'hpsalt':
            geom = hpsGeomalt.BoxGeometry(bnds)
            xl = geom.bounds[0,0]
            xr = geom.bounds[1,0]

            def bc(p):
                return np.sin(np.pi*p[:,1])*np.sinh(np.pi*p[:,0])

            def c11(p):
                return torch.ones_like(p[:,0])
            def c22(p):
                return torch.ones_like(p[:,0])
            def c(p):
                return -kh*kh*torch.ones_like(p[:,0])
            diff_op = pdoalt.PDO_2d(c11=c11,c22=c22)#,c=c)
        case 'hps':
            geom = hpsGeom.BoxGeometry(bnds)
            xl = geom.bounds[0,0]
            xr = geom.bounds[1,0]

            def bc(p):
                return np.sin(np.pi*p[...,1])*np.sinh(np.pi*p[...,0])

            def c11(p):
                return jnp.ones_like(p[...,0])
            def c22(p):
                return jnp.ones_like(p[...,0])
            def c(p):
                return -kh*kh*jnp.ones_like(p[...,0])
            diff_op = pdo.PDO2d(c11=c11,c22=c22,c=c)
            p = ny//n_boxes_y
            n_boxes_x = (int)(n_boxes_y*xr)
            ax = .5*xr/n_boxes_x
            ay = .5*1/n_boxes_y
            solver = hps.HPSMultidomain(diff_op,geom,np.array([ax,ay]),p)
            XX = solver.XX
            XXi = XX[solver._Ji,:]
            XXb = XX[solver._Jx,:]
            x,w0 = np.polynomial.legendre.leggauss(p)
            w = np.kron(np.ones(shape=(n_boxes_y,)),np.sqrt(w0))
        case 'stencil':
            geom = stencilGeom.BoxGeometry(bnds)
            xl = geom.bounds[0,0]
            xr = geom.bounds[1,0]

            def bc(p):
                return np.sin(np.pi*p[:,1])*np.sinh(np.pi*p[:,0])

            def c11(p):
                return np.ones_like(p[:,0])
            def c22(p):
                return np.ones_like(p[:,0])
            def c(p):
                return -kh*kh*np.ones_like(p[:,0])
            diff_op = pdo.PDO2d(c11=c11,c22=c22,c=c)
            nx = (int)(xr*ny)
            nx = nx - nx%2 + 1
            solver = stencil.stencilSolver(diff_op,geom,[nx,ny])
            w = np.ones(shape=(ny-2,))
            XXi = solver.XXi
            XXb = solver.XXb

        case 'spectral':
            geom = spectralGeom.BoxGeometry(bnds)
            xl = geom.bounds[0,0]
            xr = geom.bounds[1,0]

            def bc(p):
                return np.sin(np.pi*p[:,1])*np.sinh(np.pi*p[:,0])

            def c11(p):
                return np.ones_like(p[:,0])
            def c22(p):
                return np.ones_like(p[:,0])
            def c(p):
                return -kh*kh*np.ones_like(p[:,0])
            diff_op = pdo.PDO2d(c11=c11,c22=c22,c=c)
            py = ny
            px = (int)(xr*ny)
            px = px + px%2
            solver = spectral.spectralSolver(diff_op,geom,[px,py])
            x,w0 = spectral.clenshaw_curtis_compute(py+1)
            w = np.sqrt(w0[1:len(w0)-1]/2)
            XXi = solver.XXi
            XXb = solver.XXb
        case 'SOMS':
            xl = bnds[0,0]
            xr = bnds[1,0]
            p = ny//n_boxes_y
            py = p
            px = p#(int)(xr*ny)
            #px = px + px%2
            n_boxes_x = (int)(xr*n_boxes_y)
            tic = time.time()
            Sii,Sib,XX,Ii,Ib = SOMS.SOMS_solver(px,py,n_boxes_x,n_boxes_y,kh,2,1)
            toc = time.time()-tic
            print("time SOMS = ",toc)
            print("DOFS SOMS = ",XX.shape[0])
            x,w0 = spectral.clenshaw_curtis_compute(py+1)
            w = np.sqrt(w0[1:len(w0)-1]/2)
            w = np.kron(np.ones(shape=(n_boxes_y,)),w)
            XXi = XX[Ii,:]
            XXb = XX[Ib,:]

    # set up S map

    xc = (xl+xr)/2
    if not disc=='SOMS':
        Aii = np.array(solver._Aii.todense())
        Aib = np.array(solver._Aix.todense())
    else:
        Aii = Sii
        Aib = Sib
    
    xcr = .25
    Ic = np.where(np.abs(XXi[:,0]-xc)<1e-10)[0]
    Icr = np.where(np.abs(XXi[:,1]-xcr)<1e-10)[0]
    Il = np.where((np.abs(XXb[:,0]-xl)<1e-10)&(XXb[:,1]>0)&(XXb[:,1]<1))[0]
    Id = np.where((np.abs(XXb[:,1])<1e-10))[0]
    nx = len(Id)

    S = -(np.linalg.solve(Aii,Aib[:,Il]))[Ic,:]
    S = np.array(S)

    SW = np.diag(w)@S@np.diag(1./w)
    [e,V] = np.linalg.eig(SW)
    Isort = np.argsort(np.abs(e))
    e = e[Isort[::-1]]
    V = V[:,Isort[::-1]]
    
    return e,nx

ny = 1*32
xl = 0.
xr = 2.
xc = (xl+xr)/2



e_SOMS,nx= eigvals_compute('SOMS',ny,xr,1)
e_hps_2,nx= eigvals_compute('hps',ny,xr,1)
e_stencil,nx= eigvals_compute('stencil',ny,xr,1)
plt.figure(1)
plt.semilogy(np.abs(e_stencil))
plt.semilogy(np.abs(e_SOMS))
plt.semilogy(np.abs(e_hps_2))
plt.legend(['stencil','SOMS2','hps2'])


e_SOMS,nx= eigvals_compute('SOMS',ny,xr,2)
e_hps_4,nx= eigvals_compute('hps',ny,xr,2)
e_stencil,nx= eigvals_compute('stencil',ny,xr,1)
plt.figure(2)
plt.semilogy(np.abs(e_stencil))
plt.semilogy(np.abs(e_SOMS))
plt.semilogy(np.abs(e_hps_4))
plt.legend(['stencil','SOMS4','hps4'])
plt.show()
