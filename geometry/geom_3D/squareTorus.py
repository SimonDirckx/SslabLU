import numpy as np
import jax.numpy as jnp
from solver.spectralmultidomain.hps.geom              import ParametrizedGeometry3D

from solver.hpsmultidomain.hpsmultidomain.geom import ParametrizedGeometry3D as ParametrizedGeometry3Dalt
import matplotlib.pyplot as plt

import torch

const_theta = 1./(2.*np.pi)
bnds = [[0,0,0],[1.,1.,1.]]


####################################
#           NUMPY VERSION
####################################
    

r_np    = lambda zz: (zz[:,0]**2 + zz[:,1]**2)**0.5

z1_np   = lambda zz: np.multiply( 1 + 1 * zz[:,1], np.cos(zz[:,0]/const_theta) )
z2_np   = lambda zz: np.multiply( 1 + 1 * zz[:,1], np.sin(zz[:,0]/const_theta) )
z3_np   = lambda zz: zz[:,2]


y1_np   = lambda zz: (const_theta* np.atan2(zz[:,1],zz[:,0]))+.5
y2_np   = lambda zz: r_np(zz) - 1
y3_np   = lambda zz: zz[:,2]

y1_d1_np    = lambda zz: -const_theta     * np.divide(zz[:,1], r_np(zz)**2)
y1_d2_np    = lambda zz: +const_theta     * np.divide(zz[:,0], r_np(zz)**2)
y1_d1d1_np  = lambda zz: +2*const_theta   * np.divide(np.multiply(zz[:,0],zz[:,1]), r_np(zz)**4)
y1_d2d2_np  = lambda zz: -2*const_theta   * np.divide(np.multiply(zz[:,0],zz[:,1]), r_np(zz)**4)
y1_d1d1_np  = None; y1_d2d2_np = None


y2_d1_np    = lambda zz: np.divide(zz[:,0], r_np(zz))
y2_d2_np    = lambda zz: np.divide(zz[:,1], r_np(zz))
y2_d1d1_np  = lambda zz: np.divide(zz[:,1]**2, r_np(zz)**3)
y2_d2d2_np  = lambda zz: np.divide(zz[:,0]**2, r_np(zz)**3)

y3_d3_np    = lambda zz: np.ones(shape=zz[:,2].shape)
def gb_np(p):
    return (np.abs(p[:,1]-bnds[0][1])<1e-14) | (np.abs(p[:,1]-bnds[1][1])<1e-14) | (np.abs(p[:,2]-bnds[0][2])<1e-14) | (np.abs(p[:,2]-bnds[1][2])<1e-14)

####################################
#           JAX VERSION
####################################

def r_jnp(p):
    return (p[...,0]**2 + p[...,1]**2)**0.5

z1_jnp = lambda zz: jnp.multiply( 1 + 1 * zz[...,1], jnp.cos(zz[...,0]/const_theta) )
z2_jnp = lambda zz: jnp.multiply( 1 + 1 * zz[...,1], jnp.sin(zz[...,0]/const_theta) )
z3_jnp = lambda zz: zz[...,2]


y1_jnp = lambda zz: (const_theta*jnp.atan2(zz[...,1],zz[...,0]))+.5
y2_jnp = lambda zz: r_jnp(zz) - 1
y3_jnp = lambda zz: zz[...,2]

y1_d1_jnp    = lambda zz: -const_theta     * jnp.divide(zz[...,1], r_jnp(zz)**2)
y1_d2_jnp    = lambda zz: +const_theta     * jnp.divide(zz[...,0], r_jnp(zz)**2)
y1_d1d1_jnp  = lambda zz: +2*const_theta   * jnp.divide(jnp.multiply(zz[...,0],zz[...,1]), r_jnp(zz)**4)
y1_d2d2_jnp  = lambda zz: -2*const_theta   * jnp.divide(jnp.multiply(zz[...,0],zz[...,1]), r_jnp(zz)**4)
y1_d1d1_jnp = None; y1_d2d2_jnp = None


y2_d1_jnp    = lambda zz: jnp.divide(zz[...,0], r_jnp(zz))
y2_d2_jnp    = lambda zz: jnp.divide(zz[...,1], r_jnp(zz))
y2_d1d1_jnp  = lambda zz: jnp.divide(zz[...,1]**2, r_jnp(zz)**3)
y2_d2d2_jnp  = lambda zz: jnp.divide(zz[...,0]**2, r_jnp(zz)**3)

y3_d3_jnp    = lambda zz: jnp.ones(shape=zz[...,2].shape)



def gb_jnp(p):
    return ((jnp.abs(p[...,1]-bnds[0][1]))<1e-14) | ((jnp.abs(p[...,1]-bnds[1][1]))<1e-14) | (jnp.abs(p[...,2]-bnds[0][2])<1e-14) | (jnp.abs(p[...,2]-bnds[1][2])<1e-14)

####################################
#           TORCH VERSION
####################################

r_torch           = lambda zz: (zz[:,0]**2 + zz[:,1]**2)**0.5

z1_torch = lambda zz: torch.mul( 1 + 1 * zz[:,1], torch.cos(zz[:,0]/const_theta) )
z2_torch = lambda zz: torch.mul( 1 + 1 * zz[:,1], torch.sin(zz[:,0]/const_theta) )
z3_torch = lambda zz: zz[:,2]


y1_torch = lambda zz: (const_theta* torch.atan2(zz[:,1],zz[:,0]))+.5
y2_torch = lambda zz: r_np(zz) - 1
y3_torch = lambda zz: zz[:,2]

y1_d1_torch    = lambda zz: -const_theta     * torch.div(zz[:,1], r_np(zz)**2) # r_np works here
y1_d2_torch    = lambda zz: +const_theta     * torch.div(zz[:,0], r_np(zz)**2)
y1_d1d1_torch  = lambda zz: +2*const_theta   * torch.div(torch.mul(zz[:,0],zz[:,1]), r_np(zz)**4)
y1_d2d2_torch  = lambda zz: -2*const_theta   * torch.div(torch.mul(zz[:,0],zz[:,1]), r_np(zz)**4)
y1_d1d1_torch  = None; y1_d2d2 = None


y2_d1_torch    = lambda zz: torch.div(zz[:,0], r_np(zz))
y2_d2_torch    = lambda zz: torch.div(zz[:,1], r_np(zz))
y2_d1d1_torch  = lambda zz: torch.div(zz[:,1]**2, r_np(zz)**3)
y2_d2d2_torch  = lambda zz: torch.div(zz[:,0]**2, r_np(zz)**3)

y3_d3_torch    = lambda zz: torch.ones(zz[:,2].shape)

####################################
#           OVERALL
####################################


def z1(p,jax_avail = True,torch_avail=False):
    if jax_avail:
        return z1_jnp(p)
    elif torch_avail:
        return z1_torch(p)
    else:
        return z1_np(p)
def z2(p,jax_avail = True,torch_avail=False):
    if jax_avail:
        return z2_jnp(p)
    elif torch_avail:
        return z2_torch(p)
    else:
        return z2_np(p)
def z3(p,jax_avail = True,torch_avail=False):
    if jax_avail:
        return z3_jnp(p)
    elif torch_avail:
        return z3_torch(p)
    else:
        return z3_np(p)
    
def y1(p,jax_avail = True,torch_avail=False):
    if jax_avail:
        return y1_jnp(p)
    elif torch_avail:
        return y1_torch(p)
    else:
        return y1_np(p)
def y2(p,jax_avail = True,torch_avail=False):
    if jax_avail:
        return y2_jnp(p)
    elif torch_avail:
        return y2_torch(p)
    else:
        return y2_np(p)
def y3(p,jax_avail = True,torch_avail=False):
    if jax_avail:
        return y3_jnp(p)
    elif torch_avail:
        return y3_torch(p)
    else:
        return y3_np(p)
    
def y1_d1(p,jax_avail = True,torch_avail=False):
    if jax_avail:
        return y1_d1_jnp(p)
    elif torch_avail:
        return y1_d1_torch(p)
    else:
        return y1_d1_np(p)
def y1_d2(p,jax_avail = True,torch_avail=False):
    if jax_avail:
        return y1_d2_jnp(p)
    elif torch_avail:
        return y1_d2_torch(p)
    else:
        return y1_d2_np(p)

    



def y2_d1(p,jax_avail = True,torch_avail=False):
    if jax_avail:
        return y2_d1_jnp(p)
    elif torch_avail:
        return y2_d1_torch(p)
    else:
        return y2_d1_np(p)
def y2_d2(p,jax_avail = True,torch_avail=False):
    if jax_avail:
        return y2_d2_jnp(p)
    elif torch_avail:
        return y2_d2_torch(p)
    else:
        return y2_d2_np(p)

def y3_d3(p,jax_avail = True,torch_avail=False):
    if jax_avail:
        return y3_d3_jnp(p)
    elif torch_avail:
        return y3_d3_torch(p)
    else:
        return y3_d3_np(p)


    
def y2_d1d1(p,jax_avail = True,torch_avail=False):
    if jax_avail:
        return y2_d1d1_jnp(p)
    elif torch_avail:
        return y2_d1d1_torch(p)
    else:
        return y2_d1d1_np(p)
def y2_d2d2(p,jax_avail = True,torch_avail=False):
    if jax_avail:
        return y2_d2d2_jnp(p)
    elif torch_avail:
        return y2_d2d2_torch(p)
    else:
        return y2_d2d2_np(p)
def gb(p,jax_avail = True,torch_avail=False):
    if jax_avail:
        return gb_jnp(p)
    elif torch_avail:
        return gb_np(p)
    else:
        return gb_np(p)

def box_geom(jax_avail=True,torch_avail=False):
    if jax_avail:
        return jnp.array(bnds)
    elif torch_avail:
        return np.array(bnds)
    else:
        return np.array(bnds)
def bounds():
    return bnds

#
# TODO: make hpsmultidomain
#
def param_geom(jax_avail = True,torch_avail=False, hpsalt=False):
    if hpsalt:
        return ParametrizedGeometry3Dalt(
                        box_geom(jax_avail,torch_avail),\
                        z1=lambda p:z1(p,jax_avail,torch_avail),z2=lambda p:z2(p,jax_avail,torch_avail),z3=lambda p:z3(p,jax_avail,torch_avail),\
                        y1=lambda p:y1(p,jax_avail,torch_avail),y2=lambda p:y2(p,jax_avail,torch_avail),y3=lambda p:y3(p,jax_avail,torch_avail),\
                        y1_d1=lambda p:y1_d1(p,jax_avail,torch_avail), y1_d2=lambda p:y1_d2(p,jax_avail,torch_avail),\
                        y2_d1=lambda p:y2_d1(p,jax_avail,torch_avail), y2_d2=lambda p:y2_d2(p,jax_avail,torch_avail),\
                        y3_d3=lambda p:y3_d3(p,jax_avail,torch_avail),\
                        y2_d1d1=lambda p:y2_d1d1(p,jax_avail,torch_avail), y2_d2d2=lambda p:y2_d2d2(p,jax_avail,torch_avail)
                        )
    else:
        return ParametrizedGeometry3D(
                        box_geom(jax_avail,torch_avail),\
                        z1=lambda p:z1(p,jax_avail,torch_avail),z2=lambda p:z2(p,jax_avail,torch_avail),z3=lambda p:z3(p,jax_avail,torch_avail),\
                        y1=lambda p:y1(p,jax_avail,torch_avail),y2=lambda p:y2(p,jax_avail,torch_avail),y3=lambda p:y3(p,jax_avail,torch_avail),\
                        y1_d1=lambda p:y1_d1(p,jax_avail,torch_avail), y1_d2=lambda p:y1_d2(p,jax_avail,torch_avail),\
                        y2_d1=lambda p:y2_d1(p,jax_avail,torch_avail), y2_d2=lambda p:y2_d2(p,jax_avail,torch_avail),\
                        y3_d3=lambda p:y3_d3(p,jax_avail,torch_avail),\
                        y2_d1d1=lambda p:y2_d1d1(p,jax_avail,torch_avail), y2_d2d2=lambda p:y2_d2d2(p,jax_avail,torch_avail)
                        )

####################################
#       slab functions
####################################


# N is number of single slabs!!!
# convention: ith dSlab is the double slab that has the ith interface as its central interface

def dSlabs(N):
    dSlabs = []
    H = (bnds[1][0]-bnds[0][0])/N
    connectivity=[]
    for n in range(N):
        c = bnds[0][0]+n*H
        bnds_n = [[c-H,bnds[0][1],bnds[0][2]],[c+H,bnds[1][1],bnds[1][2]]]
        connectivity+=[[(n-1)%N,(n+1)%N]]
        dSlabs+=[bnds_n]
    return dSlabs,connectivity,H
    
