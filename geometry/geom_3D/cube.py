import numpy as np
import jax.numpy as jnp


bnds = [[0,0,0],[1,1,1]]

def gb_np(p):
    return ((np.abs(p[:,0]-bnds[0][0]))<1e-14) | ((np.abs(p[:,0]-bnds[1][0]))<1e-14) | ((np.abs(p[:,1]-bnds[0][1]))<1e-14) | ((np.abs(p[:,1]-bnds[1][1]))<1e-14) | ((np.abs(p[:,2]-bnds[0][2]))<1e-14) | ((np.abs(p[:,2]-bnds[1][2]))<1e-14)

def gb_jnp(p):
    return ((jnp.abs(p[...,0]-bnds[0][0]))<1e-14) | ((jnp.abs(p[...,0]-bnds[1][0]))<1e-14) | ((jnp.abs(p[...,1]-bnds[0][1]))<1e-14) | ((jnp.abs(p[...,1]-bnds[1][1]))<1e-14) | ((jnp.abs(p[...,2]-bnds[0][2]))<1e-14) | ((jnp.abs(p[...,2]-bnds[1][2]))<1e-14)

def gb(p,jax_avail = True):
    if jax_avail:
        return gb_jnp(p)
    else:
        return gb_np(p)

def box_geom(jax_avail=True):
    if jax_avail:
        return jnp.array(bnds)
    else:
        return np.array(bnds)
def bounds():
    return bnds


def dSlabs(N):
    dSlabs = []
    H = (bnds[1][0]-bnds[0][0])/N
    connectivity=[]
    for n in range(N-1):
        c = bnds[0][0]+(n+1)*H
        bnds_n = [[c-H,bnds[0][1],bnds[0][2]],[c+H,bnds[1][1],bnds[1][2]]]
        if n == 0:
            connectivity+=[[-1,1]]
        elif n == N-2:
            connectivity+=[[n-1,-1]]
        else:
            connectivity+=[[n-1,n+1]]
        dSlabs+=[bnds_n]
    return dSlabs,connectivity,H
