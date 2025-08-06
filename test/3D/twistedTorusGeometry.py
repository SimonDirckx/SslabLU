
import numpy as np
import jax.numpy as jnp
from hps.geom              import ParametrizedGeometry3D
import matplotlib.pyplot as plt

R = 1.5
bnds = [[0.,0.,0.],[1.,1.,1.]]

####################################
#           NUMPY VERSION
####################################

def z1_np(p):
    c=np.cos(np.pi*p[:,1])
    s=np.sin(np.pi*p[:,1])
    c2 = np.multiply(c,c)
    cs = np.multiply(c,s)
    q = np.multiply(c2,p[:,0])-np.multiply(cs,p[:,2])+c*(R+1)
    return q

def z2_np(p):
    c=np.cos(np.pi*p[:,1])
    s=np.sin(np.pi*p[:,1])
    s2 = np.multiply(s,s)
    cs = np.multiply(c,s)
    q = np.multiply(cs,p[:,0])-np.multiply(s2,p[:,2])+s*(R+1)
    return q
def z3_np(p):
    c=np.cos(np.pi*p[:,1])
    s=np.sin(np.pi*p[:,1])
    q = np.multiply(s,p[:,0])+np.multiply(c,p[:,2])
    return q


def y1_np(p):
    # p is a vector of points, Nx3
    th = np.arctan2(p[:,1],p[:,0])
    c=np.cos(th)
    s=np.sin(th)
    c2 = np.multiply(c,c)
    cs = np.multiply(c,s)
    q = np.multiply(p[:,0],c2)+np.multiply(p[:,1],cs)-(R+1)*c + np.multiply(s,p[:,2])
    return q

def y2_np(p):
    th = np.arctan2(p[:,1],p[:,0])
    return th/np.pi

def y3_np(p):
    th = np.arctan2(p[:,1],p[:,0])
    c=np.cos(th)
    s=np.sin(th)
    s2 = np.multiply(s,s)
    cs = np.multiply(c,s)
    q = -np.multiply(p[:,0],cs)-np.multiply(p[:,1],s2)+(R+1)*s+np.multiply(c,p[:,2])
    return q

#verified
def y1_d1_np(p):
    th = np.arctan2(p[:,1],p[:,0])
    c2t = np.cos(2*th)
    s2t = np.sin(2*th)
    s   = np.sin(th)
    c   = np.cos(th)
    r2  = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    A   = (s2t*p[:,0]*p[:,1] - c2t*p[:,1]**2 - (R+1)*s*p[:,1] - p[:,2]*p[:,1]*c)
    return (c2t+1)/2. + A/r2

#verified
def y1_d2_np(p):
    th = np.arctan2(p[:,1],p[:,0])
    c2t = np.cos(2*th)
    s2t = np.sin(2*th)
    s   = np.sin(th)
    c   = np.cos(th)
    r2  = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    A   = ( c2t*p[:,0]*p[:,1] - s2t*p[:,0]*p[:,0] + (R+1)*s*p[:,0] + p[:,2]*p[:,0]*c)
    return s2t/2. + A/r2

def y1_d3_np(p):
    th = np.arctan2(p[:,1],p[:,0])
    return np.sin(th)

#verified
def y2_d1_np(p):
    r2 = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    return -(p[:,1]/r2)/np.pi
#verified
def y2_d2_np(p):
    r2 = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    return (p[:,0]/r2)/np.pi


def y3_d1_np(p):
    th = np.arctan2(p[:,1],p[:,0])
    c2t = np.cos(2*th)
    s2t = np.sin(2*th)
    s   = np.sin(th)
    c   = np.cos(th)
    r2  = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    A   = (c2t*p[:,0]*p[:,1] + s2t*p[:,1]**2 - (R+1)*c*p[:,1] + p[:,2]*p[:,1]*s)
    return -s2t/2. + A/r2

def y3_d2_np(p):
    th = np.arctan2(p[:,1],p[:,0])
    c2t = np.cos(2*th)
    s2t = np.sin(2*th)
    s   = np.sin(th)
    c   = np.cos(th)
    r2  = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    A   = (-s2t*p[:,0]*p[:,1] - c2t*(p[:,0]**2) + (R+1)*c*p[:,0] - p[:,2]*p[:,0]*s)
    return (c2t-1.)/2. + A/r2
#verified
def y3_d3_np(p):
    th = np.arctan2(p[:,1],p[:,0])
    return np.cos(th)

#verified
def y1_d1d1_np(p):
    th = np.arctan2(p[:,1],p[:,0])
    c2t = np.cos(2*th)
    s2t = np.sin(2*th)
    s   = np.sin(th)
    c   = np.cos(th)
    r2  = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    r4=r2**2
    A   = (s2t*p[:,0]*p[:,1] - c2t*p[:,1]*p[:,1] - (R+1)*s*p[:,1] - p[:,2]*p[:,1]*c)
    dA = p[:,1]*s2t-(2*c2t*p[:,0]*(p[:,1]**2)+2*s2t*p[:,1]**3-(R+1)*c*p[:,1]**2 + p[:,2]*s*(p[:,1]**2) )/r2
    return p[:,1]*s2t/r2 + (dA*r2 - 2*A*p[:,0])/r4
#verified
def y1_d2d2_np(p):
    th = np.arctan2(p[:,1],p[:,0])
    c2t = np.cos(2*th)
    s2t = np.sin(2*th)
    s   = np.sin(th)
    c   = np.cos(th)
    r2  = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    r4 = r2**2
    A   = ( c2t*p[:,0]*p[:,1] - s2t*p[:,0]*p[:,0] + (R+1)*s*p[:,0] + p[:,2]*p[:,0]*c)
    dA  = p[:,0]*c2t-(2*s2t*p[:,1]*p[:,0]**2+2*c2t*(p[:,0]**3)-(R+1)*c*(p[:,0]**2)+p[:,2]*s*(p[:,0]**2))/r2
    return c2t*p[:,0]/r2 + (dA*r2 - 2*A*p[:,1])/r4
#verified
def y2_d1d1_np(p):
    r2 = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    return (2*p[:,1]*p[:,0]/r2)/(r2*np.pi)
#verified
def y2_d2d2_np(p):
    r2 = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    r4 = r2**2
    return -2*(p[:,0]*p[:,1])/(r4*np.pi)

#verified
def y3_d1d1_np(p):
    th = np.arctan2(p[:,1],p[:,0])
    c2t = np.cos(2*th)
    s2t = np.sin(2*th)
    s   = np.sin(th)
    c   = np.cos(th)
    r2  = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    r4 = r2**2
    A   = (c2t*p[:,0]*p[:,1] + s2t*p[:,1]**2 - (R+1)*c*p[:,1] + p[:,2]*p[:,1]*s)
    dA  = p[:,1]*c2t+(2*p[:,0]*(p[:,1]**2)*s2t-2*(p[:,1]**3)*c2t-(R+1)*(p[:,1]**2)*s-p[:,2]*(p[:,1]**2)*c)/r2
    return c2t*p[:,1]/r2 + (dA*r2 - 2*p[:,0]*A)/r4
#verified
def y3_d2d2_np(p):
    th = np.arctan2(p[:,1],p[:,0])
    c2t = np.cos(2*th)
    s2t = np.sin(2*th)
    s   = np.sin(th)
    c   = np.cos(th)
    r2  = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    r4 = r2**2
    A   = (-s2t*p[:,0]*p[:,1] - c2t*(p[:,0]**2) + (R+1)*c*p[:,0] - p[:,2]*p[:,0]*s)
    dA  = -p[:,0]*s2t+(-2*(p[:,0]**2)*p[:,1]*c2t +2*(p[:,0]**3)*s2t - (R+1)*(p[:,0]**2)*s-(p[:,0]**2)*p[:,2]*c)/r2
    return -s2t*p[:,0]/r2 + (dA*r2 - 2*p[:,1]*A)/r4
def gb_np(p):
    return ((np.abs(p[:,1]-bnds[0][1]))<1e-14) | ((np.abs(p[:,1]-bnds[1][1]))<1e-14) | (np.abs(p[:,2]-bnds[0][2])<1e-14) | (np.abs(p[:,2]-bnds[1][2])<1e-14)


####################################
#           JAX VERSION
####################################

def z1_jnp(p):
    c=jnp.cos(jnp.pi*p[...,1])
    s=jnp.sin(jnp.pi*p[...,1])
    c2 = jnp.multiply(c,c)
    cs = jnp.multiply(c,s)
    q = jnp.multiply(c2,p[...,0])-jnp.multiply(cs,p[...,2])+c*(R+1)
    return q

def z2_jnp(p):
    c=jnp.cos(jnp.pi*p[...,1])
    s=jnp.sin(jnp.pi*p[...,1])
    s2 = jnp.multiply(s,s)
    cs = jnp.multiply(c,s)
    q = jnp.multiply(cs,p[...,0])-jnp.multiply(s2,p[...,2])+s*(R+1)
    return q
def z3_jnp(p):
    c=jnp.cos(jnp.pi*p[...,1])
    s=jnp.sin(jnp.pi*p[...,1])
    q = jnp.multiply(s,p[...,0])+jnp.multiply(c,p[...,2])
    return q


def y1_jnp(p):
    # p is a vector of points, Nx3
    th = jnp.arctan2(p[...,1],p[...,0])
    c=jnp.cos(th)
    s=jnp.sin(th)
    c2 = jnp.multiply(c,c)
    cs = jnp.multiply(c,s)
    q = jnp.multiply(p[...,0],c2)+jnp.multiply(p[...,1],cs)-(R+1)*c + jnp.multiply(s,p[...,2])
    return q

def y2_jnp(p):
    th = jnp.arctan2(p[...,1],p[...,0])
    return th/jnp.pi

def y3_jnp(p):
    th = jnp.arctan2(p[...,1],p[...,0])
    c=jnp.cos(th)
    s=jnp.sin(th)
    s2 = jnp.multiply(s,s)
    cs = jnp.multiply(c,s)
    q = -jnp.multiply(p[...,0],cs)-jnp.multiply(p[...,1],s2)+(R+1)*s+jnp.multiply(c,p[...,2])
    return q

#verified
def y1_d1_jnp(p):
    th = jnp.arctan2(p[...,1],p[...,0])
    c2t = jnp.cos(2*th)
    s2t = jnp.sin(2*th)
    s   = jnp.sin(th)
    c   = jnp.cos(th)
    r2  = p[...,0]*p[...,0]+p[...,1]*p[...,1]
    A   = (s2t*p[...,0]*p[...,1] - c2t*p[...,1]**2 - (R+1)*s*p[...,1] - p[...,2]*p[...,1]*c)
    return (c2t+1)/2. + A/r2

#verified
def y1_d2_jnp(p):
    th = jnp.arctan2(p[...,1],p[...,0])
    c2t = jnp.cos(2*th)
    s2t = jnp.sin(2*th)
    s   = jnp.sin(th)
    c   = jnp.cos(th)
    r2  = p[...,0]*p[...,0]+p[...,1]*p[...,1]
    A   = ( c2t*p[...,0]*p[...,1] - s2t*p[...,0]*p[...,0] + (R+1)*s*p[...,0] + p[...,2]*p[...,0]*c)
    return s2t/2. + A/r2

def y1_d3_jnp(p):
    th = jnp.arctan2(p[...,1],p[...,0])
    return jnp.sin(th)

#verified
def y2_d1_jnp(p):
    r2 = p[...,0]*p[...,0]+p[...,1]*p[...,1]
    return -(p[...,1]/r2)/jnp.pi
#verified
def y2_d2_jnp(p):
    r2 = p[...,0]*p[...,0]+p[...,1]*p[...,1]
    return (p[...,0]/r2)/jnp.pi


def y3_d1_jnp(p):
    th = jnp.arctan2(p[...,1],p[...,0])
    c2t = jnp.cos(2*th)
    s2t = jnp.sin(2*th)
    s   = jnp.sin(th)
    c   = jnp.cos(th)
    r2  = p[...,0]*p[...,0]+p[...,1]*p[...,1]
    A   = (c2t*p[...,0]*p[...,1] + s2t*p[...,1]**2 - (R+1)*c*p[...,1] + p[...,2]*p[...,1]*s)
    return -s2t/2. + A/r2

def y3_d2_jnp(p):
    th = jnp.arctan2(p[...,1],p[...,0])
    c2t = jnp.cos(2*th)
    s2t = jnp.sin(2*th)
    s   = jnp.sin(th)
    c   = jnp.cos(th)
    r2  = p[...,0]*p[...,0]+p[...,1]*p[...,1]
    A   = (-s2t*p[...,0]*p[...,1] - c2t*(p[...,0]**2) + (R+1)*c*p[...,0] - p[...,2]*p[...,0]*s)
    return (c2t-1.)/2. + A/r2
#verified
def y3_d3_jnp(p):
    th = jnp.arctan2(p[...,1],p[...,0])
    return jnp.cos(th)

#verified
def y1_d1d1_jnp(p):
    th = jnp.arctan2(p[...,1],p[...,0])
    c2t = jnp.cos(2*th)
    s2t = jnp.sin(2*th)
    s   = jnp.sin(th)
    c   = jnp.cos(th)
    r2  = p[...,0]*p[...,0]+p[...,1]*p[...,1]
    r4=r2**2
    A   = (s2t*p[...,0]*p[...,1] - c2t*p[...,1]*p[...,1] - (R+1)*s*p[...,1] - p[...,2]*p[...,1]*c)
    dA = p[...,1]*s2t-(2*c2t*p[...,0]*(p[...,1]**2)+2*s2t*p[...,1]**3-(R+1)*c*p[...,1]**2 + p[...,2]*s*(p[...,1]**2) )/r2
    return p[...,1]*s2t/r2 + (dA*r2 - 2*A*p[...,0])/r4
#verified
def y1_d2d2_jnp(p):
    th = jnp.arctan2(p[...,1],p[...,0])
    c2t = jnp.cos(2*th)
    s2t = jnp.sin(2*th)
    s   = jnp.sin(th)
    c   = jnp.cos(th)
    r2  = p[...,0]*p[...,0]+p[...,1]*p[...,1]
    r4 = r2**2
    A   = ( c2t*p[...,0]*p[...,1] - s2t*p[...,0]*p[...,0] + (R+1)*s*p[...,0] + p[...,2]*p[...,0]*c)
    dA  = p[...,0]*c2t-(2*s2t*p[...,1]*p[...,0]**2+2*c2t*(p[...,0]**3)-(R+1)*c*(p[...,0]**2)+p[...,2]*s*(p[...,0]**2))/r2
    return c2t*p[...,0]/r2 + (dA*r2 - 2*A*p[...,1])/r4
#verified
def y2_d1d1_jnp(p):
    r2 = p[...,0]*p[...,0]+p[...,1]*p[...,1]
    return (2*p[...,1]*p[...,0]/r2)/(r2*jnp.pi)
#verified
def y2_d2d2_jnp(p):
    r2 = p[...,0]*p[...,0]+p[...,1]*p[...,1]
    r4 = r2**2
    return -2*(p[...,0]*p[...,1])/(r4*jnp.pi)

#verified
def y3_d1d1_jnp(p):
    th = jnp.arctan2(p[...,1],p[...,0])
    c2t = jnp.cos(2*th)
    s2t = jnp.sin(2*th)
    s   = jnp.sin(th)
    c   = jnp.cos(th)
    r2  = p[...,0]*p[...,0]+p[...,1]*p[...,1]
    r4 = r2**2
    A   = (c2t*p[...,0]*p[...,1] + s2t*p[...,1]**2 - (R+1)*c*p[...,1] + p[...,2]*p[...,1]*s)
    dA  = p[...,1]*c2t+(2*p[...,0]*(p[...,1]**2)*s2t-2*(p[...,1]**3)*c2t-(R+1)*(p[...,1]**2)*s-p[...,2]*(p[...,1]**2)*c)/r2
    return c2t*p[...,1]/r2 + (dA*r2 - 2*p[...,0]*A)/r4
#verified
def y3_d2d2_jnp(p):
    th = jnp.arctan2(p[...,1],p[...,0])
    c2t = jnp.cos(2*th)
    s2t = jnp.sin(2*th)
    s   = jnp.sin(th)
    c   = jnp.cos(th)
    r2  = p[...,0]*p[...,0]+p[...,1]*p[...,1]
    r4 = r2**2
    A   = (-s2t*p[...,0]*p[...,1] - c2t*(p[...,0]**2) + (R+1)*c*p[...,0] - p[...,2]*p[...,0]*s)
    dA  = -p[...,0]*s2t+(-2*(p[...,0]**2)*p[...,1]*c2t +2*(p[...,0]**3)*s2t - (R+1)*(p[...,0]**2)*s-(p[...,0]**2)*p[...,2]*c)/r2
    return -s2t*p[...,0]/r2 + (dA*r2 - 2*p[...,1]*A)/r4

def gb_jnp(p):
    return ((jnp.abs(p[...,1]-bnds[0][1]))<1e-14) | ((jnp.abs(p[...,1]-bnds[1][1]))<1e-14) | (jnp.abs(p[...,2]-bnds[0][2])<1e-14) | (jnp.abs(p[...,2]-bnds[1][2])<1e-14)

####################################
#           OVERALL
####################################


def z1(p,jax_avail = True):
    if jax_avail:
        return z1_jnp(p)
    else:
        return z1_np(p)
def z2(p,jax_avail = True):
    if jax_avail:
        return z2_jnp(p)
    else:
        return z2_np(p)
def z3(p,jax_avail = True):
    if jax_avail:
        return z3_jnp(p)
    else:
        return z3_np(p)
    
def y1(p,jax_avail = True):
    if jax_avail:
        return y1_jnp(p)
    else:
        return y1_np(p)
def y2(p,jax_avail = True):
    if jax_avail:
        return y2_jnp(p)
    else:
        return y2_np(p)
def y3(p,jax_avail = True):
    if jax_avail:
        return y3_jnp(p)
    else:
        return y3_np(p)
    
def y1_d1(p,jax_avail = True):
    if jax_avail:
        return y1_d1_jnp(p)
    else:
        return y1_d1_np(p)
def y1_d2(p,jax_avail = True):
    if jax_avail:
        return y1_d2_jnp(p)
    else:
        return y1_d2_np(p)
def y1_d3(p,jax_avail = True):
    if jax_avail:
        return y1_d3_jnp(p)
    else:
        return y1_d3_np(p)
    

def y2_d1(p,jax_avail = True):
    if jax_avail:
        return y2_d1_jnp(p)
    else:
        return y2_d1_np(p)
def y2_d2(p,jax_avail = True):
    if jax_avail:
        return y2_d2_jnp(p)
    else:
        return y2_d2_np(p)

def y3_d1(p,jax_avail = True):
    if jax_avail:
        return y3_d1_jnp(p)
    else:
        return y3_d1_np(p)
def y3_d2(p,jax_avail = True):
    if jax_avail:
        return y3_d2_jnp(p)
    else:
        return y3_d2_np(p)
def y3_d3(p,jax_avail = True):
    if jax_avail:
        return y3_d3_jnp(p)
    else:
        return y3_d3_np(p)
    
def y1_d1d1(p,jax_avail = True):
    if jax_avail:
        return y1_d1d1_jnp(p)
    else:
        return y1_d1d1_np(p)
def y1_d2d2(p,jax_avail = True):
    if jax_avail:
        return y1_d2d2_jnp(p)
    else:
        return y1_d2d2_np(p)
    
def y2_d1d1(p,jax_avail = True):
    if jax_avail:
        return y2_d1d1_jnp(p)
    else:
        return y2_d1d1_np(p)
def y2_d2d2(p,jax_avail = True):
    if jax_avail:
        return y2_d2d2_jnp(p)
    else:
        return y2_d2d2_np(p)

def y3_d1d1(p,jax_avail = True):
    if jax_avail:
        return y3_d1d1_jnp(p)
    else:
        return y3_d1d1_np(p)
def y3_d2d2(p,jax_avail = True):
    if jax_avail:
        return y3_d2d2_jnp(p)
    else:
        return y3_d2d2_np(p)
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
def param_geom(jax_avail = True):
    return ParametrizedGeometry3D(
                        box_geom(jax_avail),\
                        z1=lambda p:z1(p,jax_avail),z2=lambda p:z2(p,jax_avail),z3=lambda p:z3(p,jax_avail),\
                        y1=lambda p:y1(p,jax_avail),y2=lambda p:y2(p,jax_avail),y3=lambda p:y3(p,jax_avail),\
                        y1_d1=lambda p:y1_d1(p,jax_avail), y1_d2=lambda p:y1_d2(p,jax_avail),y1_d3=lambda p:y1_d3(p,jax_avail),\
                        y2_d1=lambda p:y2_d1(p,jax_avail), y2_d2=lambda p:y2_d2(p,jax_avail),\
                        y3_d1=lambda p:y3_d1(p,jax_avail), y3_d2=lambda p:y3_d2(p,jax_avail),y3_d3=lambda p:y3_d3(p,jax_avail),\
                        y1_d1d1=lambda p:y1_d1d1(p,jax_avail), y1_d2d2=lambda p:y1_d2d2(p,jax_avail),\
                        y2_d1d1=lambda p:y2_d1d1(p,jax_avail), y2_d2d2=lambda p:y2_d2d2(p,jax_avail),\
                        y3_d1d1=lambda p:y3_d1d1(p,jax_avail), y3_d2d2=lambda p:y3_d2d2(p,jax_avail)
                        )

####################################
#        PLOT TAYLOR ERR
####################################

#first deriv should decay as h^2
#second deriv should decay as h^3

def check_param():
    p=np.random.uniform(size=(500,3))#,low=.25,high=.75)


    def ydiff(z,ypos,zpos):
        match zpos:
            case 0:
                match ypos:
                    case 0:
                        return y1_d1(z,False)
                    case 1:
                        return y2_d1(z,False)
                    case 2:
                        return y3_d1(z,False)
            case 1:
                match ypos:
                    case 0:
                        return y1_d2(z,False)
                    case 1:
                        return y2_d2(z,False)
                    case 2:
                        return y3_d2(z,False)
            case 2:
                match ypos:
                    case 0:
                        return y1_d3(z,False)
                    case 1:
                        return np.zeros_like(z[:,1])
                    case 2:
                        return y3_d3(z,False)

                        
    def yddiff(z,ypos,zpos):
        match zpos:
            case 0:
                match ypos:
                    case 0:
                        return y1_d1d1(z,False)
                    case 1:
                        return y2_d1d1(z,False)
                    case 2:
                        return y3_d1d1(z,False)
            case 1:
                match ypos:
                    case 0:
                        return y1_d2d2(z,False)
                    case 1:
                        return y2_d2d2(z,False)
                    case 2:
                        return y3_d2d2(z,False)
            case 2:
                match ypos:
                    case 0:
                        return np.zeros_like(z[:,0])
                    case 1:
                        return np.zeros_like(z[:,1])
                    case 2:
                        return np.zeros_like(z[:,2])


    kvec = np.array([2,3,4,5,6])
    hvec = 1./(2**kvec)
    errdiff = np.zeros(shape=(len(hvec),9))
    errddiff = np.zeros(shape=(len(hvec),9))
    errTest = np.zeros(shape=(len(hvec),))
    z=np.zeros(shape= p.shape)
    z[:,0] = z1(p,False)
    z[:,1] = z2(p,False)
    z[:,2] = z3(p,False)
    for hind in range(len(hvec)):
        for ypos in range(0,3):
            for zpos in range(0,3):
                dz = hvec[hind]
                zdz=np.zeros(shape= z.shape)
                zdz[:,0] = z[:,0]
                zdz[:,1] = z[:,1]
                zdz[:,2] = z[:,2]
                zdz[:,zpos]+=dz

                y=np.zeros(shape= z.shape)
                y[:,0] = y1(z,False)
                y[:,1] = y2(z,False)
                y[:,2] = y3(z,False)

                ydy=np.zeros(shape= zdz.shape)
                ydy[:,0] = y1(zdz,False)
                ydy[:,1] = y2(zdz,False)
                ydy[:,2] = y3(zdz,False)

                diffy = ydiff(z,ypos,zpos)
                ddiffy = yddiff(z,ypos,zpos)

                #compute inf. norm of Taylor err.
                errdiff[hind,ypos+3*zpos]=np.linalg.norm(ydy[:,ypos]-y[:,ypos]-diffy*dz,ord=np.inf)
                errddiff[hind,ypos+3*zpos]=np.linalg.norm(ydy[:,ypos]-y[:,ypos]-diffy*dz-ddiffy*dz*dz/2.,ord=np.inf)


    print(errdiff[:,7])
    print(errddiff[:,7])
    for i in range(9):
        plt.figure(i)
        plt.loglog(hvec,errdiff[:,i],label='diff')
        plt.loglog(hvec,errddiff[:,i],label='ddiff')
        plt.loglog(hvec,2*errdiff[0,i]*(hvec**2)/(hvec[0]**2),label='h^2',linestyle='dashed')
        plt.loglog(hvec,2*errddiff[0,i]*(hvec**3)/(hvec[0]**3),label='h^3',linestyle='dashed')
        plt.legend()
    plt.show()