
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import torch

from solver.spectralmultidomain.hps.geom              import ParametrizedGeometry3D
from solver.hpsmultidomain.hpsmultidomain.geom import ParametrizedGeometry3D as ParametrizedGeometry3Dalt

R = 1.5
bnds = [[0,0,0],[1.,1.,1.]]


####################################
#           NUMPY VERSION
####################################

def z1_np(p):
    q = 2*p-1
    c=np.cos(np.pi*q[:,0])
    s=np.sin(np.pi*q[:,0])
    c2 = np.multiply(c,c)
    cs = np.multiply(c,s)
    z1 = np.multiply(c2,q[:,1])-np.multiply(cs,q[:,2])+c*(R+1)
    return z1

def z2_np(p):
    q = 2*p-1
    c=np.cos(np.pi*q[:,0])
    s=np.sin(np.pi*q[:,0])
    s2 = np.multiply(s,s)
    cs = np.multiply(c,s)
    z2 = np.multiply(cs,q[:,1])-np.multiply(s2,q[:,2])+s*(R+1)
    return z2
def z3_np(p):
    q=2*p-1
    c=np.cos(np.pi*q[:,0])
    s=np.sin(np.pi*q[:,0])
    z3 = np.multiply(s,q[:,1])+np.multiply(c,q[:,2])
    return z3

def y1_np(p):
    th = np.arctan2(p[:,1],p[:,0])
    return ((th/np.pi)+1)/2

def y2_np(p):
    # p is a vector of points, Nx3
    th = np.arctan2(p[:,1],p[:,0])
    c=np.cos(th)
    s=np.sin(th)
    c2 = np.multiply(c,c)
    cs = np.multiply(c,s)
    q = np.multiply(p[:,0],c2)+np.multiply(p[:,1],cs)-(R+1)*c + np.multiply(s,p[:,2])
    return (q+1)/2


def y3_np(p):
    th = np.arctan2(p[:,1],p[:,0])
    c=np.cos(th)
    s=np.sin(th)
    s2 = np.multiply(s,s)
    cs = np.multiply(c,s)
    q = -np.multiply(p[:,0],cs)-np.multiply(p[:,1],s2)+(R+1)*s+np.multiply(c,p[:,2])
    return (q+1)/2

#verified
def y2_d1_np(p):
    th = np.arctan2(p[:,1],p[:,0])
    c2t = np.cos(2*th)
    s2t = np.sin(2*th)
    s   = np.sin(th)
    c   = np.cos(th)
    r2  = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    A   = (s2t*p[:,0]*p[:,1] - c2t*p[:,1]**2 - (R+1)*s*p[:,1] - p[:,2]*p[:,1]*c)
    return ((c2t+1)/2 + A/r2)/2

#verified
def y1_d1_np(p):
    r2 = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    return -((p[:,1]/r2)/np.pi)/2
#verified
def y1_d2_np(p):
    r2 = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    return ((p[:,0]/r2)/np.pi)/2

#verified
def y2_d2_np(p):
    th = np.arctan2(p[:,1],p[:,0])
    c2t = np.cos(2*th)
    s2t = np.sin(2*th)
    s   = np.sin(th)
    c   = np.cos(th)
    r2  = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    A   = ( c2t*p[:,0]*p[:,1] - s2t*p[:,0]*p[:,0] + (R+1)*s*p[:,0] + p[:,2]*p[:,0]*c)
    return (s2t/2. + A/r2)/2

def y2_d3_np(p):
    th = np.arctan2(p[:,1],p[:,0])
    return (np.sin(th))/2




def y3_d1_np(p):
    th = np.arctan2(p[:,1],p[:,0])
    c2t = np.cos(2*th)
    s2t = np.sin(2*th)
    s   = np.sin(th)
    c   = np.cos(th)
    r2  = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    A   = (c2t*p[:,0]*p[:,1] + s2t*p[:,1]**2 - (R+1)*c*p[:,1] + p[:,2]*p[:,1]*s)
    return (-s2t/2. + A/r2)/2

def y3_d2_np(p):
    th = np.arctan2(p[:,1],p[:,0])
    c2t = np.cos(2*th)
    s2t = np.sin(2*th)
    s   = np.sin(th)
    c   = np.cos(th)
    r2  = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    A   = (s2t*p[:,0]*p[:,1] + c2t*p[:,0]*p[:,0] - (R+1)*c*p[:,0] + p[:,2]*p[:,0]*s)
    return ((c2t-1)/2. - A/r2)/2
#verified
def y3_d3_np(p):
    th = np.arctan2(p[:,1],p[:,0])
    return np.cos(th)/2


#verified
def y1_d1d1_np(p):
    r2 = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    return ((2*p[:,1]*p[:,0]/r2)/(r2*np.pi))/2
#verified
def y1_d2d2_np(p):
    r2 = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    return -(2*(p[:,0]*p[:,1]/r2)/(r2*np.pi))/2

#verified
def y2_d1d1_np(p):
    th = np.arctan2(p[:,1],p[:,0])
    c2t = np.cos(2*th)
    s2t = np.sin(2*th)
    s   = np.sin(th)
    c   = np.cos(th)
    r2  = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    A   = (s2t*p[:,0]*p[:,1] - c2t*p[:,1]*p[:,1] - (R+1)*s*p[:,1] - p[:,2]*p[:,1]*c)
    dA = p[:,1]*s2t-(2*c2t*p[:,0]*(p[:,1]**2)+2*s2t*p[:,1]**3-(R+1)*c*p[:,1]**2 + p[:,2]*s*(p[:,1]**2) )/r2
    return ((p[:,1]*s2t + dA - 2*A*p[:,0]/r2 )/r2)/2
#verified
def y2_d2d2_np(p):
    th = np.arctan2(p[:,1],p[:,0])
    c2t = np.cos(2*th)
    s2t = np.sin(2*th)
    s   = np.sin(th)
    c   = np.cos(th)
    r2  = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    A   = ( c2t*p[:,0]*p[:,1] - s2t*p[:,0]*p[:,0] + (R+1)*s*p[:,0] + p[:,2]*p[:,0]*c)
    dA  = p[:,0]*c2t-(2*s2t*p[:,1]*p[:,0]**2+2*c2t*(p[:,0]**3)-(R+1)*c*(p[:,0]**2)+p[:,2]*s*(p[:,0]**2))/r2
    return ((c2t*p[:,0] + dA - 2*A*p[:,1]/r2 )/r2)/2


#verified
def y3_d1d1_np(p):
    th = np.arctan2(p[:,1],p[:,0])
    c2t = np.cos(2*th)
    s2t = np.sin(2*th)
    s   = np.sin(th)
    c   = np.cos(th)
    r2  = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    A   = (c2t*p[:,0]*p[:,1] + s2t*p[:,1]**2 - (R+1)*c*p[:,1] + p[:,2]*p[:,1]*s)
    dA  = p[:,1]*c2t+(2*p[:,0]*(p[:,1]**2)*s2t-2*(p[:,1]**3)*c2t-(R+1)*(p[:,1]**2)*s-p[:,2]*(p[:,1]**2)*c)/r2
    return ((c2t*p[:,1] + dA - 2*p[:,0]*A/r2)/r2)/2
#verified
def y3_d2d2_np(p):
    th = np.arctan2(p[:,1],p[:,0])
    c2t = np.cos(2*th)
    s2t = np.sin(2*th)
    s   = np.sin(th)
    c   = np.cos(th)
    r2  = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    A   = (s2t*p[:,0]*p[:,1] + c2t*p[:,0]*p[:,0] - (R+1)*c*p[:,0] + p[:,2]*p[:,0]*s)
    dA  = p[:,0]*s2t+(2*(p[:,0]**2)*p[:,1]*c2t-2*(p[:,0]**3)*s2t + (R+1)*(p[:,0]**2)*s + p[:,2]*(p[:,0]**2)*c)/r2
    return -((s2t*p[:,0] + dA - 2*p[:,1]*A/r2)/r2)/2
def gb_np(p):
    return ((np.abs(p[:,1]-bnds[0][1]))<1e-14) | ((np.abs(p[:,1]-bnds[1][1]))<1e-14) | (np.abs(p[:,2]-bnds[0][2])<1e-14) | (np.abs(p[:,2]-bnds[1][2])<1e-14)


####################################
#           JAX VERSION
####################################

def z1_jnp(p):
    q = 2*p-1
    c=jnp.cos(jnp.pi*q[...,0])
    s=jnp.sin(jnp.pi*q[...,0])
    c2 = jnp.multiply(c,c)
    cs = jnp.multiply(c,s)
    z1 = jnp.multiply(c2,q[...,1])-jnp.multiply(cs,q[...,2])+c*(R+1)
    return z1

def z2_jnp(p):
    q = 2*p-1
    c=jnp.cos(jnp.pi*q[...,0])
    s=jnp.sin(jnp.pi*q[...,0])
    s2 = jnp.multiply(s,s)
    cs = jnp.multiply(c,s)
    z2 = jnp.multiply(cs,q[...,1])-jnp.multiply(s2,q[...,2])+s*(R+1)
    return z2
def z3_jnp(p):
    q=2*p-1
    c=jnp.cos(jnp.pi*q[...,0])
    s=jnp.sin(jnp.pi*q[...,0])
    z3 = jnp.multiply(s,q[...,1])+jnp.multiply(c,q[...,2])
    return z3

def y1_jnp(p):
    th = jnp.arctan2(p[...,1],p[...,0])
    return ((th/jnp.pi)+1)/2

def y2_jnp(p):
    # p is a vector of points, Nx3
    th = jnp.arctan2(p[...,1],p[...,0])
    c=jnp.cos(th)
    s=jnp.sin(th)
    c2 = jnp.multiply(c,c)
    cs = jnp.multiply(c,s)
    q = jnp.multiply(p[...,0],c2)+jnp.multiply(p[...,1],cs)-(R+1)*c + jnp.multiply(s,p[...,2])
    return (q+1)/2


def y3_jnp(p):
    th = jnp.arctan2(p[...,1],p[...,0])
    c=jnp.cos(th)
    s=jnp.sin(th)
    s2 = jnp.multiply(s,s)
    cs = jnp.multiply(c,s)
    q = -jnp.multiply(p[...,0],cs)-jnp.multiply(p[...,1],s2)+(R+1)*s+jnp.multiply(c,p[...,2])
    return (q+1)/2

#verified
def y2_d1_jnp(p):
    th = jnp.arctan2(p[...,1],p[...,0])
    c2t = jnp.cos(2*th)
    s2t = jnp.sin(2*th)
    s   = jnp.sin(th)
    c   = jnp.cos(th)
    r2  = p[...,0]*p[...,0]+p[...,1]*p[...,1]
    A   = (s2t*p[...,0]*p[...,1] - c2t*p[...,1]**2 - (R+1)*s*p[...,1] - p[...,2]*p[...,1]*c)
    return ((c2t+1)/2. + A/r2)/2

#verified
def y1_d1_jnp(p):
    r2 = p[...,0]*p[...,0]+p[...,1]*p[...,1]
    return -((p[...,1]/r2)/jnp.pi)/2
#verified
def y1_d2_jnp(p):
    r2 = p[...,0]*p[...,0]+p[...,1]*p[...,1]
    return ((p[...,0]/r2)/jnp.pi)/2

#verified
def y2_d2_jnp(p):
    th = jnp.arctan2(p[...,1],p[...,0])
    c2t = jnp.cos(2*th)
    s2t = jnp.sin(2*th)
    s   = jnp.sin(th)
    c   = jnp.cos(th)
    r2  = p[...,0]*p[...,0]+p[...,1]*p[...,1]
    A   = ( c2t*p[...,0]*p[...,1] - s2t*p[...,0]*p[...,0] + (R+1)*s*p[...,0] + p[...,2]*p[...,0]*c)
    return (s2t/2. + A/r2)/2

def y2_d3_jnp(p):
    th = jnp.arctan2(p[...,1],p[...,0])
    return (jnp.sin(th))/2




def y3_d1_jnp(p):
    th = jnp.arctan2(p[...,1],p[...,0])
    c2t = jnp.cos(2*th)
    s2t = jnp.sin(2*th)
    s   = jnp.sin(th)
    c   = jnp.cos(th)
    r2  = p[...,0]*p[...,0]+p[...,1]*p[...,1]
    A   = (c2t*p[...,0]*p[...,1] + s2t*p[...,1]**2 - (R+1)*c*p[...,1] + p[...,2]*p[...,1]*s)
    return (-s2t/2. + A/r2)/2

def y3_d2_jnp(p):
    th = jnp.arctan2(p[...,1],p[...,0])
    c2t = jnp.cos(2*th)
    s2t = jnp.sin(2*th)
    s   = jnp.sin(th)
    c   = jnp.cos(th)
    r2  = p[...,0]*p[...,0]+p[...,1]*p[...,1]
    A   = (s2t*p[...,0]*p[...,1] + c2t*p[...,0]*p[...,0] - (R+1)*c*p[...,0] + p[...,2]*p[...,0]*s)
    return ((c2t-1)/2. - A/r2)/2
#verified
def y3_d3_jnp(p):
    th = jnp.arctan2(p[...,1],p[...,0])
    return jnp.cos(th)/2


#verified
def y1_d1d1_jnp(p):
    r2 = p[...,0]*p[...,0]+p[...,1]*p[...,1]
    return ((2*p[...,1]*p[...,0]/r2)/(r2*jnp.pi))/2
#verified
def y1_d2d2_jnp(p):
    r2 = p[...,0]*p[...,0]+p[...,1]*p[...,1]
    return -(2*(p[...,0]*p[...,1]/r2)/(r2*jnp.pi))/2

#verified
def y2_d1d1_jnp(p):
    th = jnp.arctan2(p[...,1],p[...,0])
    c2t = jnp.cos(2*th)
    s2t = jnp.sin(2*th)
    s   = jnp.sin(th)
    c   = jnp.cos(th)
    r2  = p[...,0]*p[...,0]+p[...,1]*p[...,1]
    A   = (s2t*p[...,0]*p[...,1] - c2t*p[...,1]*p[...,1] - (R+1)*s*p[...,1] - p[...,2]*p[...,1]*c)
    dA = p[...,1]*s2t-(2*c2t*p[...,0]*(p[...,1]**2)+2*s2t*p[...,1]**3-(R+1)*c*p[...,1]**2 + p[...,2]*s*(p[...,1]**2) )/r2
    return ((p[...,1]*s2t + dA - 2*A*p[...,0]/r2 )/r2)/2
#verified
def y2_d2d2_jnp(p):
    th = jnp.arctan2(p[...,1],p[...,0])
    c2t = jnp.cos(2*th)
    s2t = jnp.sin(2*th)
    s   = jnp.sin(th)
    c   = jnp.cos(th)
    r2  = p[...,0]*p[...,0]+p[...,1]*p[...,1]
    A   = ( c2t*p[...,0]*p[...,1] - s2t*p[...,0]*p[...,0] + (R+1)*s*p[...,0] + p[...,2]*p[...,0]*c)
    dA  = p[...,0]*c2t-(2*s2t*p[...,1]*p[...,0]**2+2*c2t*(p[...,0]**3)-(R+1)*c*(p[...,0]**2)+p[...,2]*s*(p[...,0]**2))/r2
    return ((c2t*p[...,0] + dA - 2*A*p[...,1]/r2 )/r2)/2


#verified
def y3_d1d1_jnp(p):
    th = jnp.arctan2(p[...,1],p[...,0])
    c2t = jnp.cos(2*th)
    s2t = jnp.sin(2*th)
    s   = jnp.sin(th)
    c   = jnp.cos(th)
    r2  = p[...,0]*p[...,0]+p[...,1]*p[...,1]
    A   = (c2t*p[...,0]*p[...,1] + s2t*p[...,1]**2 - (R+1)*c*p[...,1] + p[...,2]*p[...,1]*s)
    dA  = p[...,1]*c2t+(2*p[...,0]*(p[...,1]**2)*s2t-2*(p[...,1]**3)*c2t-(R+1)*(p[...,1]**2)*s-p[...,2]*(p[...,1]**2)*c)/r2
    return ((c2t*p[...,1] + dA - 2*p[...,0]*A/r2)/r2)/2
#verified
def y3_d2d2_jnp(p):
    th = jnp.arctan2(p[...,1],p[...,0])
    c2t = jnp.cos(2*th)
    s2t = jnp.sin(2*th)
    s   = jnp.sin(th)
    c   = jnp.cos(th)
    r2  = p[...,0]*p[...,0]+p[...,1]*p[...,1]
    A   = (s2t*p[...,0]*p[...,1] + c2t*p[...,0]*p[...,0] - (R+1)*c*p[...,0] + p[...,2]*p[...,0]*s)
    dA  = p[...,0]*s2t+(2*(p[...,0]**2)*p[...,1]*c2t-2*(p[...,0]**3)*s2t + (R+1)*(p[...,0]**2)*s + p[...,2]*(p[...,0]**2)*c)/r2
    return -((s2t*p[...,0] + dA - 2*p[...,1]*A/r2)/r2)/2
def gb_jnp(p):
    return ((jnp.abs(p[...,1]-bnds[0][1]))<1e-14) | ((jnp.abs(p[...,1]-bnds[1][1]))<1e-14) | (jnp.abs(p[...,2]-bnds[0][2])<1e-14) | (jnp.abs(p[...,2]-bnds[1][2])<1e-14)


####################################
#           TORCH VERSION
####################################

def z1_torch(p):
    q = 2*p-1
    c=torch.cos(torch.pi*q[:,0])
    s=torch.sin(torch.pi*q[:,0])
    c2 = torch.multiply(c,c)
    cs = torch.multiply(c,s)
    z1 = torch.multiply(c2,q[:,1])-torch.multiply(cs,q[:,2])+c*(R+1)
    return z1

def z2_torch(p):
    q = 2*p-1
    c=torch.cos(torch.pi*q[:,0])
    s=torch.sin(torch.pi*q[:,0])
    s2 = torch.multiply(s,s)
    cs = torch.multiply(c,s)
    z2 = torch.multiply(cs,q[:,1])-torch.multiply(s2,q[:,2])+s*(R+1)
    return z2
def z3_torch(p):
    q=2*p-1
    c=torch.cos(torch.pi*q[:,0])
    s=torch.sin(torch.pi*q[:,0])
    z3 = torch.multiply(s,q[:,1])+torch.multiply(c,q[:,2])
    return z3

def y1_torch(p):
    th = torch.arctan2(p[:,1],p[:,0])
    return ((th/torch.pi)+1)/2

def y2_torch(p):
    # p is a vector of points, Nx3
    th = torch.arctan2(p[:,1],p[:,0])
    c=torch.cos(th)
    s=torch.sin(th)
    c2 = torch.multiply(c,c)
    cs = torch.multiply(c,s)
    q = torch.multiply(p[:,0],c2)+torch.multiply(p[:,1],cs)-(R+1)*c + torch.multiply(s,p[:,2])
    return (q+1)/2


def y3_torch(p):
    th = torch.arctan2(p[:,1],p[:,0])
    c=torch.cos(th)
    s=torch.sin(th)
    s2 = torch.multiply(s,s)
    cs = torch.multiply(c,s)
    q = -torch.multiply(p[:,0],cs)-torch.multiply(p[:,1],s2)+(R+1)*s+torch.multiply(c,p[:,2])
    return (q+1)/2

#verified
def y2_d1_torch(p):
    th = torch.arctan2(p[:,1],p[:,0])
    c2t = torch.cos(2*th)
    s2t = torch.sin(2*th)
    s   = torch.sin(th)
    c   = torch.cos(th)
    r2  = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    A   = (s2t*p[:,0]*p[:,1] - c2t*p[:,1]**2 - (R+1)*s*p[:,1] - p[:,2]*p[:,1]*c)
    return ((c2t+1)/2. + A/r2)/2

#verified
def y1_d1_torch(p):
    r2 = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    return -((p[:,1]/r2)/torch.pi)/2
#verified
def y1_d2_torch(p):
    r2 = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    return ((p[:,0]/r2)/torch.pi)/2

#verified
def y2_d2_torch(p):
    th = torch.arctan2(p[:,1],p[:,0])
    c2t = torch.cos(2*th)
    s2t = torch.sin(2*th)
    s   = torch.sin(th)
    c   = torch.cos(th)
    r2  = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    A   = ( c2t*p[:,0]*p[:,1] - s2t*p[:,0]*p[:,0] + (R+1)*s*p[:,0] + p[:,2]*p[:,0]*c)
    return (s2t/2. + A/r2)/2

def y2_d3_torch(p):
    th = torch.arctan2(p[:,1],p[:,0])
    return (torch.sin(th))/2




def y3_d1_torch(p):
    th = torch.arctan2(p[:,1],p[:,0])
    c2t = torch.cos(2*th)
    s2t = torch.sin(2*th)
    s   = torch.sin(th)
    c   = torch.cos(th)
    r2  = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    A   = (c2t*p[:,0]*p[:,1] + s2t*p[:,1]**2 - (R+1)*c*p[:,1] + p[:,2]*p[:,1]*s)
    return (-s2t/2. + A/r2)/2

def y3_d2_torch(p):
    th = torch.arctan2(p[:,1],p[:,0])
    c2t = torch.cos(2*th)
    s2t = torch.sin(2*th)
    s   = torch.sin(th)
    c   = torch.cos(th)
    r2  = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    A   = (s2t*p[:,0]*p[:,1] + c2t*p[:,0]*p[:,0] - (R+1)*c*p[:,0] + p[:,2]*p[:,0]*s)
    return ((c2t-1)/2. - A/r2)/2
#verified
def y3_d3_torch(p):
    th = torch.arctan2(p[:,1],p[:,0])
    return torch.cos(th)/2


#verified
def y1_d1d1_torch(p):
    r2 = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    return ((2*p[:,1]*p[:,0]/r2)/(r2*torch.pi))/2
#verified
def y1_d2d2_torch(p):
    r2 = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    return -(2*(p[:,0]*p[:,1]/r2)/(r2*torch.pi))/2

#verified
def y2_d1d1_torch(p):
    th = torch.arctan2(p[:,1],p[:,0])
    c2t = torch.cos(2*th)
    s2t = torch.sin(2*th)
    s   = torch.sin(th)
    c   = torch.cos(th)
    r2  = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    A   = (s2t*p[:,0]*p[:,1] - c2t*p[:,1]*p[:,1] - (R+1)*s*p[:,1] - p[:,2]*p[:,1]*c)
    dA = p[:,1]*s2t-(2*c2t*p[:,0]*(p[:,1]**2)+2*s2t*p[:,1]**3-(R+1)*c*p[:,1]**2 + p[:,2]*s*(p[:,1]**2) )/r2
    return ((p[:,1]*s2t + dA - 2*A*p[:,0]/r2 )/r2)/2
#verified
def y2_d2d2_torch(p):
    th = torch.arctan2(p[:,1],p[:,0])
    c2t = torch.cos(2*th)
    s2t = torch.sin(2*th)
    s   = torch.sin(th)
    c   = torch.cos(th)
    r2  = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    A   = ( c2t*p[:,0]*p[:,1] - s2t*p[:,0]*p[:,0] + (R+1)*s*p[:,0] + p[:,2]*p[:,0]*c)
    dA  = p[:,0]*c2t-(2*s2t*p[:,1]*p[:,0]**2+2*c2t*(p[:,0]**3)-(R+1)*c*(p[:,0]**2)+p[:,2]*s*(p[:,0]**2))/r2
    return ((c2t*p[:,0] + dA - 2*A*p[:,1]/r2 )/r2)/2


#verified
def y3_d1d1_torch(p):
    th = torch.arctan2(p[:,1],p[:,0])
    c2t = torch.cos(2*th)
    s2t = torch.sin(2*th)
    s   = torch.sin(th)
    c   = torch.cos(th)
    r2  = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    A   = (c2t*p[:,0]*p[:,1] + s2t*p[:,1]**2 - (R+1)*c*p[:,1] + p[:,2]*p[:,1]*s)
    dA  = p[:,1]*c2t+(2*p[:,0]*(p[:,1]**2)*s2t-2*(p[:,1]**3)*c2t-(R+1)*(p[:,1]**2)*s-p[:,2]*(p[:,1]**2)*c)/r2
    return ((c2t*p[:,1] + dA - 2*p[:,0]*A/r2)/r2)/2
#verified
def y3_d2d2_torch(p):
    th = torch.arctan2(p[:,1],p[:,0])
    c2t = torch.cos(2*th)
    s2t = torch.sin(2*th)
    s   = torch.sin(th)
    c   = torch.cos(th)
    r2  = p[:,0]*p[:,0]+p[:,1]*p[:,1]
    A   = (s2t*p[:,0]*p[:,1] + c2t*p[:,0]*p[:,0] - (R+1)*c*p[:,0] + p[:,2]*p[:,0]*s)
    dA  = p[:,0]*s2t+(2*(p[:,0]**2)*p[:,1]*c2t-2*(p[:,0]**3)*s2t + (R+1)*(p[:,0]**2)*s + p[:,2]*(p[:,0]**2)*c)/r2
    return -((s2t*p[:,0] + dA - 2*p[:,1]*A/r2)/r2)/2


####################################
#           OVERALL
####################################


def z1(p,jax_avail = True,torch_avail = False):
    if jax_avail:
        return z1_jnp(p)
    elif torch_avail:
        return z1_torch(p)
    else:
        return z1_np(p)
def z2(p,jax_avail = True,torch_avail = False):
    if jax_avail:
        return z2_jnp(p)
    elif torch_avail:
        return z2_torch(p)
    else:
        return z2_np(p)
def z3(p,jax_avail = True,torch_avail = False):
    if jax_avail:
        return z3_jnp(p)
    elif torch_avail:
        return z3_torch(p)
    else:
        return z3_np(p)
    
def y1(p,jax_avail = True,torch_avail = False):
    if jax_avail:
        return y1_jnp(p)
    elif torch_avail:
        return y1_torch(p)
    else:
        return y1_np(p)
def y2(p,jax_avail = True,torch_avail = False):
    if jax_avail:
        return y2_jnp(p)
    elif torch_avail:
        return y2_torch(p)
    else:
        return y2_np(p)
def y3(p,jax_avail = True,torch_avail = False):
    if jax_avail:
        return y3_jnp(p)
    elif torch_avail:
        return y3_torch(p)
    else:
        return y3_np(p)
    
def y1_d1(p,jax_avail = True,torch_avail = False):
    if jax_avail:
        return y1_d1_jnp(p)
    elif torch_avail:
        return y1_d1_torch(p)
    else:
        return y1_d1_np(p)
def y1_d2(p,jax_avail = True,torch_avail = False):
    if jax_avail:
        return y1_d2_jnp(p)
    elif torch_avail:
        return y1_d2_torch(p)
    else:
        return y1_d2_np(p)

    

def y2_d1(p,jax_avail = True,torch_avail = False):
    if jax_avail:
        return y2_d1_jnp(p)
    elif torch_avail:
        return y2_d1_torch(p)
    else:
        return y2_d1_np(p)
def y2_d2(p,jax_avail = True,torch_avail = False):
    if jax_avail:
        return y2_d2_jnp(p)
    elif torch_avail:
        return y2_d2_torch(p)
    else:
        return y2_d2_np(p)
def y2_d3(p,jax_avail = True,torch_avail = False):
    if jax_avail:
        return y2_d3_jnp(p)
    elif torch_avail:
        return y2_d3_torch(p)
    else:
        return y2_d3_np(p)

def y3_d1(p,jax_avail = True,torch_avail = False):
    if jax_avail:
        return y3_d1_jnp(p)
    elif torch_avail:
        return y3_d1_torch(p)
    else:
        return y3_d1_np(p)
def y3_d2(p,jax_avail = True,torch_avail = False):
    if jax_avail:
        return y3_d2_jnp(p)
    elif torch_avail:
        return y3_d2_torch(p)
    else:
        return y3_d2_np(p)
def y3_d3(p,jax_avail = True,torch_avail = False):
    if jax_avail:
        return y3_d3_jnp(p)
    elif torch_avail:
        return y3_d3_torch(p)
    else:
        return y3_d3_np(p)
    
def y1_d1d1(p,jax_avail = True,torch_avail = False):
    if jax_avail:
        return y1_d1d1_jnp(p)
    elif torch_avail:
        return y1_d1d1_torch(p)
    else:
        return y1_d1d1_np(p)
def y1_d2d2(p,jax_avail = True,torch_avail = False):
    if jax_avail:
        return y1_d2d2_jnp(p)
    elif torch_avail:
        return y1_d2d2_torch(p)
    else:
        return y1_d2d2_np(p)

    
def y2_d1d1(p,jax_avail = True,torch_avail = False):
    if jax_avail:
        return y2_d1d1_jnp(p)
    elif torch_avail:
        return y2_d1d1_torch(p)
    else:
        return y2_d1d1_np(p)
def y2_d2d2(p,jax_avail = True,torch_avail = False):
    if jax_avail:
        return y2_d2d2_jnp(p)
    elif torch_avail:
        return y2_d2d2_torch(p)
    else:
        return y2_d2d2_np(p)

def y3_d1d1(p,jax_avail = True,torch_avail = False):
    if jax_avail:
        return y3_d1d1_jnp(p)
    elif torch_avail:
        return y3_d1d1_torch(p)
    else:
        return y3_d1d1_np(p)
def y3_d2d2(p,jax_avail = True,torch_avail = False):
    if jax_avail:
        return y3_d2d2_jnp(p)
    elif torch_avail:
        return y3_d2d2_torch(p)
    else:
        return y3_d2d2_np(p)
def gb(p,jax_avail = True,torch_avail = False):
    if jax_avail:
        return gb_jnp(p)
    elif torch_avail:
        return gb_np(p)
    else:
        return gb_np(p)

def box_geom(jax_avail=True,torch_avail = False):
    if jax_avail:
        return jnp.array(bnds)
    elif torch_avail:
        return np.array(bnds)
    else:
        return np.array(bnds)
def bounds():
    return bnds


def param_geom(jax_avail = True,torch_avail=False, hpsalt=False):
    if hpsalt:
        return ParametrizedGeometry3Dalt(
                        box_geom(jax_avail,torch_avail),\
                        z1=lambda p:z1(p,jax_avail,torch_avail),z2=lambda p:z2(p,jax_avail,torch_avail),z3=lambda p:z3(p,jax_avail,torch_avail),\
                        y1=lambda p:y1(p,jax_avail,torch_avail),y2=lambda p:y2(p,jax_avail,torch_avail),y3=lambda p:y3(p,jax_avail,torch_avail),\
                        y1_d1=lambda p:y1_d1(p,jax_avail,torch_avail), y1_d2=lambda p:y1_d2(p,jax_avail,torch_avail),\
                        y2_d1=lambda p:y2_d1(p,jax_avail,torch_avail), y2_d2=lambda p:y2_d2(p,jax_avail,torch_avail),y2_d3=lambda p:y2_d3(p,jax_avail,torch_avail),\
                        y3_d1=lambda p:y3_d1(p,jax_avail,torch_avail), y3_d2=lambda p:y3_d2(p,jax_avail,torch_avail),y3_d3=lambda p:y3_d3(p,jax_avail,torch_avail),\
                        y1_d1d1=lambda p:y1_d1d1(p,jax_avail,torch_avail), y1_d2d2=lambda p:y1_d2d2(p,jax_avail,torch_avail),\
                        y2_d1d1=lambda p:y2_d1d1(p,jax_avail,torch_avail), y2_d2d2=lambda p:y2_d2d2(p,jax_avail,torch_avail),\
                        y3_d1d1=lambda p:y3_d1d1(p,jax_avail,torch_avail), y3_d2d2=lambda p:y3_d2d2(p,jax_avail,torch_avail)
                        )
    else:
        return ParametrizedGeometry3D(
                        box_geom(jax_avail,torch_avail),\
                        z1=lambda p:z1(p,jax_avail,torch_avail),z2=lambda p:z2(p,jax_avail,torch_avail),z3=lambda p:z3(p,jax_avail,torch_avail),\
                        y1=lambda p:y1(p,jax_avail,torch_avail),y2=lambda p:y2(p,jax_avail,torch_avail),y3=lambda p:y3(p,jax_avail,torch_avail),\
                        y1_d1=lambda p:y1_d1(p,jax_avail,torch_avail), y1_d2=lambda p:y1_d2(p,jax_avail,torch_avail),\
                        y2_d1=lambda p:y2_d1(p,jax_avail,torch_avail), y2_d2=lambda p:y2_d2(p,jax_avail,torch_avail),y2_d3=lambda p:y2_d3(p,jax_avail,torch_avail),\
                        y3_d1=lambda p:y3_d1(p,jax_avail,torch_avail), y3_d2=lambda p:y3_d2(p,jax_avail,torch_avail),y3_d3=lambda p:y3_d3(p,jax_avail,torch_avail),\
                        y1_d1d1=lambda p:y1_d1d1(p,jax_avail,torch_avail), y1_d2d2=lambda p:y1_d2d2(p,jax_avail,torch_avail),\
                        y2_d1d1=lambda p:y2_d1d1(p,jax_avail,torch_avail), y2_d2d2=lambda p:y2_d2d2(p,jax_avail,torch_avail),\
                        y3_d1d1=lambda p:y3_d1d1(p,jax_avail,torch_avail), y3_d2d2=lambda p:y3_d2d2(p,jax_avail,torch_avail)
                        )

####################################
#       slab functions
####################################


def slabs(N):
    slabs = []
    H = (bnds[1][0]-bnds[0][0])/N
    for n in range(N):
        bnds_n = [[bnds[0][0]+n*H,bnds[0][1],bnds[0][2]],[bnds[0][0]+(n+1)*H,bnds[1][1],bnds[1][2]]]
        slabs+=[bnds_n]
    return slabs,H
def connectivity(slabs):
    N=len(slabs)
    connectivity = [[N-1,0]]
    for i in range(N):
        connectivity+=[[i,i+1]]
    if_connectivity = []
    for i in range(N):
        if_connectivity+=[[(i-1)%N,(i+1)%N]]
    return connectivity,if_connectivity

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
    



####################################
#       validate diff
####################################


def check_param():
    p=np.random.uniform(size= (100,3))
    def diffy(indy,indz,z):
        match indz:
            case 0:
                match indy:
                    case 0:
                        return y1_d1(z,False)
                    case 1:
                        return y2_d1(z,False)
                    case 2:
                        return y3_d1(z,False)
            case 1:
                match indy:
                    case 0:
                        return y1_d2(z,False)
                    case 1:
                        return y2_d2(z,False)
                    case 2:
                        return y3_d2(z,False)
            case 2:
                match indy:
                    case 0:
                        return np.zeros_like(z[:,1])
                    case 1:
                        return y2_d3(z,False)
                    case 2:
                        return y3_d3(z,False)
    def ddiffy(indy,indz,z):
        match indz:
            case 0:
                match indy:
                    case 0:
                        return y1_d1d1(z,False)
                    case 1:
                        return y2_d1d1(z,False)
                    case 2:
                        return y3_d1d1(z,False)
            case 1:
                match indy:
                    case 0:
                        return y1_d2d2(z,False)
                    case 1:
                        return y2_d2d2(z,False)
                    case 2:
                        return y3_d2d2(z,False)
            case 2:
                match indy:
                    case 0:
                        return np.zeros_like(z[:,0])
                    case 1:
                        return np.zeros_like(z[:,1])
                    case 2:
                        return np.zeros_like(z[:,0])


    z=np.zeros(shape= p.shape)
    z[:,0] = z1(p,False)
    z[:,1] = z2(p,False)
    z[:,2] = z3(p,False)

    y=np.zeros(shape= p.shape)
    y[:,0] = y1(z,False)
    y[:,1] = y2(z,False)
    y[:,2] = y3(z,False)
    print("err y =",np.linalg.norm(p-y,ord=np.inf))



    kvec = np.array([2,3,4,5,6,7])
    hvec = 1./(2**kvec)
    errH = np.zeros(shape=(len(hvec),9))
    errH2 = np.zeros(shape=(len(hvec),9))
    for indh in range(len(hvec)):
        dz = hvec[indh]
        for indy in range(3):
            for indz in range(3):
                zdz=np.zeros(shape= z.shape)
                zdz[:,0] = z[:,0]
                zdz[:,1] = z[:,1]
                zdz[:,2] = z[:,2]
                zdz[:,indz]+=dz

                ydy=np.zeros(shape= y.shape)
                ydy[:,0] = y1(zdz,False)
                ydy[:,1] = y2(zdz,False)
                ydy[:,2] = y3(zdz,False)

                dy = diffy(indy,indz,z)
                ddy = ddiffy(indy,indz,z)
                errH[indh,indy+3*indz] = np.linalg.norm(ydy[:,indy]-y[:,indy]-dz*dy,ord=np.inf)
                errH2[indh,indy+3*indz] = np.linalg.norm(ydy[:,indy]-y[:,indy]-dz*dy-dz*dz*ddy/2,ord=np.inf)
    print(errH)
    for i in range(9):
        plt.figure(i)
        plt.loglog(hvec,errH[:,i],label='dy')
        plt.loglog(hvec,errH2[:,i],label='ddy')
        plt.loglog(hvec,2*errH[0,i]*(hvec**2)/(hvec[0]**2),label='h2',linestyle='dashed')
        plt.loglog(hvec,2*errH2[0,i]*(hvec**3)/(hvec[0]**3),label='h3',linestyle='dashed')
        plt.legend()
    plt.show()