import numpy as np
from SOMSmerge import InterfaceMap
import SOMSmerge
import matplotlib.pyplot as plt
import scipy.special as special
import numpy.polynomial.chebyshev as chebpoly
def bc(p):
    r = np.sqrt(((p[:, 0] -.25)**2) + ((p[:, 1] + 1.)**2))
    return np.log(r) / (2 * np.pi)


def cheb_interp(dx,dy,nx,ny,p,dir):
    if dir == 'horizontal':
        xpts = np.linspace(0,dx,nx)
        xpts_excl = np.concatenate((xpts[1:-1],dx+xpts[1:-1]))
        xpts = np.concatenate((xpts[1:],dx+xpts[1:-1]))
        xpts = 2*(xpts-dx)/dx
        xpts_excl = 2*(xpts_excl-dx)/dx
        Ix = chebpoly.chebvander(xpts,p)@np.linalg.pinv(chebpoly.chebvander(xpts_excl,p))
        Iy = None
        
    else:
        ypts = np.linspace(0,dy,ny)
        ypts_excl = np.concatenate((ypts[1:-1],dy+ypts[1:-1]))
        ypts = np.concatenate((ypts[1:],dy+ypts[1:-1]))
        ypts = 2*(ypts-dy)/dy
        ypts_excl = 2*(ypts_excl-dy)/dy
        Iy = chebpoly.chebvander(ypts,p)@np.linalg.pinv(chebpoly.chebvander(ypts_excl,p))
        Ix = None
    return Ix,Iy


def bc_helmholtz(p,kh):
    
    return np.sin(kh*(p[:,0]+p[:,1])/np.sqrt(2))


def interface_points(x_offset=0.0):
    """
    Return the N interface DOF coordinates for a pair of boxes,
    i.e. x = x_offset + 1 (the shared boundary), y = h,...,N*h.
    """
    xs = np.arange(1, N+1) * h
    return np.column_stack([np.full(N, x_offset + 1.0), xs])


#def laplace_stencil(nx, ny,hx,hy):
#    ex = np.ones(shape = (nx-1,))
#    ey = np.ones(shape = (ny-1,))
#    Dxx = (-2*np.identity(nx)+np.diag(ex,-1)+np.diag(ex,1))/(hx*hx)
#    Dyy = (-2*np.identity(ny)+np.diag(ey,-1)+np.diag(ey,1))/(hy*hy)
#    L = np.kron(Dxx,np.identity(ny))+np.kron(np.identity(nx),Dyy)
#    return L
def helmholtz_stencil(nx, ny,hx,hy,kh):
    ex = np.ones(shape = (nx-1,))
    ey = np.ones(shape = (ny-1,))
    Dxx = (-2*np.identity(nx)+np.diag(ex,-1)+np.diag(ex,1))/(hx*hx)
    Dyy = (-2*np.identity(ny)+np.diag(ey,-1)+np.diag(ey,1))/(hy*hy)
    L = np.kron(Dxx,np.identity(ny))+np.kron(np.identity(nx),Dyy)+kh*kh*np.kron(np.identity(nx),np.identity(ny))
    return L


def interface_map(nx, ny,dx,dy,dir,kh=0.):
    p = max(nx//2,ny//2)
    Ix,Iy = cheb_interp(dx,dy,nx,ny,p,dir)
    if dir=='horizontal':
        # nx and ny are the number of DOFs in UNMERGED boxes
        xpts = np.linspace(0,2*dx,2*nx-1)
        ypts = np.linspace(0,dy,ny)
        XY = np.zeros(shape = (len(xpts)*len(ypts),2))
        XY[:,0] = np.kron(xpts,np.ones_like(ypts))
        XY[:,1] = np.kron(np.ones_like(xpts),ypts)
        hx = xpts[1]-xpts[0]
        hy = ypts[1]-ypts[0]
        L = helmholtz_stencil(2*nx-1, ny,hx,hy,kh)
        c = (xpts[0]+xpts[-1])/2
        tol = 1e-10
        Ii  = np.where((XY[:,0]>xpts[0]) & (XY[:,0]<xpts[-1]) & (XY[:,1]>ypts[0]) & (XY[:,1]<ypts[-1]))[0]
        Il  = np.where((abs(XY[:,0]-xpts[0]) <tol) & (XY[:,1]<ypts[-1]) & (XY[:,1]>ypts[0]))[0]
        Id  = np.where((abs(XY[:,1]-ypts[0]) <tol) & (XY[:,0]<xpts[-1]) & (XY[:,0]>xpts[0]))[0]
        Idl = np.where((abs(XY[:,1]-ypts[0]) <tol) & (XY[:,0]<c-tol)      & (XY[:,0]>xpts[0]))[0]
        Idr = np.where((abs(XY[:,1]-ypts[0]) <tol) & (XY[:,0]>c+tol)      & (XY[:,0]<xpts[-1]))[0]
        Iu  = np.where((abs(XY[:,1]-ypts[-1])<tol) & (XY[:,0]<xpts[-1]) & (XY[:,0]>xpts[0]))[0]
        Iul = np.where((abs(XY[:,1]-ypts[-1])<tol) & (XY[:,0]<c-tol)      & (XY[:,0]>xpts[0]))[0]
        Iur = np.where((abs(XY[:,1]-ypts[-1])<tol) & (XY[:,0]>c+tol)      & (XY[:,0]<xpts[-1]))[0]
        Ir  = np.where((abs(XY[:,0]-xpts[-1])<tol) & (XY[:,1]<ypts[-1]) & (XY[:,1]>ypts[0]))[0]

        Ib_full = np.concatenate([Il, Id, Iu, Ir])
        Ib = np.concatenate([Il, Idl, Idr, Iul, Iur, Ir])
        seg_starts = [0,len(Il),len(Il)+len(Idl),len(Il)+len(Idl)+len(Idr),\
                    len(Il)+len(Idl)+len(Idr)+len(Iul),len(Il)+len(Idl)+len(Idr)+len(Iul)+len(Iur)]
        n_full = len(Ib_full)
        n_seg  = len(Ib)
        I_seg_to_full = np.zeros((len(Ib_full),len(Ib)))
        I_seg_to_full[:len(Il),:][:,:seg_starts[1]] = np.identity(len(Il))
        I_seg_to_full[len(Il):len(Il)+len(Id),:][:,seg_starts[1]:seg_starts[3]] = Ix
        I_seg_to_full[len(Il)+len(Id):len(Il)+len(Id)+len(Iu),:][:,seg_starts[3]:seg_starts[5]] = Ix
        I_seg_to_full[len(Il)+len(Id)+len(Iu):,:][:,seg_starts[5]:] = np.identity(len(Ir))
        c = (xpts[0]+xpts[-1])/2
        XYi = XY[Ii,:]
        Ic = np.where((abs(XYi[:,0]-c)<1e-10) & (XYi[:,1]< ypts[-1]) & (XYi[:,1]> ypts[0]))[0]
        Lii = L[:,Ii][Ii,:]
        Lib_full = L[Ii,:][:,Ib_full]
        Lib = Lib_full @ I_seg_to_full
        S = -np.linalg.solve(Lii,Lib)
        S = S[Ic,:]
    elif dir=='vertical':
        xpts = np.linspace(0, dx,   nx)
        ypts = np.linspace(0, 2*dy, 2*ny-1)
        XY = np.zeros((len(xpts)*len(ypts), 2))
        XY[:,0] = np.kron(xpts, np.ones_like(ypts))
        XY[:,1] = np.kron(np.ones_like(xpts), ypts)
        hx = xpts[1]-xpts[0];  hy = ypts[1]-ypts[0]
        L = helmholtz_stencil(nx, 2*ny-1, hx, hy,kh)
        c = (ypts[0]+ypts[-1])/2
        tol = 1e-10
        Ii  = np.where((XY[:,0]>xpts[0]) & (XY[:,0]<xpts[-1]) & (XY[:,1]>ypts[0]) & (XY[:,1]<ypts[-1]))[0]
        Il  = np.where((abs(XY[:,0]-xpts[0]) <tol) & (XY[:,1]<ypts[-1]) & (XY[:,1]>ypts[0]))[0]
        Id  = np.where((abs(XY[:,1]-ypts[0]) <tol) & (XY[:,0]<xpts[-1]) & (XY[:,0]>xpts[0]))[0]
        Iu  = np.where((abs(XY[:,1]-ypts[-1])<tol) & (XY[:,0]<xpts[-1]) & (XY[:,0]>xpts[0]))[0]
        Ir  = np.where((abs(XY[:,0]-xpts[-1])<tol) & (XY[:,1]<ypts[-1]) & (XY[:,1]>ypts[0]))[0]
        Ib_full = np.concatenate([Il, Id, Iu, Ir])
        Ilb = np.where((abs(XY[:,0]-xpts[0]) <tol) & (XY[:,1]<c-tol)      & (XY[:,1]>ypts[0]))[0]
        Ilt = np.where((abs(XY[:,0]-xpts[0]) <tol) & (XY[:,1]>c+tol)      & (XY[:,1]<ypts[-1]))[0]
        Irb = np.where((abs(XY[:,0]-xpts[-1])<tol) & (XY[:,1]<c-tol)      & (XY[:,1]>ypts[0]))[0]
        Irt = np.where((abs(XY[:,0]-xpts[-1])<tol) & (XY[:,1]>c+tol)      & (XY[:,1]<ypts[-1]))[0]
        Ib  = np.concatenate([Ilb, Ilt, Id, Iu, Irb, Irt])
        seg_starts =    [0, len(Ilb), len(Ilb)+len(Ilt),
                        len(Ilb)+len(Ilt)+len(Id),
                        len(Ilb)+len(Ilt)+len(Id)+len(Iu),
                        len(Ilb)+len(Ilt)+len(Id)+len(Iu)+len(Irb)]
        n_full = len(Ib_full);  n_seg = len(Ib)
        I_seg_to_full = np.zeros((len(Ib_full),len(Ib)))
        I_seg_to_full[:len(Il),:][:,:seg_starts[2]] = Iy
        I_seg_to_full[len(Il):len(Il)+len(Id),:][:,seg_starts[2]:seg_starts[3]] = np.identity(len(Id))
        I_seg_to_full[len(Il)+len(Id):len(Il)+len(Id)+len(Iu),:][:,seg_starts[3]:seg_starts[4]] = np.identity(len(Iu))
        I_seg_to_full[len(Il)+len(Id)+len(Iu):,:][:,seg_starts[4]:] = Iy
        XYi = XY[Ii,:]
        Ic  = np.where((abs(XYi[:,1]-c)<1e-10) & (XYi[:,0]<xpts[-1]) & (XYi[:,0]>xpts[0]))[0]
        Lii      = L[:,Ii][Ii,:]
        Lib_full = L[Ii,:][:,Ib_full]
        Lib      = Lib_full @ I_seg_to_full
        S = -np.linalg.solve(Lii, Lib)
        S = S[Ic,:]
    return InterfaceMap(S,seg_starts)
def gen_DOFs(nx, ny, dx, dy, orientation='horizontal'):
    if orientation == 'horizontal':
        xpts = np.linspace(0, 2*dx, 2*nx-1)
        ypts = np.linspace(0, dy,   ny)
        XY = np.zeros((len(xpts)*len(ypts), 2))
        XY[:,0] = np.kron(xpts, np.ones_like(ypts))
        XY[:,1] = np.kron(np.ones_like(xpts), ypts)
        c = (xpts[0]+xpts[-1])/2
        tol = 1e-10
        Il  = np.where((abs(XY[:,0]-xpts[0]) <tol) & (XY[:,1]<ypts[-1]) & (XY[:,1]>ypts[0]))[0]
        Idl = np.where((abs(XY[:,1]-ypts[0]) <tol) & (XY[:,0]<c-tol)      & (XY[:,0]>xpts[0]))[0]
        Idr = np.where((abs(XY[:,1]-ypts[0]) <tol) & (XY[:,0]>c+tol)      & (XY[:,0]<xpts[-1]))[0]
        Iul = np.where((abs(XY[:,1]-ypts[-1])<tol) & (XY[:,0]<c-tol)      & (XY[:,0]>xpts[0]))[0]
        Iur = np.where((abs(XY[:,1]-ypts[-1])<tol) & (XY[:,0]>c+tol)      & (XY[:,0]<xpts[-1]))[0]
        Ir  = np.where((abs(XY[:,0]-xpts[-1])<tol) & (XY[:,1]<ypts[-1]) & (XY[:,1]>ypts[0]))[0]
        Ii  = np.where((XY[:,0]>xpts[0])  & (XY[:,0]<xpts[-1]) & (XY[:,1]>ypts[0]) & (XY[:,1]<ypts[-1]))[0]
        XYi = XY[Ii,:]
        Ic  = np.where((abs(XYi[:,0]-c)<tol) & (XYi[:,1]<ypts[-1]) & (XYi[:,1]>ypts[0]))[0]
        segs = {k: XY[idx,:].copy() for k, idx in enumerate([Il, Idl, Idr, Iul, Iur, Ir])}
    else:  # vertical
        xpts = np.linspace(0, dx,   nx)
        ypts = np.linspace(0, 2*dy, 2*ny-1)
        XY = np.zeros((len(xpts)*len(ypts), 2))
        XY[:,0] = np.kron(xpts, np.ones_like(ypts))
        XY[:,1] = np.kron(np.ones_like(xpts), ypts)
        c = (ypts[0]+ypts[-1])/2
        tol = 1e-10
        Ilb = np.where((abs(XY[:,0]-xpts[0]) <tol) & (XY[:,1]<c-tol)      & (XY[:,1]>ypts[0]))[0]
        Ilt = np.where((abs(XY[:,0]-xpts[0]) <tol) & (XY[:,1]>c+tol)      & (XY[:,1]<ypts[-1]))[0]
        Id  = np.where((abs(XY[:,1]-ypts[0]) <tol) & (XY[:,0]<xpts[-1]) & (XY[:,0]>xpts[0]))[0]
        Iu  = np.where((abs(XY[:,1]-ypts[-1])<tol) & (XY[:,0]<xpts[-1]) & (XY[:,0]>xpts[0]))[0]
        Irb = np.where((abs(XY[:,0]-xpts[-1])<tol) & (XY[:,1]<c-tol)      & (XY[:,1]>ypts[0]))[0]
        Irt = np.where((abs(XY[:,0]-xpts[-1])<tol) & (XY[:,1]>c+tol)      & (XY[:,1]<ypts[-1]))[0]
        Ii  = np.where((XY[:,0]>xpts[0])  & (XY[:,0]<xpts[-1]) & (XY[:,1]>ypts[0]) & (XY[:,1]<ypts[-1]))[0]
        XYi = XY[Ii,:]
        Ic  = np.where((abs(XYi[:,1]-c)<1e-10) & (XYi[:,0]<xpts[-1]) & (XYi[:,0]>xpts[0]))[0]
        segs = {k: XY[idx,:].copy() for k, idx in enumerate([Ilb, Ilt, Id, Iu, Irb, Irt])}
    segs['iface'] = XYi[Ic,:].copy()
    return segs

def shift(segs, dx_off=0., dy_off=0.):
    """Return a copy of a DOF dict shifted by (dx_off, dy_off)."""
    out = {}
    for k, v in segs.items():
        s = v.copy(); s[:,0] += dx_off; s[:,1] += dy_off
        out[k] = s
    return out



def test_interface_map(S_imap, x_offset,dx,dy, label):
    xpts = np.linspace(0,2*dx,2*nx-1)
    ypts = np.linspace(0,dy,ny)
    XY = np.zeros(shape = (len(xpts)*len(ypts),2))
    XY[:,0] = np.kron(xpts,np.ones_like(ypts))
    XY[:,1] = np.kron(np.ones_like(xpts),ypts)
    c = (xpts[0]+xpts[-1])/2

    tol = 1e-10
    Ii  = np.where((XY[:,0]>xpts[0]) & (XY[:,0]<xpts[-1]) & (XY[:,1]>ypts[0]) & (XY[:,1]<ypts[-1]))[0]
    Il  = np.where((abs(XY[:,0]-xpts[0]) <tol) & (XY[:,1]<ypts[-1]) & (XY[:,1]>ypts[0]))[0]
    Idl = np.where((abs(XY[:,1]-ypts[0]) <tol) & (XY[:,0]<c-tol)      & (XY[:,0]>xpts[0]))[0]
    Idr = np.where((abs(XY[:,1]-ypts[0]) <tol) & (XY[:,0]>c+tol)      & (XY[:,0]<xpts[-1]))[0]
    Iul = np.where((abs(XY[:,1]-ypts[-1])<tol) & (XY[:,0]<c-tol)      & (XY[:,0]>xpts[0]))[0]
    Iur = np.where((abs(XY[:,1]-ypts[-1])<tol) & (XY[:,0]>c+tol)      & (XY[:,0]<xpts[-1]))[0]
    Ir  = np.where((abs(XY[:,0]-xpts[-1])<tol) & (XY[:,1]<ypts[-1]) & (XY[:,1]>ypts[0]))[0]
    Ib = np.concatenate([Il, Idl, Idr, Iul, Iur, Ir])
    
    XYb = XY[Ib,:]
    XYi = XY[Ii,:]
    Ic = np.where((abs(XYi[:,0]-c)<1e-10) & (XYi[:,1]< ypts[-1]) & (XYi[:,1]> ypts[0]))[0]
    XYc = XYi[Ic,:]
    XYb[:,0]+=x_offset
    XYc[:,0]+=x_offset
    f_bnd  = bc(XYb)
    u_pred = S_imap.S @ f_bnd
    u_true = bc(XYc)

    err = np.max(np.abs(u_pred - u_true))
    print(f"  {label}: max error = {err:.2e}")
    return err


'''
print("Building Laplace stencil and computing interface maps...")

nx = 20
ny = 20
dx = 1.
dy = 1.


# ---------------------------------------------------------------------------
# Horizontal merge test
# ---------------------------------------------------------------------------


S_ab = interface_map(nx,ny,dx,dy,'horizontal')
S_bc = interface_map(nx,ny,dx,dy,'horizontal')
S_cd = interface_map(nx,ny,dx,dy,'horizontal')
interface_maps = [S_ab, S_bc, S_cd]

print("\nTesting interface maps against bc(p) = log(r)/(2*pi):")
test_interface_map(S_ab, x_offset=0.0, dx=dx,dy=dy,label="S_ab (x=1)")
test_interface_map(S_bc, x_offset=dx, dx=dx,dy=dy,label="S_bc (x=2)")
test_interface_map(S_cd, x_offset=2*dx, dx=dx,dy=dy,label="S_cd (x=3)")
print("Done.")
a = SOMSmerge.Node(); b = SOMSmerge.Node()
c = SOMSmerge.Node(); d = SOMSmerge.Node()
tau = SOMSmerge.Node(children=(a, b))
sig = SOMSmerge.Node(children=(c, d))

_imap_registry = {(id(a),id(b)): S_ab, (id(b),id(c)): S_bc, (id(c),id(d)): S_cd}
SOMSmerge.interface_map = lambda t, s: _imap_registry[(id(t), id(s))]


print("\nRunning horizontal_merge(tau, sig)...")
merged_node, merged_imap = SOMSmerge.horizontal_merge(tau, sig)
print(f"  Merged imap shape: {merged_imap.S.shape}")
print(f"  Merged seg_starts: {merged_imap.seg_starts}")


# Pre-compute DOF sets in local coordinates
dofs_h = gen_DOFs(nx, ny, dx, dy, 'horizontal')
dofs_v = gen_DOFs(nx, ny, dx, dy, 'vertical')

ab = dofs_h;  
bc_d = shift(dofs_h, dx_off=dx);  
cd = shift(dofs_h, dx_off=2*dx)
XYb_merged = np.vstack([
    ab[0],  ab[1],  ab[2],
    bc_d[2], cd[2],
    ab[3],  ab[4],
    bc_d[4], cd[4],
    cd[5],
])
XYc_merged = shift(dofs_h, dx_off=dx)['iface']

f_outer = bc(XYb_merged)
u_pred  = merged_imap.S @ f_outer
u_true  = bc(XYc_merged)
print(f"  Outer boundary shape: {XYb_merged.shape}  (expect ({merged_imap.S.shape[1]}, 2))")
err = np.max(np.abs(u_pred - u_true))
print(f"\nMerged imap: max error = {err:.2e}")


# ---------------------------------------------------------------------------
# Vertical merge test
# ---------------------------------------------------------------------------
print("\nBuilding vertical interface maps S_ac and S_bd...")
S_ac = interface_map(nx, ny, dx, dy,'vertical')
S_bd = interface_map(nx, ny, dx, dy,'vertical')
print(f"  S_ac shape: {S_ac.S.shape}, seg_starts: {S_ac.seg_starts}")

_imap_registry[(id(a), id(c))] = S_ac
_imap_registry[(id(b), id(d))] = S_bd

print("\nRunning vertical_merge(tau, sig)...")
mu_node, mu_imap = SOMSmerge.vertical_merge(tau, sig)
print(f"  mu imap shape: {mu_imap.S.shape}")
print(f"  mu seg_starts: {mu_imap.seg_starts}")

ab_v  = dofs_h                   
cd_v  = shift(dofs_h, dy_off=dy) 
ac_v  = dofs_v                   
bd_v  = shift(dofs_v, dx_off=dx) 

XYb_mu = np.vstack([
    ab_v[0],
    cd_v[0],
    ab_v[1],
    ab_v[2],
    cd_v[3],
    cd_v[4],
    ab_v[5],
    cd_v[5],
])

XYc_mu = np.vstack([
    ac_v['iface'],          
    bd_v['iface'],          
])

print(f"  Outer boundary shape: {XYb_mu.shape}  (expect ({mu_imap.S.shape[1]}, 2))")
print(f"  Interface shape:      {XYc_mu.shape}  (expect ({mu_imap.S.shape[0]}, 2))")

f_mu   = bc(XYb_mu)
u_pred = mu_imap.S @ f_mu
u_true = bc(XYc_mu)
err    = np.max(np.abs(u_pred - u_true))
print(f"\nVertical merge imap: max error = {err:.2e}")
'''