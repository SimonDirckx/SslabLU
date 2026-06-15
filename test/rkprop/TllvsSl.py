"""
Tll at Lx=1/16 (boundary DtN self-map, plane at x=0) vs
Sl  at Lx=1/8  (interior interface map: x=0 face -> mid-plane x=1/16),
both reaching depth 1/16, both on a 128x128 plane with 16x16 leaves and a
perfect binary tree.  Sl is built with the thinSlab3D logic (double-wide box,
center plane = Jc, output padded to the full plane) using our own 7-pt stencil,
applied matrix-free.  We sweep the HBS target rank and overlay both curves.
"""
import numpy as np, gc
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
torch.set_default_dtype(torch.float64)
import HBStorch

# ---------------------------------------------------- 7-pt box operator
def build_box(Wx, nx, side):
    ny = nz = side
    hx = Wx/(nx-1); hy = 1.0/(side-1)
    def path_lap(m, h):
        e = np.ones(m)
        L = sp.diags([-e[:-1], 2*e, -e[:-1]], [-1,0,1], format="lil")
        L[0,0]=1.0; L[-1,-1]=1.0
        return (L.tocsr())/(h*h)
    Lx1,Ly1,Lz1 = path_lap(nx,hx), path_lap(ny,hy), path_lap(nz,hy)
    Ix,Iy,Iz = (sp.identity(m, format="csr") for m in (nx,ny,nz))
    A = (sp.kron(sp.kron(Lx1,Iy),Iz)+sp.kron(sp.kron(Ix,Ly1),Iz)
       + sp.kron(sp.kron(Ix,Iy),Lz1)).tocsr()
    N = nx*ny*nz
    i,j,k = np.unravel_index(np.arange(N),(nx,ny,nz))
    return A, i, j, k, ny, nz

# ---------------------------------------------------- bisection (equal leaves)
def bisection_order(coords, nleaves):
    leafsz = coords.shape[0]//nleaves; order=[]
    def rec(idx):
        if len(idx)==leafsz: order.append(idx); return
        d=int(np.argmax(coords[idx].max(0)-coords[idx].min(0)))
        s=idx[np.argsort(coords[idx,d],kind="stable")]; h=len(s)//2
        rec(s[:h]); rec(s[h:])
    rec(np.arange(coords.shape[0])); return np.concatenate(order)

# ---------------------------------------------------- HBS run + threshold
def make_hbs(N, nleaves, nl, L, rk, Om, Psi, Y, Z):
    hbs = HBStorch.HBSMAT(device='cpu', quad=False)
    hbs.perm = torch.arange(N, dtype=torch.int64)
    hbs.Nb = nleaves; hbs.nl = nl; hbs.L = L; hbs.fac = 2
    hbs.constructHBS(rk, Om, Psi, Y, Z, fast=False)
    return hbs

def hbs_error(apply_fwd, apply_adj, N, order, nleaves, nl, L, rk):
    s = max(nl+rk, 3*rk) + 40
    rng = np.random.default_rng(0)
    Om = rng.standard_normal((N, s)); Y = apply_fwd(Om)
    Psi = rng.standard_normal((N, s)); Z = apply_adj(Psi)
    hbs = make_hbs(N, nleaves, nl, L, rk, Om, Psi, Y, Z)
    Xt = np.random.default_rng(1).standard_normal((N, 16))
    ap = hbs.matmat(Xt)
    if torch.is_tensor(ap): ap = ap.cpu().numpy()
    ex = apply_fwd(Xt)
    err = np.linalg.norm(ap-ex)/np.linalg.norm(ex)
    del hbs, Om, Psi, Y, Z, Xt, ap, ex; gc.collect()
    return err

def top_rank(apply_fwd, apply_adj, N, tol=1e-6, q=460):
    half=N//2; rng=np.random.default_rng(7)
    R=np.zeros((N,q)); R[half:]=rng.standard_normal((half,q))
    W=apply_fwd(R)[:half]; Q,_=np.linalg.qr(W)
    R2=np.zeros((N,Q.shape[1])); R2[:half]=Q
    BtQ=apply_adj(R2)[half:]
    s=np.linalg.svd(BtQ, compute_uv=False)
    return int(np.sum(s>tol*s[0]))

# =================================================================== Sl @ Lx=1/8
side=128; nleaves, nl = 64, 256; L = int(np.log2(nleaves))+1
Wx_S = 1.0/8; nx_S = 5                       # hx = (1/8)/4 = 1/32, mid-plane x=1/16 = node 2
A, i, j, k, ny, nz = build_box(Wx_S, nx_S, side)
N = ny*nz
on_b = (i==0)|(i==nx_S-1)|(j==0)|(j==ny-1)|(k==0)|(k==nz-1)
iI = np.where(~on_b)[0]
imid = (nx_S-1)//2
left = np.where(i==0)[0]                                  # x=0 face (Jl), (j,k)-lex
midplane_all = np.where(i==imid)[0]                       # full mid-plane (Jc_large)
mid_int_glob = np.where((i==imid)&(~on_b))[0]             # strictly-interior mid-plane
posI = -np.ones(A.shape[0],int); posI[iI]=np.arange(iI.size)
Jc_local = posI[mid_int_glob]                            # rows in interior ordering
posP = -np.ones(A.shape[0],int); posP[midplane_all]=np.arange(midplane_all.size)
Jc_inJc = posP[mid_int_glob]                             # rows in the full-plane ordering
coords = np.column_stack([j[left], k[left]]).astype(float)
print(f"Sl: box[0,{Wx_S}], nx={nx_S}, hx={Wx_S/(nx_S-1):.4f}, mid-plane x={imid*Wx_S/(nx_S-1):.4f}")
print(f"    |Jl(face)|={left.size}, |Jc_large(plane)|={midplane_all.size}, |mid interior|={mid_int_glob.size}")

Aii = A[iI][:,iI].tocsc(); lu = spla.splu(Aii)
Aib_L = A[iI][:,left].tocsc()                            # interior <- left face
order = bisection_order(coords, nleaves)
def Sl_fwd(Vp, bs=256):                                  # face(perm) -> plane(perm)
    U=np.zeros((N,Vp.shape[1])); U[order]=Vp
    out=np.zeros((N,Vp.shape[1]))
    for c in range(0,Vp.shape[1],bs):
        sl=slice(c,min(c+bs,Vp.shape[1]))
        sol=lu.solve((Aib_L@U[:,sl]))
        out[Jc_inJc, sl] = -sol[Jc_local]
    return out[order]
def Sl_adj(Wp, bs=256):                                  # plane(perm) -> face(perm)
    U=np.zeros((N,Wp.shape[1])); U[order]=Wp
    out=np.zeros((N,Wp.shape[1]))
    for c in range(0,Wp.shape[1],bs):
        sl=slice(c,min(c+bs,Wp.shape[1]))
        t=np.zeros((iI.size, sl.stop-sl.start)); t[Jc_local]=U[Jc_inJc, sl]
        sol=lu.solve(t)                                  # Aii symmetric
        out[:, sl] = -(Aib_L.T@sol)
    return out[order]

trank_S = top_rank(Sl_fwd, Sl_adj, N)
print(f"Sl top off-diagonal numerical rank @1e-6 = {trank_S}")
print("\nSl(1/8) rk sweep:")
rks_S = [4, 8, 16, 32, 64, 128, 192, 256]
errs_S = []
for rk in rks_S:
    e = hbs_error(Sl_fwd, Sl_adj, N, order, nleaves, nl, L, rk)
    errs_S.append(e); print(f"  rk={rk:>4}  err={e:.3e}"); gc.collect()
del lu, A, Aii, Aib_L; gc.collect()

# =================================================================== Tll @ Lx=1/16
# Identical discretization & methodology to the prior run (box[0,1/16], nx=3,
# hx=1/32, boundary DtN); reused here for the overlay.
rks_T  = [16, 64, 128, 160, 192, 224, 256, 320]
errs_T = [0.221, 0.114, 0.0746, 2.615e-4, 2.545e-5, 4.426e-6, 1.420e-6, 1.955e-12]
trank_T = 254
print(f"\nTll(1/16) top off-diagonal rank @1e-6 = {trank_T} (prior identical run)")

# -------------------------------------------------------------------- plot
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.figure(figsize=(7.2,4.8))
plt.semilogy(rks_T, errs_T, "s-", color="tab:orange",
             label=f"Tll, Lx=1/16  (boundary DtN, top rank {trank_T})")
plt.semilogy(rks_S, errs_S, "o-", color="tab:blue",
             label=f"Sl, Lx=1/8  (interior interface, top rank {trank_S})")
plt.axvline(trank_T, color="tab:orange", ls=":", lw=1)
plt.axvline(trank_S, color="tab:blue", ls=":", lw=1)
plt.axhline(1e-6, color="k", ls="--", lw=.7, label="1e-6")
plt.xlabel("fixed target rank rk"); plt.ylabel("HBS matvec relative error")
plt.title("Boundary DtN vs interior interface map, reach 1/16, 128x128 face")
plt.grid(alpha=.3, which="both"); plt.legend(fontsize=9)
plt.tight_layout(); plt.savefig("/home/claude/Sl_vs_Tll.png", dpi=130)
print("\nsaved Sl_vs_Tll.png")