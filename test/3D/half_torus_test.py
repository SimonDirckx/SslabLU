"""Helmholtz on the HALF twisted elliptic torus, solved on the straight cylinder.

Physical domain: half torus (zeta in [0, pi]), major radius R0 = 3, elliptic
cross-section with semi-axes a = 1.2, b = 0.75, rotated in its poloidal plane
by alpha = 2 zeta (planar twist, not periodic, flat end caps).
Computational domain: 5-patch HPS cylinder of radius sqrt(a b) (same area) and
length pi R0 (same centreline length).
Exact solution: u = cos(k r)/r, r = |x - x0|, with the source x0 outside the
half torus and off the z-axis (the axis the cylinder is bent around).
"""
import time
import numpy as np
from stellerator import make_cylinder, build_patch, glue, solve_dirichlet
from half_torus import HalfTwistedTorus, point_source

T = HalfTwistedTorus(R0=3.0, a=1.2, b=0.75, twist=2.0)
k = 2.0
x0 = np.array([2.0, -1.0, 0.5])      # in the missing half (y < 0); 2.24 from the z-axis
u_phys = point_source(k, x0)
u_cyl = lambda Xc: u_phys(T.phi(Xc))  # exact solution pulled back to the cylinder
PDE = T.helmholtz(k)


def run(p, n=4, m=4, nz=40, flux="normal", verbose=False):
    patches = make_cylinder(s=0.5, L=T.L, n=n, m=m, nz=nz, Rc=T.Rc)
    t0 = time.time()
    built = [build_patch(P, p, PDE, flux=flux) for P in patches]
    gl = glue(built)
    if verbose:
        c = gl["counts"]; Xb = gl["Xdof"][c == 1]
        wall = np.abs(np.hypot(Xb[:, 0], Xb[:, 1]) - T.Rc) < 1e-9
        cap = (np.abs(Xb[:, 2]) < 1e-9) | (np.abs(Xb[:, 2] - T.L) < 1e-9)
        print("  cluster sizes:", dict(zip(*np.unique(c, return_counts=True))),
              "  boundary unknowns on wall/caps:", bool(np.all(wall | cap)))
    out, u = solve_dirichlet(gl, built, u_cyl)
    t = time.time() - t0
    Ic = gl["counts"] == 2
    scale = max(np.abs(u_cyl(X)).max() for X, _ in out)
    e_if = np.abs(u[Ic] - u_cyl(gl["Xdof"][Ic])).max() / scale
    e_all = max(np.abs(us - u_cyl(X)).max() for X, us in out) / scale
    nleaf = sum(int(b["hps"].nboxes) for b in built)
    return dict(e_if=e_if, e_all=e_all, t=t, ndof=int(Ic.sum()), nleaf=nleaf)


if __name__ == "__main__":
    print("Half twisted elliptic torus: R0=%.1f, a=%.2f, b=%.2f, twist alpha=%.1f*zeta"
          % (T.R0, T.a, T.b, T.twist))
    print("Cylinder: radius %.4f (area %.4f = pi a b %.4f), length %.4f = pi R0"
          % (T.Rc, np.pi * T.Rc ** 2, np.pi * T.a * T.b, T.L))
    print("Source x0 =", x0, " distance to z-axis %.2f" % np.hypot(*x0[:2]))

    print("\nSanity (p=8):")
    r = run(8, verbose=True)
    print("  rel. max error: interfaces %.2e   all leaf nodes %.2e" % (r["e_if"], r["e_all"]))

    print("\np-convergence (core 2x2, arches 2x2, 10 leaves along the cylinder):")
    print("   p   leaves  interface dofs   rel.err interfaces   rel.err all nodes   time[s]")
    for p in [6, 8, 10, 12]:          # p=12 with 200 leaves needs > 7 GB RAM on CPU
        r = run(p)
        print("  %2d   %5d    %8d          %.2e             %.2e        %.1f"
              % (p, r["nleaf"], r["ndof"], r["e_if"], r["e_all"], r["t"]))
