"""Helmholtz test on the 5-patch cylinder: -Lap u - k^2 u = 0, exact solution
u = cos(k r)/r from a point source outside the cylinder."""
import sys, time
import numpy as np
sys.path.insert(0, "/home/claude/glue")
from stellerator import (make_cylinder, build_patch, glue, solve_dirichlet,
                           physical_normals, face_layout, FACE_AXIS, FACE_SIGN)

k = 3.0
x0 = np.array([1.6, 0.4, 0.5])          # outside: distance to axis 1.65 > 1
L = 1.0

def uex(X):
    r = np.linalg.norm(X - x0, axis=1)
    return np.cos(k * r) / r

cfun = lambda X: -k ** 2 * np.ones(len(X))


def sanity(built, gl):
    counts = gl["counts"]
    print("  cluster sizes:", dict(zip(*np.unique(counts, return_counts=True))))
    Xb = gl["Xdof"][counts == 1]
    on_wall = np.abs(np.hypot(Xb[:, 0], Xb[:, 1]) - 1) < 1e-9
    on_cap = (np.abs(Xb[:, 2]) < 1e-9) | (np.abs(Xb[:, 2] - L) < 1e-9)
    print("  boundary unknowns on wall/caps:", np.all(on_wall | on_cap))
    # outward physical normals of the two slots in every interior cluster
    nrm = []
    for b in built:
        hps = b["hps"]; q = np.array(hps.q); _, offs = face_layout(q)
        Gr = physical_normals(b["P"], b["Xref"])
        loc = np.arange(len(b["Xref"])) % offs[-1]
        f = np.searchsorted(offs, loc, side="right") - 1
        kax = np.array(FACE_AXIS)[f]; sg = np.array(FACE_SIGN)[f]
        g = Gr[np.arange(len(loc)), kax, :] * sg[:, None]
        nrm.append(g / np.linalg.norm(g, axis=1, keepdims=True))
    nrm = np.vstack(nrm)
    lab = gl["labels"]
    S = np.zeros((counts.size, 3)); np.add.at(S, lab, nrm)
    print("  max |n_a + n_b| over interior clusters: %.1e"
          % np.abs(S[counts == 2]).max())


def run(p, convert=True, n=4, m=4, nz=4, verbose=False):
    patches = make_cylinder(s=0.5, L=L, n=n, m=m, nz=nz)
    t0 = time.time()
    built = [build_patch(P, p, cfun, convert=convert) for P in patches]
    gl = glue(built)
    if verbose:
        sanity(built, gl)
    out, u = solve_dirichlet(gl, built, uex)
    t1 = time.time()
    Ic = gl["counts"] == 2
    e_if = np.abs(u[Ic] - uex(gl["Xdof"][Ic])).max()
    e_all = max(np.abs(us - uex(X)).max() for X, us in out)
    scale = max(np.abs(uex(X)).max() for X, _ in out)
    return dict(p=p, ndof=int(Ic.sum()), e_if=e_if / scale, e_all=e_all / scale, t=t1 - t0)


if __name__ == "__main__":
    print("Sanity checks (p=8, physical-flux gluing):")
    r = run(8, verbose=True)
    print("  rel. max error interfaces %.2e   all nodes %.2e" % (r["e_if"], r["e_all"]))

    print("\nConvergence in p (4x4x4 core leaves, arches 4x4x4):")
    print("   p   interface dofs   rel.err (interfaces)   rel.err (all nodes)   time[s]")
    for p in [6, 8, 10]:
        r = run(p)
        print("  %2d   %8d         %.2e              %.2e           %.1f"
              % (p, r["ndof"], r["e_if"], r["e_all"], r["t"]))

    print("\nh-refinement at p=8 (leaves per direction doubled in every patch):")
    for nl in [4,8]:
        r = run(8, n=nl, m=nl, nz=nl)
        print("  %d leaves/dir: interface dofs %6d   rel.err (all nodes) %.2e   time %.1f s"
              % (nl, r["ndof"], r["e_all"], r["t"]))

    print("\nNaive gluing (reference normal derivatives, no metric conversion):")
    for p in [8, 12]:
        r = run(p, convert=False)
        print("  p=%2d  rel.err (all nodes) %.2e" % (p, r["e_all"]))