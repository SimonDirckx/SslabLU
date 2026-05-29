"""
Benchmark SOMS3D_legendre_v6 build cost vs total DOFs N.

Two refinement strategies on the unit cube:
  - h-refinement: vary nb (cubed tiling nb x nb x nb), p fixed
  - p-refinement: vary p, nb fixed

Two metrics per run:
  - compute time (wall-clock for SOMS_solver_sparse: matrix + RHS build)
  - peak memory footprint (tracemalloc peak during the build)

Plot results on log-log axes with reference complexity lines:
  O(N), O(N^{4/3}), O(N^{3/2}), O(N^2).

This script is a plotting smoke test: it uses only two small problem sizes
per refinement type to verify the plot machinery, not to make scaling claims.
"""

import sys
import time
import tracemalloc

import numpy as np
import matplotlib.pyplot as plt
import mumps
sys.path.insert(0, '/home/claude/soms')
from SOMS3D_csr import SOMS_solver_sparse


# Constant-coefficient Helmholtz (just a pre-defined problem to call the solver on).
K = 1.0
COEFFS = {'c11': 1.0, 'c22': 1.0, 'c33': 1.0, 'c': K ** 2}
def solve_with_mumps(Sii, rhs):
    """Solve Sii @ u = rhs using python-mumps."""
    ctx = mumps.Context()
    ctx.analyze(Sii)             # symbolic factorization (uses sparsity pattern only)
    ctx.factor(Sii)              # numeric factorization
    x = ctx.solve(rhs)           # solve, returns the solution
    return x

def run_one(p, nb):
    """
    Build the SOMS system once. Returns (N_dofs, elapsed_seconds, peak_bytes).
    Uses ct_pde=True with no forcing — the build cost dominates; the solve
    is not included.
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    Sii, Sib, ftild, XYtot, Ii, Ib = SOMS_solver_sparse(
        p, p, p, nb, nb, nb,
        coeffs=COEFFS, ct_pde=True, forcing=None,
    )
    
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    N = (nb*p)**3
    mem = Sii.data.nbytes + Sii.indices.nbytes + Sii.indptr.nbytes
    mem_tot = mem
    mem_tot+= Sib.data.nbytes + Sib.indices.nbytes + Sib.indptr.nbytes
    mem_tot+=   ftild.nbytes\
                +XYtot.nbytes
    tic = time.perf_counter()
    b = np.random.standard_normal((Sii.shape[0],))
    u = solve_with_mumps(Sii,b)
    elapsed_solve = time.perf_counter()-tic
    return N, elapsed,elapsed_solve, peak,mem,mem_tot


def add_reference_lines(ax, Ns, ys, exponents, labels):
    """
    For each exponent, plot a dashed reference line N^exp anchored so that
    it passes through the first data point's (N, y) pair.
    """
    if len(Ns) == 0:
        return
    N0 = Ns[0]
    y0 = ys[0]
    N_grid = np.geomspace(min(Ns) * 0.7, max(Ns) * 1.4, 50)
    for exp, lab in zip(exponents, labels):
        scale = y0 / (N0 ** exp)
        ax.loglog(N_grid, scale * N_grid ** exp,
                  '--', linewidth=0.8, alpha=0.6, label=lab)


def main():
    # --- h-refinement: vary nb, p fixed ---
    p_fixed = 8
    nb_list = [2,3,4,5,6]                       # two small sizes
    h_results = []
    print(f"h-refinement (p = {p_fixed}):")
    for nb in nb_list:
        N, elapsed,elapsed_solve, peak,mem,mem_tot = run_one(p_fixed, nb)
        print(f"    nb={nb}\
                    N={N:>6} time_construct={elapsed:.3f}s\
                    time_solve={elapsed_solve:.3f}s\
                    mem_tot={mem_tot/1e6:.1f}MB\
                    peak_mem={peak/1e6:.1f}MB    ")
        h_results.append((N, elapsed,elapsed_solve, peak,mem))

    # --- p-refinement: vary p, nb fixed ---
    nb_fixed = 4
    p_list = [6,8,10]                        # two small sizes
    p_results = []
    print(f"\np-refinement (nb = {nb_fixed}):")
    for p in p_list:
        N, elapsed,elapsed_solve, peak,mem,mem_tot = run_one(p, nb_fixed)
        print(f"    p={p}\
                    N={N:>6} time_construct={elapsed:.3f}s\
                    time_solve={elapsed_solve:.3f}s\
                    mem={mem/1e6:.1f}MB\
                    peak_mem={peak/1e6:.1f}MB    ")
        p_results.append((N, elapsed,elapsed_solve, peak,mem))

    # --- Plot ---
    fig, axes = plt.subplots(3, 2, figsize=(11, 8))
    exponents = [1.0, 4.0 / 3.0, 1.5, 2.0]
    labels = ['$O(N)$', '$O(N^{4/3})$', '$O(N^{3/2})$', '$O(N^2)$']

    def plot_panel(ax, results, title, ylabel, kk=0):
        if not results:
            return
        Ns = np.array([r[0] for r in results])
        if kk == 0:
            ys_raw = np.array([r[1] for r in results]) 
        elif kk == 1:
            ys_raw = np.array([r[4]/1e6 for r in results]) 
        elif kk == 2:
            ys_raw = np.array([r[2] for r in results]) 
        else:
            raise KeyError("which must be 0, 1 or 2")
        ax.loglog(Ns, ys_raw, 'o-', linewidth=1.8, markersize=7, label='measured')
        add_reference_lines(ax, Ns, ys_raw, exponents, labels)
        ax.set_xlabel('Total DOFs $N$')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8, loc='best')

    plot_panel(axes[0, 0], h_results, f'h-refinement (p={p_fixed}): build time',
               'time (s)')
    plot_panel(axes[0, 1], p_results, f'p-refinement (nb={nb_fixed}): build time',
               'time (s)')
    plot_panel(axes[1, 0], h_results, f'h-refinement (p={p_fixed}): peak memory',
               'peak memory (MB)', kk=1)
    plot_panel(axes[1, 1], p_results, f'p-refinement (nb={nb_fixed}): peak memory',
               'peak memory (MB)', kk=1)
    plot_panel(axes[2, 0], h_results, f'h-refinement (p={p_fixed}): factorization time',
               'factorization time (s)', kk=2)
    plot_panel(axes[2, 1], p_results, f'p-refinement (nb={nb_fixed}): factorization time',
               'factorization time (s)', kk=2)

    fig.suptitle('SOMS3D v6 — build cost scaling (smoke test, 2 points per series)',
                 fontsize=11)
    fig.tight_layout()
    out = '/home/simond/SslabLU/scaling.png'
    fig.savefig(out, dpi=120)
    #plt.show()
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()