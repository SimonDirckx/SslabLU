"""
HBS rank growth study ("comb" experiment) for the dense block-tridiagonal
direct solvers.

The question: as a direct solver eliminates, the blocks it produces (Schur
complements on the diagonal, the fill-in off-diagonals) are no longer the
original operator.  If those blocks were stored in HBS form, what rank would
each one need to be "sufficiently accurate", and how does that rank grow with
the level of the solver?

Method: factorize EXACTLY and densely with omsdirectsolve, keeping every
intermediate block (save_levels=True).  Then, block by block and level by
level, compress the exact block at a range of HBS ranks and measure the error
directly.  Ranks are never inferred from an SVD of the off-diagonal blocks:
that would give the BLR rank, which for HBS is only a lower bound -- the shared
nested bases can fail to represent a block at a rank the per-block SVD says is
enough.  Only the HBS construction itself is trusted.

Three sampled error measures per block, all relative and all using the same
fixed Gaussian probes across every rank so the curves are comparable:

  err_apply   ||(B - H) G|| / ||B G||
              fidelity of the compression itself.

  err_solve   ||B^-1 R - H.solve(R)|| / ||B^-1 R||
              fidelity of the ULV solve, which is what a diagonal block is
              actually used for.  Reported for diagonal blocks only; the
              off-diagonals are only ever applied.

  err_global  ||x_exact - x_swapped|| / ||x_exact||
              swap the compressed block into an otherwise exact solver and
              solve.  Two variants:

                per-block  one block replaced at a time.  Measures that block's
                           DIRECT contribution to the final solve.  Note it is
                           identically zero for even-indexed red-black
                           diagonals: those are never solved with at their own
                           level, they feed the construction of the next one.
                           So a zero here means "not read by the solve", not
                           "harmless to compress".

                per-level  every block at a level replaced at once.  This is
                           the number to read as "the rank level l needs".

Every probe hangs off the exact factorization and dead-ends there -- one spine,
uniform teeth -- hence "comb".  Neither err_global variant captures error
PROPAGATED from level l into the construction of level l+1, because the later
levels here were always built from the exact blocks.  What is measured is the
rank each exact block needs to be representable, not the rank a fully
compressed solver would need to stay stable.  That is the ladder experiment.

Non-cyclic only.

Sampling.  The HBS construction is randomized, so the samples it is built from
are drawn HERE rather than left to HBSMAT's internal default.  Two reasons.
First, the internal path re-draws for every rank, so consecutive points on a
rank curve would differ by the realization of Om as well as by the rank -- the
very noise the `stable` flag in minimal_ranks exists to detect.  Second, its
oversampling is a constant +10 regardless of rank, so the ratio s/rk collapses
as the rank grows and the high-rank end of every curve is sample-starved.

Instead: one Gaussian pair (Om, Psi) is drawn per solver at the widest budget
the rank grid needs, shared by every block at every level, and each rank uses
the leading n_samples(rk) columns of it.  Slicing is exact -- Y[:, :s] is
A @ Om[:, :s] columnwise, and a prefix of an i.i.d. Gaussian is a valid draw of
that width -- so every rank is measured with exactly the oversampling a
standalone compression at that rank would have given it, and no rank is
flattered by a budget sized for a larger one.

Device.  Om, Psi and the products Y, Z are built as torch tensors on
cfg.device, because HBSMAT allocates its internals there and the construction
mixes the two.  Their dtype follows torch.get_default_dtype(), which is what
the untyped torch.zeros calls inside ULVsparse_torch allocate -- so if the
process default is float32, every error here floors around 1e-7 regardless of
rank.  Set torch.set_default_dtype(torch.float64) before running.
"""

import csv
import time
from dataclasses import dataclass, field, fields, astuple

import numpy as np
import torch

import matAssembly.HBS.HBStorch as HBSnew
import gen_HH_op_cube as HHcube
from direct_solve.omsdirectsolve import ThomasSolver, RedBlackSolver, dense_to_linop


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def nleaf_from_tree(tree):
    """DOFs per leaf.

    Prefer the actual size of a leaf box, which is what HBSMAT itself uses
    (self.nl = len(perm)//Nb); _min_leaf_size is only the threshold that
    stopped the splitting and can be larger than any leaf really is.
    """
    try:
        return int(len(tree.get_box_inds(tree.get_leaves()[0])))
    except Exception:
        pass
    n = getattr(tree, '_min_leaf_size', None)
    if n is None:
        raise ValueError("tree has no _min_leaf_size; pass rk_grid explicitly.")
    return int(n)


def make_rank_grid(nleaf, step=10):
    """nleaf/2 .. 4*nleaf inclusive, in steps of `step`."""
    nleaf    = int(nleaf)
    grid_max = 4 * nleaf
    grid     = list(range(nleaf // 2, grid_max + 1, step))
    if grid[-1] != grid_max:
        grid.append(grid_max)
    return grid


def n_samples(rk, nl, fac, oversample=0.5, oversample_min=20):
    """Samples a standalone HBS compression at rank `rk` would use.

    max(fac*rk, nl) is the largest block a level presents: fac*rk at the coarse
    levels (fac = 2 for a binary tree, 4 for a quad tree, since that is how many
    children the reduction stacks), nl at the leaves.  +rk is the basis itself.
    The last term is the oversampling, held proportional to rk so that s/rk does
    not collapse as the rank grows.
    """
    return (max(int(fac)*int(rk), int(nl)) + int(rk)
            + max(int(oversample_min), int(oversample*rk)))


def default_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class ProbeConfig:
    """Everything the sweep needs beyond the operator itself.

    One tree/quad pair is reused for every block at every level, which is valid
    because the reduction keeps every block m x m.
    """
    tree:           object
    quad:           object
    rk_grid:        list  = None      # defaults to nleaf/2 .. 4*nleaf
    step:           int   = 10
    device:         str   = None      # 'cuda' when available
    torch_dtype:    object = None     # defaults to torch.get_default_dtype()
    fast:           bool  = False
    nprobe:         int   = 50        # columns for err_apply / err_solve
    nrhs:           int   = 4         # right-hand sides for err_global
    seed:           int   = 0
    oversample:     float = 0.5       # p = max(oversample_min, oversample*rk)
    oversample_min: int   = 20
    cache_samples:  bool  = True      # keep A@Om per block for a level's rank loop
    eps_list:       list  = field(default_factory=lambda: [1e-3, 1e-4, 1e-5, 1e-6])
    per_block_e2e:  bool  = True
    per_level_e2e:  bool  = True
    verbose:        bool  = True

    def __post_init__(self):
        self.nl  = nleaf_from_tree(self.tree)
        # how many children the HBS reduction stacks per level
        self.fac = 4 if self.quad else 2
        if self.rk_grid is None:
            self.rk_grid = make_rank_grid(self.nl, self.step)
        if self.device is None:
            self.device = default_device()
        if self.torch_dtype is None:
            # Match what HBSMAT allocates internally: the untyped torch.zeros
            # calls in ULVsparse_torch follow the process default, and a
            # float32 factor against a float64 sample raises rather than casts.
            self.torch_dtype = torch.get_default_dtype()

    def n_samples(self, rk):
        return n_samples(rk, self.nl, self.fac,
                         self.oversample, self.oversample_min)

    def tensor(self, X):
        """numpy -> torch, on the device and in the dtype HBSMAT expects."""
        return torch.as_tensor(np.ascontiguousarray(X),
                               dtype=self.torch_dtype, device=self.device)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class Record:
    """One (block, rank) measurement.  Field order is the CSV column order."""
    solver:           str
    level:            int
    node:             int
    role:             str
    rk:               int
    m:                int
    nsamples:         int
    block_norm:       float
    err_apply:        float
    err_solve:        float = None    # diagonal blocks only
    err_global_block: float = None    # filled by the per-block substitution
    err_global_level: float = None    # filled by the whole-level substitution
    read_by_solve:    bool  = None

    @property
    def key(self):
        return (self.solver, self.level, self.node, self.role)


@dataclass
class RankResult:
    """Smallest rank clearing a tolerance, and whether it holds above it."""
    rank:   int   = None      # None: the grid never cleared eps
    stable: bool  = None      # do all larger ranks in the grid clear it too
    best:   float = None      # smallest error seen anywhere on the grid


@dataclass
class BlockRef:
    """Exact quantities for one block, computed once and reused every rank."""
    apply: object              # A @ G
    lu:    object = None       # diagonal blocks only
    solve: object = None       # A^-1 R, diagonal blocks only


# ---------------------------------------------------------------------------
# Error measures
# ---------------------------------------------------------------------------

def _apply(op, X):
    if hasattr(op, 'matmat'):
        return np.asarray(op.matmat(X))
    return np.asarray(op @ X)


def _relerr(ref, approx):
    den = np.linalg.norm(ref)
    if den == 0:
        return np.nan
    return float(np.linalg.norm(ref - approx) / den)


def err_apply(A_dense, H, G, ref=None):
    ref = A_dense @ G if ref is None else ref
    return _relerr(ref, _apply(H, G))


def err_solve(A_lu, H, R, ref=None):
    from scipy.linalg import lu_solve
    ref = lu_solve(A_lu, R) if ref is None else ref
    try:
        got = np.asarray(H.solve(R))
    except Exception:
        return np.nan
    return _relerr(ref, got)


def _detect_read_slots(solver, RHS, x_ref, blocks, rng):
    """Which slots the solve actually reads.

    Determined by substituting a deliberately wrong operator once per block and
    seeing whether the answer moves.  Inferring this from a zero compression
    error instead would be wrong at high rank, where a block that IS read
    produces an error of exactly 0.0 and would be misread as unused.
    """
    read = {}
    for b in blocks:
        junk = dense_to_linop(3.0 * np.eye(b.shape[0])
                              + rng.standard_normal(b.shape))
        prev = solver.set_block_op(*b.key, junk)
        try:
            e = _relerr(x_ref, solver.solve(RHS))
        finally:
            solver.set_block_op(*b.key, prev)
        read[b.key] = bool(e > 1e-12)
    return read


def _system_size(solver):
    if isinstance(solver, RedBlackSolver):
        return solver.m * solver.nSlabs
    return solver.m * len(solver.B)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep_solver(solver, cfg, label=None):
    """Compress every logged block at every rank in the grid and measure.

    Returns a flat list of Records, one per (level, node, role, rk).
    """
    from scipy.linalg import lu_factor, lu_solve

    label = label or type(solver).__name__
    rng   = np.random.default_rng(cfg.seed)
    m     = solver.m

    # Fixed probes, shared by every block and every rank.
    G = rng.standard_normal((m, cfg.nprobe))
    R = rng.standard_normal((m, cfg.nprobe))

    blocks = solver.get_blocks()                    # 'general' only
    levels = sorted({b.level for b in blocks})

    N        = _system_size(solver)
    RHS      = rng.standard_normal((N, cfg.nrhs))
    need_e2e = cfg.per_block_e2e or cfg.per_level_e2e
    x_ref    = solver.solve(RHS) if need_e2e else None
    read     = (_detect_read_slots(solver, RHS, x_ref, blocks, rng)
                if need_e2e else {})

    # Exact references per block, computed once.
    ref = {}
    for b in blocks:
        ref[b.key] = BlockRef(apply=b.dense @ G)
        if b.role == 'diag':
            ref[b.key].lu    = lu_factor(b.dense)
            ref[b.key].solve = lu_solve(ref[b.key].lu, R)

    # Construction samples: one draw at the widest budget any usable rank in the
    # grid asks for, shared by every block and every level.  Each rank slices the
    # leading cfg.n_samples(rk) columns out of it.  Drawn from the seeded rng, so
    # unlike HBSMAT's internal path the whole study is reproducible.  Built on
    # cfg.device: HBSMAT allocates its internals there and multiplies the two.
    usable = [rk for rk in cfg.rk_grid if rk < m]
    smax   = max((cfg.n_samples(rk) for rk in usable), default=0)
    Om     = cfg.tensor(rng.standard_normal((m, smax))) if smax else None
    Psi    = cfg.tensor(rng.standard_normal((m, smax))) if smax else None
    if cfg.verbose and smax:
        print(f"    [{label}] samples: nl={cfg.nl} fac={cfg.fac}  "
              f"s = {cfg.n_samples(min(usable))} .. {smax}  "
              f"(block size m = {m}, {cfg.device}, {cfg.torch_dtype})")
        if smax >= m:
            print(f"    [{label}] NOTE: s reaches {smax} >= m = {m} at the top "
                  f"of the grid; those ranks see the whole block.")

    records = []

    for level in levels:
        lvl = [b for b in blocks if b.level == level]

        # Exact samples per block, once for this level's whole rank loop.
        # Y[:, :s] == A @ Om[:, :s] columnwise, so slicing these is the same as
        # sampling at width s.
        YZ = {}
        if cfg.cache_samples and smax:
            for b in lvl:
                A_t     = cfg.tensor(b.dense)
                YZ[b.key] = (A_t @ Om, A_t.T @ Psi)

        for rk in cfg.rk_grid:
            if rk >= m:
                if cfg.verbose:
                    print(f"    [{label}] level {level}: skipping rk={rk} "
                          f">= block size {m}")
                continue

            t0   = time.time()
            s    = min(cfg.n_samples(rk), smax)
            rows = []          # (block, record, compressed) for this rank

            for b in lvl:
                if b.key in YZ:
                    Y, Z = YZ[b.key]
                    Y, Z = Y[:, :s], Z[:, :s]
                else:
                    A_t  = cfg.tensor(b.dense)
                    Y, Z = A_t @ Om[:, :s], A_t.T @ Psi[:, :s]

                H = HBSnew.HBSMAT(dense_to_linop(b.dense),
                                  device=cfg.device, tree=cfg.tree, quad=cfg.quad)
                H.construct(rk, Om[:, :s], Psi[:, :s], Y, Z,
                            compute_ULV=True, fast=cfg.fast)

                rec = Record(
                    solver=label, level=level, node=b.node, role=b.role,
                    rk=rk, m=m, nsamples=s,
                    block_norm=float(np.linalg.norm(b.dense)),
                    err_apply=err_apply(b.dense, H, G, ref[b.key].apply),
                    err_solve=(err_solve(ref[b.key].lu, H, R, ref[b.key].solve)
                               if b.role == 'diag' else None),
                    read_by_solve=read.get(b.key))

                records.append(rec)
                rows.append((b, rec, H))

            # ---- per-block substitution --------------------------------
            if cfg.per_block_e2e:
                for b, rec, H in rows:
                    prev = solver.set_block_op(*b.key, H)
                    try:
                        rec.err_global_block = _relerr(x_ref, solver.solve(RHS))
                    finally:
                        solver.set_block_op(*b.key, prev)

            # ---- whole-level substitution ------------------------------
            if cfg.per_level_e2e:
                saved = [(b, solver.set_block_op(*b.key, H)) for b, _, H in rows]
                try:
                    e_lvl = _relerr(x_ref, solver.solve(RHS))
                finally:
                    for b, prev in saved:
                        solver.set_block_op(*b.key, prev)
                for _, rec, _ in rows:
                    rec.err_global_level = e_lvl

            if cfg.verbose:
                print(f"    [{label}] level {level:>2}  rk={rk:>4}  s={s:>5}  "
                      f"{len(lvl)} blocks  {time.time()-t0:6.2f}s")

        YZ.clear()

    return records


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _first_clearing(pairs, eps):
    """(rank, error) pairs -> RankResult.

    The construction is randomized, so a lone passing rank below a run of
    failures is luck; `stable` records whether every larger rank clears eps too.
    """
    pairs = sorted(p for p in pairs if p[1] is not None and not np.isnan(p[1]))
    if not pairs:
        return RankResult()
    best    = min(e for _, e in pairs)
    passing = [rk for rk, e in pairs if e <= eps]
    if not passing:
        return RankResult(rank=None, stable=False, best=best)
    rk0 = min(passing)
    return RankResult(rank=rk0, best=best,
                      stable=all(e <= eps for rk, e in pairs if rk >= rk0))


def minimal_ranks(records, metric, eps):
    """Smallest rank clearing eps, per block."""
    out = {}
    for key in {r.key for r in records}:
        out[key] = _first_clearing(
            [(r.rk, getattr(r, metric)) for r in records if r.key == key], eps)
    return out


def level_ranks(records, eps, metric='err_global_level'):
    """Smallest rank at which a whole level clears eps."""
    out = {}
    for key in {(r.solver, r.level) for r in records}:
        out[key] = _first_clearing(
            {(r.rk, getattr(r, metric)) for r in records
             if (r.solver, r.level) == key}, eps)
    return out


def print_summary(records, cfg):
    solvers = sorted({r.solver for r in records})
    for eps in cfg.eps_list:
        print(f"\n=== sufficient rank at eps = {eps:g} " + "=" * 40)

        per_block = minimal_ranks(records, 'err_apply', eps)
        per_solve = minimal_ranks(records, 'err_solve', eps)
        per_level = level_ranks(records, eps)

        for s in solvers:
            print(f"\n  {s}")
            print(f"    {'level':>5} {'blocks':>7} {'apply':>14} "
                  f"{'solve(diag)':>14} {'whole level':>12}")
            for l in sorted({r.level for r in records if r.solver == s}):
                keys  = [k for k in per_block if k[0] == s and k[1] == l]
                ap    = [per_block[k].rank for k in keys]
                sv    = [per_solve[k].rank for k in keys
                         if k[3] == 'diag' and per_solve[k].rank is not None]
                nfail = sum(1 for v in ap if v is None)
                ap_s  = ('-' if all(v is None for v in ap)
                         else f"{max(v for v in ap if v is not None)}"
                              + (f" (+{nfail} fail)" if nfail else ""))
                sv_s  = f"{max(sv)}" if sv else '-'
                lv    = per_level[(s, l)].rank if (s, l) in per_level else None
                print(f"    {l:>5} {len(keys):>7} {ap_s:>14} {sv_s:>14} "
                      f"{(lv if lv is not None else '-'):>12}")

    # blocks the solve never reads, worth stating once
    unread = sorted({r.key for r in records if r.read_by_solve is False})
    if unread:
        print(f"\n  {len(unread)} block(s) are never read by the solve, so their "
              f"per-block err_global is 0 by construction.")
        print("  Read the whole-level column for those, not the per-block one.")


def write_csv(records, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([fl.name for fl in fields(Record)])
        for r in records:
            w.writerow(astuple(r))
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_study(S_rk_list, cfg, T=None, solvers=('redblack', 'thomas'),
              csv_path=None):
    """Factorize densely, then sweep HBS ranks over every intermediate block.

    S_rk_list : list of (S^-_i, S^+_i) pairs, dense arrays or LinearOperators.
    cfg       : ProbeConfig, carrying the tree/quad the compressions use.
    T         : optional diagonal list; None means the identity diagonal.
    """
    m = S_rk_list[0][0].shape[0]
    print(f"rank grid: {cfg.rk_grid}   (block size m = {m}, "
          f"device = {cfg.device}, dtype = {cfg.torch_dtype})")
    print(f"sampling: s(rk) = max({cfg.fac}*rk, {cfg.nl}) + rk + "
          f"max({cfg.oversample_min}, {cfg.oversample:g}*rk)")

    records = []
    for which in solvers:
        if which == 'redblack':
            s = RedBlackSolver(m, cyclic=False, save_levels=True)
        elif which == 'thomas':
            s = ThomasSolver(m, cyclic=False, save_levels=True)
        else:
            raise ValueError(which)

        t0 = time.time()
        s.factorize(S_rk_list, T)
        print(f"\n{type(s).__name__}: dense factorization {time.time()-t0:.2f}s, "
              f"{len(s.get_blocks())} general blocks over {s.n_levels} levels")
        records += sweep_solver(s, cfg)

    print_summary(records, cfg)
    if csv_path:
        write_csv(records, csv_path)
        print(f"\nwrote {csv_path} ({len(records)} rows)")
    return records


def main():
    N = 17
    H = 1/N
    S_rk_list, tree = HHcube.get_HH_op_cube(25., N, 8, np.array([H/4, 1./16, 1./16]))
    cfg = ProbeConfig(tree, quad=False)
    run_study(S_rk_list, cfg=cfg, csv_path="rank_growth.csv")


if __name__ == "__main__":
    main()