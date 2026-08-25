"""
Overlapping multi-slab (OMS) decomposition.

Two front-end classes:
    oms     -- interface operators are compressed (rank-structured) via an assembler
    oms_lu  -- interface operators are kept as un-compressed local solve operators

Both share bookkeeping and the global S-operator assembly through _omsBase.

Changes relative to the original are tagged  # FIX:  (correctness) and  # OPT:
(performance / hygiene).
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

import solver.solver as solverWrap
from solver.solver import stMap

# OPT: dropped unused imports (sys, matplotlib.pyplot, scipy.sparse.linalg as
#      splinalg) and the duplicated numpy / solver imports.

__all__ = ["slab", "omsStats", "oms", "oms_lu"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

# FIX: the original compared coordinates against a hard 1e-14 absolute
#      tolerance.  That silently returns empty index sets on domains whose
#      coordinates are not O(1), or when xc = (xl+xr)/2 is not exactly
#      representable.  Tolerance is now relative to the coordinate magnitude.
_COORD_RTOL = 1e-12


def _coord_close(x, x0, scale):
    return np.abs(x - x0) <= _COORD_RTOL * max(1.0, abs(scale), abs(float(x0)))


def _as_index(block):
    """
    Contiguous global dof blocks are stored as `range` objects (they carry a
    len(), which callers rely on).  For array access a `slice` is strictly
    better: it yields a *view*, so `result[blk] += X` is genuinely in place
    instead of fancy-index read-modify-write through a temporary.
    """
    if isinstance(block, range) and block.step == 1:
        return slice(block.start, block.stop)
    if isinstance(block, slice):
        return block
    return np.asarray(block)


def _dtype_of(obj, default=np.float64):
    dt = getattr(obj, "dtype", None)
    return default if dt is None else dt


def _eval_reduced_load(reduced_load, solver, slabInd):
    """
    FIX: the original used two incompatible conventions for `reduced_load` --
    called as reduced_load(solver.solver) in construct_Stot_helper, subscripted
    as reduced_load[slabInd] in construct_rhstot -- and defaulted to
    `lambda x: 0`, which works with neither.  Both forms are accepted here.

    Must return (b_C, b_X).
    """
    if reduced_load is None:
        raise ValueError(
            "problem_type='mixed' requires `reduced_load`: either a sequence "
            "indexed by slab, or a callable taking the local solver."
        )
    if callable(reduced_load):
        out = reduced_load(solver.solver)
    else:
        out = reduced_load[slabInd]
    b_C, b_X = out
    return np.asarray(b_C), np.asarray(b_X)


# --------------------------------------------------------------------------- #
# slab
# --------------------------------------------------------------------------- #


class slab:
    """
    Class encoding source-target maps (left and right)

    @param
    geom: local geometry
    gb_vec: local boundary dirichlet data
    transform: local transform into global domain
    """

    def __init__(self, geom, gb_vec, transform=None):
        self.geom = np.asarray(geom)
        self.transform = transform
        self.gb_vec = gb_vec

        ndim = self.geom.shape[-1]
        # FIX: bare `except:` swallowed KeyboardInterrupt and masked genuine
        #      errors raised inside the user's gb.  Also, np.random.randn
        #      perturbed the global RNG stream (breaking reproducibility of
        #      anything downstream) and probed gb at points outside the slab.
        try:
            rng = np.random.default_rng(0)
            lo = np.min(self.geom, axis=0)
            hi = np.max(self.geom, axis=0)
            probe = lo + (hi - lo) * rng.random((5, ndim))
            res = np.asarray(gb_vec(probe))
        except Exception as exc:
            raise ValueError(
                "gb must accept an array of shape (numpoints, ndim)"
            ) from exc
        if res.ndim != 1 or res.shape[0] != 5:
            raise ValueError(
                "gb must return a 1d array of length numpoints; got shape %r"
                % (res.shape,)
            )

    def compute_idxs_and_pts(self, solver, XX=None, XXb=None, XXi=None):
        """
        Indices needed for the source-target maps (left, center, right,
        boundary, interior).

        `XX` (or, cheaper, `XXb` / `XXi` directly) may be supplied to override
        the solver's own coordinates -- needed when a single reference solver
        is reused for several translated slabs under stiff_mat_const.
        """
        if XXb is None or XXi is None:
            XX = solver.XX if XX is None else XX
            XXb = XX[solver.Ib, ...] if XXb is None else XXb
            XXi = XX[solver.Ii, ...] if XXi is None else XXi

        xl = float(self.geom[0][0])
        xr = float(self.geom[1][0])
        xc = 0.5 * (xl + xr)
        scale = max(abs(xl), abs(xr), abs(xr - xl))

        # OPT: gb_vec was evaluated three times on the full boundary point set.
        gb_b = np.asarray(self.gb_vec(XXb), dtype=bool)

        # OPT: np.where(mask)[0] -> np.flatnonzero(mask)
        Il = np.flatnonzero(_coord_close(XXb[..., 0], xl, scale) & ~gb_b)
        Ir = np.flatnonzero(_coord_close(XXb[..., 0], xr, scale) & ~gb_b)
        Ic = np.flatnonzero(_coord_close(XXi[..., 0], xc, scale))
        Igb = np.flatnonzero(gb_b)

        return Il, Ir, Ic, Igb, XXi, XXb


class omsStats:
    # stats for debugging and performance checks
    def __init__(self):
        self.compression = None
        self.compr_timing = None
        self.discr_timing = None
        self.sampl_timing = None
        self.n_factorizations = None   # 1 when stiff_mat_const is used
        self.n_assembled = None        # distinct interface blocks built
        self.n_reused = None           # blocks served from the cache


# --------------------------------------------------------------------------- #
# shared base
# --------------------------------------------------------------------------- #


class _omsBase:
    """
    Bookkeeping and global S-operator assembly shared by `oms` and `oms_lu`.

    The two classes were ~90% duplicated in the original, which is why the
    compute_global_dofs bug existed twice and the rmatvec bug three times.
    """

    def __init__(self, slabList: list, pdo, gb, solver_opts, connectivity,
                 stiff_mat_const=False):
        self.slabList = slabList
        self.pdo = pdo
        self.connectivity = connectivity
        self.opts = solver_opts
        self.gb = gb
        self.glob_target_dofs = []
        self.glob_source_dofs = []
        self.localSolver = None
        self.nbytes = 0
        self.densebytes = 0
        self.stats = omsStats()
        self.nc = None
        self.ncs = []          # FIX: per-slab interface sizes; self.nc alone
                               #      silently assumed every slab was the same.

        # ---- constant-stiffness-matrix reuse ----------------------------- #
        # `stiff_mat_const` asserts that every slab discretizes to *the same*
        # matrices (Aii, Aib, Abi, Abb, and for 'mixed' also M, E, JD, JN).
        # That needs constant PDE coefficients AND isomorphic discretizations
        # -- constant coefficients alone are not enough under a non-uniform
        # mesh.  Under it, one factorization and one assembly per distinct
        # face serve the whole decomposition.
        self.stiff_mat_const = bool(stiff_mat_const)
        self._ref_solver = None
        self._offsets = None
        self._block_cache = {}
        self._n_factorizations = 0
        self._n_assembled = 0
        self._n_reused = 0

    # ------------------------------------------------------------------ #
    # constant-stiffness-matrix support
    # ------------------------------------------------------------------ #

    def _compute_offsets(self, atol=None):
        """
        Verify every slab is a rigid translate of slabList[0] and return the
        translation vectors.  A stiff_mat_const run is only legitimate if this
        holds, so it is checked rather than assumed.
        """
        slabs = [np.asarray(s, dtype=float) for s in self.slabList]
        ref = slabs[0]
        ext0 = ref.max(axis=0) - ref.min(axis=0)
        scale = max(1.0, float(np.max(np.abs(ext0))), float(np.max(np.abs(ref))))
        tol = _COORD_RTOL * scale if atol is None else atol

        offsets = []
        for i, g in enumerate(slabs):
            if g.shape != ref.shape:
                raise ValueError(
                    "stiff_mat_const: slab %d has geometry shape %r, slab 0 has %r"
                    % (i, g.shape, ref.shape))
            d = g.min(axis=0) - ref.min(axis=0)
            if not np.allclose(g, ref + d, rtol=0.0, atol=tol):
                raise ValueError(
                    "stiff_mat_const: slab %d is not a rigid translate of slab 0 "
                    "(max deviation %.3e, tol %.3e).  The local stiffness "
                    "matrices will not coincide."
                    % (i, float(np.max(np.abs(g - (ref + d)))), tol))
            offsets.append(d)
        return offsets

    def _slab_solver(self, slabInd, dbg=0):
        """
        Return (solver, XXb, XXi, tDisc) for slab `slabInd`.

        stiff_mat_const=False : discretize and factorize this slab (as before).
        stiff_mat_const=True  : build slab 0 once and hand back the reference
                                solver with rigidly shifted coordinates, so
                                gb / bc are still evaluated at true *global*
                                positions.
        """
        if not self.stiff_mat_const:
            geom = np.array(self.slabList[slabInd])
            t0 = time.time()
            solver = solverWrap.solverWrapper(self.opts)
            solver.construct(geom, self.pdo, verbose=dbg)
            self._n_factorizations += 1
            return solver, solver.XXb, solver.XXi, time.time() - t0

        tDisc = 0.0
        if self._ref_solver is None:
            self._offsets = self._compute_offsets()
            geom = np.array(self.slabList[0])
            t0 = time.time()
            solver = solverWrap.solverWrapper(self.opts)
            solver.construct(geom, self.pdo, verbose=dbg)
            tDisc = time.time() - t0
            self._ref_solver = solver
            self.localSolver = solver
            self._n_factorizations += 1
            if dbg > 0:
                print("stiff_mat_const: single reference factorization built "
                      "in %5.2f s, reused for all %d slabs"
                      % (tDisc, len(self.slabList)))

        ref = self._ref_solver
        d = self._offsets[slabInd]
        # OPT: shift the cached boundary/interior point sets rather than
        #      re-slicing the full XX array for every slab.
        if np.any(d):
            XXb = ref.XXb + d
            XXi = ref.XXi + d
        else:
            XXb, XXi = ref.XXb, ref.XXi
        return ref, XXb, XXi, tDisc

    def _slab_indices(self, slabInd, solver, XXb, XXi):
        """
        Per-slab index sets, shared by both front-ends.  Returns
        (Il, Ir, Ic, Igb, XXi, XXb, pts_l, pts_r).

        For problem_type='mixed', Il/Ir index into solver.JD, so the physical
        source coordinates are XXb[JD[I]] -- see the FIX in compute_stmaps.
        """
        geom = np.array(self.slabList[slabInd])
        slab_i = slab(geom, self.gb)
        Il, Ir, Ic, Igb, XXi, XXb = slab_i.compute_idxs_and_pts(
            solver, XXb=XXb, XXi=XXi)

        pts_l = pts_r = None
        if getattr(solver.opts, "problem_type", "Dirichlet") == "mixed":
            xl = float(geom[0][0])
            xr = float(geom[1][0])
            scale = max(abs(xl), abs(xr), abs(xr - xl))
            XD = XXb[solver.JD, 0]
            Il = np.flatnonzero(_coord_close(XD, xl, scale))
            Ir = np.flatnonzero(_coord_close(XD, xr, scale))
            pts_l = XXb[solver.JD[Il], :]
            pts_r = XXb[solver.JD[Ir], :]

        return Il, Ir, Ic, Igb, XXi, XXb, pts_l, pts_r

    @staticmethod
    def _block_key(I, J, side):
        """
        Cache key for a source-target block.  The operator depends only on the
        (target, source) index pair, so identical pairs across slabs give
        bit-identical blocks when the stiffness matrices coincide.  `side` is
        carried along so that an empty left face and an empty right face --
        which have identical (empty) index arrays -- cannot collide.
        """
        return (side, np.asarray(I).tobytes(), np.asarray(J).tobytes())


    # ------------------------------------------------------------------ #

    def compute_global_dofs(self):
        """
        Bookkeeping: how local double-slab dofs relate to the 'global' dofs of
        the reduced S-system.
        """
        # FIX: `glob_source_dofs` used to be bound only inside the guard while
        #      the assignment to self sat outside it -> UnboundLocalError on a
        #      second call, and a silent wipe to [] if targets were empty.
        if self.glob_source_dofs:
            return self.glob_source_dofs
        if not self.glob_target_dofs:
            raise RuntimeError(
                "compute_global_dofs() called before glob_target_dofs was built"
            )

        glob_source_dofs = []
        for slabInd in range(len(self.connectivity)):
            IFLeft = self.connectivity[slabInd][0]
            IFRight = self.connectivity[slabInd][1]
            if IFLeft < 0:
                glob_source_dofs.append([self.glob_target_dofs[IFRight]])
            elif IFRight < 0:
                glob_source_dofs.append([self.glob_target_dofs[IFLeft]])
            else:
                glob_source_dofs.append(
                    [self.glob_target_dofs[IFLeft], self.glob_target_dofs[IFRight]]
                )
        self.glob_source_dofs = glob_source_dofs
        return glob_source_dofs

    # ------------------------------------------------------------------ #

    def _make_st_linop(self, I, J, Bcols, A_solver, nrows_full):
        """
        Build one source-target LinearOperator:  v |-> (A^{-1} B[:,J] v)[I].

        OPT: B[:,J] used to be re-sliced on *every* apply.  Sparse column
             slicing is not free and a randomized assembler applies this
             hundreds of times; it is hoisted here, together with its
             transpose.
        """
        BJ = Bcols[:, J]
        if sp.issparse(BJ):
            BJ = BJ.tocsr()
            BJT = BJ.T.tocsr()
        else:
            BJ = np.asarray(BJ)
            BJT = BJ.T
        dt = np.result_type(_dtype_of(A_solver), _dtype_of(BJ))

        def smatmat(v, transpose=False):
            v_in = np.asarray(v)
            oneD = v_in.ndim == 1
            v_tmp = v_in[:, np.newaxis] if oneD else v_in

            if not transpose:
                result = (A_solver @ (BJ @ v_tmp))[I, :]
            else:
                # FIX: the original sized this with v.shape[1] instead of
                #      v_tmp.shape[1], so every 1-D rmatvec raised IndexError.
                # FIX: dtype was hard-wired to float64, silently killing
                #      complex problems.
                tmp = np.zeros(
                    (nrows_full, v_tmp.shape[1]),
                    dtype=np.result_type(dt, v_tmp.dtype),
                )
                tmp[I, :] = v_tmp
                result = BJT @ (A_solver.T @ tmp)

            return result.ravel() if oneD else result

        # OPT: passing dtype explicitly stops SciPy from probing the operator
        #      with matvec(zeros(N)) just to infer it -- that probe is a full
        #      local solve, four of them per slab.
        return LinearOperator(
            shape=(len(I), len(J)),
            dtype=dt,
            matvec=lambda v: smatmat(v),
            rmatvec=lambda v: smatmat(v, transpose=True),
            matmat=lambda v: smatmat(v),
            rmatmat=lambda v: smatmat(v, transpose=True),
        )

    # ------------------------------------------------------------------ #

    def construct_Stot_and_rhstot_linearOperator(
        self, S_list, rhs_list, Ntot, nc=None, dbg=0
    ):
        """
        Construct the global S operator (as I + S) and the global rhs.

        EXPLAINER OF CONVENTIONS:
            - global dof ordering is inferred from the supplied connectivity
            - joined slabs are contiguous (fictitious domain extension used for
              periodic domains)
            - no domain checks are done (garbage in, garbage out)
            - contiguous blocks are used for global dofs, to improve efficiency
            - first INTERFACES (i.e. 'Ic') is assumed to be global dofs
              0...len(Ic)-1

        `nc` is retained for signature compatibility and is no longer used.
        """
        tgt = [_as_index(b) for b in self.glob_target_dofs]
        src = [[_as_index(b) for b in row] for row in self.glob_source_dofs]

        # FIX: dtype is now inferred rather than forced to float64, so complex
        #      (Helmholtz / time-harmonic) problems survive.
        dts = [_dtype_of(np.asarray(r)) for r in rhs_list]
        dts += [_dtype_of(blk) for row in S_list for blk in row]
        dtype = np.result_type(*dts)

        rhstot = np.zeros(Ntot, dtype=dtype)
        # FIX: was rhstot[i*nc:(i+1)*nc] with a single global nc, which
        #      contradicts the per-slab glob_target_dofs built alongside it.
        for i, rhs in enumerate(rhs_list):
            rhstot[tgt[i]] = rhs

        def smatmat(v, transpose=False):
            v_in = np.asarray(v)
            oneD = v_in.ndim == 1
            v_tmp = v_in[:, np.newaxis] if oneD else v_in

            # OPT: one copy (the identity term) instead of astype-then-copy.
            result = v_tmp.astype(np.result_type(v_tmp.dtype, dtype), copy=True)

            if not transpose:
                for i in range(len(tgt)):
                    ti = tgt[i]
                    for j, sj in enumerate(src[i]):
                        result[ti] += S_list[i][j] @ v_tmp[sj]
            else:
                for i in range(len(tgt)):
                    ti = tgt[i]
                    for j, sj in enumerate(src[i]):
                        result[sj] += S_list[i][j].T @ v_tmp[ti]

            return result.ravel() if oneD else result

        Linop = LinearOperator(
            shape=(Ntot, Ntot),
            dtype=dtype,
            matvec=smatmat,
            rmatvec=lambda v: smatmat(v, transpose=True),
            matmat=smatmat,
            rmatmat=lambda v: smatmat(v, transpose=True),
        )
        return Linop, rhstot

    # ------------------------------------------------------------------ #

    def _summary(self, discrTime, sampleTime, compressTime, glob_target_dofs,
                 relerrl=None, relerrr=None, compression=None):
        # FIX: averages were divided by len(connectivity)-1, which is both an
        #      off-by-one and a ZeroDivisionError for a single-slab run.
        nslabs = max(len(self.slabList), 1)
        # Under stiff_mat_const the work is not per-slab, so averaging over
        # slabs would understate it by a factor of nslabs.  Report totals.
        nfac = max(self._n_factorizations, 1)
        nasm = max(self._n_assembled, 1)
        print("============================OMS SUMMARY============================")
        if self.stiff_mat_const:
            print("stiff_mat_const              =  True")
            print("factorizations               = ", self._n_factorizations,
                  " (of", nslabs, "slabs)")
            print("blocks assembled / reused    = ", self._n_assembled, "/",
                  self._n_reused)
            print("total discr. time            = ", discrTime)
            print("total sample time            = ", sampleTime)
            print("total compr. time            = ", compressTime)
        print("avg. discr. time             = ", discrTime / nfac)
        print("avg. sample time             = ", sampleTime / nasm)
        print("avg. compr. time             = ", compressTime / nasm)
        if compression is not None:
            print("compression rate             = ", compression)
        print("total dofs                   = ",
              sum(len(dof) for dof in glob_target_dofs))
        if relerrl is not None:
            print("estim. max. err. ( l // r )  = (", relerrl, " // ", relerrr, ")")
        print("===================================================================")


# --------------------------------------------------------------------------- #
# oms : compressed interface operators
# --------------------------------------------------------------------------- #


class oms(_omsBase):
    """
    overlapping multislab class
    @param
    slablist: list of double-wide slabs
    pdo:    global partial differential operator
    solver_opts: solver options (h and p specs)
    connectivity: encodes which double slabs are connected to which
    """

    def compute_stmaps(self, Il, Ic, Ir, XXi, XXb, solver):
        A_solver = solver.solver_ii
        nrows = len(solver.Ii)

        Linop_r = self._make_st_linop(Ic, Ir, solver.Aib, A_solver, nrows)
        Linop_l = self._make_st_linop(Ic, Il, solver.Aib, A_solver, nrows)

        st_r = stMap(Linop_r, XXb[Ir, :], XXi[Ic, :],
                     A_solver.shape[0], A_solver.shape[1])
        st_l = stMap(Linop_l, XXb[Il, :], XXi[Ic, :],
                     A_solver.shape[0], A_solver.shape[1])
        return st_l, st_r

    # ------------------------------------------------------------------ #

    def _assemble_cached(self, st, I, J, assembler, slabInd, label, dbg):
        """
        Compress one source-target block, reusing an identical block when
        stiff_mat_const is set.

        Under stiff_mat_const the whole (source points, target points)
        configuration is rigidly translated from slab to slab, so the
        compressed block is numerically identical and only the *distinct*
        (Ic, J) index pairs need assembling -- two for an entire decomposition
        instead of two per slab.

        Returns (block, was_freshly_assembled, compression_ratio).
        """
        key = self._block_key(I, J, label) if self.stiff_mat_const else None
        if key is not None and key in self._block_cache:
            self._n_reused += 1
            mat, ratio = self._block_cache[key]
            if dbg > 1:
                print("%s SLAB %d: reusing cached block" % (label, slabInd))
            return mat, False, ratio

        mat = assembler.assemble(st, dbg=dbg)
        self._n_assembled += 1
        nb = assembler.stats.nbytes
        self.nbytes += nb

        dense = int(np.prod(st.A.shape)) * 8
        self.densebytes += dense
        ratio = nb / dense if dense else float("nan")

        if dbg > 0:
            # FIX: the two branches used to carry each other's LEFT/RIGHT label.
            print("%s SLAB %d compression time %5.2f s, sample time %5.2f s"
                  % (label, slabInd, assembler.stats.timeCompress,
                     assembler.stats.timeSample))
        if key is not None:
            self._block_cache[key] = (mat, ratio)
        return mat, True, ratio

    # ------------------------------------------------------------------ #

    def construct_Stot_helper(self, bc, assembler, dbg=0):
        """
        Construct S_rk_list and the other helpers needed for the S operator,
        whether it is applied iteratively or factorized directly.
        """
        connectivity = self.connectivity
        slabs = self.slabList

        Ntot = 0
        S_rk_list = []
        rhs_list = []
        glob_target_dofs = []
        startCentral = 0

        discrTime = 0.0
        compressTime = 0.0
        sampleTime = 0.0
        shapeMatch = True
        relerrl = 0.0
        relerrr = 0.0

        for slabInd in range(len(slabs)):
            solver, XXb, XXi, tDisc = self._slab_solver(slabInd, dbg=dbg)
            discrTime += tDisc
            if dbg > 1 and tDisc > 0:
                print("SLAB %2.0d discretization time = %5.2f s" % (slabInd, tDisc))

            Il, Ir, Ic, Igb, XXi, XXb, _, _ = self._slab_indices(
                slabInd, solver, XXb, XXi)
            nc = len(Ic)
            if dbg > 1:
                print("SLAB %2.0d size = %2.0d" % (slabInd, nc))
            self.nc = nc
            self.ncs.append(nc)
            Ntot += nc
            glob_target_dofs.append(range(startCentral, startCentral + nc))
            startCentral += nc

            fgb = bc(XXb[Igb, :])
            st_l, st_r = self.compute_stmaps(Il, Ic, Ir, XXi, XXb, solver)

            rhs = solver.solver_ii @ (solver.Aib[:, Igb] @ fgb)
            rhs = -rhs[Ic]
            rhs_list.append(rhs)

            bool_l = len(Il) > 0
            bool_r = len(Ir) > 0

            # FIX: the geometric emptiness of Il/Ir and the topological
            #      connectivity flags must agree, otherwise the block stored
            #      below is silently the *previous* slab's matrix (rkMat_l /
            #      rkMat_r were never cleared between iterations).
            if bool_l != (connectivity[slabInd][0] >= 0):
                raise ValueError(
                    "slab %d: left connectivity says %s but %d left-face dofs "
                    "were found" % (slabInd, connectivity[slabInd][0], len(Il))
                )
            if bool_r != (connectivity[slabInd][1] >= 0):
                raise ValueError(
                    "slab %d: right connectivity says %s but %d right-face dofs "
                    "were found" % (slabInd, connectivity[slabInd][1], len(Ir))
                )

            rkMat_l = None
            rkMat_r = None
            compression_l = None
            compression_r = None
            fresh_l = False
            fresh_r = False

            # FIX: dividing by np.prod(shape)*8 blew up (ZeroDivisionError,
            #      then nan in stats.compression) whenever a face was empty.
            #      Byte accounting now lives in _assemble_cached and counts
            #      each distinct block once, so stats.compression stays a true
            #      per-block ratio under reuse.
            if bool_r:
                rkMat_r, fresh_r, compression_r = self._assemble_cached(
                    st_r, Ic, Ir, assembler, slabInd, "RIGHT", dbg)
                if fresh_r:
                    compressTime += assembler.stats.timeCompress
                    sampleTime += assembler.stats.timeSample
            if bool_l:
                rkMat_l, fresh_l, compression_l = self._assemble_cached(
                    st_l, Ic, Il, assembler, slabInd, "LEFT", dbg)
                if fresh_l:
                    compressTime += assembler.stats.timeCompress
                    sampleTime += assembler.stats.timeSample

            # FIX: shapeMatch was printed but its computation was commented
            #      out, so it always reported True.
            if bool_l and rkMat_l is not None:
                shapeMatch = shapeMatch and (tuple(rkMat_l.shape) == tuple(st_l.A.shape))
            if bool_r and rkMat_r is not None:
                shapeMatch = shapeMatch and (tuple(rkMat_r.shape) == tuple(st_r.A.shape))

            if dbg > 0:
                errl = errr = None
                if bool_l:
                    Vl = np.random.standard_normal(
                        size=(st_l.A.shape[1], assembler.matOpts.maxRank))
                    Ul = st_l.A @ Vl
                    errl = np.linalg.norm(Ul - rkMat_l @ Vl) / np.linalg.norm(Ul)
                    relerrl = max(relerrl, errl)
                    print("LEFT ERR = ", errl)
                if bool_r:
                    Vr = np.random.standard_normal(
                        size=(st_r.A.shape[1], assembler.matOpts.maxRank))
                    Ur = st_r.A @ Vr
                    errr = np.linalg.norm(Ur - rkMat_r @ Vr) / np.linalg.norm(Ur)
                    relerrr = max(relerrr, errr)
                    print("RIGHT ERR = ", errr)

                if dbg > 1:
                    # FIX: the per-slab report used to print the running maxima
                    #      relerrl/relerrr rather than this slab's own errors.
                    if bool_l and bool_r:
                        print("SLAB %d error = %5.2e // %5.2e" % (slabInd, errl, errr))
                        print("SLAB %d compression = %5.3e // %5.3e\n"
                              % (slabInd, compression_l, compression_r))
                    elif bool_r:
                        print("SLAB %d error = %5.2e" % (slabInd, errr))
                        print("SLAB %d compression = %5.3e\n" % (slabInd, compression_r))
                    elif bool_l:
                        print("SLAB %d error = %5.2e" % (slabInd, errl))
                        print("SLAB %d compression = %5.3e\n" % (slabInd, compression_l))

            if connectivity[slabInd][0] < 0:
                S_rk_list.append([rkMat_r])
            elif connectivity[slabInd][1] < 0:
                S_rk_list.append([rkMat_l])
            else:
                S_rk_list.append([rkMat_l, rkMat_r])

            del st_l, st_r, Il, Ir, Ic, Igb, XXi, XXb
            if not self.stiff_mat_const:
                del solver

            if dbg > 0:
                print("overlapping slab ", slabInd + 1, " of ", len(slabs), " done")

        compression = (self.nbytes / self.densebytes) if self.densebytes else float("nan")
        if dbg > 0:
            self._summary(discrTime, sampleTime, compressTime, glob_target_dofs,
                          relerrl, relerrr, compression)
            print("shapes match?                = ", shapeMatch)

        nfac = max(self._n_factorizations, 1)
        nasm = max(self._n_assembled, 1)
        self.stats.compression = compression
        self.stats.sampl_timing = sampleTime / nasm
        self.stats.compr_timing = compressTime / nasm
        self.stats.discr_timing = discrTime / nfac
        self.stats.n_factorizations = self._n_factorizations
        self.stats.n_assembled = self._n_assembled
        self.stats.n_reused = self._n_reused
        self.glob_target_dofs = glob_target_dofs
        self.compute_global_dofs()

        return S_rk_list, rhs_list, Ntot, self.nc

    # ------------------------------------------------------------------ #

    def construct_Stot_and_rhstot(self, bc, assembler, dbg=0):
        S_rk_list, rhs_list, Ntot, nc = self.construct_Stot_helper(bc, assembler, dbg)
        return self.construct_Stot_and_rhstot_linearOperator(
            S_rk_list, rhs_list, Ntot, nc, dbg)

    def construct_rhstot(self, bc):
        """
        TODO IMPLEMENT RHS -- `oms` does not cache local solvers, so the rhs
        cannot be rebuilt without re-discretizing.  See oms_lu.construct_rhstot.
        """
        raise NotImplementedError(
            "oms does not cache local solvers; use oms_lu.construct_rhstot"
        )


# --------------------------------------------------------------------------- #
# oms_lu : un-compressed interface operators
# --------------------------------------------------------------------------- #


class oms_lu(_omsBase):
    """
    overlapping multislab class, no interface reduction
    @param
    slablist: list of double-wide slabs
    pdo:    global partial differential operator
    solver_opts: solver options (h and p specs)
    connectivity: encodes which double slabs are connected to which
    """

    def __init__(self, slabList: list, pdo, gb, solver_opts, connectivity,
                 stiff_mat_const=False):
        super().__init__(slabList, pdo, gb, solver_opts, connectivity,
                         stiff_mat_const=stiff_mat_const)
        self.solvers = []
        self.idx = []

    # ------------------------------------------------------------------ #

    def compute_stmaps(self, Il, Ic, Ir, XXi, XXb, solver, pts_l=None, pts_r=None):
        A_solver = solver.solver_ii
        ptype = solver.opts.problem_type

        if ptype == "Dirichlet":
            Bcols = solver.Aib
            nrows = len(solver.Ii)
        elif ptype == "mixed":
            Bcols = solver.E
            nrows = len(solver.Ii) + len(solver.JN)
        else:
            raise NameError(
                "solver problem type not recognized: must be 'Dirichlet' or 'mixed'."
            )

        Linop_r = self._make_st_linop(Ic, Ir, Bcols, A_solver, nrows)
        Linop_l = self._make_st_linop(Ic, Il, Bcols, A_solver, nrows)

        # FIX: for problem_type='mixed', Il/Ir index *into solver.JD*, not into
        #      XXb.  XXb[Ir,:] therefore handed the assembler the coordinates of
        #      the wrong points.  Callers now pass the correct source points.
        if pts_l is None:
            pts_l = XXb[Il, :]
        if pts_r is None:
            pts_r = XXb[Ir, :]

        st_r = stMap(Linop_r, pts_r, XXi[Ic, :], A_solver.shape[0], A_solver.shape[1])
        st_l = stMap(Linop_l, pts_l, XXi[Ic, :], A_solver.shape[0], A_solver.shape[1])
        return st_l, st_r

    # ------------------------------------------------------------------ #

    def _local_rhs(self, solver, bc, Ic, Igb, XXb, reduced_load, slabInd):
        ptype = solver.opts.problem_type
        if ptype == "Dirichlet":
            fgb = bc(XXb[Igb, :])
            return -(solver.solver_ii @ (solver.Aib[:, Igb] @ fgb))[Ic]
        if ptype == "mixed":
            # composition returns (b_C on C-space, b_X on X-space)
            b_C, b_X = _eval_reduced_load(reduced_load, solver, slabInd)
            b_N = b_X[solver.JN]                  # X-space load on Neumann rows
            fgb = bc(XXb[solver.JN, :])           # g_N
            rhs = solver.solver_ii @ np.concatenate([b_C, fgb + b_N])
            return rhs[Ic]
        raise NameError(
            "solver problem type not recognized, must be 'Dirichlet' or 'mixed'"
        )

    # ------------------------------------------------------------------ #

    def construct_Stot_helper(self, bc, reduced_load=None, dbg=0):
        """
        Construct S_lu_list and the other helpers needed for the S operator.
        """
        connectivity = self.connectivity
        slabs = self.slabList

        Ntot = 0
        S_lu_list = []
        rhs_list = []
        glob_target_dofs = []
        startCentral = 0

        discrTime = 0.0
        compressTime = 0.0
        sampleTime = 0.0

        for slabInd in range(len(slabs)):
            solver, XXb, XXi, tDisc = self._slab_solver(slabInd, dbg=dbg)
            discrTime += tDisc
            self.solvers.append(solver)   # same object for every slab when
                                          # stiff_mat_const is set
            if dbg > 1 and tDisc > 0:
                print("SLAB %2.0d discretization time = %5.2f s"
                      % (slabInd, tDisc))

            Il, Ir, Ic, Igb, XXi, XXb, pts_l, pts_r = self._slab_indices(
                slabInd, solver, XXb, XXi)

            self.idx.append((Il, Ir, Ic, Igb, XXi, XXb))

            nc = len(Ic)
            if dbg > 1:
                print("SLAB %2.0d size = %2.0d" % (slabInd, nc))
            self.nc = nc
            self.ncs.append(nc)
            Ntot += nc
            glob_target_dofs.append(range(startCentral, startCentral + nc))
            startCentral += nc

            # Under stiff_mat_const identical (Ic, J) pairs give the identical
            # operator, so build each distinct one once.
            key_l = self._block_key(Ic, Il, 'l') if self.stiff_mat_const else None
            key_r = self._block_key(Ic, Ir, 'r') if self.stiff_mat_const else None
            if (key_l is not None and key_l in self._block_cache
                    and key_r in self._block_cache):
                A_l = self._block_cache[key_l][0]
                A_r = self._block_cache[key_r][0]
                self._n_reused += 1
            else:
                st_l, st_r = self.compute_stmaps(Il, Ic, Ir, XXi, XXb, solver,
                                                 pts_l=pts_l, pts_r=pts_r)
                A_l, A_r = st_l.A, st_r.A
                self._n_assembled += 1
                if key_l is not None:
                    self._block_cache[key_l] = (A_l, float("nan"))
                    self._block_cache[key_r] = (A_r, float("nan"))

            rhs_list.append(
                self._local_rhs(solver, bc, Ic, Igb, XXb, reduced_load, slabInd))

            if connectivity[slabInd][0] < 0:
                S_lu_list.append([A_r])
            elif connectivity[slabInd][1] < 0:
                S_lu_list.append([A_l])
            else:
                S_lu_list.append([A_l, A_r])

            if dbg > 0:
                print("overlapping slab ", slabInd + 1, " of ", len(slabs), " done")

        if dbg > 0:
            self._summary(discrTime, sampleTime, compressTime, glob_target_dofs)

        self.stats.discr_timing = discrTime / max(self._n_factorizations, 1)
        self.stats.n_factorizations = self._n_factorizations
        self.stats.n_assembled = self._n_assembled
        self.stats.n_reused = self._n_reused
        self.glob_target_dofs = glob_target_dofs
        self.compute_global_dofs()

        return S_lu_list, rhs_list, Ntot, self.ncs

    # ------------------------------------------------------------------ #

    def construct_Stot_and_rhstot(self, bc, reduced_load=None, dbg=0):
        # FIX: the original signature was (self, bc, assembler, dbg=0) and the
        #      body called construct_Stot_helper(bc, dbg) -- so dbg landed in
        #      the reduced_load slot and `assembler` was never used at all.
        S_lu_list, rhs_list, Ntot, nc = self.construct_Stot_helper(
            bc, reduced_load, dbg)
        return self.construct_Stot_and_rhstot_linearOperator(
            S_lu_list, rhs_list, Ntot, nc, dbg)

    # ------------------------------------------------------------------ #

    def construct_rhstot(self, bc, reduced_load=None, dbg=0):
        """
        Rebuild the global rhs only, reusing the cached local solvers.
        """
        if not self.solvers:
            raise RuntimeError(
                "construct_rhstot() requires cached solvers; call "
                "construct_Stot_helper() first"
            )
        if not self.glob_target_dofs:
            raise RuntimeError("construct_rhstot() called before glob_target_dofs")

        rhs_list = []
        for slabInd in range(len(self.slabList)):
            solver = self.solvers[slabInd]
            Il, Ir, Ic, Igb, XXi, XXb = self.idx[slabInd]
            rhs_list.append(
                self._local_rhs(solver, bc, Ic, Igb, XXb, reduced_load, slabInd))

        Ntot = sum(len(b) for b in self.glob_target_dofs)
        dtype = np.result_type(*[np.asarray(r).dtype for r in rhs_list])
        rhstot = np.zeros(Ntot, dtype=dtype)
        # FIX: was a uniform i*nc slicing, inconsistent with glob_target_dofs.
        for i, rhs in enumerate(rhs_list):
            rhstot[_as_index(self.glob_target_dofs[i])] = rhs
        return rhstot

    # ------------------------------------------------------------------ #

    def uX_full(self, uhat, i, b_C, b_X):
        solver = self.solvers[i]          # cached wrapper for slab i
        if solver.opts.problem_type != "mixed":
            raise NameError("uX_full is only defined for problem_type='mixed'")

        Il, Ir, Ic, _, _, _ = self.idx[i]

        # ---- exterior trace, length nX, in I_Xtot ordering ----
        uX = np.zeros(len(solver.Ib), dtype=np.result_type(np.asarray(uhat).dtype,
                                                           np.asarray(b_C).dtype))

        # Source 1: artificial faces <- neighbour central traces
        iL, iR = self.connectivity[i][0], self.connectivity[i][1]
        # FIX: was `!= -1`; every other site in the file uses `< 0`.
        # FIX: was uhat[iL*nc:(iL+1)*nc] with the local nc, which assumed all
        #      slabs share an interface size.
        if iL >= 0:
            uX[solver.JD[Il]] = uhat[_as_index(self.glob_target_dofs[iL])]
        if iR >= 0:
            uX[solver.JD[Ir]] = uhat[_as_index(self.glob_target_dofs[iR])]

        # Source 2: physical walls <- solved Neumann values from the local solve
        b_N = np.asarray(b_X)[solver.JN]
        g_N = np.zeros(len(solver.JN))                # homogeneous Neumann data
        u_D = uX[solver.JD]                           # artificial-face data
        rhs = np.concatenate([b_C, g_N + b_N]) - solver.E @ u_D
        w = solver.solver_ii @ rhs                    # = M^{-1} rhs
        uX[solver.JN] = w[len(b_C):]                  # lower block = u_N

        return uX