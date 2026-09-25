"""
Overlapping multi-slab (OMS) decomposition.

One front-end class, `oms`, configured by two flags:

    constructHBS  keepLU   operators available   default
    ------------  ------   -------------------   -------
    True          False    HBS                   HBS
    True          True     HBS, LU               LU
    False         True     LU                    LU
    False         False    -- rejected: nothing would be kept to solve with --

Operator selection (flag rhsHBS on construct_Stot_and_rhstot):
    LU  : exact fused local-solve operator
    HBS : compressed blocks K_{i,i+-1}
The rhs is the exact local-solve rhs  -(A^{-1} A_ib g)[Ic]  in both cases,
computed during construction.  Re-evaluating it for new boundary data
(construct_rhstot) needs keepLU; keepLU also enables uX_full.

Changes relative to the original are tagged  # FIX:  (correctness) and  # OPT:
(performance / hygiene).

Fused LU operator: the left and right source faces of a slab share one local
solve,  (A^{-1} (B_l v_l + B_r v_r))[Ic],  instead of two separate solves.

stiff_mat_const: all maps of the reduced system are derived from ONE interior
double-slab set-up (both source faces present), rather than from slab 0 (right
map) plus slab 1 (left map).  Boundary slabs reuse the reference maps.

Off-loading (default: on whenever stiff_mat_const is False): if a slab's HBS
blocks and/or LU factors live on a CUDA device, they are copied to host
memory as soon as they are built and moved back one slab at a time when the
operator is applied, so device memory is O(one slab) instead of O(all
slabs).  torch and cupy are handled automatically; other device types can be
registered with register_device_mover().  After the first slab, the final
host footprint is estimated and checked against the available host memory
(MemoryError if it does not fit).
"""

from __future__ import annotations

import copy
import time
import types
import warnings
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, splu

import solver.solver as solverWrap
from solver.solver import stMap

# OPT: dropped unused imports (sys, matplotlib.pyplot, scipy.sparse.linalg as
#      splinalg) and the duplicated numpy / solver imports.

__all__ = ["slab", "omsStats", "oms", "register_device_mover",
           "register_host_sizer"]

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
        self.n_offloaded = 0           # device objects moved to host
        self.host_bytes = 0            # host bytes held by kept operators
        self.host_bytes_estimate = None  # projected after the first slab



# --------------------------------------------------------------------------- #
# off-loading: compute device (CUDA) -> host memory
# --------------------------------------------------------------------------- #
#
# Every heavy per-slab object (HBS blocks, local LU factors) is held through a
# *handle* with one method, get().
#   _Resident : object stays where it is.
#   _OnHost   : a copy of the object in which every device array has been
#               replaced by a host copy (_HostLeaf).  get() rebuilds the object
#               with fresh device arrays, so a caller that drops its reference
#               after use keeps at most one slab on the device.
# Objects are walked through containers, instance __dict__s, bound methods and
# function closures (a LinearOperator built from lambdas that capture device
# tensors is handled).  Objects without device data are left untouched.

_DEVICE_MOVERS = {}   # cls -> fn(obj, pin_memory) -> (host, restore) | None
_HOST_SIZERS = {}     # cls -> fn(obj) -> host bytes held by obj
_MAX_DEPTH = 10
_NOCHANGE = object()


def register_device_mover(cls, to_host):
    """
    Teach oms to off-load objects of type `cls` (and subclasses).

    to_host(obj, pin_memory) must return None if `obj` is not on a compute
    device, else (host_value, restore) with restore(host_value) giving back
    an equivalent device object.  torch tensors, cupy arrays and cupyx sparse
    matrices are registered automatically when those packages are importable.
    """
    _DEVICE_MOVERS[cls] = to_host


def register_host_sizer(cls, nbytes):
    """Teach the host-memory estimate the footprint of an opaque type."""
    _HOST_SIZERS[cls] = nbytes


def _lookup(table, obj):
    for c in type(obj).__mro__:
        f = table.get(c)
        if f is not None:
            return f
    return None


# ---- defaults ------------------------------------------------------------- #

def _superlu_nbytes(lu):
    # nnz of L+U with values + row indices, plus two permutations.  SuperLU
    # does not expose its dtype cheaply; assume complex128 as an upper bound
    # would be too pessimistic, so assume float64.
    return int(lu.nnz) * (8 + 4) + 2 * int(lu.shape[0]) * 4


register_host_sizer(type(splu(sp.eye(1, format="csc"))), _superlu_nbytes)

try:  # pragma: no cover - depends on the environment
    import torch as _torch

    def _torch_to_host(t, pin_memory):
        if t.device.type != "cuda":
            return None
        dev = t.device
        h = t.detach().to("cpu")
        if pin_memory:
            h = h.pin_memory()
        return h, (lambda x, dev=dev: x.to(dev, non_blocking=True))

    def _torch_nbytes(t):
        if t.device.type != "cpu":
            return 0
        if t.layout == _torch.strided:
            return t.untyped_storage().nbytes()
        if t.layout == _torch.sparse_coo:
            t = t.coalesce()
            return sum(_torch_nbytes(x) for x in (t.indices(), t.values()))
        parts = [getattr(t, a)() for a in
                 ("crow_indices", "col_indices", "ccol_indices", "row_indices", "values")
                 if hasattr(t, a)]
        total = 0
        for p in parts:
            try:
                total += _torch_nbytes(p)
            except RuntimeError:
                pass
        return total

    register_device_mover(_torch.Tensor, _torch_to_host)
    register_host_sizer(_torch.Tensor, _torch_nbytes)
except ImportError:
    pass

try:  # pragma: no cover - depends on the environment
    import cupy as _cupy

    def _cupy_to_host(a, pin_memory):
        dev = a.device.id

        def restore(x, dev=dev):
            with _cupy.cuda.Device(dev):
                return _cupy.asarray(x)
        return _cupy.asnumpy(a), restore

    register_device_mover(_cupy.ndarray, _cupy_to_host)
    register_host_sizer(_cupy.ndarray, lambda a: 0)
    try:
        import cupyx.scipy.sparse as _cusp

        def _cusp_to_host(a, pin_memory):
            dev, cls = a.device.id if hasattr(a, "device") else None, type(a)

            def restore(x, dev=dev, cls=cls):
                if dev is None:
                    return cls(x)
                with _cupy.cuda.Device(dev):
                    return cls(x)
            return a.get(), restore

        register_device_mover(_cusp.spmatrix, _cusp_to_host)
        register_host_sizer(_cusp.spmatrix, lambda a: 0)
    except ImportError:
        pass
except ImportError:
    pass


# ---- tree walking --------------------------------------------------------- #

class _HostLeaf:
    """Host copy of one device object, plus how to put it back."""
    __slots__ = ("host", "restore")

    def __init__(self, host, restore):
        self.host = host
        self.restore = restore


_OPAQUE = (str, bytes, int, float, complex, bool, type(None), range, slice,
           np.ndarray, np.generic, np.dtype, np.ufunc, type, types.ModuleType,
           types.BuiltinFunctionType, types.CodeType)


def _map_tree(obj, leaf_fn, memo, depth=0):
    """
    Return `obj` with every leaf for which leaf_fn(leaf) is not _NOCHANGE
    replaced.  Sub-trees that contain no such leaf are returned as-is (same
    identity), so untouched objects are never copied.
    """
    oid = id(obj)
    if oid in memo:
        return memo[oid]
    r = leaf_fn(obj)
    if r is not _NOCHANGE:
        memo[oid] = r
        return r
    if depth > _MAX_DEPTH or isinstance(obj, _OPAQUE) or sp.issparse(obj):
        return obj
    memo[oid] = obj            # breaks cycles; overwritten if we copy
    d = depth + 1

    if isinstance(obj, (list, tuple)):
        new = [_map_tree(x, leaf_fn, memo, d) for x in obj]
        if all(a is b for a, b in zip(new, obj)):
            return obj
        out = new if isinstance(obj, list) else (
            type(obj)(*new) if hasattr(obj, "_fields") else type(obj)(new))
    elif isinstance(obj, dict):
        new = {k: _map_tree(v, leaf_fn, memo, d) for k, v in obj.items()}
        if all(new[k] is obj[k] for k in obj):
            return obj
        out = type(obj)(new) if type(obj) is not dict else new
    elif isinstance(obj, types.FunctionType):
        cells = obj.__closure__ or ()
        vals = []
        for c in cells:
            try:
                vals.append(c.cell_contents)
            except ValueError:            # empty cell
                vals.append(_NOCHANGE)
        new_vals = [v if v is _NOCHANGE else _map_tree(v, leaf_fn, memo, d)
                    for v in vals]
        defs = obj.__defaults__ or ()
        new_defs = tuple(_map_tree(x, leaf_fn, memo, d) for x in defs)
        if all(a is b for a, b in zip(new_vals, vals)) and \
                all(a is b for a, b in zip(new_defs, defs)):
            return obj
        closure = tuple(c if nv is v else types.CellType(nv)
                        for c, v, nv in zip(cells, vals, new_vals)) or None
        out = types.FunctionType(obj.__code__, obj.__globals__, obj.__name__,
                                 new_defs or None, closure)
        out.__kwdefaults__ = obj.__kwdefaults__
        out.__dict__.update(obj.__dict__)
    elif isinstance(obj, types.MethodType):
        f = _map_tree(obj.__func__, leaf_fn, memo, d)
        s = _map_tree(obj.__self__, leaf_fn, memo, d)
        if f is obj.__func__ and s is obj.__self__:
            return obj
        out = types.MethodType(f, s)
    elif hasattr(obj, "__dict__") and not callable(getattr(obj, "__get__", None)):
        attrs = vars(obj)
        new = {k: _map_tree(v, leaf_fn, memo, d) for k, v in attrs.items()}
        if all(new[k] is attrs[k] for k in attrs):
            return obj
        try:
            out = copy.copy(obj)
            out.__dict__.update(new)
        except Exception:
            return obj                    # cannot rebuild: leave on device
    else:
        return obj
    memo[oid] = out
    return out


def _to_host(obj, pin_memory=False):
    """(host copy of obj, number of device objects moved)."""
    moved = [0]

    def leaf(x):
        f = _lookup(_DEVICE_MOVERS, x)
        if f is None:
            return _NOCHANGE
        r = f(x, pin_memory)
        if r is None:
            return _NOCHANGE
        moved[0] += 1
        return _HostLeaf(*r)

    return _map_tree(obj, leaf, {}), moved[0]


def _to_device(obj):
    def leaf(x):
        return x.restore(x.host) if isinstance(x, _HostLeaf) else _NOCHANGE
    return _map_tree(obj, leaf, {})


def _host_nbytes(obj):
    """Host bytes reachable from obj (arrays counted once by their base)."""
    seen = set()
    total = [0]

    def arr_bytes(a):
        root = a
        while isinstance(root.base, np.ndarray):
            root = root.base
        if id(root) not in seen:
            seen.add(id(root))
            total[0] += root.nbytes

    def leaf(x):
        # read-only walk: always report _NOCHANGE so nothing is ever copied
        if isinstance(x, _HostLeaf):
            x = x.host
        if isinstance(x, np.ndarray):
            arr_bytes(x)
        elif sp.issparse(x):
            for a in ("data", "indices", "indptr", "row", "col", "offsets"):
                v = getattr(x, a, None)
                if isinstance(v, np.ndarray):
                    arr_bytes(v)
        else:
            f = _lookup(_HOST_SIZERS, x)
            if f is not None and id(x) not in seen:
                seen.add(id(x))
                total[0] += int(f(x))
        return _NOCHANGE

    _map_tree(obj, leaf, {})
    return total[0]


# ---- handles -------------------------------------------------------------- #

class _Resident:
    __slots__ = ("_obj",)

    def __init__(self, obj):
        self._obj = obj

    def get(self):
        return self._obj

    def host_tree(self):
        return self._obj


class _OnHost:
    __slots__ = ("_tree",)

    def __init__(self, tree):
        self._tree = tree

    def get(self):
        return _to_device(self._tree)

    def host_tree(self):
        return self._tree


# ---- host memory ---------------------------------------------------------- #

def _available_host_memory():
    """
    Bytes of host memory this process can still allocate: MemAvailable,
    tightened by a cgroup limit if one applies (SLURM / containers).
    None if it cannot be determined.
    """
    avail = None
    try:
        import psutil
        avail = int(psutil.virtual_memory().available)
    except Exception:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        avail = int(line.split()[1]) * 1024
                        break
        except OSError:
            pass
    for lim, cur in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                     ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
                      "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            with open(lim) as f:
                s = f.read().strip()
            if s == "max":
                continue
            with open(cur) as f:
                room = int(s) - int(f.read().strip())
            if room >= 0 and room < (1 << 60):
                avail = room if avail is None else min(avail, room)
            break
        except (OSError, ValueError):
            continue
    return avail


def _fmt_bytes(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return "%.1f %s" % (n, unit)
        n /= 1024.0


# attributes of the solver wrapper that the kept LU handle actually needs;
# coordinates and the full stiffness matrix are deliberately left behind.
_LU_ATTRS = {
    "Dirichlet": ("opts", "solver_ii", "Ii", "Ib", "Aib"),
    "mixed": ("opts", "solver_ii", "Ii", "Ib", "E", "JD", "JN", "solver"),
}


def _slim_solver(solver):
    ptype = getattr(solver.opts, "problem_type", "Dirichlet")
    ns = SimpleNamespace()
    for a in _LU_ATTRS.get(ptype, _LU_ATTRS["Dirichlet"]):
        if hasattr(solver, a):
            setattr(ns, a, getattr(solver, a))
    return ns


# --------------------------------------------------------------------------- #
# oms
# --------------------------------------------------------------------------- #


class oms:
    """
    Overlapping multi-slab decomposition for  (I - K) u = f,  K block
    tridiagonal with K_{i,i-1}, K_{i,i+1} the left / right maps of double
    slab i.

    @param
    slabList:      list of double-wide slabs
    pdo:           global partial differential operator
    gb:            indicator of the global (physical) boundary
    solver_opts:   solver options (h and p specs)
    connectivity:  connectivity[i] = [left neighbour, right neighbour], <0 = none
    constructHBS:  compress K_{i,i+-1} into HBS blocks (needs an assembler)
    keepLU:        keep the local factorizations (exact operator, new rhs)
    stiff_mat_const: every slab is a rigid translate with identical local
                   matrices -> one factorization, one reference set-up
    offload:       move per-slab HBS blocks / LU factors that live on a CUDA
                   device to host memory once built.  None (default) = on
                   unless stiff_mat_const.  No-op for host-resident data.
    pin_memory:    pin the host copies (faster host->device transfer when the
                   operator is applied, but page-locked memory is scarcer).
    check_host_memory: after the first slab, project the final host
                   footprint and raise MemoryError if it will not fit.
    host_mem_fraction: fraction of the available host memory the projection
                   may use (default 0.9).
    """

    def __init__(self, slabList: list, pdo, gb, solver_opts, connectivity,
                 constructHBS=True, keepLU=False, stiff_mat_const=False,
                 offload=None, pin_memory=False,
                 check_host_memory=True, host_mem_fraction=0.9):
        if not constructHBS and not keepLU:
            raise ValueError(
                "oms(constructHBS=False, keepLU=False): you have asked for every "
                "local solve to be done and then for all of it to be thrown away. "
                "With neither the HBS blocks nor the LU factors kept there is no "
                "operator left to solve with -- only a very expensive way of "
                "heating the room. Set constructHBS=True, keepLU=True, or both."
            )
        self.slabList = slabList
        self.pdo = pdo
        self.connectivity = connectivity
        self.opts = solver_opts
        self.gb = gb
        self.constructHBS = bool(constructHBS)
        self.keepLU = bool(keepLU)

        self.glob_target_dofs = []
        self.glob_source_dofs = []
        self.localSolver = None
        self.nbytes = 0
        self.densebytes = 0
        self.stats = omsStats()
        self.ncs = []          # FIX: per-slab interface sizes; self.nc alone
                               #      silently assumed every slab was the same.

        # ---- constant-stiffness-matrix reuse ----------------------------- #
        # `stiff_mat_const` asserts that every slab discretizes to *the same*
        # matrices (Aii, Aib, Abi, Abb, and for 'mixed' also M, E, JD, JN).
        # That needs constant PDE coefficients AND isomorphic discretizations
        # -- constant coefficients alone are not enough under a non-uniform
        # mesh.  Under it, one factorization and one reference set-up serve
        # the whole decomposition.
        self.stiff_mat_const = bool(stiff_mat_const)
        self._ref_solver = None
        self._ref_ind = None      # index of the reference (interior) slab
        self._ref_faces = None    # (Il, Ir, Ic) of the reference slab
        self._ref_hbs = {}        # side -> (block, ratio, err)
        self._ref_lu = {}         # 'l' / 'r' / 'lr' -> operator
        self._offsets = None
        self._block_cache = {}
        self._n_factorizations = 0
        self._n_assembled = 0
        self._n_reused = 0

        # ---- off-loading -------------------------------------------------- #
        # FIX: without stiff_mat_const every slab's HBS blocks (oms) or LU
        #      factors (oms_lu) stayed on the compute device, so device memory
        #      grew with the number of slabs until OOM.
        self.offload = (not self.stiff_mat_const) if offload is None else bool(offload)
        self.pin_memory = bool(pin_memory)
        self.check_host_memory = bool(check_host_memory)
        self.host_mem_fraction = float(host_mem_fraction)
        self._n_offloaded = 0

        # ---- products of construct_Stot_helper ---------------------------- #
        self._S_hbs = []          # per slab: handle -> [blocks] (src order)
        self._rhs_cache = None    # (bc, reduced_load, rhs_list) from construction
        self._built = False
        self._S_lu = []           # per slab: fused LinearOperator
        self._lu_handles = []     # per slab: handle -> slim local solver
        self._slab_info = []      # per slab: compact data for rhs / uX_full
        self._dtype = np.float64

    # ------------------------------------------------------------------ #
    # sizes
    # ------------------------------------------------------------------ #

    @property
    def Ntot(self):
        """Total number of interface dofs of the reduced system (None before
        construction)."""
        return sum(self.ncs) if self.ncs else None

    @property
    def nc(self):
        """
        Interface size per slab, for code that assumes it is uniform.
        Returns the last slab's size; warns if the sizes differ between slabs
        (use `ncs` for the per-slab list, or glob_target_dofs for offsets).
        None before construction.
        """
        if not self.ncs:
            return None
        if any(n != self.ncs[0] for n in self.ncs):
            sizes = sorted(set(self.ncs))
            warnings.warn(
                "oms.nc: interface sizes differ between slabs (%d distinct "
                "values, %d..%d); returning the last one (%d).  Use oms.ncs for "
                "the per-slab sizes and oms.glob_target_dofs for offsets."
                % (len(sizes), sizes[0], sizes[-1], self.ncs[-1]),
                UserWarning, stacklevel=2)
        return self.ncs[-1]

    # ------------------------------------------------------------------ #
    # lifetime
    # ------------------------------------------------------------------ #

    def close(self):
        """Release all kept operators (host and device)."""
        self._S_hbs, self._S_lu, self._lu_handles = [], [], []
        self._ref_hbs, self._ref_lu, self._block_cache = {}, {}, {}
        self._ref_solver = self.localSolver = None
        self._rhs_cache = None
        self._built = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _offload(self, obj):
        """
        Handle for `obj`.  With off-loading on, device arrays inside it are
        copied to host memory now and the device copies are released as soon
        as the caller drops `obj`; get() moves them back on demand.
        """
        if not self.offload:
            return _Resident(obj)
        host, moved = _to_host(obj, self.pin_memory)
        if moved == 0:
            return _Resident(obj)         # nothing on a device: nothing to do
        self._n_offloaded += moved
        return _OnHost(host)

    # ------------------------------------------------------------------ #
    # host-memory projection
    # ------------------------------------------------------------------ #

    def _check_host_memory(self, n_faces_first, dbg=0):
        """
        Called once the first slab is complete.  Projects the host memory
        the finished decomposition will hold (kept LUs, HBS blocks,
        bookkeeping), scaling LU / bookkeeping by the number of slabs and HBS blocks by the number of source faces, and compares
        the *remaining* growth against the available host memory.

        The projection assumes the other slabs are about as large as the
        first; for strongly non-uniform decompositions it is indicative only.
        """
        nslabs = len(self.slabList)
        total_faces = sum((c[0] >= 0) + (c[1] >= 0) for c in self.connectivity)

        lu_b = _host_nbytes(self._lu_handles[0].host_tree()) if self._lu_handles else 0
        hbs_b = _host_nbytes(self._S_hbs[0].host_tree()) if self._S_hbs else 0
        info_b = _host_nbytes(self._slab_info[0])

        per_face = hbs_b / max(n_faces_first, 1)
        estimate = int((lu_b + info_b) * nslabs + per_face * total_faces)
        held = lu_b + hbs_b + info_b
        self.stats.host_bytes_estimate = estimate

        avail = _available_host_memory()
        if dbg > 0:
            print("host memory: first slab holds %s, projected total %s, "
                  "available %s" % (_fmt_bytes(held), _fmt_bytes(estimate),
                                    "unknown" if avail is None else _fmt_bytes(avail)))
        if not self.check_host_memory:
            return
        if avail is None:
            warnings.warn("oms: cannot determine available host memory; "
                          "skipping the host-memory check", RuntimeWarning,
                          stacklevel=3)
            return
        need = estimate - held
        budget = self.host_mem_fraction * avail
        if need > budget:
            parts = []
            if lu_b:
                parts.append("LU factors %s/slab" % _fmt_bytes(lu_b))
            if hbs_b:
                parts.append("HBS blocks %s/face" % _fmt_bytes(per_face))
            raise MemoryError(
                "oms: after slab 1 of %d the decomposition is projected to need "
                "%s more host memory (total %s: %s), but only %s is available "
                "(%.0f%% of %s).  Options: keepLU=False if the HBS operator "
                "suffices, stiff_mat_const=True if the slabs are translates, "
                "fewer / smaller slabs, or check_host_memory=False to try anyway."
                % (nslabs, _fmt_bytes(need), _fmt_bytes(estimate),
                   ", ".join(parts) or "bookkeeping", _fmt_bytes(budget),
                   100 * self.host_mem_fraction, _fmt_bytes(avail)))

    # ------------------------------------------------------------------ #
    # constant-stiffness-matrix support
    # ------------------------------------------------------------------ #

    def _reference_index(self):
        """
        FIX: the reference used to be slabList[0].  That is a *boundary*
             double slab: its left face lies on the physical boundary, so it
             only carries the right map K_{i,i+1}.  The left map K_{i,i-1} then
             had to come from a second, overlapping slab (slabList[1]) with its
             own complete set-up.  An interior double slab carries both source
             faces, so one set-up yields both maps for the whole system.
             Falls back to slab 0 only if no interior slab exists.
        """
        if self._ref_ind is None:
            self._ref_ind = 0
            for i, conn in enumerate(self.connectivity):
                if conn[0] >= 0 and conn[1] >= 0:
                    self._ref_ind = i
                    break
        return self._ref_ind

    def _compute_offsets(self, atol=None):
        """
        Verify every slab is a rigid translate of the reference slab and
        return the translation vectors.  A stiff_mat_const run is only
        legitimate if this holds, so it is checked rather than assumed.
        """
        slabs = [np.asarray(s, dtype=float) for s in self.slabList]
        r = self._reference_index()
        ref = slabs[r]
        ext0 = ref.max(axis=0) - ref.min(axis=0)
        scale = max(1.0, float(np.max(np.abs(ext0))), float(np.max(np.abs(ref))))
        tol = _COORD_RTOL * scale if atol is None else atol

        offsets = []
        for i, g in enumerate(slabs):
            if g.shape != ref.shape:
                raise ValueError(
                    "stiff_mat_const: slab %d has geometry shape %r, reference "
                    "slab %d has %r" % (i, g.shape, r, ref.shape))
            d = g.min(axis=0) - ref.min(axis=0)
            if not np.allclose(g, ref + d, rtol=0.0, atol=tol):
                raise ValueError(
                    "stiff_mat_const: slab %d is not a rigid translate of "
                    "reference slab %d (max deviation %.3e, tol %.3e).  The "
                    "local stiffness matrices will not coincide."
                    % (i, r, float(np.max(np.abs(g - (ref + d)))), tol))
            offsets.append(d)
        return offsets

    def _build_local_solver(self, slabInd, dbg=0):
        geom = np.array(self.slabList[slabInd])
        solver = solverWrap.solverWrapper(self.opts)
        solver.construct(geom, self.pdo, verbose=dbg)
        self._n_factorizations += 1
        return solver

    def _slab_solver(self, slabInd, dbg=0):
        """
        Return (solver, XXb, XXi, tDisc) for slab `slabInd`.

        stiff_mat_const=False : discretize and factorize this slab.
        stiff_mat_const=True  : build the reference (interior) slab once and
                                hand back its solver with rigidly shifted
                                coordinates, so gb / bc are still evaluated
                                at true *global* positions.
        """
        if not self.stiff_mat_const:
            t0 = time.time()
            solver = self._build_local_solver(slabInd, dbg)
            return solver, solver.XXb, solver.XXi, time.time() - t0

        tDisc = 0.0
        if self._ref_solver is None:
            self._offsets = self._compute_offsets()
            t0 = time.time()
            self._ref_solver = self._build_local_solver(self._reference_index(), dbg)
            tDisc = time.time() - t0
            self.localSolver = self._ref_solver
            if dbg > 0:
                print("stiff_mat_const: single reference factorization "
                      "(slab %d) built in %5.2f s, reused for all %d slabs"
                      % (self._ref_ind, tDisc, len(self.slabList)))

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
        Per-slab index sets.  Returns (Il, Ir, Ic, Igb, XXi, XXb, pts_l, pts_r),
        pts_l / pts_r being the physical source coordinates of each face.

        For problem_type='mixed', Il/Ir index into solver.JD, so the source
        coordinates are XXb[JD[I]], not XXb[I].
        """
        geom = np.array(self.slabList[slabInd])
        slab_i = slab(geom, self.gb)
        Il, Ir, Ic, Igb, XXi, XXb = slab_i.compute_idxs_and_pts(
            solver, XXb=XXb, XXi=XXi)

        if getattr(solver.opts, "problem_type", "Dirichlet") == "mixed":
            xl = float(geom[0][0])
            xr = float(geom[1][0])
            scale = max(abs(xl), abs(xr), abs(xr - xl))
            XD = XXb[solver.JD, 0]
            Il = np.flatnonzero(_coord_close(XD, xl, scale))
            Ir = np.flatnonzero(_coord_close(XD, xr, scale))
            pts_l = XXb[solver.JD[Il], :]
            pts_r = XXb[solver.JD[Ir], :]
        else:
            pts_l = XXb[Il, :]
            pts_r = XXb[Ir, :]

        return Il, Ir, Ic, Igb, XXi, XXb, pts_l, pts_r

    def _reference_faces(self, dbg=0):
        """
        Under stiff_mat_const: (solver, Il, Ir, Ic, XXi, pts_l, pts_r, tDisc)
        of the reference double slab, in the reference solver's local
        numbering.  Every map of the reduced system is derived from these.
        """
        r = self._reference_index()
        solver, XXb, XXi, tDisc = self._slab_solver(r, dbg=dbg)
        Il, Ir, Ic, _, XXi, _, pts_l, pts_r = self._slab_indices(
            r, solver, XXb, XXi)
        self._ref_faces = (np.asarray(Il), np.asarray(Ir), np.asarray(Ic))
        return solver, Il, Ir, Ic, XXi, pts_l, pts_r, tDisc

    def _matches_reference(self, Ic, J, side):
        """
        True if this slab's (target, source) index pair coincides with the
        reference slab's pair for `side` ('l' or 'r').  Normally always true;
        it can fail only if gb trims a face differently on some slab, in which
        case the caller falls back to building that block explicitly.
        """
        if self._ref_faces is None:
            return False
        Il, Ir, Ic_ref = self._ref_faces
        J_ref = Il if side == "l" else Ir
        return (len(J_ref) > 0 and np.array_equal(np.asarray(Ic), Ic_ref)
                and np.array_equal(np.asarray(J), J_ref))

    @staticmethod
    def _block_key(I, J, side):
        """
        Cache key for a source-target block.  `side` is carried along so that
        an empty left face and an empty right face -- which have identical
        (empty) index arrays -- cannot collide.
        """
        return (side, np.asarray(I).tobytes(), np.asarray(J).tobytes())

    # ------------------------------------------------------------------ #
    # bookkeeping
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
    # local operators
    # ------------------------------------------------------------------ #

    @staticmethod
    def _coupling(solver):
        """(Bcols, nrows) of the local solve, by problem type."""
        ptype = getattr(solver.opts, "problem_type", "Dirichlet")
        if ptype == "Dirichlet":
            return solver.Aib, len(solver.Ii)
        if ptype == "mixed":
            return solver.E, len(solver.Ii) + len(solver.JN)
        raise NameError(
            "solver problem type not recognized: must be 'Dirichlet' or 'mixed'."
        )

    def _make_st_linop(self, I, J, solver, handle=None):
        """
        Build one source-target LinearOperator:  v |-> (A^{-1} B[:,J] v)[I].

        `solver` must be resident at call time (used for B and the dtype).
        The local solve itself goes through `handle`, so an off-loaded
        factorization is only loaded while the operator is being applied.

        OPT: B[:,J] used to be re-sliced on *every* apply.  It is hoisted
             here, together with its transpose (it is a thin face coupling,
             cheap to keep resident).
        """
        Bcols, nrows_full = self._coupling(solver)
        if handle is None:
            handle = _Resident(solver)
        BJ = Bcols[:, J]
        if sp.issparse(BJ):
            BJ = BJ.tocsr()
            BJT = BJ.T.tocsr()
        else:
            BJ = np.asarray(BJ)
            BJT = BJ.T
        dt = np.result_type(_dtype_of(solver.solver_ii), _dtype_of(BJ))

        def smatmat(v, transpose=False):
            v_in = np.asarray(v)
            oneD = v_in.ndim == 1
            v_tmp = v_in[:, np.newaxis] if oneD else v_in
            A_solver = handle.get().solver_ii

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
            del A_solver

            return result.ravel() if oneD else result

        # OPT: passing dtype explicitly stops SciPy from probing the operator
        #      with matvec(zeros(N)) just to infer it -- a full local solve.
        return LinearOperator(
            shape=(len(I), len(J)),
            dtype=dt,
            matvec=lambda v: smatmat(v),
            rmatvec=lambda v: smatmat(v, transpose=True),
            matmat=lambda v: smatmat(v),
            rmatmat=lambda v: smatmat(v, transpose=True),
        )

    def _stmap(self, Ic, J, pts, XXi, solver):
        """One source-target map  J -> Ic  of the local double slab."""
        A_solver = solver.solver_ii
        return stMap(self._make_st_linop(Ic, J, solver), pts, XXi[Ic, :],
                     A_solver.shape[0], A_solver.shape[1])

    def compute_stmaps(self, Il, Ic, Ir, XXi, XXb, solver, pts_l=None, pts_r=None):
        """Separate left / right source-target maps of one double slab."""
        pts_l = XXb[Il, :] if pts_l is None else pts_l
        pts_r = XXb[Ir, :] if pts_r is None else pts_r
        return (self._stmap(Ic, Il, pts_l, XXi, solver),
                self._stmap(Ic, Ir, pts_r, XXi, solver))

    @staticmethod
    def _column_restriction(op, start, n):
        """
        x |-> op @ [0; x; 0], with x occupying columns start:start+n of op.
        Lets a boundary slab reuse the reference slab's fused [Il, Ir]
        operator instead of owning a separate one.  The zero block costs a
        sparse multiply, not an extra local solve.
        """
        ncols = op.shape[1]
        sl = slice(start, start + n)

        def mv(x):
            x = np.asarray(x)
            full = np.zeros((ncols,) + x.shape[1:],
                            dtype=np.result_type(op.dtype, x.dtype))
            full[sl] = x
            return op @ full

        def rmv(y):
            y = np.asarray(y)
            out = op.rmatvec(y) if y.ndim == 1 else op.rmatmat(y)
            return out[sl]

        return LinearOperator(shape=(op.shape[0], n), dtype=op.dtype,
                              matvec=mv, rmatvec=rmv, matmat=mv, rmatmat=rmv)

    # ------------------------------------------------------------------ #
    # HBS blocks
    # ------------------------------------------------------------------ #

    @staticmethod
    def _block_error(st, mat, assembler):
        V = np.random.standard_normal(size=(st.A.shape[1], assembler.matOpts.maxRank))
        U = st.A @ V
        return np.linalg.norm(U - mat @ V) / np.linalg.norm(U)

    def _assemble_cached(self, st, I, J, assembler, slabInd, label, dbg):
        """
        Compress one source-target block.  Under stiff_mat_const an identical
        (I, J) pair is served from the cache.

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

    def _hbs_block(self, side, Ic, J, pts, slabInd, solver, XXi, assembler, dbg):
        """
        One HBS block of slab `slabInd`.  Under stiff_mat_const it is taken
        from the reference set-up (explicit build only if this slab's face
        differs from the reference face).

        Returns (block, compression_ratio, err, tCompress, tSample, shape_ok).
        """
        if self.stiff_mat_const and side in self._ref_hbs \
                and self._matches_reference(Ic, J, side):
            if slabInd != self._ref_ind:
                self._n_reused += 1
            mat, ratio, err = self._ref_hbs[side]
            return mat, ratio, err, 0.0, 0.0, True

        label = "LEFT" if side == "l" else "RIGHT"
        if self.stiff_mat_const and dbg > 0:
            print("stiff_mat_const: %s face of slab %d differs from the "
                  "reference face; building it explicitly" % (label, slabInd))
        st = self._stmap(Ic, J, pts, XXi, solver)
        mat, fresh, ratio = self._assemble_cached(
            st, Ic, J, assembler, slabInd, label, dbg)
        tC = assembler.stats.timeCompress if fresh else 0.0
        tS = assembler.stats.timeSample if fresh else 0.0
        # FIX: shapeMatch was printed but its computation was commented out.
        shape_ok = tuple(mat.shape) == tuple(st.A.shape)
        err = None
        if dbg > 0:
            err = self._block_error(st, mat, assembler)
            print("%s ERR = " % label, err)
        return mat, ratio, err, tC, tS, shape_ok

    # ------------------------------------------------------------------ #
    # reference set-up (stiff_mat_const)
    # ------------------------------------------------------------------ #

    def _build_reference(self, assembler, dbg=0):
        """
        stiff_mat_const: derive everything from ONE interior double slab --
        the HBS blocks K_{i,i-1}, K_{i,i+1} (constructHBS) and / or the fused
        exact operator [K_{i,i-1}  K_{i,i+1}] (keepLU).

        Returns (tDisc, tCompress, tSample, shapes_match).
        """
        self._ref_hbs, self._ref_lu = {}, {}
        solver, Il, Ir, Ic, XXi, pts_l, pts_r, tDisc = self._reference_faces(dbg)
        tC = tS = 0.0
        shapes_ok = True

        if self.constructHBS:
            for side, J, pts, label in (("l", Il, pts_l, "LEFT"),
                                        ("r", Ir, pts_r, "RIGHT")):
                if len(J) == 0:
                    continue          # only if no interior slab exists
                st = self._stmap(Ic, J, pts, XXi, solver)
                mat, _, ratio = self._assemble_cached(
                    st, Ic, J, assembler, self._ref_ind, label, dbg)
                tC += assembler.stats.timeCompress
                tS += assembler.stats.timeSample
                shapes_ok = shapes_ok and tuple(mat.shape) == tuple(st.A.shape)
                err = None
                if dbg > 0:
                    err = self._block_error(st, mat, assembler)
                    print("%s ERR (reference slab %d) = " % (label, self._ref_ind), err)
                self._ref_hbs[side] = (mat, ratio, err)
                del st

        if self.keepLU and len(Il) > 0 and len(Ir) > 0:
            op = self._make_st_linop(
                Ic, np.concatenate([np.asarray(Il), np.asarray(Ir)]), solver)
            self._n_assembled += 1
            self._ref_lu = {
                "lr": op,
                "l": self._column_restriction(op, 0, len(Il)),
                "r": self._column_restriction(op, len(Il), len(Ir)),
            }
        return tDisc, tC, tS, shapes_ok

    def _ref_lu_operator(self, tag, Ic, Il, Ir):
        """Reference-derived LU operator for a slab with source faces `tag`."""
        if not self._ref_lu:
            return None
        ok = ((tag in ("l", "lr")) <= self._matches_reference(Ic, Il, "l")) and \
             ((tag in ("r", "lr")) <= self._matches_reference(Ic, Ir, "r"))
        return self._ref_lu[tag] if ok else None

    # ------------------------------------------------------------------ #
    # right-hand side
    # ------------------------------------------------------------------ #

    @staticmethod
    def _local_rhs(solver, bc, info, reduced_load, slabInd):
        Ic = info.Ic
        ptype = getattr(solver.opts, "problem_type", "Dirichlet")
        if ptype == "Dirichlet":
            fgb = bc(info.pts_gb)
            return -(solver.solver_ii @ (solver.Aib[:, info.Igb] @ fgb))[Ic]
        if ptype == "mixed":
            # composition returns (b_C on C-space, b_X on X-space)
            b_C, b_X = _eval_reduced_load(reduced_load, solver, slabInd)
            b_N = b_X[solver.JN]                  # X-space load on Neumann rows
            fgb = bc(info.pts_N)                  # g_N
            rhs = solver.solver_ii @ np.concatenate([b_C, fgb + b_N])
            return rhs[Ic]
        raise NameError(
            "solver problem type not recognized, must be 'Dirichlet' or 'mixed'"
        )

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #

    def construct_Stot_helper(self, bc, assembler=None, reduced_load=None, dbg=0):
        """
        Build the per-slab interface operators and local right-hand sides.
        Returns (rhs_list, Ntot).  The operators are kept on the object and
        combined into I - K by global_operator().
        """
        if self.constructHBS and assembler is None:
            raise ValueError("constructHBS=True needs an `assembler`")

        # OPT: calling this twice used to keep a second full set of local
        #      solvers alive; everything from a previous call is dropped.
        self.close()
        self._slab_info = []
        self._n_offloaded = 0
        self.ncs = []
        self.glob_source_dofs = []
        connectivity = self.connectivity
        slabs = self.slabList

        Ntot = 0
        rhs_list = []
        glob_target_dofs = []
        startCentral = 0
        dtypes = []

        discrTime = 0.0
        compressTime = 0.0
        sampleTime = 0.0
        shapeMatch = True
        relerrl = 0.0
        relerrr = 0.0

        if self.stiff_mat_const:
            # One interior double slab -> every map of the whole system.
            tDisc, tC, tS, shapeMatch = self._build_reference(assembler, dbg)
            discrTime += tDisc
            compressTime += tC
            sampleTime += tS

        for slabInd in range(len(slabs)):
            solver, XXb, XXi, tDisc = self._slab_solver(slabInd, dbg=dbg)
            discrTime += tDisc
            if dbg > 1 and tDisc > 0:
                print("SLAB %2.0d discretization time = %5.2f s" % (slabInd, tDisc))

            Il, Ir, Ic, Igb, XXi, XXb, pts_l, pts_r = self._slab_indices(
                slabInd, solver, XXb, XXi)
            nc = len(Ic)
            if dbg > 1:
                print("SLAB %2.0d size = %2.0d" % (slabInd, nc))
            self.ncs.append(nc)
            Ntot += nc
            glob_target_dofs.append(range(startCentral, startCentral + nc))
            startCentral += nc

            L, R = connectivity[slabInd][0], connectivity[slabInd][1]
            # FIX: the geometric emptiness of Il/Ir and the topological
            #      connectivity flags must agree, otherwise the operator is
            #      applied to the wrong dofs.
            if (len(Il) > 0) != (L >= 0):
                raise ValueError(
                    "slab %d: left connectivity says %s but %d left-face dofs "
                    "were found" % (slabInd, L, len(Il)))
            if (len(Ir) > 0) != (R >= 0):
                raise ValueError(
                    "slab %d: right connectivity says %s but %d right-face dofs "
                    "were found" % (slabInd, R, len(Ir)))

            # OPT: keep only what rhs / uX_full need; the original stored the
            #      full XXi / XXb point arrays of every slab.
            ptype = getattr(solver.opts, "problem_type", "Dirichlet")
            info = SimpleNamespace(
                Il=np.asarray(Il), Ir=np.asarray(Ir), Ic=np.asarray(Ic),
                Igb=np.asarray(Igb),
                pts_gb=XXb[Igb, :] if ptype == "Dirichlet" else None,
                pts_N=XXb[solver.JN, :] if ptype == "mixed" else None,
            )
            # FIX (regression): the rhs is the exact local solve, as in the
            #      original oms / oms_lu.  A compressed boundary-to-interface
            #      map was briefly built here; its source set (the slab's
            #      physical walls) is not a planar interface, so the HBS tree
            #      cannot be built on it.
            rhs = self._local_rhs(solver, bc, info, reduced_load, slabInd)
            rhs_list.append(rhs)
            dtypes.append(_dtype_of(np.asarray(rhs)))

            # source faces in glob_source_dofs order: left, then right
            faces = []
            if L >= 0:
                faces.append(("l", Il, pts_l))
            if R >= 0:
                faces.append(("r", Ir, pts_r))

            # ---- HBS blocks ------------------------------------------------ #
            if self.constructHBS:
                blocks, errs, ratios = [], {}, {}
                for side, J, pts in faces:
                    mat, ratio, err, tC, tS, ok = self._hbs_block(
                        side, Ic, J, pts, slabInd, solver, XXi, assembler, dbg)
                    compressTime += tC
                    sampleTime += tS
                    shapeMatch = shapeMatch and ok
                    blocks.append(mat)
                    dtypes.append(_dtype_of(mat))
                    errs[side], ratios[side] = err, ratio

                if dbg > 0:
                    if errs.get("l") is not None:
                        relerrl = max(relerrl, errs["l"])
                    if errs.get("r") is not None:
                        relerrr = max(relerrr, errs["r"])
                    if dbg > 1:
                        # FIX: the per-slab report used to print the running
                        #      maxima rather than this slab's own errors.
                        sides = [f[0] for f in faces]
                        print("SLAB %d error = %s" % (slabInd, " // ".join(
                            "%5.2e" % errs[s] for s in sides)))
                        print("SLAB %d compression = %s\n" % (slabInd, " // ".join(
                            "%5.3e" % ratios[s] for s in sides)))

                # Reference blocks are shared -> keep resident.  Otherwise the
                # slab's blocks go to the store right away.
                if self.stiff_mat_const:
                    self._S_hbs.append(_Resident(blocks))
                else:
                    self._S_hbs.append(self._offload(blocks))
                del blocks

            # ---- kept LU / fused exact operator --------------------------- #
            if self.keepLU:
                tag = "".join(f[0] for f in faces)
                J = np.concatenate([np.asarray(f[1]) for f in faces])
                S_i = None
                if self.stiff_mat_const:
                    handle = _Resident(solver)
                    S_i = self._ref_lu_operator(tag, Ic, Il, Ir)
                    if S_i is not None:
                        if slabInd != self._ref_ind:
                            self._n_reused += 1
                    else:
                        key = self._block_key(Ic, J, tag)
                        if key in self._block_cache:
                            S_i = self._block_cache[key][0]
                            self._n_reused += 1
                        else:
                            S_i = self._make_st_linop(Ic, J, solver, handle)
                            self._n_assembled += 1
                            self._block_cache[key] = (S_i, float("nan"))
                else:
                    # linop needs B / dtype from the resident solver, then
                    # the factorization itself moves off the device.
                    handle = self._offload(_slim_solver(solver))
                    S_i = self._make_st_linop(Ic, J, solver, handle)
                    self._n_assembled += 1
                self._S_lu.append(S_i)
                self._lu_handles.append(handle)
                dtypes.append(S_i.dtype)

            self._slab_info.append(info)
            if slabInd == 0 and not self.stiff_mat_const:
                self._check_host_memory(len(faces), dbg)

            del Il, Ir, Ic, Igb, XXi, XXb, pts_l, pts_r
            if not self.stiff_mat_const:
                del solver        # its factorization is on disk or discarded

            if dbg > 0:
                print("overlapping slab ", slabInd + 1, " of ", len(slabs), " done")

        # Without keepLU nothing needs the reference factorization any more.
        if self.stiff_mat_const and not self.keepLU:
            self._ref_solver = None
            self.localSolver = None

        self._dtype = np.result_type(*dtypes) if dtypes else np.float64
        compression = (self.nbytes / self.densebytes) if self.densebytes else float("nan")

        nfac = max(self._n_factorizations, 1)
        nasm = max(self._n_assembled, 1)
        self.stats.compression = compression if self.constructHBS else None
        self.stats.sampl_timing = sampleTime / nasm
        self.stats.compr_timing = compressTime / nasm
        self.stats.discr_timing = discrTime / nfac
        self.stats.n_factorizations = self._n_factorizations
        self.stats.n_assembled = self._n_assembled
        self.stats.n_reused = self._n_reused
        self.stats.n_offloaded = self._n_offloaded
        self.stats.host_bytes = self.host_nbytes()

        if dbg > 0:
            self._summary(discrTime, sampleTime, compressTime, glob_target_dofs,
                          relerrl if self.constructHBS else None,
                          relerrr if self.constructHBS else None,
                          compression if self.constructHBS else None)
            if self.constructHBS:
                print("shapes match?                = ", shapeMatch)

        self.glob_target_dofs = glob_target_dofs
        self.compute_global_dofs()
        self._rhs_cache = (bc, reduced_load, rhs_list)
        self._built = True
        return rhs_list, Ntot

    # ------------------------------------------------------------------ #
    # global operator
    # ------------------------------------------------------------------ #

    def _resolve_hbs(self, rhsHBS):
        """
        One switch selects operator AND rhs together:
            False -> exact LU operator  (needs keepLU)
            True  -> HBS operator       (needs constructHBS)
            None  -> LU when the factorizations are kept, else HBS.
        The rhs is the exact local-solve rhs on both sides.
        """
        if rhsHBS is None:
            return not self.keepLU
        if rhsHBS and not self.constructHBS:
            raise ValueError("rhsHBS=True needs constructHBS=True")
        if not rhsHBS and not self.keepLU:
            raise ValueError("rhsHBS=False (the LU versions) needs keepLU=True")
        return bool(rhsHBS)

    def _resolve_operator(self, operator):
        if operator is None:
            return "hbs" if self._resolve_hbs(None) else "lu"
        if operator not in ("hbs", "lu"):
            raise ValueError("operator must be 'hbs' or 'lu'")
        self._resolve_hbs(operator == "hbs")          # validates availability
        return operator

    def global_operator(self, operator=None):
        """
        I + S as a LinearOperator (S = -K, sign carried by the local maps).

        operator='lu'  : exact fused local solves (default if keepLU)
        operator='hbs' : compressed HBS blocks    (default otherwise)

        Only the operator: no rhs, no construction.  Use it to get a second
        view of an already-built decomposition (e.g. the LU operator next to
        the HBS one to measure compression error); construct_Stot_and_rhstot
        is the normal entry point.

        With off-loading, each application streams one slab's blocks or
        factorization from the store at a time.

        EXPLAINER OF CONVENTIONS:
            - global dof ordering is inferred from the supplied connectivity
            - joined slabs are contiguous (fictitious domain extension used for
              periodic domains)
            - contiguous blocks are used for global dofs, to improve efficiency
        """
        operator = self._resolve_operator(operator)
        if not self.glob_target_dofs:
            raise RuntimeError("global_operator() called before construct_Stot_helper()")

        tgt = [_as_index(b) for b in self.glob_target_dofs]
        src = [[_as_index(b) for b in row] for row in self.glob_source_dofs]
        Ntot = sum(len(b) for b in self.glob_target_dofs)
        dtype = self._dtype

        def _blen(b):
            return (b.stop - b.start) if isinstance(b, slice) else len(b)

        lens = [[_blen(b) for b in row] for row in src]

        if operator == "hbs":
            handles = self._S_hbs

            def smatmat(v, transpose=False):
                v_in = np.asarray(v)
                oneD = v_in.ndim == 1
                v_tmp = v_in[:, np.newaxis] if oneD else v_in
                # OPT: one copy (the identity term) instead of astype-then-copy.
                result = v_tmp.astype(np.result_type(v_tmp.dtype, dtype), copy=True)
                for i, ti in enumerate(tgt):
                    blocks = handles[i].get()      # one slab resident at a time
                    if not transpose:
                        for blk, sj in zip(blocks, src[i]):
                            result[ti] += blk @ v_tmp[sj]
                    else:
                        for blk, sj in zip(blocks, src[i]):
                            result[sj] += blk.T @ v_tmp[ti]
                    del blocks
                return result.ravel() if oneD else result
        else:
            ops = self._S_lu

            def smatmat(v, transpose=False):
                v_in = np.asarray(v)
                oneD = v_in.ndim == 1
                v_tmp = v_in[:, np.newaxis] if oneD else v_in
                result = v_tmp.astype(np.result_type(v_tmp.dtype, dtype), copy=True)
                if not transpose:
                    for i, ti in enumerate(tgt):
                        srcs = src[i]
                        if len(srcs) == 1:
                            x = v_tmp[srcs[0]]
                        else:
                            x = np.concatenate([v_tmp[sj] for sj in srcs], axis=0)
                        result[ti] += ops[i] @ x          # ONE local solve
                else:
                    for i, ti in enumerate(tgt):
                        y = ops[i].T @ v_tmp[ti]          # ONE local solve
                        off = 0
                        for sj, n in zip(src[i], lens[i]):
                            result[sj] += y[off:off + n]
                            off += n
                return result.ravel() if oneD else result

        return LinearOperator(
            shape=(Ntot, Ntot),
            dtype=dtype,
            matvec=smatmat,
            rmatvec=lambda v: smatmat(v, transpose=True),
            matmat=smatmat,
            rmatmat=lambda v: smatmat(v, transpose=True),
        )

    def _assemble_rhs(self, rhs_list):
        Ntot = sum(len(b) for b in self.glob_target_dofs)
        dtype = np.result_type(*[np.asarray(r).dtype for r in rhs_list])
        rhstot = np.zeros(Ntot, dtype=dtype)
        # FIX: was a uniform i*nc slicing, inconsistent with glob_target_dofs.
        for i, rhs in enumerate(rhs_list):
            rhstot[_as_index(self.glob_target_dofs[i])] = rhs
        return rhstot

    def construct_Stot_and_rhstot(self, bc, assembler=None, reduced_load=None,
                                  dbg=0, rhsHBS=None):
        """
        Return (I + S as LinearOperator, global rhs), both from the same side:

            rhsHBS=False : exact LU operator   (needs keepLU)
            rhsHBS=True  : HBS operator        (needs constructHBS)
            rhsHBS=None  : LU if keepLU, else HBS

        The rhs is the exact local-solve rhs  -(A^{-1} A_ib g)[Ic]  on both
        sides (as in the original oms and oms_lu).  It is computed during
        construction; for different boundary data it needs keepLU.

        The decomposition (factorizations, compression) is built on the first
        call only; later calls reuse it.  Call construct_Stot_helper() to
        force a rebuild.
        """
        hbs = self._resolve_hbs(rhsHBS)          # fail fast, before any work
        if not self._built:
            self.construct_Stot_helper(bc, assembler, reduced_load, dbg)
        return (self.global_operator("hbs" if hbs else "lu"),
                self.construct_rhstot(bc, reduced_load, rhsHBS=hbs))

    # ------------------------------------------------------------------ #
    # things that need the kept factorizations
    # ------------------------------------------------------------------ #

    def _require_lu(self, what):
        if not self.keepLU:
            raise RuntimeError(
                "%s needs the local factorizations; construct with keepLU=True" % what)
        if not self._lu_handles:
            raise RuntimeError("%s called before construct_Stot_helper()" % what)

    @property
    def hbs_blocks(self):
        """
        The old `S_rk_list`: one list per slab, each holding that slab's HBS
        blocks ordered like glob_source_dofs[slabInd] (left face, then right;
        end slabs have a single block).  So  hbs_blocks[0][0].tree  is the
        tree of slab 0's block, as before.

        With off-loading this brings every slab's blocks back to the device at
        once, which is what off-loading exists to avoid -- use
        hbs_blocks_for(slabInd) to walk them one slab at a time.  Under
        stiff_mat_const all slabs share the reference blocks.
        """
        self._require_hbs()
        if self._n_offloaded and self.offload:
            warnings.warn(
                "oms.hbs_blocks materializes the blocks of all %d slabs at "
                "once, undoing the off-loading; use hbs_blocks_for(slabInd) "
                "to take them one slab at a time."
                % len(self._S_hbs), RuntimeWarning, stacklevel=2)
        return [h.get() for h in self._S_hbs]

    def hbs_blocks_for(self, slabInd):
        """One slab's HBS blocks: hbs_blocks[slabInd], without materializing
        the others."""
        self._require_hbs()
        return self._S_hbs[slabInd].get()

    def _require_hbs(self):
        if not self.constructHBS:
            raise RuntimeError("hbs_blocks needs constructHBS=True")
        if not self._S_hbs:
            raise RuntimeError("hbs_blocks used before construct_Stot_helper()")

    def host_nbytes(self):
        """Host bytes currently held by the kept operators and bookkeeping."""
        trees = [h.host_tree() for h in self._S_hbs + self._lu_handles]
        return _host_nbytes([trees, self._slab_info, self._ref_hbs])

    def construct_rhstot(self, bc, reduced_load=None, dbg=0, rhsHBS=None):
        """
        Global (exact) rhs for boundary data `bc`.  `rhsHBS` is accepted for
        symmetry with construct_Stot_and_rhstot and only validated: the rhs
        does not depend on the operator choice.

        For the `bc` the decomposition was built with, the rhs computed during
        construction is returned (no local solves).  Other `bc` need the kept
        factorizations (keepLU=True).
        """
        self._resolve_hbs(rhsHBS)
        if not self._built:
            raise RuntimeError("construct_rhstot() called before construct_Stot_helper()")
        cache = self._rhs_cache
        if cache is not None and cache[0] is bc and cache[1] is reduced_load:
            return self._assemble_rhs(cache[2])
        if not self.keepLU:
            raise RuntimeError(
                "construct_rhstot() for new boundary data needs the local "
                "factorizations; construct with keepLU=True")
        self._require_lu("construct_rhstot()")
        rhs_list = []
        for slabInd, (h, info) in enumerate(zip(self._lu_handles, self._slab_info)):
            solver = h.get()
            rhs_list.append(self._local_rhs(solver, bc, info, reduced_load, slabInd))
            del solver
        return self._assemble_rhs(rhs_list)

    def uX_full(self, uhat, i, b_C, b_X):
        self._require_lu("uX_full()")
        solver = self._lu_handles[i].get()
        if solver.opts.problem_type != "mixed":
            raise NameError("uX_full is only defined for problem_type='mixed'")

        info = self._slab_info[i]
        Il, Ir = info.Il, info.Ir

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

    # ------------------------------------------------------------------ #

    def _summary(self, discrTime, sampleTime, compressTime, glob_target_dofs,
                 relerrl=None, relerrr=None, compression=None):
        # FIX: averages were divided by len(connectivity)-1, which is both an
        #      off-by-one and a ZeroDivisionError for a single-slab run.
        nslabs = max(len(self.slabList), 1)
        nfac = max(self._n_factorizations, 1)
        nasm = max(self._n_assembled, 1)
        print("============================OMS SUMMARY============================")
        print("constructHBS / keepLU        = ", self.constructHBS, "/", self.keepLU)
        if self.stiff_mat_const:
            print("stiff_mat_const              =  True (reference slab %d)"
                  % self._ref_ind)
            print("factorizations               = ", self._n_factorizations,
                  " (of", nslabs, "slabs)")
            print("blocks assembled / reused    = ", self._n_assembled, "/",
                  self._n_reused)
            print("total discr. time            = ", discrTime)
            print("total sample time            = ", sampleTime)
            print("total compr. time            = ", compressTime)
        print("available (operator + rhs)   = ", " / ".join(
            k for k, ok in (("LU", self.keepLU), ("HBS", self.constructHBS))
            if ok), " (default %s)" % ("LU" if self.keepLU else "HBS"))
        if self.offload:
            print("device objects off-loaded    = ", self._n_offloaded)
        print("host bytes held              = ", _fmt_bytes(self.host_nbytes()))
        if self.stats.host_bytes_estimate is not None:
            print("host bytes projected (slab 1)= ",
                  _fmt_bytes(self.stats.host_bytes_estimate))
        print("avg. discr. time             = ", discrTime / nfac)
        if self.constructHBS:
            print("avg. sample time             = ", sampleTime / nasm)
            print("avg. compr. time             = ", compressTime / nasm)
        if compression is not None:
            print("compression rate             = ", compression)
        print("total dofs                   = ",
              sum(len(dof) for dof in glob_target_dofs))
        if relerrl is not None:
            print("estim. max. err. ( l // r )  = (", relerrl, " // ", relerrr, ")")
        print("===================================================================")
