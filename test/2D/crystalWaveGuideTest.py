# basic packages
# Default behavior is reduced-system conditioning/reporting only.
# Useful overrides:
#   SSLABLU_PDE=constant, SSLABLU_KH=0: Laplace with known data.
#   SSLABLU_PDE=constant, SSLABLU_KH=<nonzero>: constant Helmholtz.
#   SSLABLU_PDE=crystal: variable-coefficient crystal Helmholtz.
#   SSLABLU_RUN_VALIDATION=1: run the more expensive validation solves.
#   SSLABLU_RUN_GMRES=1: run GMRES diagnostics on the meaningful direct-solve RHS.
#   SSLABLU_COND_NIT=20: use 20 randomized power iterations in condition estimates.
#   SSLABLU_NPAN_X=<nx>, SSLABLU_NPAN_Y=<ny>: refine the local panel grid with fixed subdomains.
#   SSLABLU_CHECK_ITI_CAYLEY=1: verify local ItI maps against DtN maps by Cayley transform.
#   SSLABLU_REF_PATH=/path/ref.npy: compare variable-coefficient runs to a refined reference.
#   SSLABLU_SAVE_REF_PATH=/path/ref.npy: save the current variable-coefficient S solution as reference.
#   SSLABLU_SHARED_POINT_VALIDATION=1: compare S interface traces to a saved refined trace.
#   SSLABLU_SHARED_REF_PATH=/path/ref.npz: refined interface trace for 1D trace interpolation.
#   SSLABLU_SAVE_SHARED_REF_PATH=/path/ref.npz: save the current S interface trace.
import os
import sys
from pathlib import Path
import numpy as np
import torch
import scipy
import scipy.sparse as sp

_HPSMULTIDOMAIN_ROOT = Path(__file__).resolve().parents[2] / "solver" / "hpsmultidomain"
if str(_HPSMULTIDOMAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_HPSMULTIDOMAIN_ROOT))

# oms packages
import solver.solver as solverWrap
import matAssembly.matAssembler as mA
import multislab.oms as oms
import hpsmultidomain.pdo as pdo
import hpsmultidomain.sparse_utils as sparse_utils

# validation&testing
import time
from scipy.sparse.linalg import aslinearoperator, gmres

import geometry.geom_2D.square as square
class gmres_info(object):
    def __init__(self, disp=False):
        self._disp = disp
        self.niter = 0
        self.resList=[]
    def __call__(self, rk=None):
        self.niter += 1
        self.resList+=[rk]
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))


def timed_step(label, fn):
    tic = time.time()
    result = fn()
    toc = time.time() - tic
    print("TIMING %-28s = %8.3f s" % (label, toc))
    return result, toc


def gmres_probe(op, rhs, rtol=1e-8, restart=200, maxiter=500):
    info_cb = gmres_info()
    rhs = np.asarray(rhs)
    if rhs.ndim == 2 and rhs.shape[1] == 1:
        rhs = rhs[:, 0]

    kwargs = dict(callback=info_cb, maxiter=maxiter, restart=restart)
    try:
        sol, info = gmres(op, rhs, rtol=rtol, callback_type="pr_norm", **kwargs)
    except TypeError:
        sol, info = gmres(op, rhs, tol=rtol, **kwargs)

    relres = np.linalg.norm(op @ sol - rhs) / np.linalg.norm(rhs)
    return info_cb.niter, info, relres


def add_dense_block(builder, rows, cols, block, shape):
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    block = np.asarray(block)
    if block.size == 0:
        return

    rr = np.repeat(rows, cols.shape[0])
    cc = np.tile(cols, rows.shape[0])
    builder.add_data(sp.coo_matrix((block.reshape(-1), (rr, cc)), shape=shape))


def build_s_sparse_matrix(OMS, S_rk_list, Ntot, dtype):
    nnz = Ntot
    for row_blocks in S_rk_list:
        for block in row_blocks:
            nnz += block.shape[0] * block.shape[1]

    builder = sparse_utils.CSRBuilder(Ntot, Ntot, nnz, dtype=dtype)
    builder.add_data(sp.eye(Ntot, format="coo", dtype=dtype))
    shape = (Ntot, Ntot)
    for i in range(len(OMS.glob_target_dofs)):
        rows = np.asarray(list(OMS.glob_target_dofs[i]), dtype=int)
        for j in range(len(OMS.glob_source_dofs[i])):
            cols = np.asarray(list(OMS.glob_source_dofs[i][j]), dtype=int)
            add_dense_block(builder, rows, cols, S_rk_list[i][j], shape)
    return builder.tocsr()


def sparse_direct_inverse_operator(A, label):
    solver, toc_factor = timed_step(
        "%s sparse solver build" % label,
        lambda: sparse_utils.SparseSolver(A.tocsr()),
    )
    print("%s sparse solver backend = %s" % (label, solver.backend))
    return solver.solve_op, toc_factor, solver.backend


def condition_t_system(system, seed):
    label = system['label']
    Aii = system['A']
    rhs = system['rhs']
    Aii_op = aslinearoperator(Aii)
    if sp.issparse(Aii):
        Aii_inv, toc_factor, backend = sparse_direct_inverse_operator(Aii, label)
    else:
        Aii_inv, toc_factor = timed_step(
            "%s dense LU build" % label,
            lambda: sparse_utils.dense_lu_inverse_operator(Aii),
        )
        backend = "dense_lu"
    sol, toc_solve = timed_step(
        "%s direct solve" % label,
        lambda: Aii_inv.matvec(rhs),
    )
    relres = np.linalg.norm(Aii @ sol - rhs) / np.linalg.norm(rhs)
    (op_norm, inv_norm, cond), _ = timed_step(
        "%s condition estimate" % label,
        lambda: sparse_utils.estimate_condition_number(Aii_op, Aii_inv, nit=cond_nit, seed=seed),
    )
    eff_cond = sparse_utils.estimate_effective_condition_number(
        Aii_op,
        rhs,
        solution=sol,
        op_norm=op_norm,
    )
    if run_t_gmres_probe:
        gmres_stats, _ = timed_step(
            "%s GMRES probe" % label,
            lambda: gmres_probe(Aii_op, rhs, rtol=gmres_rtol, restart=gmres_restart, maxiter=gmres_maxiter),
        )
    else:
        gmres_stats = (-1, -1, np.nan)

    system['sol'] = sol
    system['timing'] = {
        'toc_sparse_assembly': system.get('toc_sparse_assembly', np.nan),
        'toc_sparse_solver': toc_factor,
        'toc_system_solve': toc_solve,
        'solver_backend': backend,
    }
    return {
        'system': system,
        'shape': Aii.shape,
        'solver_backend': backend,
        'relres': relres,
        'op_norm': op_norm,
        'inv_norm': inv_norm,
        'cond': cond,
        'eff_cond': eff_cond,
        'gmres': gmres_stats,
    }


def condition_t_formulation(diff_op, kh, N, H, opts, formulation, bc_func, s_coords, seed):
    builders = {
        'dtn': build_macro_t_dtn_system,
        'iti_schur': build_macro_t_iti_schur_system,
    }
    labels = {
        'dtn': 'T-DtN',
        'iti_schur': 'T-ItI-SC',
    }
    label = labels[formulation]
    system_builder = builders[formulation]
    system, _ = timed_step(
        "%s macro assembly" % label,
        lambda: system_builder(N, H, opts, diff_op, bc_func, s_coords, kh),
    )
    return condition_t_system(system, seed)


def print_t_conditioning(label, stats):
    niter, info, relres = stats['gmres']
    print(
        "condest(%s)  = %5.5e   effcond=%5.5e   gmres=(%d,%d,%5.2e)   "
        "[||T||=%5.5e, ||T^-1||=%5.5e, relres=%5.2e, backend=%s, size=%s]"
        % (
            label,
            stats['cond'],
            stats['eff_cond'],
            niter,
            info,
            relres,
            stats['op_norm'],
            stats['inv_norm'],
            stats['relres'],
            stats['solver_backend'],
            stats['shape'],
        )
    )


#nwaves = 24.623521102434587
nwaves = 24.673521102434584
kh = float(os.environ.get("SSLABLU_KH", (nwaves+0.03)*2*np.pi+1.8))
pde_mode = os.environ.get("SSLABLU_PDE", "crystal")
if pde_mode not in ("crystal", "constant"):
    raise ValueError("SSLABLU_PDE must be either 'crystal' or 'constant'")
bc_mode = os.environ.get("SSLABLU_BC", "known" if pde_mode == "constant" else "ones")
if bc_mode not in ("ones", "known"):
    raise ValueError("SSLABLU_BC must be either 'ones' or 'known'")
known_center = np.array([-0.75, 1.35])


def constant_helmholtz_exact(points):
    if torch.is_tensor(points):
        center = torch.as_tensor(known_center, dtype=points.dtype, device=points.device)
        dist = torch.linalg.norm(points[:, :2] - center, dim=1).detach().cpu().numpy()
        vals = np.log(dist) if kh == 0 else scipy.special.j0(kh * dist)
        return torch.as_tensor(vals, dtype=points.dtype, device=points.device)

    points = np.asarray(points)
    dist = np.linalg.norm(points[:, :2] - known_center, axis=1)
    if kh == 0:
        return np.log(dist)
    return scipy.special.j0(kh * dist)


def to_numpy_vector(values):
    if torch.is_tensor(values):
        return values.detach().cpu().numpy().reshape(-1)
    return np.asarray(values).reshape(-1)


def to_numpy_points(points):
    if torch.is_tensor(points):
        return points.detach().cpu().numpy()
    return np.asarray(points)


def coord_key(point):
    return tuple(np.round(np.asarray(point, dtype=float), 12))


def local_dtn_boundary_block(solver, row_face, col_face):
    Axx = solver.Abb[row_face[:, None], col_face].toarray()
    Axi = solver.Abi[row_face].toarray()
    Aix = solver.Aib[:, col_face].toarray()
    return Axx - Axi @ solver.solver_ii.matmat(Aix)


def get_s_target_coords(dslabs, opts, diff_op):
    coords = []
    for geom in dslabs:
        geom = np.asarray(geom)
        slab_i = oms.slab(geom, lambda p: square.gb(p, jax_avail=False, torch_avail=True))
        slab_solver = solverWrap.solverWrapper(opts)
        slab_solver.construct(geom, diff_op, verbose=False, compute_inverse=False)
        _, _, Ic, _, XXi, _ = slab_i.compute_idxs_and_pts(slab_solver)
        coords.append(to_numpy_points(XXi[Ic]))
    return np.vstack(coords)


def face_global_indices(XXb, face, coord_to_ind):
    return np.array([coord_to_ind[coord_key(x)] for x in to_numpy_points(XXb[face])])


def build_unit_slab_solver(slab_ind, H, opts, diff_op):
    geom = np.array([[slab_ind * H, 0.0], [(slab_ind + 1) * H, 1.0]])
    slab_i = oms.slab(geom, lambda p: square.gb(p, jax_avail=False, torch_avail=True))
    slab_solver = solverWrap.solverWrapper(opts)
    slab_solver.construct(geom, diff_op, verbose=False)
    Il, Ir, _, Igb, _, XXb = slab_i.compute_idxs_and_pts(slab_solver)
    return {
        'geom': geom,
        'solver': slab_solver,
        'Il': Il,
        'Ir': Ir,
        'Igb': Igb,
        'XXb': XXb,
    }


def build_macro_t_dtn_system(N, H, opts, diff_op, bc_func, s_coords, eta=None):
    coord_to_ind = {coord_key(x): i for i, x in enumerate(s_coords)}
    ndofs = s_coords.shape[0]
    rhs = np.zeros((ndofs,), dtype=np.complex128)
    slab_data = []
    blocks = []
    nnz = 0

    for slab_ind in range(N):
        data = build_unit_slab_solver(slab_ind, H, opts, diff_op)
        slab_data.append(data)
        XXb = data['XXb']
        Igb = data['Igb']
        faces = []
        if slab_ind > 0:
            faces.append(data['Il'])
        if slab_ind < N - 1:
            faces.append(data['Ir'])

        bdry_vals = to_numpy_vector(bc_func(XXb[Igb])).astype(np.complex128, copy=False)
        for row_face in faces:
            row_inds = face_global_indices(XXb, row_face, coord_to_ind)
            for col_face in faces:
                col_inds = face_global_indices(XXb, col_face, coord_to_ind)
                block = local_dtn_boundary_block(
                    data['solver'],
                    row_face,
                    col_face,
                )
                blocks.append((row_inds, col_inds, block))
                nnz += block.size
            rhs[row_inds] -= local_dtn_boundary_block(data['solver'], row_face, Igb) @ bdry_vals

    builder = sparse_utils.CSRBuilder(ndofs, ndofs, max(nnz, 1), dtype=np.complex128)
    for row_inds, col_inds, block in blocks:
        add_dense_block(builder, row_inds, col_inds, block, (ndofs, ndofs))
    Tmacro = builder.tocsr()

    return {
        'label': 'T-DtN',
        'kind': 'dtn',
        'A': Tmacro,
        'rhs': rhs,
        's_coords': s_coords,
        'slab_data': slab_data,
    }


def check_iti_cayley_transform(Tcc, Tcx, Rloc, bloc, bdry_vals, eta, seed):
    rng = np.random.default_rng(seed)
    u_probe = rng.standard_normal(Tcc.shape[0]) + 1j * rng.standard_normal(Tcc.shape[0])
    flux = Tcc @ u_probe + Tcx @ bdry_vals
    incoming = 1j * eta * u_probe - flux
    outgoing = 1j * eta * u_probe + flux
    outgoing_from_iti = Rloc @ incoming + bloc
    denom = max(np.linalg.norm(outgoing), 1.0)
    return np.linalg.norm(outgoing_from_iti - outgoing) / denom


def build_macro_t_iti_directed_data(N, H, opts, diff_op, bc_func, s_coords, eta):
    if eta == 0:
        raise ValueError("T-ItI is Helmholtz-only and needs nonzero impedance eta.")

    check_cayley = os.environ.get("SSLABLU_CHECK_ITI_CAYLEY", "0") != "0"
    cayley_tol = float(os.environ.get("SSLABLU_CHECK_ITI_CAYLEY_TOL", "1e-8"))
    cayley_errs = []
    coord_to_ind = {coord_key(x): i for i, x in enumerate(s_coords)}
    ndofs = s_coords.shape[0]
    ndirected = 2 * ndofs
    copy1 = np.arange(ndofs)
    copy2 = ndofs + np.arange(ndofs)
    bdir = np.zeros((ndirected,), dtype=np.complex128)
    slab_data = []
    blocks = []
    nnz = 0

    for slab_ind in range(N):
        data = build_unit_slab_solver(slab_ind, H, opts, diff_op)
        XXb = data['XXb']
        Igb = data['Igb']
        active_faces = []
        active_dir_inds = []

        if slab_ind > 0:
            face = data['Il']
            active_faces.append(face)
            active_dir_inds.append(ndofs + face_global_indices(XXb, face, coord_to_ind))
        if slab_ind < N - 1:
            face = data['Ir']
            active_faces.append(face)
            active_dir_inds.append(face_global_indices(XXb, face, coord_to_ind))

        active_face = np.concatenate(active_faces)
        dir_inds = np.concatenate(active_dir_inds)
        Tcc = local_dtn_boundary_block(data['solver'], active_face, active_face)
        Tcx = local_dtn_boundary_block(data['solver'], active_face, Igb)
        bdry_vals = to_numpy_vector(bc_func(XXb[Igb])).astype(np.complex128, copy=False)

        eye = np.eye(active_face.shape[0], dtype=np.complex128)
        M_inv = np.linalg.inv(1j * eta * eye - Tcc)
        Rloc = (Tcc + 1j * eta * eye) @ M_inv
        bloc = 2j * eta * (M_inv @ (Tcx @ bdry_vals))
        if check_cayley:
            cayley_errs.append(
                check_iti_cayley_transform(
                    Tcc,
                    Tcx,
                    Rloc,
                    bloc,
                    bdry_vals,
                    eta,
                    seed=7919 + slab_ind,
                )
            )

        blocks.append((dir_inds, dir_inds, Rloc))
        nnz += Rloc.size
        bdir[dir_inds] += bloc
        data.update(
            {
                'active_face': active_face,
                'dir_inds': dir_inds,
                'M_inv': M_inv,
                'Tcx': Tcx,
            }
        )
        slab_data.append(data)

    builder = sparse_utils.CSRBuilder(ndirected, ndirected, max(nnz, 1), dtype=np.complex128)
    for row_inds, col_inds, block in blocks:
        add_dense_block(builder, row_inds, col_inds, block, (ndirected, ndirected))
    Rdir = builder.tocsr()
    if check_cayley:
        max_err = max(cayley_errs) if cayley_errs else 0.0
        print("SANITY local ItI Cayley-vs-DtN max relerr = %5.5e" % max_err)
        if max_err > cayley_tol:
            raise AssertionError(
                "local ItI Cayley transform check failed: relerr=%5.5e > tol=%5.5e" % (max_err, cayley_tol)
            )

    return {
        'Rdir': Rdir,
        'bdir': bdir,
        'ndofs': ndofs,
        'ndirected': ndirected,
        'copy1': copy1,
        'copy2': copy2,
        's_coords': s_coords,
        'slab_data': slab_data,
        'eta': eta,
    }


def build_macro_t_iti_schur_system(N, H, opts, diff_op, bc_func, s_coords, eta):
    data = build_macro_t_iti_directed_data(N, H, opts, diff_op, bc_func, s_coords, eta)
    Rdir = data['Rdir'].toarray()
    bdir = data['bdir']
    ndofs = data['ndofs']
    copy1 = data['copy1']
    copy2 = data['copy2']

    R11 = Rdir[np.ix_(copy1, copy1)]
    R12 = Rdir[np.ix_(copy1, copy2)]
    R21 = Rdir[np.ix_(copy2, copy1)]
    R22 = Rdir[np.ix_(copy2, copy2)]
    b1 = bdir[copy1]
    b2 = bdir[copy2]

    # The local ItI maps are side-local.  The interface merge is the Schur
    # complement that enforces incoming data on one side to equal outgoing
    # data from the neighboring side, leaving one unknown per physical trace.
    K = np.eye(ndofs, dtype=np.complex128) - R12
    Kinv_R11 = np.linalg.solve(K, R11)
    Kinv_b1 = np.linalg.solve(K, b1)
    Amacro = np.eye(ndofs, dtype=np.complex128) - R21 - R22 @ Kinv_R11
    rhs = b2 + R22 @ Kinv_b1

    data.update(
        {
            'label': 'T-ItI-SC',
            'kind': 'iti_schur',
            'A': Amacro,
            'rhs': rhs,
            'R11': R11,
            'K': K,
            'b1': b1,
        }
    )
    return data


def apply_block_left_precondition_to_vector(Tmacro, s_coords, rhs):
    out = np.zeros_like(rhs, dtype=np.result_type(rhs.dtype, Tmacro.dtype, np.complex128))
    for xval in np.unique(np.round(s_coords[:, 0], 12)):
        block = np.where(np.abs(s_coords[:, 0] - xval) < 1e-12)[0]
        Tblock = Tmacro[block, :][:, block].toarray()
        out[block] = np.linalg.solve(Tblock, rhs[block])
    return out


def check_preconditioned_t_matches_s(Sop, dslabs, N, H, opts, diff_op, tol=1e-8):
    s_coords = get_s_target_coords(dslabs, opts, diff_op)
    zero_bc = lambda points: np.zeros(points.shape[0], dtype=np.complex128)
    Tmacro = build_macro_t_dtn_system(N, H, opts, diff_op, zero_bc, s_coords)['A']
    rng = np.random.default_rng(2601)
    # The SlabLU S linear operator currently applies real-valued probes.
    probe = rng.standard_normal(Tmacro.shape[1])
    S_probe = Sop @ probe
    T_probe = Tmacro @ probe
    Tprec_probe = apply_block_left_precondition_to_vector(Tmacro, s_coords, T_probe)
    probe_norm = np.linalg.norm(probe)
    relerr = np.linalg.norm(Tprec_probe - S_probe) / max(np.linalg.norm(S_probe), 1.0)
    print("SANITY random-probe preconditioned macro T-DtN vs S relerr = %5.5e" % relerr)
    if probe_norm < 1e-12:
        raise AssertionError("preconditioned macro T-DtN vs S check used a near-zero probe.")
    if relerr > tol:
        raise AssertionError(
            "preconditioned macro T-DtN did not match S: relerr=%5.5e > tol=%5.5e" % (relerr, tol)
        )


def expand_macro_t_iti_solution(system, sol):
    g2 = np.linalg.solve(system['K'], system['R11'] @ sol + system['b1'])
    gdir = np.zeros((2 * sol.shape[0],), dtype=np.complex128)
    gdir[system['copy1']] = sol
    gdir[system['copy2']] = g2
    return gdir


def macro_t_slab_boundary(system, data, sol, bc_func):
    XXb = data['XXb']
    Igb = data['Igb']
    bdry = np.zeros((XXb.shape[0],), dtype=np.complex128)
    bdry[Igb] = to_numpy_vector(bc_func(XXb[Igb])).astype(np.complex128, copy=False)

    if system['kind'] == 'dtn':
        coord_to_ind = {coord_key(x): i for i, x in enumerate(system['s_coords'])}
        if len(data['Il']) > 0:
            inds = face_global_indices(XXb, data['Il'], coord_to_ind)
            bdry[data['Il']] = sol[inds]
        if len(data['Ir']) > 0:
            inds = face_global_indices(XXb, data['Ir'], coord_to_ind)
            bdry[data['Ir']] = sol[inds]
        return bdry

    gdir = system.get('gdir')
    if gdir is None:
        gdir = expand_macro_t_iti_solution(system, sol)
        system['gdir'] = gdir
    bdry_vals = bdry[Igb]
    g_active = gdir[data['dir_inds']]
    # The macro ItI unknown is incoming impedance data.  This converts that
    # trace back to Dirichlet values before reusing the existing local solve.
    bdry[data['active_face']] = data['M_inv'] @ (g_active + data['Tcx'] @ bdry_vals)
    return bdry


def solve_macro_t_slabs(system, sol, bc_func):
    slab_solutions = []
    for data in system['slab_data']:
        bdry = macro_t_slab_boundary(system, data, sol, bc_func)
        bdry = bdry[:, np.newaxis]
        if torch_avail:
            uu = data['solver'].solver.solve_dir_full(torch.from_numpy(bdry))
        else:
            uu = data['solver'].solver.solve_dir_full(bdry)
        slab_solutions.append((data, to_numpy_vector(uu)))
    return slab_solutions


def known_solution_error_from_t_system(system, exact_func, label):
    sol = system['sol']
    err_num = 0.0
    err_den = 0.0
    for data, uu in solve_macro_t_slabs(system, sol, exact_func):
        uref = to_numpy_vector(exact_func(data['solver'].XXfull))
        err_num += np.linalg.norm(uref - uu, ord=2) ** 2
        err_den += np.linalg.norm(uref, ord=2) ** 2

    relerr = np.sqrt(err_num / err_den)
    print("%s known error = %5.5e" % (label, relerr))
    return relerr


def reference_grid_error_from_t_system(system, bc_func, points, ref_values, label):
    sol = system['sol']
    vals = np.zeros((points.shape[0],), dtype=np.result_type(ref_values, np.complex128))
    for data, uu in solve_macro_t_slabs(system, sol, bc_func):
        geom = data['geom']
        I0 = np.where(
            (points[:, 0] >= geom[0, 0])
            & (points[:, 0] <= geom[1, 0])
            & (points[:, 1] >= geom[0, 1])
            & (points[:, 1] <= geom[1, 1])
        )[0]
        vals[I0] = data['solver'].interp(points[I0], uu)

    relerr = np.linalg.norm(ref_values - vals, ord=2) / np.linalg.norm(ref_values, ord=2)
    print("%s reference-grid error = %5.5e" % (label, relerr))
    return relerr


def shared_interface_values_from_s_solution(dslabs, opts, diff_op, uhat):
    s_coords = get_s_target_coords(dslabs, opts, diff_op)
    coord_to_vals = {}
    for coord, val in zip(s_coords, uhat):
        coord_to_vals.setdefault(coord_key(coord), []).append(val)

    points = []
    values = []
    spreads = []
    for key, vals in coord_to_vals.items():
        vals = np.asarray(vals)
        avg = np.mean(vals)
        points.append(key)
        values.append(avg)
        spreads.append(np.max(np.abs(vals - avg)))

    return (
        np.asarray(points, dtype=float),
        np.asarray(values, dtype=np.complex128),
        max(spreads) if spreads else 0.0,
    )


def maybe_shared_interface_validation(dslabs, opts, diff_op, uhat):
    if os.environ.get("SSLABLU_SHARED_POINT_VALIDATION", "0") == "0":
        return

    points, values, max_spread = shared_interface_values_from_s_solution(
        dslabs,
        opts,
        diff_op,
        uhat,
    )
    print("================SHARED INTERFACE ERR=================")
    print("shared interface points = %d" % points.shape[0])
    print("shared duplicate max spread = %5.5e" % max_spread)

    save_path = os.environ.get("SSLABLU_SAVE_SHARED_REF_PATH")
    if save_path:
        np.savez(save_path, points=points, values=values)
        print("saved shared interface reference to %s" % save_path)

    ref_path_shared = os.environ.get("SSLABLU_SHARED_REF_PATH")
    if ref_path_shared:
        ref = np.load(ref_path_shared)
        ref_points = ref["points"]
        ref_values = ref["values"]
        ref_map = {coord_key(point): val for point, val in zip(ref_points, ref_values)}
        val_map = {coord_key(point): val for point, val in zip(points, values)}
        shared_keys = sorted(set(ref_map).intersection(val_map))
        if shared_keys:
            ref_shared = np.asarray([ref_map[key] for key in shared_keys])
            val_shared = np.asarray([val_map[key] for key in shared_keys])
            print("exact shared comparison points = %d" % len(shared_keys))
        else:
            ref_shared = interpolate_interface_reference(ref_points, ref_values, points)
            val_shared = values
            print("exact shared comparison points = 0")
            print("interface-interpolated comparison points = %d" % points.shape[0])
        relerr = np.linalg.norm(ref_shared - val_shared, ord=2) / np.linalg.norm(ref_shared, ord=2)
        print("S interface-trace error = %5.5e" % relerr)
    print("=====================================================")


def interpolate_interface_reference(ref_points, ref_values, points):
    # Refinement changes the Chebyshev trace nodes, so exact shared interface
    # nodes are usually empty.  Compare only along each vertical interface,
    # using a 1D spline so the validation is not dominated by linear
    # interpolation error.
    ref_groups = {}
    for point, value in zip(ref_points, ref_values):
        ref_groups.setdefault(round(float(point[0]), 12), []).append((float(point[1]), value))

    target_groups = {}
    for ind, point in enumerate(points):
        target_groups.setdefault(round(float(point[0]), 12), []).append((ind, float(point[1])))

    out = np.zeros((points.shape[0],), dtype=np.complex128)
    for xkey, targets in target_groups.items():
        group = ref_groups.get(xkey)
        if group is None:
            raise ValueError("missing reference interface x=%s" % xkey)
        group = sorted(group, key=lambda item: item[0])
        y_ref = np.asarray([item[0] for item in group])
        v_ref = np.asarray([item[1] for item in group])
        inds = np.asarray([item[0] for item in targets], dtype=int)
        y = np.asarray([item[1] for item in targets])
        if y_ref.shape[0] >= 4:
            real_interp = scipy.interpolate.CubicSpline(y_ref, v_ref.real)
            imag_interp = scipy.interpolate.CubicSpline(y_ref, v_ref.imag)
            out[inds] = real_interp(y) + 1j * imag_interp(y)
        else:
            out[inds] = np.interp(y, y_ref, v_ref.real) + 1j * np.interp(y, y_ref, v_ref.imag)
    return out

jax_avail=False
torch_avail=True
if jax_avail:
    import jax.numpy as jnp

    def bfield(xx):
        
        mag   = 0.930655
        width = 2500; 
        
        b = jnp.zeros_like(xx[...,0])
        
        dist = 0.04
        x0=0.1+0.5*dist; x1 = 0.50; x2 = x1+2.5*dist; x3= 0.9
        y0=0.1+0.5*dist; y1 = 0.50; y2 = y1+2.5*dist; y3= 0.9
        
        # box of points [x0,x1] x [y0,y1]
        for x in np.arange(x0,x1,dist):
            for y in np.arange(y0,y1,dist):
                xx_sq_c = (xx[...,0] - x)**2 + (xx[...,1] - y)**2
                b += mag * jnp.exp(-width * xx_sq_c)

        # box of points [x0,x1] x [y0,y2]
        for x in np.arange(x2,x3,dist):
            for y in np.arange(y0,y2-0.5*dist,dist):
                xx_sq_c = (xx[...,0] - x)**2 + (xx[...,1] - y)**2
                b += mag * jnp.exp(-width * xx_sq_c)
                
        # box of points [x0,x3] x [y2,y3]
        for x in np.arange(x0,x3,dist):
            for y in np.arange(y2,y3,dist):
                xx_sq_c = (xx[...,0] - x)**2 + (xx[...,1] - y)**2
                b += mag * jnp.exp(-width * xx_sq_c)    
        
        kh_fun = -kh**2 * (1 - b)
        return kh_fun


    def c11(p):
        return jnp.ones_like(p[...,0])
    def c22(p):
        return jnp.ones_like(p[...,0])
    def c(p):
        if pde_mode == "constant":
            return -kh**2 * jnp.ones_like(p[...,0])
        return bfield(p)
    Lapl=pdo.PDO2d(c11,c22,None,None,None,c)

elif torch_avail:
    def bfield(xx):
        
        mag   = 0.930655
        width = 2500; 
        
        b = torch.zeros_like(xx[:,0])
        
        dist = 0.04
        x0=0.1+0.5*dist; x1 = 0.50; x2 = x1+2.5*dist; x3= 0.9
        y0=0.1+0.5*dist; y1 = 0.50; y2 = y1+2.5*dist; y3= 0.9
        
        # box of points [x0,x1] x [y0,y1]
        for x in np.arange(x0,x1,dist):
            for y in np.arange(y0,y1,dist):
                xx_sq_c = (xx[:,0] - x)**2 + (xx[:,1] - y)**2
                b += mag * torch.exp(-width * xx_sq_c)

        # box of points [x0,x1] x [y0,y2]
        for x in np.arange(x2,x3,dist):
            for y in np.arange(y0,y2-0.5*dist,dist):
                xx_sq_c = (xx[:,0] - x)**2 + (xx[:,1] - y)**2
                b += mag * torch.exp(-width * xx_sq_c)
                
        # box of points [x0,x3] x [y2,y3]
        for x in np.arange(x0,x3,dist):
            for y in np.arange(y2,y3,dist):
                xx_sq_c = (xx[:,0] - x)**2 + (xx[:,1] - y)**2
                b += mag * torch.exp(-width * xx_sq_c)    
        
        kh_fun = -kh**2 * (1 - b)
        return kh_fun


    def c11(p):
        return torch.ones_like(p[:,0])
    def c22(p):
        return torch.ones_like(p[:,0])
    def c(p):
        if pde_mode == "constant":
            return -kh**2 * torch.ones_like(p[:,0])
        return bfield(p)
    Lapl=pdo.PDO_2d(c11=c11,c22=c22,c=c)

else:
    def bfield(xx):
        
        mag   = 0.930655
        width = 2500; 
        
        b = np.zeros(shape = (xx.shape[0],))
        
        dist = 0.04
        x0=0.1+0.5*dist; x1 = 0.50; x2 = x1+2.5*dist; x3= 0.9
        y0=0.1+0.5*dist; y1 = 0.50; y2 = y1+2.5*dist; y3= 0.9
        
        # box of points [x0,x1] x [y0,y1]
        for x in np.arange(x0,x1,dist):
            for y in np.arange(y0,y1,dist):
                xx_sq_c = (xx[:,0] - x)**2 + (xx[:,1] - y)**2
                b += mag * np.exp(-width * xx_sq_c)

        # box of points [x0,x1] x [y0,y2]
        for x in np.arange(x2,x3,dist):
            for y in np.arange(y0,y2-0.5*dist,dist):
                xx_sq_c = (xx[:,0] - x)**2 + (xx[:,1] - y)**2
                b += mag * np.exp(-width * xx_sq_c)
                
        # box of points [x0,x3] x [y2,y3]
        for x in np.arange(x0,x3,dist):
            for y in np.arange(y2,y3,dist):
                xx_sq_c = (xx[:,0] - x)**2 + (xx[:,1] - y)**2
                b += mag * np.exp(-width * xx_sq_c)    
        
        kh_fun = -kh**2 * (1 - b)
        return kh_fun


    def c11(p):
        return np.ones_like(p[:,0])
    def c22(p):
        return np.ones_like(p[:,0])
    def c(p):
        if pde_mode == "constant":
            return -kh**2 * np.ones_like(p[:,0])
        return bfield(p)
    Lapl=pdo.PDO2d(c11,c22,None,None,None,c)


def bc(p):
    if bc_mode == "known":
        return constant_helmholtz_exact(p)
    if torch.is_tensor(p):
        return torch.ones_like(p[:,0])
    return np.ones_like(p[:,0])


# Validation uses direct solves; GMRES probes below are diagnostics only.
run_validation = os.environ.get("SSLABLU_RUN_VALIDATION", "0") == "1"
cond_nit = int(os.environ.get("SSLABLU_COND_NIT", "20"))
dbg = int(os.environ.get("SSLABLU_DBG", "2"))
run_gmres_probe = os.environ.get("SSLABLU_RUN_GMRES", "0") != "0"
run_s_gmres_probe = os.environ.get("SSLABLU_RUN_S_GMRES", "1" if run_gmres_probe else "0") != "0"
run_t_gmres_probe = os.environ.get("SSLABLU_RUN_T_GMRES", "1" if run_gmres_probe else "0") != "0"
gmres_rtol = float(os.environ.get("SSLABLU_GMRES_RTOL", "1e-8"))
gmres_restart = int(os.environ.get("SSLABLU_GMRES_RESTART", "200"))
gmres_maxiter = int(os.environ.get("SSLABLU_GMRES_MAXITER", "500"))
validation_grid_n = int(os.environ.get("SSLABLU_VALIDATION_GRID_N", "200"))
ref_path = os.environ.get("SSLABLU_REF_PATH", "ref_sol_waveguide.npy")
save_ref_path = os.environ.get("SSLABLU_SAVE_REF_PATH")

N = int(os.environ.get("SSLABLU_N", "8"))
dSlabs,connectivity,H = square.dSlabs(N)
print(connectivity)
pvec = np.array([int(os.environ.get("SSLABLU_P", "30"))],dtype = np.int64)
npan_x = int(os.environ.get("SSLABLU_NPAN_X", "8"))
npan_y = int(os.environ.get("SSLABLU_NPAN_Y", "16"))
#pvec = np.array([8,10,12,14,16,18,20],dtype = np.int64)
err=np.zeros(shape = (len(pvec),))
discr_time=np.zeros(shape = (len(pvec),))
compr_time=np.zeros(shape = (len(pvec),))
sampl_time=np.zeros(shape = (len(pvec),))
for indp in range(len(pvec)):
    p = pvec[indp]
    formulation = "hpsalt"
    p_disc = p + 2 # To handle different conventions between hps and hpsalt
    a = np.array([H/npan_x,0.5/npan_y])
    assembler = mA.denseMatAssembler()
    opts = solverWrap.solverOptions(formulation,[p_disc,p_disc],a)
    OMS = oms.oms(dSlabs,Lapl,lambda p :square.gb(p,jax_avail=jax_avail,torch_avail=torch_avail),opts,connectivity)
    print("computing S blocks & rhs's...")
    (S_rk_list, rhs_list, Ntot, nc), _ = timed_step(
        "S block/rhs assembly",
        lambda: OMS.construct_Stot_helper(bc, assembler, dbg=dbg),
    )
    print("done")
    (Stot,rhstot), _ = timed_step(
        "S linear operator assembly",
        lambda: OMS.construct_Stot_and_rhstot_linearOperator(S_rk_list,rhs_list,Ntot,nc,dbg=dbg),
    )
    rhs_dtype = np.result_type(*[rhs.dtype for rhs in rhs_list])
    rhstot = np.zeros(shape = (Ntot,), dtype=rhs_dtype)
    for i in range(len(rhs_list)):
        rhstot[i*nc:(i+1)*nc] = rhs_list[i]
    block_dtypes = [getattr(block, "dtype", np.asarray(block).dtype) for row in S_rk_list for block in row]
    s_dtype = np.result_type(rhs_dtype, *block_dtypes)
    Ssparse, _ = timed_step("S sparse CSR assembly", lambda: build_s_sparse_matrix(OMS, S_rk_list, Ntot, s_dtype))
    if os.environ.get("SSLABLU_CHECK_T_PRECOND_EQUALS_S", "0") != "0":
        check_preconditioned_t_matches_s(
            Stot,
            dSlabs,
            N,
            H,
            opts,
            Lapl,
            tol=float(os.environ.get("SSLABLU_CHECK_T_PRECOND_TOL", "1e-8")),
        )
    Sop = aslinearoperator(Ssparse)
    Sinv, s_solver_toc, s_solver_backend = sparse_direct_inverse_operator(Ssparse, "S")
    uhat, _ = timed_step("S direct solve", lambda: Sinv.matvec(rhstot))
    (s_norm, s_inv_norm, s_cond), _ = timed_step(
        "S condition estimate",
        lambda: sparse_utils.estimate_condition_number(Sop, Sinv, nit=cond_nit, seed=101),
    )
    s_eff_cond = sparse_utils.estimate_effective_condition_number(
        Sop,
        rhstot,
        solution=uhat,
        op_norm=s_norm,
    )
    if run_s_gmres_probe:
        (s_gmres_niter, s_gmres_info, s_gmres_relres), _ = timed_step(
            "S block GMRES probe",
            lambda: gmres_probe(Stot, rhstot, rtol=gmres_rtol, restart=gmres_restart, maxiter=gmres_maxiter),
        )
    else:
        s_gmres_niter, s_gmres_info, s_gmres_relres = -1, -1, np.nan

    if os.environ.get("SSLABLU_SHARED_VALIDATION_ONLY", "0") != "0":
        maybe_shared_interface_validation(dSlabs, opts, Lapl, uhat)
        raise SystemExit(0)

    s_coords = get_s_target_coords(dSlabs, opts, Lapl)
    stats_t_dtn = condition_t_formulation(Lapl, kh, N, H, opts, formulation='dtn', bc_func=bc, s_coords=s_coords, seed=202)

    if kh != 0:
        stats_t_iti_sc = condition_t_formulation(Lapl, kh, N, H, opts, formulation='iti_schur', bc_func=bc, s_coords=s_coords, seed=303)
    else:
        stats_t_iti_sc = None

    if stats_t_dtn['shape'] != Ssparse.shape:
        raise AssertionError("macro T-DtN size %s does not match S size %s" % (stats_t_dtn['shape'], Ssparse.shape))
    if stats_t_iti_sc is not None and stats_t_iti_sc['shape'] != Ssparse.shape:
        raise AssertionError("macro T-ItI-SC size %s does not match S size %s" % (stats_t_iti_sc['shape'], Ssparse.shape))

    print("=============REDUCED SYSTEM CONDITIONING==============")
    print("kh              = %5.12e" % kh)
    print("PDE mode        = %s" % pde_mode)
    print("BC mode         = %s" % bc_mode)
    print("Validation      = %s" % run_validation)
    print("GMRES settings  = rtol=%5.2e, restart=%d, maxiter=%d" % (gmres_rtol, gmres_restart, gmres_maxiter))
    print("GMRES probes    = S:%s T:%s" % (run_s_gmres_probe, run_t_gmres_probe))
    print("condest(S)      = %5.5e   effcond=%5.5e   gmres=(%d,%d,%5.2e)   [||S||=%5.5e, ||S^-1||=%5.5e, backend=%s, size=%s]" % (s_cond, s_eff_cond, s_gmres_niter, s_gmres_info, s_gmres_relres, s_norm, s_inv_norm, s_solver_backend, Ssparse.shape))
    print_t_conditioning("T-DtN", stats_t_dtn)
    if stats_t_iti_sc is not None:
        print_t_conditioning("T-ItI-SC", stats_t_iti_sc)
    else:
        print("condest(T-ItI)  = skipped for kh=0 because ItI maps are Helmholtz-only in Domain_Driver")
    print("======================================================")
    
    res = Stot@uhat-rhstot

    
    print("=============SUMMARY==============")
    print("H                        = ",'%10.3E'%H)
    print("N                        = ",N)
    print("kh                       = ",'%10.3E'%kh)
    print("ord                      = ",p)
    print("npan_dim                 = ",(int)(H/a[0]),',',(int)(.5/a[1]))
    print("nc                       = ",OMS.nc)
    print("L2 rel. res              = ", np.linalg.norm(res)/np.linalg.norm(rhstot))
    print("==================================")

    if not run_validation:
        maybe_shared_interface_validation(dSlabs, opts, Lapl, uhat)
        print("Validation skipped. Set SSLABLU_RUN_VALIDATION=1 to run known/reference checks.")
        raise SystemExit(0)

    nc = OMS.nc

    if pde_mode == "constant" and bc_mode == "known":
        err_num = 0.0
        err_den = 0.0
        for slabInd in range(len(dSlabs)):
            geom = np.array(dSlabs[slabInd])
            slab_i = oms.slab(geom,lambda p : square.gb(p,jax_avail,torch_avail))
            solver = oms.solverWrap.solverWrapper(opts)
            solver.construct(geom,Lapl)
            Il,Ir,Ic,Igb,XXi,XXb = slab_i.compute_idxs_and_pts(solver)
            startL = slabInd-1
            startR = slabInd+1
            g = np.zeros(shape=(XXb.shape[0],), dtype=uhat.dtype)
            g[Igb] = bc(XXb[Igb,:])
            if startL>-1:
                g[Il] = uhat[startL*nc:(startL+1)*nc]
            if startR<len(dSlabs):
                g[Ir] = uhat[startR*nc:(startR+1)*nc]
            g = g[:,np.newaxis]
            if torch_avail:
                uu = solver.solver.solve_dir_full(torch.from_numpy(g))
            else:
                uu = solver.solver.solve_dir_full(g)
            uu = to_numpy_vector(uu)
            uref = to_numpy_vector(constant_helmholtz_exact(solver.XXfull))
            err_num += np.linalg.norm(uref - uu, ord=2)**2
            err_den += np.linalg.norm(uref, ord=2)**2

        ref_err = np.sqrt(err_num / err_den)
        print("===================KNOWN SOLUTION ERR===================")
        print("err known = ", ref_err)
        known_solution_error_from_t_system(stats_t_dtn['system'], constant_helmholtz_exact, "T-DtN")
        if kh != 0:
            known_solution_error_from_t_system(stats_t_iti_sc['system'], constant_helmholtz_exact, "T-ItI-SC")
        print("========================================================")
        err[indp] = ref_err
        sampl_time[indp] = OMS.stats.sampl_timing
        compr_time[indp] = OMS.stats.compr_timing
        discr_time[indp] = OMS.stats.discr_timing
        continue


    nx=validation_grid_n
    ny=validation_grid_n

    xpts = np.linspace(0,1,nx)
    ypts = np.linspace(0,1,ny)

    YY = np.zeros(shape=(nx*ny,2))
    YY[:,0] = np.kron(xpts,np.ones_like(ypts))
    YY[:,1] = np.kron(np.ones_like(xpts),ypts)

    gYY = np.zeros(shape=(YY.shape[0],))

    try:
        for slabInd in range(len(dSlabs)):
            geom    = np.array(dSlabs[slabInd])
            I0 = np.where(  (YY[:,0]>=geom[0,0]) & (YY[:,0]<=geom[1,0]) & (YY[:,1]>=geom[0,1]) & (YY[:,1]<=geom[1,1]))[0]
            YY0 = YY[I0,:]
            slab_i  = oms.slab(geom,lambda p : square.gb(p,jax_avail,torch_avail))
            solver  = oms.solverWrap.solverWrapper(opts)
            solver.construct(geom,Lapl)
            Il,Ir,Ic,Igb,XXi,XXb = slab_i.compute_idxs_and_pts(solver)
            startL = slabInd-1
            startR = slabInd+1
            g = np.zeros(shape=(XXb.shape[0],))
            g[Igb] = bc(XXb[Igb,:])
            if startL>-1:
                g[Il] = uhat[startL*nc:(startL+1)*nc]
            if startR<len(dSlabs):
                g[Ir] = uhat[startR*nc:(startR+1)*nc]
            g=g[:,np.newaxis]
            if torch_avail:
                uu = solver.solver.solve_dir_full(torch.from_numpy(g))
            else:
                uu = solver.solver.solve_dir_full(g)
            uu=uu.numpy().flatten()
            ghat = solver.interp(YY0,uu)
            gYY[I0] = ghat

        if save_ref_path:
            np.save(save_ref_path, gYY)
            print("saved reference solution to %s" % save_ref_path)
        gref = gYY if save_ref_path else np.load(ref_path)

        print("===================REF SUP ERR===================")
        s_ref_err = np.linalg.norm(gref-gYY,ord = 2)/np.linalg.norm(gref,ord = 2)
        print("S reference-grid error = ",s_ref_err)
        maybe_shared_interface_validation(dSlabs, opts, Lapl, uhat)
        t_dtn_ref_err = reference_grid_error_from_t_system(stats_t_dtn['system'], bc, YY, gref, "T-DtN")
        if kh != 0:
            t_iti_sc_ref_err = reference_grid_error_from_t_system(stats_t_iti_sc['system'], bc, YY, gref, "T-ItI-SC")
        print("=================================================")
        err[indp] = s_ref_err
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        print("===================REF SUP ERR===================")
        print("reference validation skipped: %s" % exc)
        print("=================================================")
        err[indp] = np.nan
    sampl_time[indp] = OMS.stats.sampl_timing
    compr_time[indp] = OMS.stats.compr_timing
    discr_time[indp] = OMS.stats.discr_timing


fileName = 'crystal_waveguide.csv'
errMat = np.zeros(shape=(len(pvec),5))
errMat[:,0] = pvec
errMat[:,1] = err
errMat[:,2] = sampl_time
errMat[:,3] = compr_time
errMat[:,4] = discr_time
try:
    with open(fileName,'w') as f:
        f.write('p,err,sample,compr,discr\n')
        np.savetxt(f,errMat,fmt='%.16e',delimiter=',')
except PermissionError as exc:
    print("Skipping CSV write: %s" % exc)
