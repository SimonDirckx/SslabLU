# Handoff: nonlinear Poisson–Boltzmann on the twisted elliptic torus (OMS / five-patch HPS)

This brief condenses a long design conversation. The companion write-up
`pb_fixed_operator.tex` / `.pdf` has the full math; section numbers below refer to it.
**The task: implement the single-slab first pass described in §6.** §1–5 are the
context you need to do that without re-deriving it.

---

## 0. Working rules (from the user — follow them)

- **Do not run tests yourself unless asked.** The user runs experiments on their own
  machine (CPU and GPU). Deliver code, a short note on what to run, and what output to expect.
- **Prefer minimal edits** to existing files. Match the existing code style and
  structure. New geometry follows the *cube / twisted-square-torus* design: forward and
  inverse maps with analytic first and second derivatives (`z`, `y_k_dj`, `y_k_djdj`),
  `gb`, `box_geom`, `bounds`, `param_geom(hpsalt=True)`, `dSlabs`, `check_param`. The
  PDO comes from `param_geom.transform_helmholtz_pdo(...)` or an analogous transform.
- **`max_level`:** the user sets
  `max_level = int(np.log2(n_core**2)) - 2` (for example 10 at n=32, giving 1024 tree
  leaves). **Never assume another value** (for example 5). Read it from the script.
  Full tree depth is `log2(2 n^2) = 1 + 2 log2 n`, and
  `max_level = int(log2(2 n^2)) - k` gives 2^k HPS faces per HBS leaf.
- Communicate results concisely. No recaps of steps.

## 1. Codebase map (user's repo `SslabLU`; confirm exact paths on disk)

| File | Role |
|---|---|
| `solver/cylinder_slab.py` | `CylinderSlab(geom, pde, p, Rc, s, n, m, nz=2, flux, device, glue_tol)`: five-patch HPS double slab. Requires nz even. Builds patches via `make_cylinder` (box[2]=[z0,z1]), `build_patch` per patch, glue. Canonical point order `np.lexsort((z,y,x))`; `XX = Xcyl[:,[2,0,1]]`; `Ii = counts==2`, `Ib = counts==1`; exposes `Aii` (csc), `Aib`, `Abi`, `Abb`. `pdo3d_to_cyl(pdo)` converts a PDO_3d in OMS coordinates w=(z,x,y) to PDOPhys (a_kk=c_kk, a_kl=c_kl/2, b_k=c_k, permuted by `_W_OF_CYL=[1,2,0]`, cached). **Leaf-level operators are discarded after gluing.** |
| `solver/solver.py` | `solverOptions(..., cyl=None)`; `'cylhps'` branch builds `CylinderSlab` and factors Aii via `_factor_mumps`. GPU: `cyl.setdefault('device', cuda if opts.reduced_gpu else cpu)`. |
| `stellerator.py` | `Patch.pdo` closures moved to `xx.device` (GPU routing). |
| `oms.py`, `oms_input.py` | OMS assembly; `OMSInputMixin` chooses the red–black tree: a `tree=` override, else the HBS block tree, else `_geometric_tree`. |
| `omsdirectsolveHBS_torch_new2.py` | Red–black (cyclic) solver/preconditioner on HBS-compressed interface blocks (HBSMAT from HBStorch_new: complete binary tree, equal block sizes per level, one host master buffer with views). Verified: the custom tree is not overwritten anywhere. |
| `matAssembler.py` | Accepts `tree=` (a factory or an object). |
| `circInterfaceTree.py` | Tree for the disk interface: pie levels separate the core and four arches (needs n = 4m, m a power of 2), then index-space binary dissection (`_dissect_axis`: halve the side with more leaves, alternate on ties). |
| `geometry/geom_3D/full_twisted_torus.py` (user's), `half_twisted_torus.py` | Twisted elliptic torus. R0=3, a_e=1.2, b_e=0.75, integer twist, Rc=√(a_e b_e), periodic L=2πR0 (full) or πR0 (half), core half-width `s` (use **s≈0.55**; s=2/3 gives thin corner arches). |
| `test/full_twisted_torus_oms.py` | Current Helmholtz driver (kh=4.8π, N=64, p=8, n=16, m=4, flux='conormal', rk0=100, reduced_gpu=True, cyclic=True). Template for the new PB test. |

**Known facts and issues**
- Interface sizes: total faces = n² + 4mn. Interior Dirichlet Helmholtz at kh≳15 is
  near-resonant, with error amplification ~200 over the single-slab error. That is
  unrelated to the PB task.
- `oms._host_nbytes` over-counts HBS (views of one buffer; dedup by storage) and does not
  count MUMPS factors at all. Real MUMPS memory at n=64, m=16 is about 43.5 GB per slab.
  Keep test sizes small (n ≤ 16).
- MUMPS options if needed: BLR `icntl[35]=2, cntl[7]=1e-10`; OOC `icntl[22]=1`;
  factor size from INFOG(9)/(10).

## 2. The model (§1–3)

Normalized PB, model (A) (Λ, ψ, N, τ prescribed; Dirichlet data):

    -Δφ + G[φ] = f*  in Ω,   φ = φ* on ∂Ω

- Pointwise nonlinearity: `g(x,s) = Λ² N(ψ(x)) exp(s/τ(ψ(x)))`, `∂_s g = g/τ > 0`.
- Field notation (use consistently): `G[φ](x) = g(x, φ(x))`, `c[φ](x) = ∂_s g(x, φ(x))`.
- Flux label: `ψ(χ(t,x,y)) = (x²+y²)/Rc²`. Profiles: `N(ψ)=1-ψ/2`, `τ(ψ)=1-0.4ψ`.
- Manufactured solution (physical coordinates):
  `φ* = A sin(κ1 x1 + θ) cos(κ2 x2) exp(κ3 x3) + B`, with
  `-Δφ* = (κ1² + κ2² - κ3²)(φ* - B)`. Choose A so φ*/τ varies by about 3.
  Then `f* = -Δφ* + G[φ*]`. Sweep Λ ∈ {1, 10, 30}.
- Body loads and coefficients are evaluated at χ(p) for reference points p. The Laplacian
  is transformed by the geometry PDO.

## 3. Algorithm 1: fixed-point iteration on the interface system (§5)

- State: the interface traces u (the OMS unknowns). For each double slab i, solve the local
  **nonlinear** Dirichlet problem with the neighbours' traces and wall data φ*, and read
  its midplane trace `E_i(u)`. Residual `R(u) = u - E(u)`.
- `R(u)=0` holds iff u is the true trace (uniqueness on the overlap cells).
  `E'(u) = -S(c[φ_u])`, so the OMS system `I + S(c_ref)` built once is a frozen Jacobian.
- Iteration: `u_{k+1} = u_k - P R(u_k)`, with P = one red–black application
  ≈ (I+S_ref)^{-1}. **The fixed point is exact for any nonsingular P** (P only sets the
  rate), provided the local solves are converged to the final tolerance.
- No slab refactorizations ever. Needs `keepLU=True` and the leaf data (§4).

## 4. Local nonlinear solve in statically condensed form (§5.3)

Only the condensed slab operator `A_ii(c_ref)` is factored. Per leaf ℓ keep:
`L_II = K_II + C_ref` (its factors), `Y_ℓ = -L_II^{-1} L_IF`, and the flux extraction
`D_ℓ`. Particular flux: `q_ℓ(h) = D_ℓ [L_II^{-1} h_ℓ; 0]`. Skeleton: `A x = -Q h`.

**Local chord iteration (exactly equal to φ_{j+1} = M_i^{-1}(f* - G[φ_j] + c_ref φ_j)):**

    h_j       = f* - G[φ_j] + c_ref φ_j            (at leaf interior nodes)
    x_{j+1,I} = A_ii^{-1}( -(Q h_j)_I - A_ib b_i ),   x_{j+1,b} = b_i
    φ_{j+1}|ℓ = Y_ℓ E_ℓ x_{j+1} + L_II,ℓ^{-1} h_{j,ℓ}

- Rate: `ρ_i = sup_{x, s ∈ I(x)} |1 - ∂_s g(x,s)/c_ref(x)|`, with I(x) the range of the
  iterates (never sup over all s, which is infinite). For c_ref = c[φ_c]:
  ρ ≈ sup|φ* - φ_c|/τ.
- Reconstruction is needed every step (G needs every node). The midplane trace is read
  directly from the skeleton x.
- The reducer **must** use the same `pdo3d_to_cyl` flux conversion and patch gluing as
  the A_ii build. Otherwise it applies a different operator and the fixed point shifts.
- Optional (Remark 1, later): leaf-level nonlinear elimination. Batched dense Newton per
  leaf for given face data, then a skeleton chord
  `x_{j+1} = x_j - A_ii(c_ref)^{-1} R_skel(x_j)`. Same fixed point; no separate
  reconstruction.
- Energy remarks (hold exactly only for symmetric/Galerkin discretizations; the current
  collocation HPS is non-symmetric, so they hold only approximately):
  - energy-norm chord rate θρ, with θ = 1/(1+λ_min((W C_ref)^{-1} K));
  - skeleton error ≤ volume energy error;
  - leaf elimination = minimizing a convex condensed energy, so a line search would
    make it globally convergent.

## 5. Coarse solve for c_ref (must be h-coarse and nested)

- p-only coarsening is rejected: it keeps the fine leaf count, so it is useless on the
  full torus.
- Coarse grid `(n_c, m_c, nz_c) = (n, m, nz)/2^a`, order p_c, **same patch maps and same
  s**, so every fine leaf lies inside one coarse parent leaf in patch-parameter space.
  The coarse solve is direct, so the n=4m tree restriction does not apply to it.
- Prolongation: precompute one small matrix per child position (2^{3a} of them). Each
  evaluates the parent's tensor-Chebyshev polynomial at the child's nodes. No χ⁻¹, no
  search, batched.
- `c_ref = c[P φ_c]` is smooth inside fine leaves. Coarse jumps sit on fine leaf faces,
  which HPS handles exactly. Nesting is the reason this works.
- Coarse accuracy of about 0.1τ (one digit) gives ρ ≈ 0.1.
- Coarse solver: damped Newton, refactoring each step on the coarse condensed system (same
  `CylinderSlab` path, rebuilt with c = c[φ_c^k]); continuation in Λ if needed.
- Single double slab: t-coarsening needs fine nz ≥ 4 (nz must stay even). Otherwise
  coarsen only the cross-section.
- Verify that `make_cylinder` uses uniform leaves in patch-parameter space. If the arches
  are graded, the grading must nest.

## 6. TASK: single-slab first pass

**Problem:** one double slab `[0, 2H] × D_Rc` of the full twisted elliptic torus
(H = L/N_s, N_s = 8). Exact Dirichlet data φ* on both end faces and the wall. The exact
local solution is φ*.

**Stage 0: building blocks**
1. `pb/manufactured.py`: φ*, Δφ* (closed form), g, ∂_s g, f*. Check Δφ* against finite
   differences.
2. `transform_pdo(c_fun)` next to `transform_helmholtz_pdo`: the zero-order term is
   c_fun(χ(p)). Check: c_fun ≡ -kh² reproduces the Helmholtz PDO.
3. `CylinderSlab(..., keep_leaf=True)`: keep L_II factors, Y_ℓ, D_ℓ, and the leaf node
   coordinates (reference and physical). Add:
   - `nodes()` (fixed order);
   - `reduce_body(h)` → β = -(Qh)_I - A_ib b;
   - `reconstruct(x, h)`;
   - `trace(x, plane)`.

   Check: with h=0 this reproduces the existing Dirichlet path.
4. `assemble_full()`: the sparse uncondensed collocation system built from the **same**
   leaf operators (leaf-interior rows, flux-continuity rows, Dirichlet rows).
5. `prolongate(coarse_slab, fine_slab, vals)` with child matrices. Check: exact for
   polynomials of degree ≤ p_c.

**Stage 1: linear verification.** c_ref analytic (e.g. Λ²N(ψ)), h = (-Δ+c_ref)φ*.
Solve along the full path (MUMPS on `assemble_full`) and the condensed path.
- The two paths agree to ~1e-12.
- The error vs φ* converges spectrally as p goes 6→8→10, at nodes and off-grid.
- Report the midplane error separately.
- Sizes: (n,m,nz) = (8,2,2), then (16,4,2) or nz=4.

**Stage 2: nested coarse solve.** a ∈ {1,2}, p_c ∈ {4,6}, plus the constant-c̄ baseline.
Report the coarse error after prolongation, the predicted ρ, and the coarse cost relative
to the fine setup.

**Stage 3: fine factorizations with c_ref.** Full and condensed. Time MUMPS analysis and
factorization separately; record factor nnz and peak memory. Repeat the Stage 1 agreement
check.

**Stage 4: local chord, both paths.** One generic driver
`chord(solve, φ0, tol=1e-12, maxit=200)` taking a `solve(h, b)` callable. Start from
P φ_c. Log per iteration: increment, collocation residual, volume and midplane error
vs φ*, observed rate vs predicted ρ_i.

Reference: full Newton on the fine uncondensed system, refactoring each step.

Pass criteria:
- identical iterates on both paths;
- both converge to the Newton solution;
- observed rate ≲ ρ_i;
- final error equals the Stage 1 discretization error.

**Stage 5: tables (CSV).** Sweep:
- Λ ∈ {1,10,30};
- c_ref ∈ {c̄, coarse a=1, coarse a=2} × p_c;
- p ∈ {6,8,10};
- (n,m) ∈ {(8,2),(16,4)}.

For each path record setup time, memory, per-iteration time split (G eval / reduce / skeleton solve / reconstruct), iteration counts (chord, Newton), and final errors. Timing hygiene: warm-up run, `torch.cuda.synchronize()` before each
timer stop on GPU, median of 3 runs, fixed MUMPS threads, exclude one-off assembly.

**Stage 6 (later):** leaf-level nonlinear elimination with the same driver interface.
Compare at Λ=30 with c̄.

**Suggested layout:** `pb/manufactured.py`, `pb/coarse.py`, `pb/local_chord.py`,
`pb/prolong.py`, the extensions to `solver/cylinder_slab.py`, `transform_pdo` in the
geometry module, and `test/pb_single_slab.py` (runs Stages 1–5, writes CSV).

**Biggest risk:** Stage 0.3–0.4, exposing and keeping the leaf operators through the
five-patch build, including the patch-to-physical flux conversion. Read `build_patch`
and the leaf-level HPS routines it calls before designing these. Ask the user for any
file you cannot find.

## 7. Open risks for later (full torus)

- Physical PB at large Λ has Debye layers of width ~1/Λ at the wall. Resolving them
  needs radial arch refinement, which the n=4m tree constraint blocks on the fine grid.
  A ring-peeling generalization (m = (n/4)(2^c − 1)) was proposed but not implemented.
- Global coarse problem: nested, with direct Newton, or Algorithm 1 with c̄ and
  continuation (two-level).
- oms.py memory estimate fixes (storage dedup, MUMPS sizer) were proposed but not applied.
