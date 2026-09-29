# =============================================================================
# reentrant_channel_sslablu.jl
#
# Oceananigans.jl (v0.112) re-entrant channel configured as a close-to-1-to-1
# comparison for the SslabLU barotropic implicit-free-surface channel
# (test/validation/channel_barotropic_timestep.py).
#
# Structure copied from the baroclinic template reentrant_channel_baroclinic.jl,
# then SIMPLIFIED toward the SslabLU configuration (mostly by REMOVING things):
#   * f-plane (not beta-plane), southern hemisphere        f = -1e-4
#   * IMPLICIT free surface (the SslabLU elliptic solve), not split-explicit
#   * LINEAR dynamics: momentum_advection = nothing, tracer_advection = nothing
#   * a single buoyancy tracer b (BuoyancyTracer) -- no T/S/EOS, no TKE/CATKE,
#     no biharmonic closure, no stretched grid, no surface buoyancy flux
#   * steric height as an honest, depth-independent MERIDIONAL buoyancy gradient
#     imposed by a domain-wide Newtonian relaxation of b toward b_target(y).
#     This is the honest analog of SslabLU's GAMMA_S relaxation of the SSH.
#   * von Mises ridge + meridional gap bathymetry (same H(x,y) as SslabLU)
#   * eastward (westerly) wind stress  tau0 * sin(pi y / Ly)
#   * linear bottom drag
#   * uniform SSH at rest initial condition (all fields 0); the tilt is INDUCED
#
# Because Nz > 1 with active buoyancy, this is a genuinely baroclinic model; the
# apples-to-apples comparison against (barotropic) SslabLU is at the level of
# the free-surface / barotropic-mode response, not the full 3D field. Keeping
# b_target depth-independent makes the buoyancy-driven flow barotropic-
# equivalent (no thermal-wind shear), which is as close to SslabLU as this
# model class gets.
#
# Run:  julia --project reentrant_channel_sslablu.jl [Nspinup] [Nxy]
#   Nxy (default 80) sets Nx = Ny, for the resolution study in
#   channel_ssh_compare.py. Output goes to one directory per configuration,
#     run_oceananigans_channel_n<Nxy>_dt<Δt>s_nsteps<Nspinup>[_ridgectr]/
#   (Δt in seconds, %g-formatted exactly as channel_barotropic_timestep.py
#   formats its dt, so paired runs share the dt/nsteps tokens; _ridgectr marks
#   SSLABLU_RIDGE_MIDPANEL=0). Rerunning identical settings overwrites.
# =============================================================================

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: ynode
using Oceananigans.Operators: Δzᶠᶜᶜ, Δzᶜᶠᶜ
using Oceananigans.ImmersedBoundaries: static_column_depthᶜᶜᵃ, static_column_depthᶠᶜᵃ
using Printf
using JLD2

Oceananigans.defaults.FloatType = Float64

# ---- number of spin-up steps (CLI arg, like the template) -------------------
Nspinup = 400
if length(ARGS) >= 1
    Nspinup = parse(Int, ARGS[1])
end

# ---- resolution / geometry --------------------------------------------------
# Nx = Ny, optionally from the 2nd CLI arg
const Nxy = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 80
const Nx = Nxy
const Ny = Nxy
const Nz = 1 #16                # uniform in z (SslabLU has no vertical structure)

const Lx = 1000kilometers    # = 1e6 m  (SslabLU LCHAN)
const Ly = 1000kilometers    # square channel (SslabLU domain is the unit square)
const Lz = 4000.0            # m  (SslabLU H0)

const halo_size = 4          # 4 for immersed grids

# ---- physical constants -----------------------------------------------------
const f  = -1e-4             # f-plane, southern hemisphere (SslabLU FCOR)
const ρ0 = 1025.0            # reference density (SslabLU RHO0)

# ---- SslabLU ridge + gap bathymetry -----------------------------------------
# H(x,y)/Lz = 1 - hr * gap(y) * bump(x);  bottom z_b = -Lz * (H/Lz).
# bump is a 1-periodic von Mises crest at x = RIDGE_XC * Lx; gap cuts the crest to full
# depth inside the meridional band [GAP_Y0, GAP_Y1] * Ly.
const RIDGE_HR  = 0.8        # ridge height as a fraction of Lz
const RIDGE_KB  = 40.0       # von Mises concentration (narrow crest)
# crest position x/Lx: SAME env var and values as channel_barotropic_timestep.py
# (1, default: 0.5 + 1/32, mid-panel in SslabLU's default leaf grid;
#  0: 0.5, the original centered ridge on a SslabLU slab/panel edge)
const RIDGE_MIDPANEL = get(ENV, "SSLABLU_RIDGE_MIDPANEL", "1") != "0"
const RIDGE_XC  = 0.5 + (RIDGE_MIDPANEL ? 1 / 32 : 0.0)
const GAP_DEPTH = 1.0        # 1 => crest fully cut to full depth in the gap
const GAP_Y0    = 1 / 6      # gap band edges (fractions of Ly)
const GAP_Y1    = 1 / 2
const GAP_W     = 0.05       # tanh edge width (fraction of Ly)

@inline bump_x(x) = exp(RIDGE_KB * (cos(2π * (x - RIDGE_XC * Lx) / Lx) - 1))
@inline gap_y(y)  = 1 - 0.5 * GAP_DEPTH *
                        (tanh((y - GAP_Y0 * Ly) / (GAP_W * Ly)) -
                         tanh((y - GAP_Y1 * Ly) / (GAP_W * Ly)))
@inline depth_frac(x, y) = 1 - RIDGE_HR * gap_y(y) * bump_x(x)
@inline z_bottom(x, y)   = -Lz * depth_frac(x, y)   # bottom RISES over the ridge

# ---- forcing / parameter bundle ---------------------------------------------
parameters = (
    Lx = Lx, Ly = Ly, Lz = Lz,
    τ    = 0.15 / ρ0,   # surface kinematic wind stress [m^2/s^2]  (SslabLU TAU0)
    μ    = 1e-5,        # linear bottom-drag rate [1/s]            (SslabLU RDRAG)
    Bamp = 0.0, #1.2e-3,      # meridional buoyancy half-amplitude [m/s^2] (~0.5 m steric)
    λb   = 1e5,         # buoyancy relaxation timescale [s]  (SslabLU 1/GAMMA_S)
)

# ---- grid with immersed ridge -----------------------------------------------
function make_grid(arch)
    underlying = RectilinearGrid(arch,
        topology = (Periodic, Bounded, Bounded),   # re-entrant x; closed y-walls
        size = (Nx, Ny, Nz),
        halo = (halo_size, halo_size, halo_size),
        x = (0, Lx),
        y = (0, Ly),
        z = (-Lz, 0))

    bottom = Field{Center, Center, Nothing}(underlying)
    set!(bottom, z_bottom)
    # GridFittedBottom snaps to cell faces; PartialCellBottom(bottom) gives a
    # smoother H(x,y) closer to SslabLU's continuous coefficient (see notes).
    return ImmersedBoundaryGrid(underlying, PartialCellBottom(bottom))
end

# ---- model ------------------------------------------------------------------
function build_model(grid, parameters)

    # eastward (westerly) wind stress on u at the surface: tau0 * sin(pi y / Ly)
    @inline u_wind(x, y, t, p) = -p.τ * sin(π * y / p.Ly)
    u_top = FluxBoundaryCondition(u_wind, parameters = parameters)

    # linear bottom drag: stress -mu * H(x,y) * u, i.e. tendency -mu * u (SslabLU
    # -RDRAG * u). Oceananigans divides a bottom flux by the bottom cell's Δz,
    # which on this PartialCellBottom grid with Nz = 1 IS the discrete column
    # depth H at the u/v point, so using that same Δz makes the tendency exactly
    # -mu * u, including over the ridge. (An `immersed =` BC would never fire
    # here: with Nz = 1 the k = 1 bottom face is the underlying domain boundary,
    # not an immersed face.) For Nz > 1 this is drag on the bottom cell only,
    # and k = 1 lies inside the ridge -- see the warning below.
    @inline u_drag(i, j, grid, clock, fields, p) = @inbounds -p.μ * Δzᶠᶜᶜ(i, j, 1, grid) * fields.u[i, j, 1]
    @inline v_drag(i, j, grid, clock, fields, p) = @inbounds -p.μ * Δzᶜᶠᶜ(i, j, 1, grid) * fields.v[i, j, 1]
    Nz == 1 || @warn "bottom drag is only mu*H*u (SslabLU-equivalent) for Nz = 1"
    u_bot = FluxBoundaryCondition(u_drag, discrete_form = true, parameters = parameters)
    v_bot = FluxBoundaryCondition(v_drag, discrete_form = true, parameters = parameters)

    u_bcs = FieldBoundaryConditions(top = u_top, bottom = u_bot)
    v_bcs = FieldBoundaryConditions(bottom = v_bot)   # no wind on v

    # steric height <-> honest meridional buoyancy gradient. Relax b toward a
    # depth-independent, north-high target everywhere in the domain (the analog
    # of SslabLU's GAMMA_S SSH relaxation toward eta_s(y) = A(2y - 1)).
    @inline b_target(y, p) = p.Bamp * (2 * y / p.Ly - 1)     # warm equatorward (north-high)
    @inline function b_relax(i, j, k, grid, clock, fields, p)
        y = ynode(j, grid, Center())
        @inbounds return -(fields.b[i, j, k] - b_target(y, p)) / p.λb
    end
    Fb = Forcing(b_relax, discrete_form = true, parameters = parameters)

    # minimal constant diffusivity, for numerical stability only (NOT in SslabLU)
    horizontal_closure = HorizontalScalarDiffusivity(ν = 0.0, κ = 100.0)
    vertical_closure   = VerticalScalarDiffusivity(ν = 1e-3, κ = 1e-4)

    @info "Building the model..."
    model = HydrostaticFreeSurfaceModel(grid;
        free_surface       = ImplicitFreeSurface(solver_method = :PreconditionedConjugateGradient),
        # Fallback if ImplicitFreeSurface is unhappy with the immersed grid in
        # v0.112 (this loses the SslabLU-style implicit elliptic solve):
        # free_surface     = SplitExplicitFreeSurface(grid; substeps = 10),
        coriolis           = FPlane(f = f),
        buoyancy           = BuoyancyTracer(),
        tracers            = (:b,),
        momentum_advection = nothing,   # LINEAR momentum (SslabLU has no advection)
        tracer_advection   = nothing,   # static steric target (b not advected)
        closure            = (horizontal_closure, vertical_closure),
        boundary_conditions = (u = u_bcs, v = v_bcs),
        forcing            = (b = Fb,))

    return model
end

# ---- spin-up loop -----------------------------------------------------------
function spinup!(model, Δt, nsteps)
    for _ in 1:nsteps
        time_step!(model, Δt)
    end
    return nothing
end

# ---- run --------------------------------------------------------------------
arch = CPU()
Δt   = 225            # 900 s (SslabLU dt = 0.25 h); the implicit free
                            # surface removes the fast-gravity-wave CFL limit

# one output directory per configuration (see the header), so runs don't
# overwrite each other
graph_directory = @sprintf("run_oceananigans_channel_n%d_dt%gs_nsteps%d%s/",
                           Nxy, Δt, Nspinup, RIDGE_MIDPANEL ? "" : "_ridgectr")

grid  = make_grid(arch)
model = build_model(grid, parameters)

@info @sprintf("Built model on %d x %d x %d grid; spinning up %d steps (Δt = %.0f s)...",
               Nx, Ny, Nz, Nspinup, Δt)

# Uniform SSH at rest: all prognostic fields default to 0, so NO initial
# condition is set -- the meridional tilt + gap finger are INDUCED by the
# forcing, exactly as in the SslabLU FORCED scenario.

# run in two halves so the midpoint matches SslabLU's NSTEPS//2 snapshot
Nmid = Nspinup ÷ 2
snapshot(model) = (ssh = convert(Array, interior(model.free_surface.displacement)),
                   u   = convert(Array, interior(model.velocities.u)),
                   v   = convert(Array, interior(model.velocities.v)),
                   t   = model.clock.time)

tic = time()
spinup!(model, Δt, Nmid)
mid = snapshot(model)
spinup!(model, Δt, Nspinup - Nmid)
spinup_toc = time() - tic
@info @sprintf("spin-up done in %.1f s", spinup_toc)

# ---- save -------------------------------------------------------------------
isdir(graph_directory) || mkdir(graph_directory)
filename = graph_directory * "data_final.jld2"

# cell-center coordinates and the column depth the model ACTUALLY used
# (PartialCellBottom-adjusted, incl. the minimum-fractional-cell-height cap);
# channel_ssh_compare.py differences this against the analytic SslabLU H(x,y)
xc = collect(xnodes(grid, Center()))
yc = collect(ynodes(grid, Center()))
Hc = [static_column_depthᶜᶜᵃ(i, j, grid) for i in 1:Nx, j in 1:Ny]
# u-face depth min(H[i-1], H[i]): what the momentum/drag/continuity terms use
xf = collect(xnodes(grid, Face()))
Hu = [static_column_depthᶠᶜᵃ(i, j, grid) for i in 1:Nx, j in 1:Ny]

jldsave(filename;
    Nx, Ny, Nz, Lx, Ly, Lz, xc, yc, Hc, xf, Hu, ridge_xc = RIDGE_XC,
    dt = Δt, nsteps = Nspinup, nmid = Nmid,
    t = model.clock.time, t_mid = mid.t,
    ssh_mid = mid.ssh, u_mid = mid.u, v_mid = mid.v,
    ssh = convert(Array, interior(model.free_surface.displacement)),  # template used .displacement
    b   = convert(Array, interior(model.tracers.b)),
    u   = convert(Array, interior(model.velocities.u)),
    v   = convert(Array, interior(model.velocities.v)),
    w   = convert(Array, interior(model.velocities.w)))

@info "wrote $filename"
