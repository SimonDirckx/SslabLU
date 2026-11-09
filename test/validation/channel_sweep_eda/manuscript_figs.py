"""Manuscript figures for the centered-ridge barotropic channel (solver + rankiso
sweeps, see manuscript_load.py). Writes SSLABLU_EDA_OUT/manuscript_figs/*.pdf
and *.png and prints the numbers quoted with them.

  fig1 problem (bathymetry, slabs, eta and velocity of the reference run)
  fig2 spin-up of the reference run
  fig3 self-convergence, slab-count invariance, volume drift / wall flux
  fig4 cost vs DoF and run-time composition (N = 8, dense rb)
  fig5 slab count at a fixed discretization (the finest npan_y of the solver
       sweep with every p)
  fig6 HBS compression error vs rank (rankiso sweep, npan_y = 32)
  fig7 error vs run time (solver sweep)
  fig8 compression across interface sizes: error vs interface size and vs N,
       run time compressed / dense

Reference: the dense N = 8 run with the most DoF (manuscript_common.REF); the
next finest bounds its error (REF_ERR). A layout's discretization error is its
dense N = 8 run vs REF. Run times: setup (assemble/factor/keep) + 800 steps of
the conservative form; the solver preset steps both forms, rankiso one, so
total wall times of the two presets are not comparable."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from manuscript_common import *

FIG = os.path.join(OUT, "manuscript_figs", "")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8.5,
                     "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "axes.linewidth": 0.6, "lines.linewidth": 1.3, "lines.markersize": 4.5,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
PC = {8: "#0072B2", 12: "#E69F00", 16: "#009E73"}          # Okabe-Ito, one color per p
NM = {8: "o", 16: "s", 32: "^", 64: "D", 128: "v"}          # one marker per slab count N
GREY = "#777777"
PS, RANKS = (8, 12, 16), (4, 6, 8, 12)
FLOOR = (1e-9, 1e-8)       # spread of repeated compressed runs (printed below), shaded


def save(fig, name):
    fig.savefig(FIG + name + ".pdf"); fig.savefig(FIG + name + ".png", dpi=220); plt.close(fig)


def panel(a, s):
    a.text(0.0, 1.02, s, transform=a.transAxes, fontsize=9, fontweight="bold", va="bottom")


def tidy(a):
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


def leaves(n):
    return "%d$\\times$%d" % (2 * n, n)


# dense N = 8 layouts and their discretization error vs REF
LAY = sorted({(r["p"], r["npy"]) for r in rows if r["kind"] == "rb" and r["N"] == 8})
NPYS = sorted({n for _, n in LAY})
REFLAY = (REF["p"], REF["npy"])
ERR = {(p, n): rel(fld(rbany(n, p)), fld(REF)) for p, n in LAY if (p, n) != REFLAY}
disc = lambda p, n: ERR.get((p, n), REF_ERR)
NSOF = lambda n: [N for N in (8, 16, 32, 64, 128) if N <= n]

# ---------------------------------------------------------------- 1. the problem
RHR, RKB, GY0, GY1, GW = 0.8, 40.0, 1 / 6, 0.5, 0.05
xs = (np.arange(320) + 0.5) / 320
X, Y = np.meshgrid(xs, xs, indexing="ij")
gap = 1 - 0.5 * (np.tanh((Y - GY0) / GW) - np.tanh((Y - GY1) / GW))
Hf = 1 - RHR * gap * np.exp(RKB * (np.cos(2 * np.pi * (X - 0.5)) - 1))
eta, u, v = fld(REF, "eta"), fld(REF, "u"), fld(REF, "v")
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.45), gridspec_kw=dict(wspace=0.62))
ext = (0, 1, 0, 1)
im = ax[0].imshow(Hf.T, origin="lower", extent=ext, cmap="cividis", vmin=0.2, vmax=1)
N_, npy_ = 8, 16
for n in range(N_):
    ax[0].axvline(n / N_, color="w", lw=0.7, ls=(0, (3, 2)))
c, hw = 0.5, 1 / N_
ax[0].add_patch(plt.Rectangle((c - hw, 0), 2 * hw, 1, fill=False, ec="#d55e00", lw=1.3))
for k in range(1, 4 * npy_ // N_):
    ax[0].plot([c - hw + k * 2 * hw / (4 * npy_ // N_)] * 2, [0, 1], color="#d55e00", lw=0.35)
for k in range(1, npy_):
    ax[0].plot([c - hw, c + hw], [k / npy_] * 2, color="#d55e00", lw=0.35)
fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.03).set_label("$H/H_0$")
ax[0].set_title("(a) bathymetry and slabs", fontsize=8.5, loc="left")
m = np.abs(eta).max()
im = ax[1].imshow(eta.T * 100, origin="lower", extent=ext, cmap="RdBu_r", vmin=-m * 100, vmax=m * 100)
ax[1].contour(xs, xs, Hf.T, levels=[0.5], colors="k", linewidths=0.5, linestyles="--")
fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03).set_label(r"$\eta - \bar\eta$  [cm]")
ax[1].set_title(r"(b) sea-surface height, $t$ = 100 h", fontsize=8.5, loc="left")
sp = np.hypot(u, v)
im = ax[2].imshow(sp.T * 100, origin="lower", extent=ext, cmap="magma_r", vmin=0)
ax[2].streamplot(xs, xs, u.T, v.T, color="w", linewidth=0.45, density=1.1, arrowsize=0.5)
ax[2].contour(xs, xs, Hf.T, levels=[0.5], colors="#56b4e9", linewidths=0.6, linestyles="--")
fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.03).set_label(r"$|\mathbf{u}|$  [cm/s]")
ax[2].set_title("(c) velocity, $t$ = 100 h", fontsize=8.5, loc="left")
for i, a in enumerate(ax):
    a.set_xlabel("$x/L$"); a.set_ylabel("$y/L$") if i == 0 else None
    a.set_xticks([0, 0.5, 1]); a.set_yticks([0, 0.5, 1])
save(fig, "fig1_problem")

# ---------------------------------------------------------------- 2. spin-up (reference run)
T = REF["ts"]
th = T["t_hours"]
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.1), gridspec_kw=dict(wspace=0.38))
ax[0].plot(th, T["energy"] * 1e3, color="k")
ax[0].set_ylabel(r"energy  [$10^{-3}$ J/$\rho_0$]")
ax[1].plot(th, T["max_u"] * 100, color="k", label=r"max $|u|$")
ax[1].plot(th, T["max_v"] * 100, color=GREY, ls="--", label=r"max $|v|$")
ax[1].set_ylabel("[cm/s]"); ax[1].legend(loc="upper left")
ax[2].plot(th, T["tilt_NmS"] * 100, color="k")
ax[2].set_ylabel("N$-$S SSH tilt  [cm]")
for i, a in enumerate(ax):
    a.set_xlabel("$t$  [h]"); a.set_xlim(0, 100); tidy(a); panel(a, "(%s)" % "abc"[i])
save(fig, "fig2_spinup")

# ---------------------------------------------------------------- 3. accuracy
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(wspace=0.38))
a = ax[0]
for p in PS:
    ns = [n for n in NPYS if (p, n) in ERR]
    d = [rbany(n, p)["dof"] for n in ns]
    a.loglog(d, [ERR[(p, n)] for n in ns], "-o", color=PC[p])
    a.loglog(d, [rel(fld(rbany(n, p), "u"), fld(REF, "u")) for n in ns], ":", marker="o", mfc="none", color=PC[p])
    for n in ns:                                              # same discretization, other slab counts
        for N in NSOF(n)[1:]:
            a.plot(rbany(n, p, N)["dof"], rel(fld(rbany(n, p, N)), fld(REF)), NM[N], color=PC[p], ms=6.5, mfc="none", mew=0.7)
a.plot(REF["dof"], REF_ERR, "o", color=PC[REF["p"]], mfc="none", mew=0.9)
a.set_xlabel("degrees of freedom"); a.set_ylabel("relative $L^2$ error vs. reference")
a.add_artist(a.legend(handles=[Line2D([], [], color=PC[p], marker="o", label="$p$ = %d" % p) for p in PS], loc="lower left"))
a.legend(handles=[Line2D([], [], color=GREY, ls="-", label=r"$\eta$"), Line2D([], [], color=GREY, ls=":", marker="o", mfc="none", label="$u$")]
         + [Line2D([], [], color=GREY, ls="", marker=NM[N], mfc="none", ms=6.5, label="$N$ = %d" % N) for N in (16, 32, 64, 128)],
         loc="upper right", handlelength=1.6, fontsize=7)
a.set_ylim(1e-12, 1)
a = ax[1]
for p, n in LAY:
    b = rbany(n, p)
    for N in NSOF(n)[1:]:
        x = rbany(n, p, N)
        a.semilogy(b["dof"], max(rel(fld(x, f), fld(b, f)) for f in ("eta", "u", "v")), NM[N], color=PC[p], mfc="none", ms=6, mew=0.8)
    a.semilogy(b["dof"], disc(p, n), "o", color=PC[p], ms=4.5, mfc=PC[p] if (p, n) in ERR else "none")
a.set_xscale("log")
a.axhspan(1e-16, 1e-11, color="#eeeeee", lw=0, zorder=0)
a.text(0.97, 0.04, "round-off band", transform=a.transAxes, ha="right", fontsize=7, color=GREY)
a.set_ylim(1e-14, 1); a.set_xlabel("degrees of freedom")
a.set_ylabel("relative difference / error")
a.legend(handles=[Line2D([], [], color=GREY, ls="", marker="o", label="discretization error ($N$ = 8)"),
                  Line2D([], [], color=GREY, ls="", marker="s", mfc="none", label="$N$ = 16 ... 128 vs $N$ = 8")],
         loc="upper right")
a = ax[2]
for p in PS:
    rr = [rbany(n, p) for n in NPYS if (p, n) in LAY]
    a.loglog([r["dof"] for r in rr], [abs(r["mass_rel"]) for r in rr], "-o", color=PC[p])
    a.loglog([r["dof"] for r in rr], [r["wall_v"] / r["ts"]["max_u"][-1] for r in rr], ":", marker="v", mfc="none", color=PC[p])
a.set_xlabel("degrees of freedom"); a.set_ylabel("relative error")
a.legend(handles=[Line2D([], [], color=GREY, marker="o", label=r"volume drift $|\Delta V| / \int|\eta|$"),
                  Line2D([], [], color=GREY, ls=":", marker="v", mfc="none", label=r"wall $\max|v| \,/\, \max|u|$")],
         loc="center left", bbox_to_anchor=(0, 0.42))
a.set_ylim(1e-17, 3)
for i, a in enumerate(ax):
    tidy(a); panel(a, "(%s)" % "abc"[i])
save(fig, "fig3_accuracy")

# ---------------------------------------------------------------- 4. cost scaling, N = 8, dense rb
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw=dict(wspace=0.3, width_ratios=(1, 1.25)))
a = ax[0]
for p in PS:
    rr = [rbany(n, p) for n in NPYS if (p, n) in LAY]
    d = [r["dof"] for r in rr]
    a.loglog(d, [r["leaf"] for r in rr], "-o", color=PC[p])
    a.loglog(d, [r["once"] for r in rr], "--s", color=PC[p], mfc="none")
    a.loglog(d, [r["t_solve"] for r in rr], ":^", color=PC[p], mfc="none")
dd = np.array([1e4, 6e6])
a.loglog(dd, 0.003 * dd / 1e4, color=GREY, lw=0.7)
a.text(1.2e5, 0.012, r"$\propto$ DoF", color=GREY, fontsize=7)
a.set_xlabel("degrees of freedom"); a.set_ylabel("wall time  [s]")
a.legend(handles=[Line2D([], [], color=PC[p], marker="o", label="$p$ = %d" % p) for p in PS]
         + [Line2D([], [], color=GREY, ls="--", marker="s", mfc="none", label="setup, once"),
            Line2D([], [], color=GREY, ls="-", marker="o", label="rhs + recon., per step"),
            Line2D([], [], color=GREY, ls=":", marker="^", mfc="none", label="interface solve, per step")],
         loc="upper left", ncol=2, columnspacing=0.8, handlelength=1.8)
a.set_ylim(3e-4, 3e4)
a = ax[1]
parts = (("S-map assembly", "asm", "#999999"), ("factorization", "fac", "#000000"), ("slab storage", "keep", "#cccccc"),
         ("body-load rhs", "t_rhs", "#56B4E9"), ("interface solve", "t_solve", "#D55E00"), ("reconstruction", "t_recon", "#0072B2"))
_rr = [rbany(n, p) for p, n in LAY]
SHARE = {"fac": " ($\\leq$%.2f%%)" % max(100 * r["fac"] / r["cost"] for r in _rr),
         "t_solve": " ($\\leq$%.1f%%)" % max(100 * 800 * r["t_solve"] / r["cost"] for r in _rr)}
labs, ys, k = [], [], 0
for p in PS:
    for n in NPYS:
        if (p, n) not in LAY:
            continue
        r = rbany(n, p)
        left = 0.0
        for nm, f, col in parts:
            w = 100 * (r[f] * (800 if f.startswith("t_") else 1)) / r["cost"]
            a.barh(k, w, left=left, color=col, height=0.75, lw=0)
            left += w
        a.text(101, k, "%.0f s" % r["cost"], va="center", fontsize=6.5)
        labs.append("$p$ = %d, %s" % (p, leaves(n))); ys.append(k); k += 1
    k += 0.5
a.set_yticks(ys); a.set_yticklabels(labs, fontsize=6.8); a.invert_yaxis()
a.set_xlim(0, 100); a.set_xlabel("share of run time, setup + 800 steps  [%]")
a.legend(handles=[Patch(color=c, label=nm + SHARE.get(f, "")) for nm, f, c in parts], loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
for i, a in enumerate(ax):
    tidy(a); panel(a, "(%s)" % "ab"[i])
save(fig, "fig4_cost")

# ---------------------------------------------------------------- 5. slab count at fixed discretization
NPY5 = max(n for n in NPYS if all(one(sweep="solver", npy=n, p=p, N=8, kind="rb") for p in PS))
NS = NSOF(NPY5)
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw=dict(wspace=0.4))
for p in PS:
    rb_ = [one(sweep="solver", npy=NPY5, p=p, N=N, kind="rb") for N in NS]
    hb_ = [one(sweep="solver", npy=NPY5, p=p, N=N, kind="both", rk=p) for N in NS]
    ax[0].plot(NS, [r["leaf"] for r in rb_], "-o", color=PC[p])
    ax[1].semilogy(NS, [r["t_solve"] for r in rb_], "-o", color=PC[p])
    ax[1].semilogy(NS, [r["t_solve"] for r in hb_], "--s", color=PC[p], mfc="none")
    ax[2].semilogy(NS, [r["asm"] for r in rb_], "-o", color=PC[p])
    ax[2].semilogy(NS, [r["asm"] for r in hb_], "--s", color=PC[p], mfc="none")
    ax[2].semilogy(NS, [r["fac"] for r in rb_], "-.o", color=PC[p], ms=3)
    ax[2].semilogy(NS, [r["fac"] for r in hb_], ":s", color=PC[p], mfc="none", ms=3)
ax[0].set_ylabel("rhs + reconstruction per step  [s]"); ax[0].set_ylim(0)
ax[0].legend(handles=[Line2D([], [], color=PC[p], marker="o", label="$p$ = %d" % p) for p in PS], loc="lower right")
ax[1].set_ylabel("interface solve per step  [s]")
ax[1].legend(handles=[Line2D([], [], color=GREY, marker="o", label="dense red-black"),
                      Line2D([], [], color=GREY, ls="--", marker="s", mfc="none", label="HBS red-black, rank $p$")], loc="upper left")
ax[1].set_ylim(2e-3, 3)
ax[2].set_ylabel("one-time cost  [s]")
ax[2].legend(handles=[Line2D([], [], color=GREY, marker="o", label="assembly, dense"),
                      Line2D([], [], color=GREY, ls="--", marker="s", mfc="none", label="assembly, HBS"),
                      Line2D([], [], color=GREY, ls="-.", marker="o", ms=3, label="factor, dense"),
                      Line2D([], [], color=GREY, ls=":", marker="s", mfc="none", ms=3, label="factor, HBS")],
             loc="lower left", ncol=2, columnspacing=0.6, handlelength=1.8, bbox_to_anchor=(0, -0.02))
ax[2].set_ylim(1e-2, 2e3)
for i, a in enumerate(ax):
    a.set_xscale("log", base=2); a.set_xticks(NS); a.set_xticklabels([str(N) for N in NS]); a.minorticks_off()
    a.set_xlabel("slabs $N$ (%s leaves)" % leaves(NPY5)); tidy(a); panel(a, "(%s)" % "abc"[i])
save(fig, "fig5_slabcount")


# ---------------------------------------------------------------- 6. HBS rank (rankiso, npan_y = 32)
def herr(k, n, p, N, r):
    x = (one(sweep="rankiso", npy=n, p=p, N=N, kind="smap", rk=r) if k == "smap" else
         one(sweep="rankiso", npy=n, p=p, N=N, solver="rbhbs", rk=int(1.5 * p), rbrk=r))
    return rel(fld(x), fld(rbof(x))) if x else np.nan


fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.3), sharex=True, sharey=True, gridspec_kw=dict(wspace=0.08, hspace=0.12))
for i, (k, nm) in enumerate((("smap", "S-map compression\n(dense solver)"), ("solver", "solver compression\n(S-maps at rank 1.5$p$)"))):
    for j, N in enumerate((8, 16, 32)):
        a = ax[i, j]
        a.axhspan(*FLOOR, color="#eeeeee", lw=0, zorder=-5)
        for p in PS:
            a.semilogy(RANKS, [herr(k, 32, p, N, r) for r in RANKS], "-o", color=PC[p], zorder=3)
            a.axhline(disc(p, 32), color=PC[p], ls=":", lw=0.9, zorder=2)
        a.set_xticks(RANKS); a.set_ylim(1e-10, 3e-2); tidy(a)
        if i == 0:
            a.set_title("$N$ = %d slabs" % N)
        if j == 0:
            a.set_ylabel(nm + "\n" + r"rel. error in $\eta$ vs. dense")
        if i == 1:
            a.set_xlabel("rank")
        panel(a, "(%s)" % "abcdef"[3 * i + j])
ax[0, 2].text(11.8, 3e-9, "repeat-run spread", ha="right", va="center", fontsize=7, color=GREY)
fig.legend(handles=[Line2D([], [], color=PC[p], marker="o", label="$p$ = %d" % p) for p in PS]
           + [Line2D([], [], color=GREY, ls=":", label="discretization error of that $p$, %s leaves" % leaves(32))],
           loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
save(fig, "fig6_rank")

# ---------------------------------------------------------------- 7. accuracy vs cost (solver sweep)
fig, a = plt.subplots(figsize=(4.8, 3.0))
for x in find(sweep="solver"):
    a.loglog(x["cost"], rel(fld(x), fld(REF)), "o" if x["kind"] == "rb" else "x", color=PC[x["p"]],
             ms=4 if x["kind"] == "rb" else 4.5, mew=0.9, alpha=1 if x["kind"] == "rb" else 0.7)
for p in PS:
    ns = [n for n in NPYS if one(sweep="solver", npy=n, p=p, N=8, kind="rb")]
    a.plot([one(sweep="solver", npy=n, p=p, N=8, kind="rb")["cost"] for n in ns], [disc(p, n) for n in ns], "-", color=PC[p], lw=0.8, zorder=0)
a.axhspan(*FLOOR, color="#eeeeee", lw=0, zorder=-5)
a.axhline(REF_ERR, color=GREY, lw=0.6, ls="--")
a.text(max(x["cost"] for x in find(sweep="solver")), REF_ERR * 1.5, "reference error bound", fontsize=6.5, color=GREY, ha="right")
TK = [t for t in (10, 30, 100, 300, 1000, 3000, 10000) if min(x["cost"] for x in rows) / 1.5 < t < max(x["cost"] for x in rows) * 1.5]
a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a.set_xticks(TK); a.set_xticklabels([str(t) for t in TK])
a.set_xlabel("run time, setup + 800 steps  [s]"); a.set_ylabel(r"relative error in $\eta$ vs. reference")
a.legend(handles=[Line2D([], [], color=PC[p], marker="o", ls="-", lw=0.8, label="$p$ = %d" % p) for p in PS]
         + [Line2D([], [], color=GREY, marker="o", ls="", label="dense"),
            Line2D([], [], color=GREY, marker="x", ls="", label="HBS, rank\n$p/2, p, 3p/2$")],
         loc="center left", bbox_to_anchor=(1.0, 0.5))
tidy(a)
save(fig, "fig7_error_cost")

# ---------------------------------------------------------------- 8. compression across interface sizes (rankiso)
ISO = sorted({(r["p"], r["npy"]) for r in find(sweep="rankiso", kind="rb", N=8)}, key=lambda t: t[0] * t[1])
NPYB = max(n for _, n in ISO)
PB = max(p for p, n in ISO if n == NPYB)
NC = {8: "#0072B2", 32: "#D55E00"}
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5), gridspec_kw=dict(wspace=0.42))
a = ax[0]
for N in (8, 32):
    for k, mk in (("smap", "o"), ("solver", "s")):
        pts = sorted((p * n, herr(k, n, p, N, 8)) for p, n in ISO)
        a.loglog(*zip(*pts), mk, color=NC[N], mfc=NC[N] if k == "smap" else "none", mew=0.9, ls="-" if k == "smap" else "--", lw=0.7)
a.axhspan(*FLOOR, color="#eeeeee", lw=0, zorder=-5)
a.set_xlabel("interface size $n_c$ = npan$_y\\,p$"); a.set_ylabel(r"compression error in $\eta$, rank 8")
NCT = sorted({p * n for p, n in ISO})
a.legend(handles=[Line2D([], [], color=NC[N], marker="o", ls="", label="$N$ = %d" % N) for N in (8, 32)]
         + [Line2D([], [], color=GREY, marker="o", ls="-", lw=0.7, label="S-maps"),
            Line2D([], [], color=GREY, marker="s", mfc="none", ls="--", lw=0.7, label="solver")], loc="upper left", ncol=2, fontsize=7)
a.set_ylim(1e-9, 1e-2)
a = ax[1]
NSB = NSOF(NPYB)
for k, ls in (("smap", "-"), ("solver", "--")):
    for r, mk in ((8, "o"), (12, "s")):
        a.loglog(NSB, [herr(k, NPYB, PB, N, r) for N in NSB], ls, marker=mk, color=PC[PB], mfc=PC[PB] if r == 8 else "none", mew=0.9)
a.axhspan(*FLOOR, color="#eeeeee", lw=0, zorder=-5)
a.axhline(disc(PB, NPYB), color=PC[PB], ls=":", lw=0.9)
a.text(NSB[-1], disc(PB, NPYB) * 2, "discretization error", fontsize=6.5, color=PC[PB], ha="right")
a.set_xscale("log", base=2); a.set_xticks(NSB); a.set_xticklabels([str(N) for N in NSB]); a.minorticks_off()
a.set_xlabel("slabs $N$ (%s leaves, $p$ = %d)" % (leaves(NPYB), PB)); a.set_ylabel(r"compression error in $\eta$")
a.legend(handles=[Line2D([], [], color=GREY, ls="-", label="S-maps"), Line2D([], [], color=GREY, ls="--", label="solver"),
                  Line2D([], [], color=GREY, marker="o", ls="", label="rank 8"), Line2D([], [], color=GREY, marker="s", mfc="none", ls="", label="rank 12")],
         loc="upper left", ncol=2, fontsize=7)
a = ax[2]
for k, mk in (("smap", "o"), ("solver", "s")):
    for p in PS:
        pts = []
        for pp, n in ISO:
            if pp != p:
                continue
            x = (one(sweep="rankiso", npy=n, p=p, N=8, kind="smap", rk=8) if k == "smap" else
                 one(sweep="rankiso", npy=n, p=p, N=8, solver="rbhbs", rk=int(1.5 * p), rbrk=8))
            pts.append((p * n, x["wall"] / rbof(x)["wall"]))
        if pts:
            a.semilogx(*zip(*sorted(pts)), "-" if k == "smap" else "--", marker=mk, color=PC[p], mfc=PC[p] if k == "smap" else "none", mew=0.9)
a.axhline(1.0, color=GREY, lw=0.6)
a.set_xlabel("interface size $n_c$"); a.set_ylabel("run time, compressed / dense  ($N$ = 8)")
a.legend(handles=[Line2D([], [], color=PC[p], marker="o", label="$p$ = %d" % p) for p in PS]
         + [Line2D([], [], color=GREY, ls="-", marker="o", label="S-maps, rank 8"), Line2D([], [], color=GREY, ls="--", marker="s", mfc="none", label="solver, rank 8")],
         loc="lower left", fontsize=7)
for a in (ax[0], ax[2]):
    a.set_xticks(NCT); a.set_xticklabels([str(t) for t in NCT], rotation=45, fontsize=6.5)
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
for i, a in enumerate(ax):
    tidy(a); panel(a, "(%s)" % "abc"[i])
save(fig, "fig8_rank_scaling")

# ---------------------------------------------------------------- numbers quoted with the figures
g = lambda v: "%.1e" % v
print("data:", ROOTS)
print("reference: dense p = %d, %s leaves, N = 8 (%s sweep); closest other dense run p = %d, %s differs by %s (reference error bound)"
      % (REF["p"], leaves(REF["npy"]).replace("$\\times$", "x"), REF["sweep"], REF2["p"], leaves(REF2["npy"]).replace("$\\times$", "x"), g(REF_ERR)))
print("discretization error in eta, t = 100 h:")
for p in PS:
    print("  p=%2d: %s" % (p, "  ".join("%s %s" % (leaves(n).replace("$\\times$", "x"), g(ERR[(p, n)])) for n in NPYS if (p, n) in ERR)))
print("observed orders, finest two grids per p:", {p: "%.1f" % np.log2(ERR[(p, ns[-2])] / ERR[(p, ns[-1])])
      for p in PS for ns in [[n for n in NPYS if (p, n) in ERR]] if len(ns) > 1})
print("slab-count invariance, max over layouts/fields: %s" % g(max(max(rel(fld(rbany(n, p, N), f), fld(rbany(n, p), f)) for f in ("eta", "u", "v"))
      for p, n in LAY for N in NSOF(n)[1:])))
print("GATE max %s | rb vs Thomas max %s | cached leaf path vs hpsalt max %s | wall max|v| %s m/s" % (
    g(max(max(r["gate"].values()) for r in rows)), g(max(r["check"] for r in rows if r["solver"] == "rb")),
    g(np.nanmax([r["leaf_check"] for r in rows])), g(max(r["wall_v"] for r in rows))))
print("per step and run composition, dense rb, N = 8:")
for p, n in LAY:
    r = rbany(n, p)
    print("  p=%2d %8s DoF %7d: step %.4f s (rhs %.4f, interface %.5f, recon %.4f) | one-time %.1f s = %.0f steps | run %.0f s: setup %.0f%%, interface solve %.1f%%" % (
        p, leaves(n).replace("$\\times$", "x"), r["dof"], r["step"], r["t_rhs"], r["t_solve"], r["t_recon"], r["once"], r["once"] / r["step"], r["cost"],
        100 * r["once"] / r["cost"], 100 * 800 * r["t_solve"] / r["cost"]))
print("%s leaves, step [s] for N = %s (dense | HBS rank p) and run time HBS / dense:" % (leaves(NPY5).replace("$\\times$", "x"), "/".join(map(str, NS))))
for p in PS:
    a_ = [one(sweep="solver", npy=NPY5, p=p, N=N, kind="rb") for N in NS]
    b_ = [one(sweep="solver", npy=NPY5, p=p, N=N, kind="both", rk=p) for N in NS]
    print("  p=%2d: %s | %s | %s" % (p, " / ".join("%.3f" % x["step"] for x in a_), " / ".join("%.3f" % x["step"] for x in b_),
                                   " / ".join("%.2f" % (y["wall"] / x["wall"]) for x, y in zip(a_, b_))))
for k in ("smap", "solver"):
    sl = [-np.polyfit(RANKS, np.log10([herr(k, 32, p, N, r) for r in RANKS]), 1)[0] for p in PS for N in (8, 16, 32)]
    print("%-6s compression error decay (npan_y = 32): %.2f..%.2f decades per unit rank" % (k, min(sl), max(sl)))
    e8 = [herr(k, n, p, 8, 8) for p, n in ISO]
    print("       rank 8, N = 8, interface sizes %d..%d: error %s..%s" % (min(p * n for p, n in ISO), max(p * n for p, n in ISO), g(min(e8)), g(max(e8))))
    gr = [herr(k, NPYB, PB, 2 * N, r) / herr(k, NPYB, PB, N, r) for r in (8, 12) for N in NSB[:-1]]
    print("       growth per doubling of N (%s, p = %d, ranks 8/12): median x%.1f" % (leaves(NPYB).replace("$\\times$", "x"), PB, np.median(gr)))
print("repeated compressed runs (solver sweep RK = RB_RK = 12 vs rankiso solver:12, p = 8, same settings):")
for n in NPYS:
    for N in NSOF(n):
        x = one(sweep="solver", npy=n, p=8, N=N, solver="rbhbs", rk=12)
        y = one(sweep="rankiso", npy=n, p=8, N=N, solver="rbhbs", rk=12, rbrk=12)
        if x and y:
            print("  %8s N=%3d: error %s / %s, difference %s" % (leaves(n).replace("$\\times$", "x"), N, g(rel(fld(x), fld(rbof(x)))), g(rel(fld(y), fld(rbof(y)))), g(rel(fld(x), fld(y)))))
print("wrote", FIG)
