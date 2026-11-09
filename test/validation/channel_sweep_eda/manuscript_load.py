"""Load the two channel sweeps used by the manuscript figures into one table
(pickled to SSLABLU_EDA_OUT/manuscript_rows.pkl). Run from anywhere:
  python test/validation/channel_sweep_eda/manuscript_load.py
  python test/validation/channel_sweep_eda/manuscript_figs.py

Run directories and sweep stdout files, relative to the repo root (defaults:
the cached-leaf-path sweeps, SSLABLU_FASTLEAF=1):
  SSLABLU_MS_SOLVER       solver-preset run directory
  SSLABLU_MS_RANKISO      rankiso-preset run directory
  SSLABLU_MS_SOLVER_OUT   that sweep's tee'd stdout files (SUMMARY wall times):
                          comma-separated paths or glob patterns, merged; a
                          sweep run in several invocations (e.g. one per
                          npan_y) has one file each
  SSLABLU_MS_RANKISO_OUT  likewise for rankiso
Every run directory needs a wall time from one of its preset's files.
"""
import glob, os, re, pickle
import numpy as np
from eda_common import REPO, OUT

os.chdir(REPO)
ROOTS = {"solver": os.environ.get("SSLABLU_MS_SOLVER", "central_ridge_solver_gpu_runs_fastleaf"),
         "rankiso": os.environ.get("SSLABLU_MS_RANKISO", "central_ridge_rankiso_gpu_runs_fastleaf")}
OUTS = {k: sorted({f for pat in os.environ.get(v, d).split(",") if pat for f in glob.glob(pat.strip())})
        for k, v, d in (("solver", "SSLABLU_MS_SOLVER_OUT", "solver_sweep_fastleaf*.out"),
                        ("rankiso", "SSLABLU_MS_RANKISO_OUT", "rankiso_sweep_fastleaf*.out"))}
for _p in ROOTS.values():
    if not os.path.isdir(_p):
        raise SystemExit("%s not found (see the SSLABLU_MS_* variables)" % _p)
for _k, _v in OUTS.items():
    if not _v:
        raise SystemExit("no sweep stdout file for %s (see the SSLABLU_MS_* variables)" % _k)
ROWS_PKL = os.path.join(OUT, "manuscript_rows.pkl")
FIELDS = [f + "_" + t + "_320" for f in ("eta", "u", "v") for t in ("mid", "final")]


def wall_times(out):
    """{(p, N, npx, npy, solver, RK, RB_RK): seconds} from a sweep's SUMMARY table."""
    W = {}
    for line in open(out).read().split("SUMMARY  (")[-1].splitlines():
        f = line.split()
        if len(f) in (9, 10) and f[0].isdigit() and f[1].isdigit() and ("ok" in f or "skipped" in f):
            if len(f) == 10:                         # ... solver RK RB_RK status time dir
                npy, p, N, npx, solver, rk, rbrk = int(f[0]), int(f[1]), int(f[2]), int(f[3]), f[4], int(f[5]), f[6]
                rbrk = int(rbrk) if rbrk != "-" else 0
                t = float(f[8])
            else:                                    # ... solver RK=RB_RK status time dir
                npy, p, N, npx, solver = int(f[0]), int(f[1]), int(f[2]), int(f[3]), f[4]
                rk = 0 if f[5] == "-" else int(f[5]); rbrk = rk if solver == "rbhbs" else 0
                t = float(f[7])
            W[(p, N, npx, npy, solver, rk, rbrk)] = t
    return W


rows = []
for sweep, root in ROOTS.items():
    W = {}
    for _o in OUTS[sweep]:
        W.update(wall_times(_o))
    logs = {os.path.basename(l): l for l in glob.glob(root + "/logs/**/*.log", recursive=True)}
    for d in sorted(glob.glob(root + "/run_sslablu_channel_*_nsteps800_*")):
        S = np.load(d + "/channel_timestep_ssh.npz")
        r = dict(sweep=sweep, dir=os.path.basename(d),
                 p=int(S["p"]), N=int(S["N"]), npx=int(S["npan_x"]), npy=int(S["npan_y"]),
                 solver=str(S["solver"]), rk=int(S["rk"]), rbrk=int(S["rb_rk"]) if str(S["solver"]) == "rbhbs" else 0,
                 check=float(S["solver_check"]), mean_mid=float(S["eta_mean_mid"]), mean_final=float(S["eta_mean_final"]))
        for f in FIELDS:
            r[f] = S[f].copy()
        C = np.genfromtxt(d + "/channel_timestep_diag.csv", delimiter=",", names=True)
        r["ts"] = {k: C[k].copy() for k in C.dtype.names}
        r.update(t_rhs=C["t_rhs"].mean(), t_solve=C["t_solve"].mean(), t_recon=C["t_recon"].mean())
        tag = ("rb" if r["solver"] == "rb" and r["rk"] == 0 else
               "rb_rk%d" % r["rk"] if r["solver"] == "rb" else
               "rbhbs_rk%d" % r["rk"] if (r["rk"] == r["rbrk"] and sweep == "solver") else
               "rbhbs_rk%d_rbrk%d" % (r["rk"], r["rbrk"]))
        txt = open(logs["p%d_N%d_pan%dx%d_%s.log" % (r["p"], r["N"], r["npx"], r["npy"], tag)]).read()
        m = re.search(r"\[divergence form\]\s+assemble/factor/keep-slabs = ([\d.]+) / ([\d.]+) / ([\d.]+)", txt)
        r["asm"], r["fac"], r["keep"] = (float(x) for x in m.groups())
        r["forms"] = 2 if "[non-conservative]" in txt else 1
        r["setup"] = float(re.search(r"setup \(both forms\)\s+=\s+([\d.]+)", txt).group(1))
        mr = re.search(r"residual \(wall leakage\)=\s+(\S+)\s+\((\S+) rel\)", txt)
        r["mass_abs"], r["mass_rel"] = float(mr.group(1)), float(mr.group(2))
        r["gate"] = {k: float(v) for k, v in re.findall(
            r"(leaf d/dx matrix|leaf d/dy matrix|skeleton body-rhs|reconstruction|wall du/dn vs data) rel\. err\s+=\s+(\S+)", txt)}
        lc = re.findall(r"cached leaf path vs hpsalt, random data: rel\. diff = (\S+)", txt)
        r["leaf_check"] = max(map(float, lc)) if lc else np.nan
        r["wall_v"] = float(re.search(r"final max\|v\| at walls\s+=\s+(\S+)", txt).group(1))
        key = (r["p"], r["N"], r["npx"], r["npy"], r["solver"], r["rk"], r["rbrk"])
        if key not in W:
            raise SystemExit("%s/%s has no wall time in %s" % (root, r["dir"], OUTS[sweep]))
        r["wall"] = W[key]
        rows.append(r)
pickle.dump(dict(rows=rows, roots=ROOTS, outs=OUTS), open(ROWS_PKL, "wb"))
print(len(rows), "runs", {s: sum(r["sweep"] == s for r in rows) for s in ROOTS}, "from", OUTS, "->", ROWS_PKL)
