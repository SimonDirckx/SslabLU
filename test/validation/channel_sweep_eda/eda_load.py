"""Load every 800-step run of the 1013 / 1018 sweeps into one table (pickled to
SSLABLU_EDA_OUT/eda_rows.pkl). Edit ROOTS to point at other run directories."""
import glob, os, re, pickle
import numpy as np
from eda_common import REPO, ROWS_PKL

os.chdir(REPO)
# run directories, relative to the repo root (SSLABLU_EDA_DATA overrides the parent)
DATA = os.environ.get("SSLABLU_EDA_DATA", "central_ridge_rank_old_hps_mem_runs")
ROOTS = {("solver", "1013"): os.path.join(DATA, "central_ridge_solver_gpu_runs_1013"),
         ("solver", "1018"): os.path.join(DATA, "central_ridge_solver_gpu_runs_1018"),
         ("rankiso", "1013"): os.path.join(DATA, "central_ridge_rankiso_gpu_runs_1013"),
         ("rankiso", "1018"): os.path.join(DATA, "central_ridge_rankiso_gpu_runs_1018")}
for _r in ROOTS.values():
    if not os.path.isdir(_r):
        raise SystemExit("run directory %s not found (set SSLABLU_EDA_DATA or edit ROOTS)" % _r)
FIELDS = [f + "_" + t + "_320" for f in ("eta", "u", "v") for t in ("mid", "final")]
rows = []
for (sweep, rep), root in ROOTS.items():
    logs = {os.path.basename(l): l for l in glob.glob(root + "/logs/**/*.log", recursive=True)}
    for d in sorted(glob.glob(root + "/run_sslablu_channel_*_nsteps800_*")):
        S = np.load(d + "/channel_timestep_ssh.npz")
        r = dict(sweep=sweep, rep=rep, dir=os.path.basename(d),
                 p=int(S["p"]), N=int(S["N"]), npx=int(S["npan_x"]), npy=int(S["npan_y"]),
                 solver=str(S["solver"]), rk=int(S["rk"]), rbrk=int(S["rb_rk"]) if str(S["solver"]) == "rbhbs" else 0,
                 check=float(S["solver_check"]), mean_mid=float(S["eta_mean_mid"]), mean_final=float(S["eta_mean_final"]))
        for f in FIELDS:
            r[f] = S[f].copy()
        C = np.genfromtxt(d + "/channel_timestep_diag.csv", delimiter=",", names=True)
        r.update(t_rhs=C["t_rhs"].mean(), t_solve=C["t_solve"].mean(), t_recon=C["t_recon"].mean(),
                 t_rhs_cv=np.std(C["t_rhs"]) / C["t_rhs"].mean(), mass_series=C["mass_resid"].copy(),
                 energy=C["energy"][-1])
        tag = ("rb" if r["solver"] == "rb" and r["rk"] == 0 else
               "rb_rk%d" % r["rk"] if r["solver"] == "rb" else
               "rbhbs_rk%d" % r["rk"] if (r["rk"] == r["rbrk"] and sweep == "solver") else
               "rbhbs_rk%d_rbrk%d" % (r["rk"], r["rbrk"]))
        lname = "p%d_N%d_pan%dx%d_%s.log" % (r["p"], r["N"], r["npx"], r["npy"], tag)
        txt = open(logs[lname]).read() if lname in logs else ""
        m = re.search(r"\[divergence form\]\s+assemble/factor/keep-slabs = ([\d.]+) / ([\d.]+) / ([\d.]+)", txt)
        r["asm"], r["fac"], r["keep"] = (float(x) for x in m.groups()) if m else (np.nan,) * 3
        r["forms"] = 2 if "[non-conservative]" in txt else 1
        su = re.search(r"setup \(both forms\)\s+=\s+([\d.]+)", txt)
        r["setup"] = float(su.group(1)) if su else np.nan
        mr = re.search(r"residual \(wall leakage\)=\s+(\S+)\s+\((\S+) rel\)", txt)
        r["mass_rel"] = float(mr.group(2)) if mr else np.nan
        ns = re.search(r"HBS red-black.*?(\d+) samples", txt)
        r["log_found"] = bool(txt)
        rows.append(r)
pickle.dump(rows, open(ROWS_PKL, "wb"))
print(len(rows), "runs;", sum(not r["log_found"] for r in rows), "without a log;",
      sum(np.isnan(r["asm"]) for r in rows), "without timings")
