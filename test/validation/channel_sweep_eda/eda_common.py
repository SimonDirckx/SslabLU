"""Shared paths and helpers for the channel sweep EDA scripts.

Exploratory analysis of the channel_ridgectr_sweep.py runs (solver and rankiso
presets, two replicates each). Run from anywhere, in order:
  python test/validation/channel_sweep_eda/eda_load.py      # runs -> eda_rows.pkl
  python test/validation/channel_sweep_eda/eda_analyze.py   # tables A-E (stdout)
  python test/validation/channel_sweep_eda/eda_extra.py     # noise floor, rank needed, clean cost
  python test/validation/channel_sweep_eda/eda_figs.py      # three PNGs

Paths: run directories are read from SSLABLU_EDA_DATA (default
<repo>/central_ridge_rank_old_hps_mem_runs/, see eda_load.ROOTS); every
intermediate pickle and figure goes to SSLABLU_EDA_OUT (default
<repo>/channel_sweep_eda_out/). `rows` is the table eda_load.py writes, loaded
on import when it exists; each row is one run, tagged here with
  kind  rb | smap (rb on RK > 0 S-maps) | solver (rbhbs, S-maps at RK = 1.5p)
        | both (rbhbs with RK = RB_RK)
  rank  the rank that kind varies (0 for rb).
"""
import os
import pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.environ.get("SSLABLU_EDA_OUT", os.path.join(REPO, "channel_sweep_eda_out"))
os.makedirs(OUT, exist_ok=True)
ROWS_PKL = os.path.join(OUT, "eda_rows.pkl")
rows = pickle.load(open(ROWS_PKL, "rb")) if os.path.exists(ROWS_PKL) else []



def kind(r):
    if r["solver"] == "rb":
        return "rb" if r["rk"] == 0 else "smap"
    if r["sweep"] == "solver":
        return "both"
    return "solver" if r["rk"] != r["rbrk"] or r["rk"] == int(1.5 * r["p"]) else "both"


for r in rows:
    r["kind"] = kind(r)
    r["rank"] = {"rb": 0, "smap": r["rk"], "solver": r["rbrk"], "both": r["rk"]}[r["kind"]]


def fld(r, f="eta", t="final"):
    a = r["%s_%s_320" % (f, t)]
    return a - r["mean_" + t] if f == "eta" else a


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def find(**kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def one(**kw):
    x = find(**kw)
    return x[0] if x else None


def rbof(r, rep=None):
    return one(sweep=r["sweep"], rep=rep or r["rep"], npy=r["npy"], p=r["p"], N=r["N"], kind="rb")
