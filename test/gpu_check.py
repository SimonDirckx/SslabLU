#!/usr/bin/env python3
# =============================================================================
# gpu_check.py
#
# Does PyTorch actually run on the GPU here, and does hpsmultidomain use it?
#   1. environment   CUDA_VISIBLE_DEVICES / CUDA_DEVICE_ORDER, nvidia-smi's GPU
#                    list vs what torch enumerates, device compute capability
#                    vs the sm_XX kernels this torch build ships
#   2. kernels       float64 elementwise, matmul and batched linalg.solve on
#                    every visible device (catches "no kernel image")
#   3. hpsalt DtNs   HPS_Multidomain.get_DtNs(device, 'build') on a small 2D
#                    Poisson tiling, on CPU and on CUDA: the two must agree,
#                    and the CUDA run must allocate device memory
#   4. SslabLU path  what the slab solvers request: hpsalt builds on the GPU
#                    only with reduced_gpu=True (solverWrap default False);
#                    HBStorch (rkHBS, RK > 0) picks cuda:0 whenever
#                    torch.cuda.is_available()
# Exit status 0 only if every check that ran passed.
#
# Usage (from the repo root):  python test/gpu_check.py
#   SSLABLU_GPUCHECK_P     leaf order p for check 3 (default 12)
#   SSLABLU_GPUCHECK_NPAN  panels per dimension    (default 16)
# =============================================================================

import os
import sys
import time
import subprocess

import numpy as np
import torch

torch.set_default_dtype(torch.double)
results = []       # (check, ok, detail)


def record(name, ok, detail=""):
    results.append((name, ok, detail))
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, (": " + detail) if detail else ""))


# ---- 1. environment -------------------------------------------------------------
print("1. environment")
print("  CUDA_VISIBLE_DEVICES = %r, CUDA_DEVICE_ORDER = %r"
      % (os.environ.get("CUDA_VISIBLE_DEVICES"), os.environ.get("CUDA_DEVICE_ORDER")))
try:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=index,name,compute_cap,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True, check=True)
    for line in smi.stdout.strip().splitlines():
        print("  nvidia-smi (PCI order, ignores CUDA_VISIBLE_DEVICES): " + line)
except (OSError, subprocess.CalledProcessError) as e:
    print("  nvidia-smi unavailable: %s" % e)
print("  torch %s, CUDA build %s, driver-visible devices %d"
      % (torch.__version__, torch.version.cuda, torch.cuda.device_count() if torch.cuda.is_available() else 0))
record("torch.cuda.is_available()", torch.cuda.is_available())
arch = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
ndev = torch.cuda.device_count() if torch.cuda.is_available() else 0
for i in range(ndev):
    cc = torch.cuda.get_device_capability(i)
    sm = "sm_%d%d" % cc
    # a compute_XY PTX entry can JIT for devices with CC >= XY
    ptx = [a for a in arch if a.startswith("compute_") and int(a[8:]) <= 10 * cc[0] + cc[1]]
    record("cuda:%d %s (%s) has kernels in this torch build" % (i, torch.cuda.get_device_name(i), sm),
           sm in arch or bool(ptx), "build ships " + " ".join(arch))

# ---- 2. kernels -----------------------------------------------------------------
print("2. kernels")
for i in range(ndev):
    dev = torch.device("cuda:%d" % i)
    try:
        A = torch.randn(64, 48, 48, device=dev) + 48 * torch.eye(48, device=dev)
        B = torch.randn(64, 48, 8, device=dev)
        X = torch.linalg.solve(A, B)
        Xc = torch.linalg.solve(A.cpu(), B.cpu())
        e = float((X.cpu() - Xc).abs().max() / Xc.abs().max())
        torch.cuda.synchronize(dev)
        record("cuda:%d float64 batched solve" % i, e < 1e-12, "rel diff vs CPU %.1e" % e)
    except Exception as ex:
        record("cuda:%d float64 batched solve" % i, False, "%s: %s" % (type(ex).__name__, str(ex).splitlines()[0]))
gpu_ok = ndev > 0 and all(ok for n, ok, _ in results if n.startswith("cuda:"))

# ---- 3. hpsmultidomain DtN build ----------------------------------------------------
print("3. hpsmultidomain get_DtNs(device, 'build')")
from hpsmultidomain import pdo
from hpsmultidomain.domain_driver import Domain_Driver
from hpsmultidomain.geom import BoxGeometry
import hpsmultidomain
print("  hpsmultidomain imported from %s" % os.path.dirname(hpsmultidomain.__file__))

p = int(os.environ.get("SSLABLU_GPUCHECK_P", "12"))
npan = int(os.environ.get("SSLABLU_GPUCHECK_NPAN", "16"))
box = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
a = np.array([0.5 / npan, 0.5 / npan])
drv = Domain_Driver(BoxGeometry(box), pdo.PDO_2d(c11=pdo.ones, c22=pdo.ones), 0, a,
                    p=np.array([p, p]), d=2)
print("  %d leaves, p = %d" % (drv.hps.nboxes, p))
tic = time.time()
D_cpu = drv.hps.get_DtNs(torch.device("cpu"), "build")
t_cpu = time.time() - tic
print("  cpu : %.3f s" % t_cpu)
if not gpu_ok:
    record("hpsalt DtNs on CUDA", False, "skipped: checks 1-2 failed, no usable GPU")
else:
    dev = torch.device("cuda")   # get_DtN_chunksize compares against exactly this
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        drv.hps.get_DtNs(dev, "build")           # warm-up (CUDA context, cuBLAS handles)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        tic = time.time()
        D_gpu = drv.hps.get_DtNs(dev, "build")
        torch.cuda.synchronize()
        t_gpu = time.time() - tic
        peak = torch.cuda.max_memory_allocated() / 2**20
        e = float((D_gpu.cpu() - D_cpu).abs().max() / D_cpu.abs().max())
        print("  cuda: %.3f s (%.1fx), peak device memory %.1f MiB, on %s"
              % (t_gpu, t_cpu / t_gpu, peak, torch.cuda.get_device_name()))
        record("hpsalt DtNs on CUDA match CPU", e < 1e-10, "rel diff %.1e" % e)
        record("hpsalt DtNs on CUDA allocated device memory", peak > 0, "%.1f MiB" % peak)
    except Exception as ex:
        record("hpsalt DtNs on CUDA", False, "%s: %s" % (type(ex).__name__, str(ex).splitlines()[0]))

# ---- 4. what the SslabLU slab solvers request -------------------------------------------
print("4. SslabLU device selection (informational)")
import inspect
import solver.solver as solverWrap
sig = inspect.signature(solverWrap.solverOptions.__init__) if hasattr(solverWrap, "solverOptions") else None
default = sig.parameters["reduced_gpu"].default if sig and "reduced_gpu" in sig.parameters else "?"
print("  hpsalt slab build: 'reduced_gpu' if opts.reduced_gpu else 'reduced_cpu' "
      "(default reduced_gpu=%s) -> %s" % (default, "GPU" if default is True else "CPU unless a script sets it"))
print("  HBStorch (rkHBS, RK > 0): cuda:0 if torch.cuda.is_available() -> %s"
      % ("cuda:0 (" + torch.cuda.get_device_name(0) + ")" + ("" if gpu_ok else ", which CANNOT run kernels")
         if torch.cuda.is_available() else "cpu"))

print("")
print("=============SUMMARY=============")
for name, ok, detail in results:
    print("  %s  %s" % ("PASS" if ok else "FAIL", name))
print("=================================")
sys.exit(0 if all(ok for _, ok, _ in results) else 1)
