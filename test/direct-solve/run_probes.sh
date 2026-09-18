#!/usr/bin/env bash
#
# run_probes.sh -- drive precision_probe.py to answer two questions:
#
#   (A) does RIGHT ERR decay with HBS rank at fixed kh, or is it floored?
#   (B) what is the local discretization error at production p / leaf size?
#
# Discretization is held fixed at production settings throughout; only the HBS
# rank varies in (A). Logs are kept per run so nothing has to be rerun to look
# at something that scrolled past.
#
# Usage:
#   ./run_probes.sh                       # defaults below
#   ./run_probes.sh --ranks "64 128 256"  # custom sweep
#   ./run_probes.sh --skip-sweep          # only the discretization probe
#   ./run_probes.sh --skip-local          # only the rank sweep
#
set -uo pipefail

# ---------------------------------------------------------------- defaults ---
PY=${PY:-python}
SCRIPT=${SCRIPT:-precision_probe.py}
OUTDIR=${OUTDIR:-probe_results_$(date +%Y%m%d_%H%M%S)}

KH=99.7
N=33
P=10
LEAF=0.03125          # discretization leaf width in y,z (production)
LEAF_SIZE=800         # HBS leaf
RANKS="32 64 128 192 256"
REDUCED_GPU=1
DO_SWEEP=1
DO_LOCAL=1

# ------------------------------------------------------------------- args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --kh)          KH="$2";          shift 2 ;;
    --N)           N="$2";           shift 2 ;;
    --p)           P="$2";           shift 2 ;;
    --leaf)        LEAF="$2";        shift 2 ;;
    --leaf-size)   LEAF_SIZE="$2";   shift 2 ;;
    --ranks)       RANKS="$2";       shift 2 ;;
    --reduced-gpu) REDUCED_GPU="$2"; shift 2 ;;
    --outdir)      OUTDIR="$2";      shift 2 ;;
    --skip-sweep)  DO_SWEEP=0;       shift   ;;
    --skip-local)  DO_LOCAL=0;       shift   ;;
    -h|--help)     sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -f "$SCRIPT" ]]; then
  echo "cannot find $SCRIPT (set SCRIPT=/path/to/precision_probe.py)" >&2
  exit 1
fi

mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.txt"

{
  echo "run_probes.sh  $(date)"
  echo "kh=$KH  N=$N  p=$P  leaf=$LEAF  leaf_size=$LEAF_SIZE  reduced_gpu=$REDUCED_GPU"
  echo
} | tee "$SUMMARY"

COMMON=(--kh "$KH" --N "$N" --p "$P" --leaf "$LEAF"
        --leaf_size "$LEAF_SIZE" --reduced_gpu "$REDUCED_GPU")

# ------------------------------------------------------- (A) rank sweep -----
if [[ "$DO_SWEEP" == "1" ]]; then
  {
    echo "=== (A) RIGHT ERR vs HBS rank, kh=$KH, discretization fixed ==="
    printf "%8s  %14s  %14s\n" "rank" "RIGHT_ERR" "LEFT_ERR"
  } | tee -a "$SUMMARY"

  for r in $RANKS; do
    LOG="$OUTDIR/sweep_rank${r}.log"
    "$PY" "$SCRIPT" "${COMMON[@]}" --rank "$r" >"$LOG" 2>&1
    rc=$?

    RE=$(grep -m1 "RIGHT ERR" "$LOG" | awk -F'=' '{print $2}' | tr -d ' ')
    LE=$(grep -m1 "LEFT ERR"  "$LOG" | awk -F'=' '{print $2}' | tr -d ' ')
    [[ -z "$RE" ]] && RE="(rc=$rc, see $(basename "$LOG"))"
    [[ -z "$LE" ]] && LE="-"

    printf "%8s  %14s  %14s\n" "$r" "$RE" "$LE" | tee -a "$SUMMARY"
  done
  echo | tee -a "$SUMMARY"
fi

# ----------------------------------------- (B) discretization probe ---------
if [[ "$DO_LOCAL" == "1" ]]; then
  {
    echo "=== (B) local Dirichlet solve vs exact, p=$P, leaf=$LEAF ==="
  } | tee -a "$SUMMARY"

  LOG="$OUTDIR/local_solve.log"
  "$PY" "$SCRIPT" "${COMMON[@]}" --rank 500 --skip_compression >"$LOG" 2>&1

  grep -E "points per wavelength|u_h - u|dtype=" "$LOG" | tee -a "$SUMMARY"
  echo | tee -a "$SUMMARY"
fi

# ---------------------------------------------------------- how to read -----
{
  echo "=== how to read this ==="
  echo "(A) log(RIGHT ERR) roughly linear in rank -> genuine spectral decay;"
  echo "    more rank buys more digits, predictably but expensively."
  echo "(A) curve flattens hard              -> real floor; look inside"
  echo "    HBStorch's error measurement next."
  echo "(B) ~1e-8  -> discretization is not the constraint; rank is the knob."
  echo "(B) ~1e-6  -> the 8.6e-6 local error is mostly discretization"
  echo "    amplified by the cube factor; raise p, rank will not help."
  echo
  echo "logs: $OUTDIR/"
} | tee -a "$SUMMARY"
