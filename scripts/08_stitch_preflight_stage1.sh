#!/bin/bash
#SBATCH --job-name=ls_stitch_preflight_reg
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/vast/scratch/users/kriel.j/output.preflight_reg.%j.%N.log
#SBATCH --error=/vast/scratch/users/kriel.j/output.preflight_reg.%j.%N.log

# Phase 2.5 SUBSTACK PRE-FLIGHT — Stage 1 register-ONLY
#
# Differs from the production 08_stitch_stage1_register.sh in two ways:
#   1. Writes stitch_positions.json + stitch_diagnostics.json to the preflight
#      subdir, NOT the production scratch root.
#   2. Does NOT recreate fused_direct.zarr — the Phase 2 zarr is left alone
#      so it can be used as a side-by-side reference in 08_fusion_dev.ipynb.
#
# Preflight-specific speed-ups vs. production Stage 1 (goal: <30 min):
#   --z-slab-half 16       32-plane slab (vs. 64 in production) — 2x less
#                          data per pair; registration is compute-bound on
#                          Z × 1920 × 1920 phase-correlation FFTs.
#                          NB: an earlier preflight used --z-slab-half 8
#                          --reg-z-bin 2 (effective Z = 8 planes). That
#                          starved phase correlation of out-of-plane signal
#                          and produced quality=0.0 on every edge. See debug
#                          session preflight-stage2-canvas-too-small for the
#                          full investigation.
#   --reg-z-bin 1          No Z binning during phase correlation. Full
#                          out-of-plane signal preserved (binning was 2x in
#                          the prior preflight; reverting per debug session
#                          findings).
#   --pre-reg-pruning-method keep_axis_aligned
#                          Restricts pairwise candidates to axis-aligned
#                          neighbors only (49 pairs for a 6x5 grid), vs. the
#                          library default "alternating_pattern" which can
#                          leave additional diagonal pairs in a grid with
#                          non-zero diagonal overlap.
#
# Expected pair count and runtime after these changes:
#   49 pairs × ~15s/pair (32 planes, no binning) ≈ 12 min registration
#   Total Stage 1 wall time: ~20-25 min (I/O + graph build + register + write)
#
# Pre-flight purpose: catch algorithmic regressions in ~25 min of GPU time,
# vs. hours wasted on a full-scale production submission.

set -euo pipefail

SCRATCH="/vast/scratch/users/kriel.j/KL018_lightsheet"
PREFLIGHT="$SCRATCH/preflight"
CZI_PATH="$SCRATCH/KL018_85_D7_CT2AvIII_Overview.czi"
SCRIPT_DIR="/vast/projects/BCRL_Multi_Omics/scripts/lightsheet_pipeline"
ENV_DIR="/vast/scratch/users/kriel.j/mvstitch_env"

mkdir -p "$PREFLIGHT"

if ! command -v module &>/dev/null; then
    source /etc/profile.d/modules.sh
fi
module load CUDA/12.1

# Use the env's Python directly so the script works on any node regardless of
# whether the stornext conda init path is mounted (conda activate is a no-op
# when conda.sh is unavailable, but returns exit 0 — set -e won't catch it).
PYTHON="$ENV_DIR/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: $PYTHON not found — is $ENV_DIR on VAST?" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1

echo "=== Pre-flight Stage 1: global registration (substack scope) ==="
echo "SLURM job:    ${SLURM_JOB_ID:-interactive}  Node: ${SLURMD_NODENAME:-local}"
echo "Pre-flight out dir: $PREFLIGHT"
echo "Production zarr (UNTOUCHED): $SCRATCH/fused_direct.zarr"
"$PYTHON" -c "import multiview_stitcher, dask; print('multiview-stitcher', multiview_stitcher.__version__, ' dask', dask.__version__)"

# --out-zarr is required by argparse but stage_register doesn't open/write to it.
# Point it at a placeholder path under preflight so any future stage_register
# code change that DOES touch it cannot reach the production zarr.
"$PYTHON" -u "$SCRIPT_DIR/08_stitch.py" \
    --stage register \
    --czi "$CZI_PATH" \
    --out-dir "$PREFLIGHT" \
    --out-zarr "$PREFLIGHT/_unused_placeholder.zarr" \
    --z-slab-half 16 \
    --reg-z-bin 1 \
    --pre-reg-pruning-method keep_axis_aligned

test -f "$PREFLIGHT/stitch_positions.json"   || { echo "ERROR: positions JSON not produced"; exit 1; }
test -f "$PREFLIGHT/stitch_diagnostics.json" || { echo "ERROR: diagnostics JSON not produced"; exit 1; }

echo ""
echo "=== Pre-flight Stage 1 done ==="
echo "Positions:   $PREFLIGHT/stitch_positions.json"
echo "Diagnostics: $PREFLIGHT/stitch_diagnostics.json"

# Quick gate check inline so the SLURM log surfaces converged + max_residual.
# Also checks median pairwise quality (>0.2) and canvas extent (within 1000 px
# of layout cache) — both regressions that were silent in the prior preflight.
"$PYTHON" - <<PYEOF
import json
import statistics
import sys

d = json.load(open("$PREFLIGHT/stitch_diagnostics.json"))
g = d.get("groupwise", {})
pairs = d.get("pairwise", [])
n_pairs = len(pairs)
qualities = [p.get("quality", 0.0) for p in pairs]
median_q = statistics.median(qualities) if qualities else 0.0
canvas = d.get("canvas_summary", {})
reg_canvas = canvas.get("from_registration_px", {})
lay_canvas = canvas.get("from_layout_cache_px", {})

print(f"  n_tiles:        {d.get('n_tiles')}")
print(f"  z_slab_used:    {d.get('z_slab_used')}")
print(f"  pairwise edges: {n_pairs}")
print(f"  median quality: {median_q:.3f}")
print(f"  converged:      {g.get('converged')}")
print(f"  rms_residual:   {g.get('rms_residual_px')}")
print(f"  max_residual:   {g.get('max_residual_px')}")
print(f"  n_constraints:  {g.get('n_constraints')}")
print(f"  canvas (reg):   H={reg_canvas.get('h')} W={reg_canvas.get('w')}")
print(f"  canvas (layout):H={lay_canvas.get('h')} W={lay_canvas.get('w')}")

checks = []
checks.append(("converged", bool(g.get("converged"))))
checks.append(("max_residual<5",
               float(g.get("max_residual_px", 99)) < 5.0))
checks.append(("median_quality>0.2", median_q > 0.2))
checks.append(("n_pairs>=10", n_pairs >= 10))
# Canvas must be close to the layout's intrinsic canvas (CZI stage metadata).
# Allow ±1000 px slack for cumulative registration corrections.
if reg_canvas and lay_canvas:
    dh = abs(int(reg_canvas.get("h", 0)) - int(lay_canvas.get("h", 0)))
    dw = abs(int(reg_canvas.get("w", 0)) - int(lay_canvas.get("w", 0)))
    checks.append(("canvas_h_within_1000px_of_layout", dh < 1000))
    checks.append(("canvas_w_within_1000px_of_layout", dw < 1000))

ok = all(v for _, v in checks)
print("  Gate checks:")
for name, v in checks:
    print(f"    {'PASS' if v else 'FAIL'}  {name}")
print(f"  pre-flight gate: {'PASS' if ok else 'FAIL'}")
if not ok:
    sys.exit(1)
PYEOF
