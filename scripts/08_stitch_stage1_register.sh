#!/bin/bash
#SBATCH --job-name=ls_stitch_reg
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/vast/scratch/users/kriel.j/output.%j.%N.log
#SBATCH --error=/vast/scratch/users/kriel.j/output.%j.%N.log

# Phase 2.5 Stage 1: globally-optimised tile registration for KL018.
#
# This job:
#   1. Activates mvstitch_env (multiview-stitcher 0.1.52; NOT lightsheet_env).
#   2. Runs 08_stitch.py --stage register, which reads a 64-plane Z-slab around
#      mid-Z for each of 30 tiles, applies the validated dual-side fuse_sides
#      path (imported unchanged from 07_direct_fuse.py — STITCH-06), and calls
#      multiview_stitcher.registration.register with
#      groupwise_resolution_method="global_optimization" (STITCH-01).
#   3. Writes stitch_positions.json + stitch_diagnostics.json to $SCRATCH.
#   4. Recreates fused_direct.zarr with the layout-cache canvas (the CZI
#      stage-metadata mosaic bbox is the authoritative canvas; registered
#      positions are stage + correction, where corrections are nominally
#      a few pixels). Stage 2 array tasks then write into pre-allocated
#      chunks via mode="r+".
#
# After this job succeeds, submit Stage 2:
#   sbatch --dependency=afterok:$SLURM_JOB_ID 08_stitch_stage2_blend.sh
#
# Usage:
#   sbatch 08_stitch_stage1_register.sh

set -euo pipefail

SCRATCH="/vast/scratch/users/kriel.j/KL018_lightsheet"
CZI_PATH="$SCRATCH/KL018_85_D7_CT2AvIII_Overview.czi"
OUT_ZARR="$SCRATCH/fused_direct.zarr"
POS_JSON="$SCRATCH/stitch_positions.json"
DIAG_JSON="$SCRATCH/stitch_diagnostics.json"
LAYOUT_CACHE="$SCRATCH/czi_layout_cache.json"
SCRIPT_DIR="/vast/projects/BCRL_Multi_Omics/scripts/lightsheet_pipeline"
ENV_DIR="/vast/scratch/users/kriel.j/mvstitch_env"

if ! command -v module &>/dev/null; then
    source /etc/profile.d/modules.sh
fi
module load CUDA/12.1

source /stornext/System/data/apps/anaconda3/anaconda3-latest/etc/profile.d/conda.sh
conda activate "$ENV_DIR"

echo "=== Stage 1: global registration ==="
echo "SLURM job:  ${SLURM_JOB_ID:-interactive}  Node: ${SLURMD_NODENAME:-local}"
echo "Scratch:    $SCRATCH"
echo "CZI:        $CZI_PATH"
echo "Out zarr:   $OUT_ZARR"
echo "Env:        $ENV_DIR"
python -c "import multiview_stitcher, dask; print('multiview-stitcher', multiview_stitcher.__version__, ' dask', dask.__version__)"

# --------------------------------------------------------------------------
# 1) Register
# --------------------------------------------------------------------------
python "$SCRIPT_DIR/08_stitch.py" \
    --stage register \
    --czi "$CZI_PATH" \
    --out-dir "$SCRATCH" \
    --out-zarr "$OUT_ZARR"

# Sanity-check Stage 1 outputs before recreating the zarr.
test -f "$POS_JSON"     || { echo "ERROR: $POS_JSON not produced"; exit 1; }
test -f "$DIAG_JSON"    || { echo "ERROR: $DIAG_JSON not produced"; exit 1; }
test -f "$LAYOUT_CACHE" || { echo "ERROR: $LAYOUT_CACHE not produced"; exit 1; }

# --------------------------------------------------------------------------
# 2) Recreate fused_direct.zarr at the layout-cache canvas size.
#    Safe here: Stage 1 is a single task; no Stage 2 array task is reading
#    the zarr yet. Stage 2 always uses mode="r+".
#
#    Canvas authority: the CZI's own mosaic bounding box (layout cache).
#    Registered positions from stitch_positions.json are the source of
#    truth for tile placement during blending, but for canvas pre-allocation
#    we use the layout cache directly. Rationale: in multiview-stitcher
#    0.1.52 the global solver's behaviour around the reference tile can
#    produce surprising canvas extents from naive max-min math; using the
#    layout cache decouples canvas size from solver internals. This caught
#    a regression in the preflight (debug session preflight-stage2-canvas-
#    too-small) where stage_register stored re-anchored deltas instead of
#    absolute positions; the layout-cache path would have pre-allocated
#    the correct canvas even in that scenario.
# --------------------------------------------------------------------------
python3 - <<PYEOF
import json
import sys
import numpy as np
import zarr

POS_JSON     = "$POS_JSON"
LAYOUT_CACHE = "$LAYOUT_CACHE"
OUT_ZARR     = "$OUT_ZARR"
TILE_H       = 1920
TILE_W       = 1920

with open(LAYOUT_CACHE) as f:
    layout = json.load(f)
with open(POS_JSON) as f:
    pos = json.load(f)

H_layout = int(layout["canvas_h"])
W_layout = int(layout["canvas_w"])
N_Z      = int(layout["n_z"])

# Cross-check the registered positions against the layout canvas.
xs = [float(p["translation_px"]["x"]) for p in pos.values()]
ys = [float(p["translation_px"]["y"]) for p in pos.values()]
H_reg = int(round(max(ys) + TILE_H - min(ys)))
W_reg = int(round(max(xs) + TILE_W - min(xs)))

print(f"Canvas (layout cache):        H={H_layout}  W={W_layout}  Z={N_Z}")
print(f"Canvas (from registered pos): H={H_reg}     W={W_reg}     (n_tiles={len(pos)})")

# Defensive: layout canvas should be in the expected KL018 range.
if H_layout > 12000 or W_layout > 14000 or H_layout < 4000 or W_layout < 4000:
    raise SystemExit(
        f"ERROR: layout canvas H={H_layout} W={W_layout} outside expected range; "
        "refusing to recreate fused_direct.zarr"
    )

# Warn (do not fail) if registered canvas diverges substantially from layout.
if abs(H_reg - H_layout) > 1000 or abs(W_reg - W_layout) > 1000:
    print(
        f"WARNING: registered canvas differs from layout canvas by >1000 px. "
        f"Layout canvas is being used for pre-allocation, but investigate "
        f"stitch_positions.json before trusting Stage 2 output.",
        file=sys.stderr,
    )

print(f"Recreating zarr at {OUT_ZARR}: shape=(1, 2, {N_Z}, {H_layout}, {W_layout})")
zarr.open(
    OUT_ZARR,
    mode="w",
    shape=(1, 2, N_Z, H_layout, W_layout),
    chunks=(1, 1, 64, 512, 512),
    dtype=np.uint16,
)
print(f"Recreated zarr at {OUT_ZARR}")
PYEOF

echo "=== Stage 1 complete. Submit Stage 2:"
echo "  sbatch --dependency=afterok:\${SLURM_JOB_ID:-<this_jobid>} 08_stitch_stage2_blend.sh"

# STITCH-01: globally-optimised registration via multiview_stitcher
# STITCH-06: 07_direct_fuse.py fuse_sides / _gaussian_ramp / read_tile_zchunk untouched
