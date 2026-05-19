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
#   4. Recreates fused_direct.zarr with the registered canvas (H,W computed
#      from stitch_positions.json) so Stage 2 array tasks can write into
#      pre-allocated chunks.
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
test -f "$POS_JSON"  || { echo "ERROR: $POS_JSON not produced"; exit 1; }
test -f "$DIAG_JSON" || { echo "ERROR: $DIAG_JSON not produced"; exit 1; }

# --------------------------------------------------------------------------
# 2) Recreate fused_direct.zarr at the registered canvas size.
#    Safe here: Stage 1 is a single task; no Stage 2 array task is reading
#    the zarr yet. Stage 2 always uses mode="r+".
# --------------------------------------------------------------------------
python3 - <<PYEOF
import json
import numpy as np
import zarr

POS_JSON = "$POS_JSON"
OUT_ZARR = "$OUT_ZARR"
N_Z      = 1557
TILE_H   = 1920   # layout["tile_h"] verified for KL018 (RESEARCH §Architecture / Phase 2)
TILE_W   = 1920   # layout["tile_w"]

with open(POS_JSON) as f:
    pos = json.load(f)

xs = [float(p["translation_px"]["x"]) for p in pos.values()]
ys = [float(p["translation_px"]["y"]) for p in pos.values()]
H  = int(round(max(ys) + TILE_H - min(ys)))
W  = int(round(max(xs) + TILE_W - min(xs)))

# Defensive: stage canvas was 8585x10095 - registered canvas should be
# within +/- 32 px of that. If it explodes, error out before destroying the file.
if H > 12000 or W > 14000 or H < 4000 or W < 4000:
    raise SystemExit(
        f"ERROR: registered canvas H={H} W={W} outside expected range; "
        "refusing to recreate fused_direct.zarr"
    )

print(f"Registered canvas: H={H}  W={W}  (n_tiles={len(pos)})")
zarr.open(
    OUT_ZARR,
    mode="w",
    shape=(1, 2, N_Z, H, W),
    chunks=(1, 1, 64, 512, 512),
    dtype=np.uint16,
)
print(f"Recreated zarr at {OUT_ZARR} with shape (1, 2, {N_Z}, {H}, {W})")
PYEOF

echo "=== Stage 1 complete. Submit Stage 2:"
echo "  sbatch --dependency=afterok:\${SLURM_JOB_ID:-<this_jobid>} 08_stitch_stage2_blend.sh"

# STITCH-01: globally-optimised registration via multiview_stitcher
# STITCH-06: 07_direct_fuse.py fuse_sides / _gaussian_ramp / read_tile_zchunk untouched
