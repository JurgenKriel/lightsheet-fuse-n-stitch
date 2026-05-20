#!/bin/bash
#SBATCH --job-name=ls_stitch_preflight_blend
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=150G
#SBATCH --time=00:45:00
#SBATCH --output=/vast/scratch/users/kriel.j/output.preflight_blend.%j.%N.log
#SBATCH --error=/vast/scratch/users/kriel.j/output.preflight_blend.%j.%N.log

# Phase 2.5 SUBSTACK PRE-FLIGHT — Stage 2 blend on 100-plane substack.
#
# Differs from the production 08_stitch_stage2_blend.sh:
#   - Not an array — single task that blends ONE 100-plane substack.
#   - Reads positions from PREFLIGHT subdir (produced by 08_stitch_preflight_stage1.sh).
#   - Writes to substack_z728_828_stitched.zarr (NOT fused_direct.zarr) so the
#     Phase 2 reference zarr stays intact for side-by-side comparison in
#     08_fusion_dev.ipynb Phase 2.5 cell.
#
# Z range = 728:828 — same 100-plane mid-volume range as Phase 1 substack work,
# centred on z=778 (the seam-quality reference plane).
#
# Wall time: ~15-25 min on A30. Submit with afterok dependency:
#   sbatch --dependency=afterok:<preflight_stage1_jobid> 08_stitch_preflight_stage2.sh

set -euo pipefail

SCRATCH="/vast/scratch/users/kriel.j/KL018_lightsheet"
PREFLIGHT="$SCRATCH/preflight"
CZI_PATH="$SCRATCH/KL018_85_D7_CT2AvIII_Overview.czi"
SUBSTACK_ZARR="$SCRATCH/substack_z728_828_stitched.zarr"
POS_JSON="$PREFLIGHT/stitch_positions.json"
SCRIPT_DIR="/vast/projects/BCRL_Multi_Omics/scripts/lightsheet_pipeline"
ENV_DIR="/vast/scratch/users/kriel.j/mvstitch_env"

export CUDA_VISIBLE_DEVICES=0

if ! command -v module &>/dev/null; then
    source /etc/profile.d/modules.sh
fi
module load CUDA/12.1

source /stornext/System/data/apps/anaconda3/anaconda3-latest/etc/profile.d/conda.sh
conda activate "$ENV_DIR"

test -f "$POS_JSON" || { echo "ERROR: $POS_JSON missing — did pre-flight Stage 1 finish?"; exit 1; }

echo "=== Pre-flight Stage 2: substack blend [z=728:828) ==="
echo "SLURM job:      ${SLURM_JOB_ID:-interactive}  Node: ${SLURMD_NODENAME:-local}"
echo "Positions:      $POS_JSON"
echo "Out substack:   $SUBSTACK_ZARR"
echo "Production zarr (UNTOUCHED): $SCRATCH/fused_direct.zarr"
python -c "import multiview_stitcher, dask; print('multiview-stitcher', multiview_stitcher.__version__, ' dask', dask.__version__)"

# --------------------------------------------------------------------------
# 1) Pre-allocate substack_z728_828_stitched.zarr at the registered canvas.
#    Full Z=1557 so stage_blend's absolute-index writes (z=728:828) land in
#    the right place; planes outside that range stay zero. This matches the
#    08_fusion_dev.ipynb cell's mid-Z crop logic (z_arr.shape[2]//2 = 778).
# --------------------------------------------------------------------------
python3 - <<PYEOF
import json
import numpy as np
import zarr

POS_JSON      = "$POS_JSON"
SUBSTACK_ZARR = "$SUBSTACK_ZARR"
N_Z           = 1557
TILE_H        = 1920
TILE_W        = 1920

with open(POS_JSON) as f:
    pos = json.load(f)

xs = [float(p["translation_px"]["x"]) for p in pos.values()]
ys = [float(p["translation_px"]["y"]) for p in pos.values()]
H  = int(round(max(ys) + TILE_H - min(ys)))
W  = int(round(max(xs) + TILE_W - min(xs)))

if H > 12000 or W > 14000 or H < 4000 or W < 4000:
    raise SystemExit(f"ERROR: registered canvas H={H} W={W} outside expected range")

print(f"Registered canvas: H={H}  W={W}  (n_tiles={len(pos)})")
zarr.open(
    SUBSTACK_ZARR,
    mode="w",
    shape=(1, 2, N_Z, H, W),
    chunks=(1, 1, 64, 512, 512),
    dtype=np.uint16,
)
print(f"Pre-allocated substack zarr at {SUBSTACK_ZARR}")
PYEOF

# --------------------------------------------------------------------------
# 2) Blend z=728:828 (chunk-aligned at 64-plane boundaries: 704, 768, 832 →
#    the [728:828] window straddles chunks 11 + 12. stage_blend handles this
#    automatically via its --z-chunk 64 internal loop.)
# --------------------------------------------------------------------------
python "$SCRIPT_DIR/08_stitch.py" \
    --stage blend \
    --czi "$CZI_PATH" \
    --out-dir "$PREFLIGHT" \
    --out-zarr "$SUBSTACK_ZARR" \
    --z-start 728 \
    --z-end   828 \
    --z-chunk 64

echo "=== Pre-flight Stage 2 done ==="
echo "Inspect in 08_fusion_dev.ipynb Phase 2.5 cell — set OUT_DIR=$PREFLIGHT, OUT_ZARR=$SUBSTACK_ZARR"
