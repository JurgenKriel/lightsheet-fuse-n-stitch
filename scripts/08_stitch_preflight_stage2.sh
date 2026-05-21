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
LAYOUT_CACHE="$PREFLIGHT/czi_layout_cache.json"
SCRIPT_DIR="/vast/projects/BCRL_Multi_Omics/scripts/lightsheet_pipeline"
ENV_DIR="/vast/scratch/users/kriel.j/mvstitch_env"

export CUDA_VISIBLE_DEVICES=0

if ! command -v module &>/dev/null; then
    source /etc/profile.d/modules.sh
fi
module load CUDA/12.1

source /stornext/System/data/apps/anaconda3/anaconda3-latest/etc/profile.d/conda.sh
conda activate "$ENV_DIR"

test -f "$POS_JSON"     || { echo "ERROR: $POS_JSON missing — did pre-flight Stage 1 finish?"; exit 1; }
test -f "$LAYOUT_CACHE" || { echo "ERROR: $LAYOUT_CACHE missing — did pre-flight Stage 1 finish?"; exit 1; }

echo "=== Pre-flight Stage 2: substack blend [z=728:828) ==="
echo "SLURM job:      ${SLURM_JOB_ID:-interactive}  Node: ${SLURMD_NODENAME:-local}"
echo "Positions:      $POS_JSON"
echo "Layout cache:   $LAYOUT_CACHE"
echo "Out substack:   $SUBSTACK_ZARR"
echo "Production zarr (UNTOUCHED): $SCRATCH/fused_direct.zarr"
python -c "import multiview_stitcher, dask; print('multiview-stitcher', multiview_stitcher.__version__, ' dask', dask.__version__)"

# --------------------------------------------------------------------------
# 1) Pre-allocate substack_z728_828_stitched.zarr at the layout-cache canvas.
#
#    The canvas dimensions come from the CZI's own stage-metadata mosaic
#    bounding box (layout["canvas_w"]/canvas_h), NOT from a recomputation
#    over stitch_positions.json. Reason: in multiview-stitcher 0.1.52 the
#    global_optimization solver may re-anchor the registered frame at one
#    reference tile, making naive max(x)-min(x) over registered positions
#    unreliable (this was the root cause of the prior canvas-too-small
#    failure — see debug session preflight-stage2-canvas-too-small).
#
#    stitch_positions.json now stores absolute world placements (stage +
#    correction) thanks to the 08_stitch.py:stage_register fix, so the
#    canvas computed from those positions also matches the layout cache
#    (±a few px of registration correction). We use the layout cache as
#    the authoritative source and cross-check against the registration
#    canvas as a sanity guard.
#
#    Full Z=1557 so stage_blend's absolute-index writes (z=728:828) land in
#    the right place; planes outside that range stay zero. This matches the
#    08_fusion_dev.ipynb cell's mid-Z crop logic (z_arr.shape[2]//2 = 778).
# --------------------------------------------------------------------------
python3 - <<PYEOF
import json
import sys
import numpy as np
import zarr

POS_JSON      = "$POS_JSON"
LAYOUT_CACHE  = "$LAYOUT_CACHE"
SUBSTACK_ZARR = "$SUBSTACK_ZARR"
TILE_H        = 1920
TILE_W        = 1920

with open(LAYOUT_CACHE) as f:
    layout = json.load(f)
with open(POS_JSON) as f:
    pos = json.load(f)

# Authoritative canvas from CZI stage-metadata mosaic bbox.
H_layout = int(layout["canvas_h"])
W_layout = int(layout["canvas_w"])
N_Z      = int(layout["n_z"])

# Cross-check against registered positions for an early warning.
xs = [float(p["translation_px"]["x"]) for p in pos.values()]
ys = [float(p["translation_px"]["y"]) for p in pos.values()]
H_reg = int(round(max(ys) + TILE_H - min(ys)))
W_reg = int(round(max(xs) + TILE_W - min(xs)))

print(f"Canvas (layout cache): H={H_layout}  W={W_layout}  Z={N_Z}")
print(f"Canvas (from registered positions): H={H_reg}  W={W_reg}  (n_tiles={len(pos)})")

# Widened guard — accept anything within reasonable range for the KL018 6x5 grid.
# Layout canvas = (8448, 10079); guard window comfortably brackets that.
if not (3500 <= H_layout <= 12000 and 3500 <= W_layout <= 14000):
    raise SystemExit(
        f"ERROR: layout canvas H={H_layout} W={W_layout} outside expected range "
        f"(3500-12000, 3500-14000)"
    )

# Warn if the registered canvas disagrees materially with the layout canvas
# (>1000 px on either dimension). This means stage_register's positions and
# the layout cache are inconsistent — likely a registration bug to investigate.
if abs(H_reg - H_layout) > 1000 or abs(W_reg - W_layout) > 1000:
    print(
        f"WARNING: registered canvas (H={H_reg} W={W_reg}) differs from layout "
        f"canvas (H={H_layout} W={W_layout}) by >1000 px. Stage 2 will pre-"
        f"allocate at the layout canvas, but the divergence suggests stage_"
        f"register may not be producing absolute world positions. Inspect "
        f"stitch_positions.json before trusting the blended output.",
        file=sys.stderr,
    )

zarr.open(
    SUBSTACK_ZARR,
    mode="w",
    shape=(1, 2, N_Z, H_layout, W_layout),
    chunks=(1, 1, 64, 512, 512),
    dtype=np.uint16,
)
print(f"Pre-allocated substack zarr at {SUBSTACK_ZARR}")
print(f"  shape: (1, 2, {N_Z}, {H_layout}, {W_layout})  dtype: uint16")
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
