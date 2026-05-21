#!/bin/bash
#SBATCH --job-name=ls_stitch_blend
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=04:00:00
#SBATCH --array=1-8
#SBATCH --output=/vast/scratch/users/kriel.j/output.%j.%a.%N.log
#SBATCH --error=/vast/scratch/users/kriel.j/output.%j.%a.%N.log

# Phase 2.5 Stage 2: globally-blended fusion for KL018 (1557 planes).
#
# Each of the 8 array tasks fuses one disjoint Z-slab of the full volume using
# the pre-allocated fused_direct.zarr (created by 08_stitch_stage1_register.sh)
# and the registered transforms in stitch_positions.json.
#
# Z partition (all boundaries are multiples of zarr chunk size 64 - no two
# tasks touch the same chunk, so r+ writes are parallel-safe; STITCH-05):
#   Task 1: z=0:256     (256 planes, 4 zarr chunks: 0-3)
#   Task 2: z=256:448   (192 planes, 3 zarr chunks: 4-6)
#   Task 3: z=448:640   (192 planes, 3 zarr chunks: 7-9)
#   Task 4: z=640:832   (192 planes, 3 zarr chunks: 10-12)
#   Task 5: z=832:1024  (192 planes, 3 zarr chunks: 13-15)
#   Task 6: z=1024:1216 (192 planes, 3 zarr chunks: 16-18)
#   Task 7: z=1216:1408 (192 planes, 3 zarr chunks: 19-21)
#   Task 8: z=1408:1557 (149 planes, 3 zarr chunks: 22-24)
#
# Submit workflow:
#   JOB1=$(sbatch --parsable 08_stitch_stage1_register.sh)
#   sbatch --dependency=afterok:$JOB1 08_stitch_stage2_blend.sh
#
# Usage (array task - do not submit directly without the Stage 1 dependency):
#   sbatch --dependency=afterok:<stage1_jobid> 08_stitch_stage2_blend.sh

set -euo pipefail

# --------------------------------------------------------------------------
# Z-partition lookup table (identical to Phase 2 02-02 - chunk-aligned).
# Index = SLURM_ARRAY_TASK_ID - 1 (0-based)
# --------------------------------------------------------------------------
Z_STARTS=(0 256 448 640 832 1024 1216 1408)
Z_ENDS=(256 448 640 832 1024 1216 1408 1557)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
Z_START=${Z_STARTS[$IDX]}
Z_END=${Z_ENDS[$IDX]}

SCRATCH="/vast/scratch/users/kriel.j/KL018_lightsheet"
CZI_PATH="$SCRATCH/KL018_85_D7_CT2AvIII_Overview.czi"
OUT_ZARR="$SCRATCH/fused_direct.zarr"
POS_JSON="$SCRATCH/stitch_positions.json"
SCRIPT_DIR="/vast/projects/BCRL_Multi_Omics/scripts/lightsheet_pipeline"
ENV_DIR="/vast/scratch/users/kriel.j/mvstitch_env"

# Belt-and-suspenders GPU pinning (SLURM cgroup already isolates).
export CUDA_VISIBLE_DEVICES=0
# Stream stdout so per-chunk blend progress appears in the SLURM log as it runs
# — mirrors preflight Stage 2 (see debug session preflight-stage2-canvas-too-
# small, where a silent 1h run looked indistinguishable from a hang).
export PYTHONUNBUFFERED=1

if ! command -v module &>/dev/null; then
    source /etc/profile.d/modules.sh
fi
module load CUDA/12.1

source /stornext/System/data/apps/anaconda3/anaconda3-latest/etc/profile.d/conda.sh
conda activate "$ENV_DIR"

# Sanity-check Stage 1 outputs exist and the zarr has been recreated.
test -f "$POS_JSON"   || { echo "ERROR: $POS_JSON missing - did Stage 1 run?"; exit 1; }
test -d "$OUT_ZARR"   || { echo "ERROR: $OUT_ZARR missing - did Stage 1 recreate it?"; exit 1; }

echo "=== Stage 2 array task $SLURM_ARRAY_TASK_ID/8 ==="
echo "SLURM job:  $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Z range:    [$Z_START, $Z_END)"
echo "Scratch:    $SCRATCH"
echo "Out zarr:   $OUT_ZARR (mode=r+)"
echo "Env:        $ENV_DIR"
python -c "import multiview_stitcher, dask; print('multiview-stitcher', multiview_stitcher.__version__, ' dask', dask.__version__)"

python "$SCRIPT_DIR/08_stitch.py" \
    --stage blend \
    --czi "$CZI_PATH" \
    --out-dir "$SCRATCH" \
    --out-zarr "$OUT_ZARR" \
    --z-start "$Z_START" \
    --z-end   "$Z_END" \
    --z-chunk 64 \
    --workers 24

echo "=== Stage 2 task $SLURM_ARRAY_TASK_ID done: z=[$Z_START:$Z_END) ==="

# STITCH-05: SLURM array writes disjoint Z-slabs into shared fused_direct.zarr
