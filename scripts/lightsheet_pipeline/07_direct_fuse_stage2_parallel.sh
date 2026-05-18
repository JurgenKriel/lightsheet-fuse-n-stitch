#!/bin/bash
#SBATCH --job-name=ls_fuse_par
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --array=1-8
#SBATCH --output=/vast/scratch/users/kriel.j/output.%j.%a.%N.log
#SBATCH --error=/vast/scratch/users/kriel.j/output.%j.%a.%N.log

# Stage 2 of 2-stage parallel fusion for KL018 (1557 planes).
#
# Each of the 8 array tasks fuses one disjoint Z-slab of the full volume
# using a pre-allocated zarr (created by Stage 1). Tasks run concurrently,
# each on its own A30 GPU, writing to non-overlapping zarr chunk files.
#
# Z partition (all boundaries are multiples of 64 — zarr chunk size):
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
#   JOB1=$(sbatch --parsable 07_direct_fuse_stage1_ncc.sh)
#   sbatch --dependency=afterok:$JOB1 07_direct_fuse_stage2_parallel.sh
#
# Validated parameters (from 08_fusion_dev.ipynb sweep, 2026-05-18):
#   sigma_frac=0.9  taper_px=288  ncc_threshold=0.5  fusion_axis=2
#
# Usage (array task only — do not submit directly without stage 1 dependency):
#   sbatch --dependency=afterok:<stage1_jobid> 07_direct_fuse_stage2_parallel.sh

# --------------------------------------------------------------------------
# Z-partition lookup table
# Index: SLURM_ARRAY_TASK_ID - 1 (0-based)
# All boundaries are multiples of 64 (zarr chunk size) to prevent chunk collision.
# --------------------------------------------------------------------------
set -euo pipefail

Z_STARTS=(0 256 448 640 832 1024 1216 1408)
Z_ENDS=(256 448 640 832 1024 1216 1408 1557)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
Z_START=${Z_STARTS[$IDX]}
Z_END=${Z_ENDS[$IDX]}

SCRATCH="/vast/scratch/users/kriel.j/KL018_lightsheet"
CZI_PATH="$SCRATCH/KL018_85_D7_CT2AvIII_Overview.czi"
OUT_ZARR="$SCRATCH/fused_direct.zarr"
NCC_JSON="$SCRATCH/ncc_scores.json"
SCRIPT_DIR="/vast/projects/BCRL_Multi_Omics/scripts/lightsheet_pipeline"

# Belt-and-suspenders GPU assignment alongside SLURM GPU cgroup isolation.
# SLURM --gres=gpu:A30:1 assigns each task its own GPU slot; within the task's
# cgroup, device 0 is the assigned GPU.
export CUDA_VISIBLE_DEVICES=0

if ! command -v module &>/dev/null; then
    source /etc/profile.d/modules.sh
fi
module load CUDA/12.1

source /stornext/System/data/apps/anaconda3/anaconda3-latest/etc/profile.d/conda.sh
conda activate /vast/scratch/users/kriel.j/lightsheet_env

echo "=== Stage 2 Task ${SLURM_ARRAY_TASK_ID}/8 ==="
echo "SLURM job: ${SLURM_JOB_ID}  Array task: ${SLURM_ARRAY_TASK_ID}  Node: ${SLURMD_NODENAME}"
echo "Z range: ${Z_START}:${Z_END}  ($(( Z_END - Z_START )) planes)"
echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NCC positions: ${NCC_JSON}"
echo "Output zarr:   ${OUT_ZARR}"
echo ""

# Verify pre-allocated zarr exists and has correct shape before starting fusion.
python3 -c "
import zarr, sys
try:
    z = zarr.open('${OUT_ZARR}', 'r')
    expected = (1, 2, 1557, 8585, 10095)
    if z.shape != expected:
        print(f'FAIL: zarr shape {z.shape} != expected {expected}')
        sys.exit(1)
    print(f'Zarr pre-check OK: shape={z.shape}')
except Exception as e:
    print(f'FAIL: cannot open zarr: {e}')
    sys.exit(1)
"
if [ $? -ne 0 ]; then
    echo "ERROR: zarr pre-check failed. Ensure Stage 1 completed successfully."
    exit 1
fi

# --------------------------------------------------------------------------
# Run fusion for this Z-slab
# --------------------------------------------------------------------------
python "${SCRIPT_DIR}/07_direct_fuse.py" \
    --czi            "$CZI_PATH" \
    --out            "$OUT_ZARR" \
    --load-positions "$NCC_JSON" \
    --zarr-mode      r+ \
    --z-start        "$Z_START" \
    --z-end          "$Z_END" \
    --sigma-frac     0.9 \
    --taper-px       288 \
    --ncc-threshold  0.5 \
    --fusion-axis    2 \
    --z-chunk        64 \
    --workers        12

FUSE_EXIT=$?
echo ""
if [ $FUSE_EXIT -ne 0 ]; then
    echo "ERROR: fusion failed with exit code $FUSE_EXIT"
    exit $FUSE_EXIT
fi

echo "Task ${SLURM_ARRAY_TASK_ID}: fusion complete for Z=${Z_START}:${Z_END}."
echo ""

# Post-task spot-check: verify zarr still has correct shape (not truncated by this task).
python3 -c "
import zarr
z = zarr.open('${OUT_ZARR}', 'r')
print(f'Post-task zarr shape: {z.shape}  chunks: {z.chunks}')
expected_shape = (1, 2, 1557, 8585, 10095)
if z.shape != expected_shape:
    print(f'WARN: shape mismatch — expected {expected_shape}')
else:
    print('Shape OK.')
"

echo "=== Task ${SLURM_ARRAY_TASK_ID} done. ==="
