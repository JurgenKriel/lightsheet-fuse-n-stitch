#!/bin/bash
#SBATCH --job-name=ls_fuse_ncc
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --output=/vast/scratch/users/kriel.j/output.%j.%N.log
#SBATCH --error=/vast/scratch/users/kriel.j/output.%j.%N.log

# Stage 1 of 2-stage parallel fusion for KL018 (1557 planes).
#
# This job:
#   1. Validates that ncc_scores.json has ncc_threshold=0.5 (the confirmed value
#      from the Phase 1 notebook sweep). If the threshold differs, re-runs NCC.
#   2. Recreates fused_direct.zarr from scratch with shape (1,2,1557,8585,10095)
#      and chunks (1,1,64,512,512) — allocates metadata only, no pixel data.
#   3. Prints the zarr shape for verification.
#
# On success, submit Stage 2:
#   sbatch --dependency=afterok:$SLURM_JOB_ID 07_direct_fuse_stage2_parallel.sh
#
# Validated parameters (from 08_fusion_dev.ipynb sweep, 2026-05-18):
#   sigma_frac=0.9  taper_px=288  ncc_threshold=0.5  fusion_axis=2
#
# Usage:
#   sbatch 07_direct_fuse_stage1_ncc.sh

set -euo pipefail

SCRATCH="/vast/scratch/users/kriel.j/KL018_lightsheet"
CZI_PATH="$SCRATCH/KL018_85_D7_CT2AvIII_Overview.czi"
OUT_ZARR="$SCRATCH/fused_direct.zarr"
NCC_JSON="$SCRATCH/ncc_scores.json"
SCRIPT_DIR="/vast/projects/BCRL_Multi_Omics/scripts/lightsheet_pipeline"
EXPECTED_THRESHOLD="0.5"

if ! command -v module &>/dev/null; then
    source /etc/profile.d/modules.sh
fi
module load CUDA/12.1

source /stornext/System/data/apps/anaconda3/anaconda3-latest/etc/profile.d/conda.sh
conda activate /vast/scratch/users/kriel.j/lightsheet_env

echo "=== Stage 1: NCC validation + zarr pre-allocation ==="
echo "SLURM job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Scratch: $SCRATCH"
echo "CZI:     $CZI_PATH"
echo "Zarr:    $OUT_ZARR"
echo "NCC:     $NCC_JSON"

# --------------------------------------------------------------------------
# Step 1: Validate or re-run NCC
# --------------------------------------------------------------------------
if [ -f "$NCC_JSON" ]; then
    NCC_THRESHOLD=$(python3 -c "import json; d=json.load(open('$NCC_JSON')); print(d.get('ncc_threshold', 'MISSING'))")
    echo "ncc_scores.json threshold: $NCC_THRESHOLD (expected: $EXPECTED_THRESHOLD)"

    if [ "$NCC_THRESHOLD" != "$EXPECTED_THRESHOLD" ]; then
        echo "WARNING: threshold mismatch — re-running NCC with ncc_threshold=0.5"
        python "${SCRIPT_DIR}/07_direct_fuse.py" \
            --czi            "$CZI_PATH" \
            --out            "${SCRATCH}/ncc_probe_rerun.zarr" \
            --sigma-frac     0.9 \
            --taper-px       288 \
            --ncc-threshold  0.5 \
            --fusion-axis    2 \
            --z-chunk        64 \
            --workers        16 \
            --z-start        0 \
            --z-end          1
        echo "NCC re-run complete."
    else
        echo "NCC threshold matches — reusing existing ncc_scores.json."
    fi
else
    echo "ncc_scores.json not found — running NCC now (z=728:729 midpoint plane)..."
    python "${SCRIPT_DIR}/07_direct_fuse.py" \
        --czi            "$CZI_PATH" \
        --out            "$OUT_ZARR" \
        --sigma-frac     0.9 \
        --taper-px       288 \
        --ncc-threshold  0.5 \
        --fusion-axis    2 \
        --z-chunk        64 \
        --workers        16 \
        --z-start        728 \
        --z-end          729
    echo "NCC run complete — ncc_scores.json written."
fi

# --------------------------------------------------------------------------
# Step 2: Recreate fused_direct.zarr (full shape, empty — metadata only)
# --------------------------------------------------------------------------
echo ""
echo "Recreating fused_direct.zarr with full shape (1, 2, 1557, 8585, 10095)..."

python3 - <<'PYEOF'
import zarr
import numpy as np
import json

out_path = "/vast/scratch/users/kriel.j/KL018_lightsheet/fused_direct.zarr"

# Read canvas dims from tile_manifest_corrected.json if it exists,
# else fall back to verified hard-coded values (H=8585, W=10095).
import os
manifest_path = "/vast/scratch/users/kriel.j/KL018_lightsheet/tile_manifest_corrected.json"
if os.path.exists(manifest_path):
    man = json.load(open(manifest_path))
    H = man["canvas_shape"]["H"]
    W = man["canvas_shape"]["W"]
    print(f"  Canvas dims from manifest: H={H}  W={W}")
else:
    H, W = 8585, 10095
    print(f"  Canvas dims (hardcoded fallback): H={H}  W={W}")

z = zarr.open(
    out_path,
    mode="w",
    shape=(1, 2, 1557, H, W),
    chunks=(1, 1, 64, 512, 512),
    dtype=np.uint16,
)
print(f"  zarr created: shape={z.shape}  chunks={z.chunks}  dtype={z.dtype}")
print(f"  Location: {out_path}")
PYEOF

echo ""
echo "=== Stage 1 complete. ==="
echo ""
echo "Verify zarr shape:"
python3 -c "
import zarr
z = zarr.open('/vast/scratch/users/kriel.j/KL018_lightsheet/fused_direct.zarr', 'r')
print(f'  shape={z.shape}  chunks={z.chunks}')
expected = (1, 2, 1557, 8585, 10095)
if z.shape == expected:
    print('  PASS: shape matches expected', expected)
else:
    print('  FAIL: expected', expected, 'got', z.shape)
    exit(1)
"

echo ""
echo "Submit Stage 2 with:"
echo "  sbatch --dependency=afterok:\${SLURM_JOB_ID} ${SCRIPT_DIR}/07_direct_fuse_stage2_parallel.sh"
