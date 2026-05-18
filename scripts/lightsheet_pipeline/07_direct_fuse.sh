#!/bin/bash
#SBATCH --job-name=ls_direct_fuse
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=/vast/scratch/users/kriel.j/output.%j.%N.log
#SBATCH --error=/vast/scratch/users/kriel.j/output.%j.%N.log

# Direct CZI → fused.zarr (bypasses bioformats2raw raw.zarr intermediate).
# Pre-stage the CZI to /vast/scratch before submitting this job for best
# throughput (avoids stornext read bottleneck during parallel subblock reads).
#
# Usage:
#   sbatch 07_direct_fuse.sh
#   sbatch 07_direct_fuse.sh /vast/scratch/users/kriel.j/KL018_lightsheet/KL018.czi

# Substack example (100 planes for parameter tuning — ~16 min on A100):
#   python "${SCRIPT_DIR}/07_direct_fuse.py" \
#       --czi      "$CZI_ARG" \
#       --out      /vast/scratch/users/kriel.j/KL018_lightsheet/substack_z728_828.zarr \
#       --z-start  728 \
#       --z-end    828 \
#       --taper-px 128 \
#       --sigma-frac 0.3 \
#       --ncc-threshold 0.05 \
#       --z-chunk  64 \
#       --workers  8
#
# Full-stack flags (edit values here after substack validation):
#   --taper-px 64  --sigma-frac 0.3  --ncc-threshold 0.05  --fusion-axis 2

MANIFEST="/vast/scratch/users/kriel.j/KL018_lightsheet/tile_manifest.json"
# Default CZI: use the staged copy on /vast/scratch if it exists, else stornext.
STAGED_CZI="/vast/scratch/users/kriel.j/KL018_lightsheet/KL018_85_D7_CT2AvIII_Overview.czi"
STORNEXT_CZI="/stornext/Img/data/prkfs1/m/Microscopy/KylieLuong/Lightsheet/KL018_KL260427/KL018_85_D7_CT2AvIII_Overview.czi"
CZI_ARG="${1:-}"

if [ -z "$CZI_ARG" ]; then
    if [ -f "$STAGED_CZI" ]; then
        CZI_ARG="$STAGED_CZI"
        echo "Using staged CZI on /vast/scratch: $CZI_ARG"
    else
        CZI_ARG="$STORNEXT_CZI"
        echo "WARNING: staged CZI not found — reading from stornext (slower)."
        echo "  Consider running the stage job first: sbatch 07a_stage_czi.sh"
    fi
fi

SCRIPT_DIR="/vast/projects/BCRL_Multi_Omics/scripts/lightsheet_pipeline"

# CuPy 14 (cupy-cuda12x) requires CUDA 12+; the cluster default is 11.8.
# Load a matching toolkit before activating conda so LD_LIBRARY_PATH/CUDA_PATH
# point at CUDA 12.x when Python imports cupy.
if ! command -v module &>/dev/null; then
    source /etc/profile.d/modules.sh
fi
module load CUDA/12.1

source /stornext/System/data/apps/anaconda3/anaconda3-latest/etc/profile.d/conda.sh
conda activate /vast/scratch/users/kriel.j/lightsheet_env

echo "Starting direct CZI → fused.zarr (z-chunk=64, workers=8)..."
python "${SCRIPT_DIR}/07_direct_fuse.py" \
    --czi      "$CZI_ARG" \
    --z-chunk  64 \
    --workers  8
echo "Done."
