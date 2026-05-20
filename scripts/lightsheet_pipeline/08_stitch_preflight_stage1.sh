#!/bin/bash
#SBATCH --job-name=ls_stitch_preflight_reg
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=00:45:00
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
# Same 64-plane Z-slab as the production register call (z_mid = n_z // 2 = 778,
# z_slab_half = 32 → window [746, 810)). If this produces a converged solver
# result with max_residual_px < 5, the production Stage 1 will too.
#
# Pre-flight purpose: catch algorithmic regressions in ~30 min of GPU time,
# vs. a 2h wasted full-scale array submission.

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

source /stornext/System/data/apps/anaconda3/anaconda3-latest/etc/profile.d/conda.sh
conda activate "$ENV_DIR"

echo "=== Pre-flight Stage 1: global registration (substack scope) ==="
echo "SLURM job:    ${SLURM_JOB_ID:-interactive}  Node: ${SLURMD_NODENAME:-local}"
echo "Pre-flight out dir: $PREFLIGHT"
echo "Production zarr (UNTOUCHED): $SCRATCH/fused_direct.zarr"
python -c "import multiview_stitcher, dask; print('multiview-stitcher', multiview_stitcher.__version__, ' dask', dask.__version__)"

# --out-zarr is required by argparse but stage_register doesn't open/write to it.
# Point it at a placeholder path under preflight so any future stage_register
# code change that DOES touch it cannot reach the production zarr.
python "$SCRIPT_DIR/08_stitch.py" \
    --stage register \
    --czi "$CZI_PATH" \
    --out-dir "$PREFLIGHT" \
    --out-zarr "$PREFLIGHT/_unused_placeholder.zarr"

test -f "$PREFLIGHT/stitch_positions.json"   || { echo "ERROR: positions JSON not produced"; exit 1; }
test -f "$PREFLIGHT/stitch_diagnostics.json" || { echo "ERROR: diagnostics JSON not produced"; exit 1; }

echo ""
echo "=== Pre-flight Stage 1 done ==="
echo "Positions:   $PREFLIGHT/stitch_positions.json"
echo "Diagnostics: $PREFLIGHT/stitch_diagnostics.json"

# Quick gate check inline so the SLURM log surfaces converged + max_residual.
python3 - <<PYEOF
import json
d = json.load(open("$PREFLIGHT/stitch_diagnostics.json"))
g = d.get("groupwise", {})
n_pairs = len(d.get("pairwise", []))
print(f"  n_tiles:        {d.get('n_tiles')}")
print(f"  z_slab_used:    {d.get('z_slab_used')}")
print(f"  pairwise edges: {n_pairs}")
print(f"  converged:      {g.get('converged')}")
print(f"  rms_residual:   {g.get('rms_residual_px')}")
print(f"  max_residual:   {g.get('max_residual_px')}")
ok = bool(g.get("converged")) and float(g.get("max_residual_px", 99)) < 5.0
print(f"  pre-flight gate (converged + max_residual<5): {'PASS' if ok else 'FAIL'}")
PYEOF
