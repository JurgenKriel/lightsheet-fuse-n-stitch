#!/bin/bash
#SBATCH --job-name=ls_output_omezarr
#SBATCH --partition=regular
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/vast/scratch/users/kriel.j/output.%j.%N.log
#SBATCH --error=/vast/scratch/users/kriel.j/output.%j.%N.log

# Step 09: Build the multi-resolution OME-Zarr pyramid from the stitched zarr
# produced by Stage 2 of 08_stitch (fused_direct.zarr).
#
# This is the PRIMARY output of the pipeline (fusion → stitch → output_ome_zarr).
# An OME-TIFF export for Imaris / Fiji is optional and lives in
# 10_export_ome_tiff.sh — submit that AFTER this one if you need it.
#
# Resource sizing rationale: pyramid build is dask-lazy + chunk-streaming. Peak
# RAM is bounded by n_workers × chunk_size (~1×1×64×512×512 × 2 B = 32 MB per
# chunk), so 64 GB is comfortable. Wall time is ~30-90 min depending on Z depth.
#
# Submit:
#   sbatch 09_output_ome_zarr.sh

set -euo pipefail

WORK_DIR="/vast/scratch/users/kriel.j/KL018_lightsheet"
MANIFEST="${WORK_DIR}/tile_manifest_corrected.json"
LEGACY_MANIFEST="${WORK_DIR}/tile_manifest.json"
[ -f "$MANIFEST" ] || MANIFEST="$LEGACY_MANIFEST"
FUSED_ZARR="${WORK_DIR}/fused_direct.zarr"
PYRAMID_ZARR="${WORK_DIR}/fused_pyramid.ome.zarr"

if ! command -v module &>/dev/null; then
    source /etc/profile.d/modules.sh
fi

source /stornext/System/data/apps/anaconda3/anaconda3-latest/etc/profile.d/conda.sh
conda activate /vast/projects/BCRL_Multi_Omics/spatialdata_env_2

# Stream stdout so dask progress lines appear in the SLURM log immediately.
export PYTHONUNBUFFERED=1

test -d "$FUSED_ZARR" || { echo "ERROR: $FUSED_ZARR missing — has 08_stitch_stage2_blend completed?"; exit 1; }
test -f "$MANIFEST"   || { echo "ERROR: $MANIFEST missing — required for voxel scaling"; exit 1; }

echo "=== Step 09: Build OME-Zarr pyramid ==="
echo "SLURM job:  ${SLURM_JOB_ID:-interactive}  Node: ${SLURMD_NODENAME:-local}"
echo "Input zarr: $FUSED_ZARR"
echo "Manifest:   $MANIFEST"
echo "Output:     $PYRAMID_ZARR"

# Use dask.array.from_zarr for lazy loading so downsampled levels are never
# materialised in RAM; write each level chunk-by-chunk via da.to_zarr. An
# earlier naive approach (slicing zarr arrays directly) materialised 100s of
# GB at once and triggered the OOM kill in output.27799311.
MANIFEST="$MANIFEST" LEGACY_MANIFEST="$LEGACY_MANIFEST" SRC="$FUSED_ZARR" DST="$PYRAMID_ZARR" python - <<'PYEOF'
import json, os, zarr, numpy as np
import dask.array as da
from pathlib import Path

manifest_path = os.environ["MANIFEST"]
legacy_path = os.environ.get("LEGACY_MANIFEST", "")
src = os.environ["SRC"]
dst = os.environ["DST"]

with open(manifest_path) as f:
    manifest = json.load(f)

vox = manifest.get("voxel_size_um") or {}
if not any(vox.get(k) for k in ("x", "y", "z")) and legacy_path and os.path.exists(legacy_path):
    print(f"  voxel_size_um missing in {manifest_path}; reading from {legacy_path}")
    with open(legacy_path) as f:
        vox = (json.load(f).get("voxel_size_um") or {})
dx = vox.get("x") or 1.0
dy = vox.get("y") or 1.0
dz = vox.get("z") or 1.0
if dx == 1.0 and dy == 1.0 and dz == 1.0:
    print("  WARNING: no voxel sizes found — pyramid will use 1 µm placeholder scales.")
else:
    print(f"  voxel size (µm): x={dx} y={dy} z={dz}")

z_src = da.from_zarr(src)
print(f"  Source shape: {z_src.shape}  chunks: {z_src.chunks}")

n_levels = 5
pyramid_dask = [z_src]
for lvl in range(1, n_levels):
    factor = 2 ** lvl
    coarse = z_src[::1, ::1, ::factor, ::factor, ::factor]
    pyramid_dask.append(coarse)

for lvl, arr in enumerate(pyramid_dask):
    print(f"  Writing level {lvl}  shape={arr.shape} ...")
    da.to_zarr(arr, url=dst, component=str(lvl), overwrite=True)

coordinate_transformations = [
    [{"type": "scale", "scale": [1.0, 1.0, dz * (2**lvl), dy * (2**lvl), dx * (2**lvl)]}]
    for lvl in range(n_levels)
]
store = zarr.open_group(dst, mode="a")
store.attrs["multiscales"] = [{
    "version": "0.4",
    "name": "",
    "axes": [
        {"name": "t", "type": "time"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ],
    "datasets": [
        {"path": str(lvl), "coordinateTransformations": coordinate_transformations[lvl]}
        for lvl in range(n_levels)
    ],
}]
print(f"Pyramid written: {dst} ({n_levels} levels)")
PYEOF

echo "=== Step 09 done ==="
echo "Next (optional): sbatch 10_export_ome_tiff.sh"
