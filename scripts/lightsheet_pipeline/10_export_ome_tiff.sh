#!/bin/bash
#SBATCH --job-name=ls_export_ometiff
#SBATCH --partition=regular
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/vast/scratch/users/kriel.j/output.%j.%N.log
#SBATCH --error=/vast/scratch/users/kriel.j/output.%j.%N.log

# Optional Step 10: Convert the OME-Zarr pyramid into an OME-TIFF for
# Imaris / Fiji compatibility, then rsync to the lab's stornext share.
#
# This is OPTIONAL — the OME-Zarr pyramid from 09_output_ome_zarr.sh is the
# primary deliverable. Run this only when a downstream tool needs OME-TIFF.
#
# Resource sizing rationale: OME-TIFF write is I/O-bound, plane-by-plane.
# Memory budget = 1 full XY plane × n_workers ≈ 100 MB × 8 = ~1 GB peak;
# 128 GB is over-provisioned to absorb LZW buffer + tifffile internals.
# Wall time dominated by writing ~3000+ planes serially → 8-12 h.
#
# Submit (requires 09_output_ome_zarr to have completed):
#   sbatch --dependency=afterok:<step09_jobid> 10_export_ome_tiff.sh

set -euo pipefail

WORK_DIR="/vast/scratch/users/kriel.j/KL018_lightsheet"
MANIFEST="${WORK_DIR}/tile_manifest_corrected.json"
LEGACY_MANIFEST="${WORK_DIR}/tile_manifest.json"
[ -f "$MANIFEST" ] || MANIFEST="$LEGACY_MANIFEST"
PYRAMID_ZARR="${WORK_DIR}/fused_pyramid.ome.zarr"
OMETIFF_OUT="${WORK_DIR}/KL018_85_D7_CT2AvIII_Overview-Fused-Stitched.ome.tiff"
DEST_DIR="/stornext/Img/data/prkfs1/m/Microscopy/KylieLuong/Lightsheet/KL018_KL260427"

if ! command -v module &>/dev/null; then
    source /etc/profile.d/modules.sh
fi

source /stornext/System/data/apps/anaconda3/anaconda3-latest/etc/profile.d/conda.sh
conda activate /vast/projects/BCRL_Multi_Omics/spatialdata_env_2

export PYTHONUNBUFFERED=1

test -d "$PYRAMID_ZARR" || { echo "ERROR: $PYRAMID_ZARR missing — run 09_output_ome_zarr.sh first"; exit 1; }
test -f "$MANIFEST"     || { echo "ERROR: $MANIFEST missing — required for voxel scaling"; exit 1; }

echo "=== Step 10: Export OME-TIFF + copy to stornext ==="
echo "SLURM job:  ${SLURM_JOB_ID:-interactive}  Node: ${SLURMD_NODENAME:-local}"
echo "Input pyr:  $PYRAMID_ZARR"
echo "Output:     $OMETIFF_OUT"
echo "Dest:       $DEST_DIR"

# raw2ometiff (a) wasn't on PATH and (b) expects bioformats2raw intermediate
# format, not OME-Zarr.  Write OME-TIFF directly with tifffile instead, reading
# the pyramid zarr one z-plane at a time to stay within the memory budget.
MANIFEST="$MANIFEST" LEGACY_MANIFEST="$LEGACY_MANIFEST" SRC="$PYRAMID_ZARR" DST="$OMETIFF_OUT" python - <<'PYEOF'
import json, os, zarr, numpy as np, tifffile
import dask.array as da

manifest_path = os.environ["MANIFEST"]
legacy_path = os.environ.get("LEGACY_MANIFEST", "")
src = os.environ["SRC"]
dst = os.environ["DST"]

with open(manifest_path) as f:
    manifest = json.load(f)
vox = manifest.get("voxel_size_um") or {}
if not any(vox.get(k) for k in ("x", "y", "z")) and legacy_path and os.path.exists(legacy_path):
    with open(legacy_path) as f:
        vox = (json.load(f).get("voxel_size_um") or {})
dx = vox.get("x") or 1.0
dy = vox.get("y") or 1.0
dz = vox.get("z") or 1.0

# Use full-resolution level 0 only; pyramid viewers handle sub-resolution
full_res = da.from_zarr(src, component="0")
T, C, Z, Y, X = full_res.shape
dtype = full_res.dtype
print(f"  Shape: {full_res.shape}  dtype: {dtype}")
print(f"  Voxel (µm): x={dx} y={dy} z={dz}")

omexml = tifffile.OmeXml()
omexml.addimage(
    dtype=dtype,
    shape=(T, C, Z, Y, X),
    storedshape=(T * C * Z, 1, 1, Y, X, 1),
    axes="TCZYX",
    PhysicalSizeX=dx,
    PhysicalSizeXUnit="um",
    PhysicalSizeY=dy,
    PhysicalSizeYUnit="um",
    PhysicalSizeZ=dz,
    PhysicalSizeZUnit="um",
)
description = omexml.tostring()

n_planes = T * C * Z
print(f"  Writing {n_planes} planes to BigTIFF...")
opts = dict(photometric="minisblack", compression="lzw")
with tifffile.TiffWriter(dst, bigtiff=True) as tif:
    i = 0
    for t in range(T):
        for c in range(C):
            for z_idx in range(Z):
                plane = np.asarray(full_res[t, c, z_idx])
                tif.write(
                    plane,
                    description=description if i == 0 else None,
                    contiguous=(i > 0),
                    **opts,
                )
                i += 1
                if i % 100 == 0 or i == n_planes:
                    print(f"  {i}/{n_planes} planes written")

print(f"OME-TIFF written: {dst}")
PYEOF

echo "Copying to stornext..."
rsync -av --progress \
    "$OMETIFF_OUT" \
    "${DEST_DIR}/"

echo "=== Step 10 done ==="
echo "Final OME-TIFF: ${DEST_DIR}/$(basename "$OMETIFF_OUT")"
