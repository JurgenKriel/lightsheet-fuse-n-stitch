#!/bin/bash
#SBATCH --job-name=ls_output
#SBATCH --partition=regular
#SBATCH --cpus-per-task 24
#SBATCH --mem=256G
#SBATCH --time 24:00:00
#SBATCH --output /vast/scratch/users/kriel.j/output.%j.%N.log
#SBATCH --error  /vast/scratch/users/kriel.j/output.%j.%N.log

set -euo pipefail

WORK_DIR="/vast/scratch/users/kriel.j/KL018_lightsheet"
MANIFEST="${WORK_DIR}/tile_manifest_corrected.json"
LEGACY_MANIFEST="${WORK_DIR}/tile_manifest.json"
[ -f "$MANIFEST" ] || MANIFEST="$LEGACY_MANIFEST"
FUSED_ZARR="${WORK_DIR}/fused_direct.zarr"
PYRAMID_ZARR="${WORK_DIR}/fused_pyramid.zarr"
OMETIFF_OUT="${WORK_DIR}/KL018_85_D7_CT2AvIII_Overview-Fused-Stitched.ome.tiff"
DEST_DIR="/stornext/Img/data/prkfs1/m/Microscopy/KylieLuong/Lightsheet/KL018_KL260427"

if ! command -v module &>/dev/null; then
    source /etc/profile.d/modules.sh
fi

source /stornext/System/data/apps/anaconda3/anaconda3-latest/etc/profile.d/conda.sh
conda activate /vast/projects/BCRL_Multi_Omics/spatialdata_env_2

echo "Input  zarr : $FUSED_ZARR"
echo "Manifest    : $MANIFEST"
echo "Pyramid out : $PYRAMID_ZARR"

# ── Step 1: Build multi-resolution OME-Zarr pyramid ──────────────────────────
# Fix: use dask.array.from_zarr for lazy loading so downsampled levels are
# never materialised in RAM; write each level chunk-by-chunk via da.to_zarr.
# The previous approach sliced zarr arrays directly, loading 100s of GB at once
# and triggering the OOM kill seen in output.27799311.
echo "Building OME-Zarr pyramid..."
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

# Lazy read — no data loaded into RAM yet
z_src = da.from_zarr(src)
print(f"  Source shape: {z_src.shape}  chunks: {z_src.chunks}")

n_levels = 5
pyramid_dask = [z_src]
for lvl in range(1, n_levels):
    factor = 2 ** lvl
    # Strided slice is still lazy; dask computes one chunk at a time when writing
    coarse = z_src[::1, ::1, ::factor, ::factor, ::factor]
    pyramid_dask.append(coarse)

# Write each level to zarr chunk-by-chunk — peak RAM ≈ n_workers × chunk_size
for lvl, arr in enumerate(pyramid_dask):
    print(f"  Writing level {lvl}  shape={arr.shape} ...")
    da.to_zarr(arr, url=dst, component=str(lvl), overwrite=True)

# Write OME-Zarr multiscales metadata
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

# ── Step 2: Convert to OME-TIFF for Imaris / Fiji compatibility ──────────────
# Fix: raw2ometiff (a) wasn't on PATH and (b) expects bioformats2raw intermediate
# format, not OME-Zarr.  Write OME-TIFF directly with tifffile instead, reading
# the pyramid zarr one z-plane at a time to stay within the memory budget.
echo "Converting to OME-TIFF..."
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

# Build OME-XML once; attach to the first IFD only
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
opts = dict(photometric="minisblack", compression="lzw", compressionargs={"level": 6})
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

# ── Step 3: Copy to stornext ──────────────────────────────────────────────────
echo "Copying to stornext..."
rsync -av --progress \
    "$OMETIFF_OUT" \
    "${DEST_DIR}/"

echo "Copy complete: ${DEST_DIR}/$(basename "$OMETIFF_OUT")"
echo "All steps done."
