# 07_direct_fuse — Direct CZI → fused stitched zarr

End-to-end fusion + stitching from a Zeiss mosaic CZI to a single OME-zarr, in
one job. Replaces pipeline steps 02 (`bioformats2raw`) + 03 (deskew) + 04
(dual-side fusion) + 05 (stitching) for Z.1 light sheet datasets that do not
need deskewing.

For the KL018 reference dataset (`KL018_85_D7_CT2AvIII_Overview.czi`, 1.3 TB):

| Dimension | Size |
|---|---|
| Mosaic tiles (M) | 30 |
| Illumination sides (I) | 2 (Gaussian-blended) |
| Channels (C) | 2 |
| Timepoints (T) | 1 |
| Tile geometry | 1920 × 1920 × 1557 |
| Final canvas | ~10830 × 8701 pixels |

---

## Quick start

```bash
# 1. Stage CZI to /vast/scratch (one-off, sbatch 07a_stage_czi.sh)
# 2. Submit fusion
sbatch 07_direct_fuse.sh

# Or run interactively on a GPU node
conda activate /vast/scratch/users/kriel.j/lightsheet_env
module load CUDA/12.1
python 07_direct_fuse.py \
    --czi /vast/scratch/users/kriel.j/KL018_lightsheet/KL018_85_D7_CT2AvIII_Overview.czi \
    --z-chunk 64 \
    --workers 8
```

The SLURM wrapper (`07_direct_fuse.sh`) already loads `CUDA/12.1` before
activating the conda environment. This is required because `cupy-cuda12x`
needs CUDA 12+ for NVRTC kernel compilation, while the cluster default is 11.8.

### CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--czi` | KL018 staged path | Input CZI path. Use `/vast/scratch` copy for I/O throughput. |
| `--out` | `fused_direct.zarr` | Output OME-zarr path. |
| `--z-start` | 0 | First Z plane to process (inclusive). Substack: use 728. |
| `--z-end` | None (full stack) | Last Z plane to process (exclusive). Substack: use 828. |
| `--z-chunk` | 64 | Z planes processed per slab. Controls peak RAM (≈21 GB at 64). |
| `--workers` | 16 | Threads for parallel subblock reads. |
| `--skip-refine` | off | Skip NCC phase-correlation refinement; trust raw stage positions. |
| `--taper-px` | 64 | Cosine-taper width at tile edges for overlap blending. Increase to 128–144 to reduce seams. |
| `--sigma-frac` | 0.3 | Gaussian blend sigma fraction for dual-side fusion. Sweep 0.2–0.6. |
| `--fusion-axis` | 2 | Axis for dual-side blend: 0=Z, 1=Y, 2=X. KL018 uses X (2). |
| `--ncc-threshold` | 0.05 | NCC quality threshold for phase-correlation refinement. Raise to 0.1–0.2 to skip weak overlaps. |
| `--dump-layout` | off | Print tile bounding boxes and exit (no processing). |

---

## Substack workflow

To run fusion on a 100-slice substack (z=728:828, the central planes of KL018)
for rapid parameter tuning before committing to a full 1557-plane run, use
`--z-start` and `--z-end` to select the slice range. For KL018 (Z.1 orthogonal,
X-axis illumination), also set `--fusion-axis 2` when sweeping blend parameters:

```bash
# Interactive GPU node
conda activate /vast/scratch/users/kriel.j/lightsheet_env
module load CUDA/12.1
python 07_direct_fuse.py \
    --czi  /vast/scratch/users/kriel.j/KL018_lightsheet/KL018_85_D7_CT2AvIII_Overview.czi \
    --out  /vast/scratch/users/kriel.j/KL018_lightsheet/substack_z728_828.zarr \
    --z-start 728 --z-end 828 \
    --taper-px 128 \
    --sigma-frac 0.3 \
    --ncc-threshold 0.05 \
    --z-chunk 64 --workers 8
```

Expected runtime: ~16 min on A100 (2 channels × 2 Z-chunks × ~4 min each).
Output: `substack_z728_828.zarr` shape (1, 2, 100, H, W) + `ncc_scores.json`.
Use `08_fusion_dev.ipynb` (`Python (spatialdata_2025)` kernel) to inspect the
output and sweep parameters interactively.

---

## Pipeline stages

The script walks through six stages. Each is summarised below; section
headers map to functions in `07_direct_fuse.py`.

### 1. CZI introspection — `get_czi_layout()`

`aicspylibczi.CziFile.get_dims_shape()` and
`get_all_mosaic_tile_bounding_boxes()` are parsed for the volume dimensions,
mosaic canvas extent, and per-tile bounding boxes (stage positions converted
to pixel coordinates).

Parsing the CZI header is slow (~minutes) for 1.3 TB files, so the result is
cached as `czi_layout_cache.json` next to the output zarr. Re-runs load
instantly from cache.

The mosaic tile keys returned by `get_all_mosaic_tile_bounding_boxes()` are
`_aicspylibczi.TileInfo` objects — the M-index is read via the `.m_index`
attribute, not via `.values()` (a previous version of this script assumed
dict-like keys and crashed).

### 2. Tile position refinement — `refine_tile_positions()`

Raw stage positions from the microscope are typically accurate to within a
few pixels, but encoder slop and stage backlash can leave seams. The script
follows a ClearMap-inspired refinement:

1. **Reference planes.** A single mid-Z plane is read for every tile at
   `I=0, C=0, T=0`. This gives a 2-D "fingerprint" of each tile suitable
   for cross-correlation.

2. **Pairwise NCC quality scores.** For every tile pair `(i, j)` with a
   stage-predicted overlap of at least 16 px in both axes, the overlap
   region is cropped from each tile's reference plane and normalised cross
   correlation is computed (`_ncc()`). High NCC means real shared content;
   low NCC means the predicted overlap is wrong or featureless.

3. **Sub-pixel shifts.** For pairs with NCC > 0.05, scikit-image's
   `phase_cross_correlation` (10× upsample) measures the residual shift
   needed to align the two crops. Stored symmetrically: `shifts[(i, j)]`
   and `shifts[(j, i)]` are sign-flipped duals.

4. **Minimum spanning tree.** A weighted graph is built where edges are
   tile pairs, weights are `-NCC` (so the MST minimises `-quality`, i.e.,
   maximises quality). `scipy.sparse.csgraph.minimum_spanning_tree` picks
   the best `n_tiles − 1` connections. This guarantees a single connected
   tile graph rooted on the strongest pairwise registrations and avoids
   propagating noise through weak overlaps.

5. **BFS shift propagation.** Starting from tile 0, the BFS walks the MST
   and applies the phase shift to each neighbour:
   `pos[nbr] += shifts[(src, nbr)]`. Original relative offsets are
   preserved; only the refinement delta is added.

The refined canvas extent (`refined_w × refined_h`) is recomputed from the
updated tile positions, so the output canvas exactly bounds the refined
mosaic.

Pass `--skip-refine` to bypass stages 2–5 entirely and use raw stage
positions. Useful for debugging or when overlaps are too sparse for
correlation to converge.

### 3. Output zarr allocation

A single OME-zarr is opened in `(T, C, Z, Y, X)` layout, dtype `uint16`,
with chunks `(1, 1, z_chunk, 512, 512)`. The full Z extent is allocated up
front but lazily — only the chunks written by stage 5 land on disk.

### 4. Z-chunk streaming

The main loop iterates `T × C × (Z / z_chunk)`. For each slab:

```
canvas       = float32 (z_chunk, canvas_H, canvas_W)
weight_canvas = float32 (z_chunk, canvas_H, canvas_W)
```

At `z_chunk=64` and a 10830×8701 canvas, each pair of buffers is about
21 GB of RAM. After all tiles are placed, the canvas is normalised by the
weight canvas, clipped to `[0, 65535]`, cast back to `uint16`, and written
to the corresponding `out_z[t, c, z0:z1]` slice.

This Z-chunked design keeps peak memory bounded regardless of total Z
extent — the full 1557-plane volume never lives in RAM at once.

### 5. Dual-side fusion — `fuse_sides()`

The Z.1 light sheet illuminates each tile from two opposing sides
(`I=0`, `I=1`). Each side has good signal on its illumination edge and
fades toward the opposite edge as scattering attenuates the laser. A
Gaussian-weighted blend along the X axis (`axis=2`) recombines them into
a single tile with uniform illumination:

```
w_a(x) = exp(-x² / (2·σ²))            # bright at left edge,  fades right
w_b(x) = w_a(-x)                       # bright at right edge, fades left
fused(z, y, x) = (w_a · side_a + w_b · side_b) / (w_a + w_b + ε)
```

`σ = 0.3 · n / 2` where `n` is the tile width. The blend runs on the GPU
via CuPy when available, falling back to NumPy otherwise. Backend
selection (`xp = cp if _GPU else np`) is the only difference between
paths; the kernels are identical.

Read I/O is parallelised: for each tile, the two illumination volumes
are read in separate thread pools (`read_tile_zchunk()`), then handed
off to `fuse_sides()`. CZI subblocks are independent so threaded reads
scale well.

If the dataset is single-sided (`n_i == 1`), this stage is skipped and
the single illumination is passed straight through.

### 6. Stitching — `place_tile_slab()`

Once a tile slab is fused, it is accumulated onto the canvas with a
cosine-taper blending mask:

```
wy(y) = ½ · (1 − cos(π · y / taper_px))   for y in edge region
       1                                   in interior
wx similarly along X
w2d  = wy · wxᵀ
canvas[z, y, x]         += tile[z, y, x] · w2d[y, x]
weight_canvas[z, y, x]  += w2d[y, x]
```

The taper width defaults to 64 pixels. Inside the taper, two overlapping
tiles each contribute a weight between 0 and 1; the canvas and weight
sums let the post-loop divide produce a smooth crossfade with no seam
line at the tile boundary.

Bounding-box clipping (`cx0 = max(x0, 0)`, etc.) handles edge tiles that
extend past the canvas after position refinement.

Tile reading + fusion runs in a `ThreadPoolExecutor` (`max_workers=4`),
but `place_tile_slab` is called serially as futures complete — the
canvas is not thread-safe for concurrent accumulation.

---

## Outputs

| Path | Contents |
|---|---|
| `fused_direct.zarr` | Final stitched volume, `(T, C, Z, Y, X)` uint16. |
| `czi_layout_cache.json` | Cached CZI dimensions, stage positions, and voxel sizes (µm). |
| `tile_manifest_corrected.json` | Final per-tile positions after refinement, plus canvas dims and `voxel_size_um`. Used directly by step 06 to build the OME-Zarr pyramid. |

### Inspecting the output

```python
import zarr
z = zarr.open("/vast/scratch/users/kriel.j/KL018_lightsheet/fused_direct.zarr", "r")
print(z.shape)   # (1, 2, 1557, 8701, 10830)
print(z.chunks)  # (1, 1, 64, 512, 512)
mid = z[0, 0, 778, :, :]   # mid-Z plane, channel 0
```

The zarr is plain (not OME-zarr multiscale). To build a pyramid for napari,
run pipeline step 06 (`06_output.sh`) afterwards.

---

## Performance notes

- **Throughput.** On A100 with 8 read-workers, a single Z-chunk of 64
  planes × 30 tiles takes ~2–4 min wall time on staged `/vast/scratch`
  data. Total run for KL018: ~2 hours per channel.
- **RAM.** Peak ≈ 2 × `z_chunk × canvas_H × canvas_W × 4 bytes` plus the
  per-tile fused slab buffers. Lower `--z-chunk` to reduce.
- **Read I/O is the dominant cost.** Always stage the CZI to
  `/vast/scratch` before submission — reads from stornext are ~3× slower
  and the threaded subblock fetches contend on the network filesystem.
- **CUDA version.** The conda env ships `cupy-cuda12x`, which requires
  CUDA 12+. The SLURM wrapper handles `module load CUDA/12.1` so this
  matters only for interactive runs.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `NVRTC_ERROR_COMPILATION ... CUDA versions below 12 not supported` | Cluster default CUDA (11.8) is in path; CuPy needs 12+. | `module load CUDA/12.1` before activating conda. |
| `ArrayMemoryError: Unable to allocate <huge> TiB` | Tile positions diverged during refinement and canvas grew unbounded. | Re-run with `--skip-refine` to confirm; otherwise inspect `tile_manifest_corrected.json` for outlier positions. |
| `'_aicspylibczi.TileInfo' object has no attribute ...` | aicspylibczi API mismatch; the code accesses `.m_index`. | Confirm `aicspylibczi >= 3.x`. |
| Slow first run | CZI header parse is uncached. | Subsequent runs use `czi_layout_cache.json` and start instantly. |
| Visible seams in the fused volume | Taper too narrow for overlap size. | Increase `--taper-px` (try 96–128). |
| Crash mid-chunk on OOM | `z_chunk` too large for canvas size. | Drop `--z-chunk` to 32 or 16. |

---

## Code map

| Function | Lines | Role |
|---|---|---|
| `parse_args()` | 65–105 | CLI definition. |
| `get_czi_layout()` | 112–174 | CZI introspection + caching. |
| `read_plane()`, `read_tile_zchunk()` | 181–212 | Parallel subblock reads. |
| `_gaussian_ramp()`, `fuse_sides()` | 219–255 | Dual-side illumination blend. |
| `_cosine_taper()`, `place_tile_slab()` | 262–315 | Tile-to-canvas accumulation. |
| `_ncc()`, `refine_tile_positions()` | 322–425 | NCC + MST + BFS position refinement. |
| `main()` | 432–579 | Orchestration: open CZI, refine, allocate zarr, stream Z-chunks. |
