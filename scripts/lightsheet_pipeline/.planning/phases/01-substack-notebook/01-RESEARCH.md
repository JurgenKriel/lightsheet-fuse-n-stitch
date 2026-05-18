# Phase 1: Substack & Notebook — Research

**Researched:** 2026-05-18
**Domain:** Zeiss CZI lightsheet fusion parameter tuning — CLI extension + Jupyter notebook
**Confidence:** HIGH

---

## Summary

Phase 1 adds `--z-start`/`--z-end` substack extraction to `07_direct_fuse.py` plus three
additional CLI parameters (`--sigma-frac`, `--fusion-axis`, `--ncc-threshold`) that are
currently hardcoded. It then builds `08_fusion_dev.ipynb` — an interactive parameter-sweep
notebook that runs the modified script via subprocess on a 100-slice substack and renders
stitching-quality diagnostics using a pre-existing kernel that already has all required
dependencies.

The critical environment finding: `lightsheet_env` has **no** matplotlib, ipykernel, or
ipywidgets. `spatialdata_env_2` has all three (matplotlib 3.10.7, ipywidgets 8.1.7,
ipympl 0.9.7, zarr 2.15.0, aicspylibczi 3.3.1) and is already registered as the
`"Python (spatialdata_2025)"` Jupyter kernel. The notebook should use this kernel and
invoke `07_direct_fuse.py` via `subprocess.run()` under `lightsheet_env` — this separates
GPU execution (lightsheet_env + CuPy) from visualization (spatialdata_env_2 + matplotlib).

The NCC refinement mid-Z choice is a non-issue: `refine_tile_positions()` reads Z=778
from the full CZI (independent of any substack range), and the target substack z=728:828
has its midpoint at exactly 778. No changes to `refine_tile_positions()` are needed.

**Primary recommendation:** Extend `07_direct_fuse.py` with five new CLI flags, adjust
`main()` to use absolute Z coordinates in the fusion loop, and build the notebook on the
`spatialdata_2025` kernel using subprocess + zarr + matplotlib — no environment
modification required.

---

## Project Constraints (from CLAUDE.md)

- Heavy computation runs on HPC via SLURM (`sbatch`) or interactive GPU nodes (`srun`)
- Activate `lightsheet_env` (`/vast/scratch/users/kriel.j/lightsheet_env`) for GPU pipeline
- Dask is required for whole-slide image processing (less relevant here — already using
  Z-chunked streaming)
- SLURM output logs: `/vast/scratch/users/kriel.j/output.%j.%N.log`
- Scratch space for large intermediate files: `/vast/scratch/users/kriel.j/`
- Check `scripts/archive/` before rewriting — no conflicting 08_* scripts found

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SUBSTACK-01 | `07_direct_fuse.py` supports `--z-start`/`--z-end` without breaking other params | Five new CLI flags; `main()` loop uses absolute Z coords; `refine_tile_positions()` unchanged |
| SUBSTACK-02 | 100-slice substack produced in <30 min on GPU node | 2 Z-chunks × ~4 min each × 2 channels ≈ 16 min upper bound; well within limit |
| NOTEBOOK-01 | `08_fusion_dev.ipynb` runs all cells to produce fused substack | `spatialdata_2025` kernel + subprocess call to `07_direct_fuse.py --z-start 728 --z-end 828` |
| NOTEBOOK-02 | Notebook exposes `sigma_frac`, `taper_px`, `fusion_axis`, `ncc_threshold`, `skip_refine` | Python dict `PARAMS = {...}` at top of notebook; passed as CLI args to subprocess |
| NOTEBOOK-03 | Notebook renders mid-Z MIP mosaic of all 30 stitched tiles | `zarr.open(substack_path)[0,0,50,:,:]` at 1/8 downsample → `matplotlib.imshow` |
| NOTEBOOK-04 | Notebook renders seam-quality heatmap (per-tile NCC scores on canvas) | Extract NCC matrix from modified `refine_tile_positions()` return; scatter/pcolor on tile grid |
| NOTEBOOK-05 | Notebook renders per-tile illumination uniformity (mean X-profile per tile) | For each tile bbox, slice zarr mid-Z crop → `np.mean(axis=0)` → line plot per tile |
| NOTEBOOK-06 | Notebook outputs `best_params` dict for copy-paste into SLURM script | Final cell prints formatted dict; user edits `07_direct_fuse.sh` manually |
</phase_requirements>

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Substack CLI extension | `07_direct_fuse.py` (CPU/GPU script) | — | All fusion logic lives here; CLI is its interface |
| Parameter exposure (sigma_frac etc.) | `07_direct_fuse.py` parse_args() | notebook PARAMS dict | Script is ground truth; notebook mirrors values |
| GPU dual-side fusion | `07_direct_fuse.py` + lightsheet_env | — | CuPy available only here |
| Visualization / diagnostics | `08_fusion_dev.ipynb` + spatialdata_env_2 | — | matplotlib/ipywidgets live here; no GPU needed |
| Subprocess orchestration | notebook cell | — | Bridges visualization kernel → GPU script |
| NCC quality data export | `07_direct_fuse.py` (new JSON output) | notebook read | Notebook cannot recompute NCC without GPU env |

---

## Standard Stack

### Core (verified in environment)

| Library | Version | Purpose | Confidence |
|---------|---------|---------|------------|
| zarr | 2.15.0 | Read substack output in notebook | VERIFIED: pip list |
| tifffile | 2023.2.28 | Present; used Phase 3 only | VERIFIED: pip list |
| aicspylibczi | 3.3.1 | CZI reading in `07_direct_fuse.py` | VERIFIED: pip list |
| cupy-cuda12x | 14.0.1 | GPU fusion in lightsheet_env | VERIFIED: pip list |
| scipy | 1.17.1 | phase_cross_correlation, csgraph MST | VERIFIED: pip list |
| scikit-image | 0.26.0 | phase_cross_correlation | VERIFIED: pip list |
| matplotlib | 3.10.7 | Notebook visualization | VERIFIED: spatialdata_env_2 pip list |
| ipywidgets | 8.1.7 | Available in spatialdata_env_2 | VERIFIED: spatialdata_env_2 pip list |
| ipympl | 0.9.7 | Interactive matplotlib in notebook | VERIFIED: spatialdata_env_2 pip list |
| numpy | 2.4.4 (lightsheet) / 2.4.5 (spatialdata) | Array ops | VERIFIED: both envs |

### Key Kernel

| Kernel display name | Internal name | Python path | Confidence |
|--------------------|---------------|-------------|------------|
| Python (spatialdata_2025) | `spatialdata_2025` | `/vast/projects/BCRL_Multi_Omics/spatialdata_env_2/bin/python` | VERIFIED: kernel.json |

This is the kernel to use for `08_fusion_dev.ipynb`. It has all visualization
dependencies and is already registered system-wide.

### No Installation Required

The standard approach of installing matplotlib+ipykernel into `lightsheet_env` is
**unnecessary** because:
1. `spatialdata_env_2` already has aicspylibczi (for any local imports if needed)
2. The notebook uses subprocess for GPU fusion — cupy is never imported in-kernel
3. The `spatialdata_2025` kernel is already registered and tested

---

## Architecture Patterns

### System Architecture Diagram

```
User edits PARAMS dict
       |
       v
[08_fusion_dev.ipynb]  (spatialdata_2025 kernel — no GPU needed)
       |
       |-- Cell 1: PARAMS dict (sigma_frac, taper_px, etc.)
       |
       |-- Cell 2: subprocess.run(
       |       ['conda', 'run', '-p', lightsheet_env_path,
       |        'python', '07_direct_fuse.py',
       |        '--z-start', '728', '--z-end', '828',
       |        '--out', substack_zarr_path,
       |        '--sigma-frac', str(PARAMS['sigma_frac']),
       |        '--taper-px', str(PARAMS['taper_px']),
       |        '--ncc-threshold', str(PARAMS['ncc_threshold']),
       |        '--fusion-axis', str(PARAMS['fusion_axis']),
       |        '--skip-refine' if PARAMS['skip_refine'] else ''
       |       ])
       |       (runs in lightsheet_env, uses CuPy GPU)
       |
       |-- Cell 3: zarr.open(substack_zarr_path)
       |       -> mid-Z MIP mosaic (matplotlib imshow, 1/8 downsample)
       |
       |-- Cell 4: Load ncc_scores.json (written by 07_direct_fuse.py)
       |       -> seam heatmap on tile grid canvas (matplotlib scatter/pcolormesh)
       |
       |-- Cell 5: For each tile bbox, slice zarr mid-Z
       |       -> np.mean(axis=0) per tile -> line plot of X-profile
       |
       |-- Cell 6: print(best_params) -> copy into 07_direct_fuse.sh
       |
[/vast/scratch/.../substack_z728_828.zarr]  shape: (1, 2, 100, 8701, 10830)
[/vast/scratch/.../ncc_scores.json]         NCC quality matrix + tile positions
```

### Recommended Project Structure

```
scripts/lightsheet_pipeline/
├── 07_direct_fuse.py          # MODIFIED: +5 CLI flags, z-start/end loop, ncc JSON output
├── 07_direct_fuse.sh          # MODIFIED: document new flags (not changed functionally yet)
├── 07_direct_fuse_README.md   # MODIFIED: document new flags
├── 08_fusion_dev.ipynb        # NEW: parameter sweep notebook
└── .planning/phases/01-substack-notebook/
    └── 01-RESEARCH.md
```

Output scratch paths:
```
/vast/scratch/users/kriel.j/KL018_lightsheet/
├── substack_z728_828.zarr         # substack output (34 GB, T=1,C=2,Z=100)
├── ncc_scores.json                # NCC quality matrix written by modified script
├── czi_layout_cache.json          # already exists — reused as-is
└── tile_manifest_corrected.json   # written by each run
```

---

## Detailed Technical Findings

### Finding 1: Changes Required to `07_direct_fuse.py`

**Five new CLI arguments** (none exist today):

```python
# In parse_args():
p.add_argument("--z-start", type=int, default=0,
    help="First Z plane to process (inclusive). Default=0 (full stack).")
p.add_argument("--z-end", type=int, default=None,
    help="Last Z plane to process (exclusive). Default=None means n_z.")
p.add_argument("--sigma-frac", type=float, default=0.3,
    help="Gaussian blend sigma fraction for dual-side fusion (default 0.3).")
p.add_argument("--fusion-axis", type=int, default=2,
    help="Axis along which dual-side illumination varies: 0=Z,1=Y,2=X (default 2=X).")
p.add_argument("--ncc-threshold", type=float, default=0.05,
    help="NCC quality threshold for phase-correlation refinement (default 0.05).")
```

**Changes to `_gaussian_ramp()`**: add `sigma_frac` as a parameter (remove the hardcoded
default-only signature).

**Changes to `fuse_sides()`**: accept `sigma_frac` and `axis` from args, pass to
`_gaussian_ramp()`.

**Changes to `refine_tile_positions()`**: accept `ncc_threshold` from args (replaces
hardcoded `0.05` on line 421). Also accept a `ncc_out_path` parameter so the NCC quality
matrix is saved to JSON for notebook consumption.

**Changes to `main()` fusion loop**:

```python
# After layout parse:
z_start = args.z_start
z_end = args.z_end if args.z_end is not None else n_z
n_z_proc = z_end - z_start

# Output zarr shape uses n_z_proc instead of n_z:
out_shape = (n_t, n_c, n_z_proc, refined_h, refined_w)

# Inner loop uses absolute Z addresses for CZI reads,
# but local (0-based) indices for zarr writes:
for chunk_idx in range(math.ceil(n_z_proc / args.z_chunk)):
    z0_abs = z_start + chunk_idx * args.z_chunk   # absolute CZI Z
    z1_abs = min(z0_abs + args.z_chunk, z_end)
    nz = z1_abs - z0_abs
    z0_loc = chunk_idx * args.z_chunk              # local zarr Z
    # read_tile_zchunk(... z0_abs, z1_abs ...) — already takes absolute Z
    # out_z[t, c, z0_loc : z0_loc + nz] = ...
```

**NCC matrix JSON output** (new, written by `refine_tile_positions()`):

```python
# ncc_scores.json written alongside tile_manifest_corrected.json:
{
  "quality_matrix": [[row0_scores...], ...],   # n_tiles x n_tiles float32
  "mid_z": 778,
  "ncc_threshold": 0.05,
  "tile_positions_refined": [...]
}
```

[VERIFIED: code read of 07_direct_fuse.py lines 353-455]

### Finding 2: NCC Mid-Z is Safe for Substack

`refine_tile_positions()` reads `mid_z = layout["n_z"] // 2 = 778` from the **full CZI**
regardless of `--z-start`/`--z-end`. This is correct behavior: position refinement should
use a representative plane independent of the substack being processed.

The target substack z=728:828 has midpoint at 778 — an exact match. Even for substacks
at other Z ranges, always using the full-stack mid-Z is the right choice for consistency.

**No changes to `refine_tile_positions()` Z-reading logic are needed.**

[VERIFIED: code read of 07_direct_fuse.py line 379]

### Finding 3: Tile Grid and Overlap Geometry

From `czi_layout_cache.json`:

| Property | Value | Source |
|----------|-------|--------|
| Grid | 6 columns × 5 rows = 30 tiles | VERIFIED: cache |
| Tile size | 1920 × 1920 px | VERIFIED: cache |
| X step | 1632 px | VERIFIED: computed |
| Y step | 1632 px | VERIFIED: computed |
| Overlap (X and Y) | 288 px (15% of 1920) | VERIFIED: computed |
| Current taper_px | 64 px = 22% of overlap | VERIFIED: README |
| Recommended taper_px | 128-144 px = 44-50% of overlap | ASSUMED |
| Canvas (raw) | 10079 × 8448 px | VERIFIED: cache |
| Canvas (refined, from existing run) | 10830 × 8701 px | VERIFIED: tile_manifest_corrected.json |

The current `taper_px=64` covers only 22% of the 288 px overlap — this explains the
visible seams. A taper of 128-144 px (covering 44-50% of overlap) is the typical standard
for light sheet mosaic stitching.

[VERIFIED: czi_layout_cache.json + computed]

### Finding 4: Existing `fused_direct.zarr` Is Navigable

```
Shape: (1, 2, 1557, 8701, 10830)   # TCZYX
Chunks: (1, 1, 64, 512, 512)
dtype: uint16
Mid-Z (Z=778, C=0) pixel range: 0 – 478 (uint16, low dynamic range)
Mean intensity: ~194 counts
```

A 1/8 thumbnail of a single mid-Z plane is 1088 × 1354 px — suitable for `matplotlib.imshow`
in a notebook cell without memory pressure.

[VERIFIED: zarr.open + z.shape/z.chunks]

### Finding 5: Kernel and Environment Strategy

**Use the `spatialdata_2025` kernel** (display name "Python (spatialdata_2025)") for
`08_fusion_dev.ipynb`. This kernel:

- Points to `/vast/projects/BCRL_Multi_Omics/spatialdata_env_2/bin/python`
- Has: matplotlib 3.10.7, ipywidgets 8.1.7, ipympl 0.9.7, zarr 2.15.0, aicspylibczi 3.3.1
- Is already registered in JupyterLab (`~/.local/share/jupyter/kernels/spatialdata_2025/`)
- Does NOT need CuPy (fusion runs via subprocess in lightsheet_env)

**Subprocess invocation pattern** for the fusion cell:

```python
import subprocess, sys, os

lightsheet_env = "/vast/scratch/users/kriel.j/lightsheet_env"
script = "/vast/projects/BCRL_Multi_Omics/scripts/lightsheet_pipeline/07_direct_fuse.py"
out_zarr = "/vast/scratch/users/kriel.j/KL018_lightsheet/substack_z728_828.zarr"

cmd = [
    "conda", "run", "--no-capture-output", "-p", lightsheet_env,
    "bash", "-c",
    f"module load CUDA/12.1 && python {script} "
    f"--z-start {PARAMS['z_start']} --z-end {PARAMS['z_end']} "
    f"--out {out_zarr} "
    f"--sigma-frac {PARAMS['sigma_frac']} "
    f"--taper-px {PARAMS['taper_px']} "
    f"--ncc-threshold {PARAMS['ncc_threshold']} "
    f"--fusion-axis {PARAMS['fusion_axis']} "
    + ("--skip-refine " if PARAMS['skip_refine'] else "")
]
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout[-3000:])  # tail of output
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])
```

**Alternative (simpler) subprocess invocation** using the lightsheet_env Python directly:

```python
python_bin = "/vast/scratch/users/kriel.j/lightsheet_env/bin/python"
# CUDA must be loaded in the parent shell before launching JupyterLab for this to work
cmd = [python_bin, script, "--z-start", "728", ...]
```

The `conda run + bash -c "module load CUDA/12.1 && ..."` form is more reliable on SLURM
nodes where the shell profile may not have loaded CUDA automatically.

[VERIFIED: kernel.json read + pip list of spatialdata_env_2]

### Finding 6: Visualization Patterns

#### Mid-Z MIP mosaic (NOTEBOOK-03)

```python
import zarr, numpy as np, matplotlib.pyplot as plt

z = zarr.open(out_zarr, "r")
# Substack zarr shape: (1, 2, 100, H, W)
mid_loc = z.shape[2] // 2   # local Z index = 50 for 100-slice
plane = z[0, 0, mid_loc, ::8, ::8]   # 1/8 downsample, channel 0

fig, ax = plt.subplots(figsize=(14, 10))
ax.imshow(plane, cmap="gray",
          vmin=np.percentile(plane, 1), vmax=np.percentile(plane, 99))
ax.set_title(f"Mid-Z MIP (z={PARAMS['z_start'] + mid_loc}), CH0 — 1/8 downsample")
plt.tight_layout()
```

[VERIFIED: zarr shape confirmed, numpy/matplotlib available in spatialdata_env_2]

#### Seam quality heatmap (NOTEBOOK-04)

The NCC quality matrix is 30×30 (tile i, tile j → quality score). For the heatmap,
display as a seam diagram: for each adjacent tile pair with non-zero quality, plot a
colored rectangle at the seam midpoint between their bounding boxes.

```python
import json, matplotlib.pyplot as plt, numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

with open(".../ncc_scores.json") as f:
    ncc_data = json.load(f)

quality = np.array(ncc_data["quality_matrix"])
tile_pos = ncc_data["tile_positions_refined"]  # list of {M, x, y, w, h}

# Build tile center dict
centers = {p["M"]: (p["x"] + p["w"]//2, p["y"] + p["h"]//2) for p in tile_pos}

fig, ax = plt.subplots(figsize=(12, 10))
norm = Normalize(vmin=0, vmax=quality.max())
cmap = plt.cm.RdYlGn

for i in range(len(tile_pos)):
    for j in range(i+1, len(tile_pos)):
        q = quality[i][j]
        if q > 0:
            cx = (centers[i][0] + centers[j][0]) / 2
            cy = (centers[i][1] + centers[j][1]) / 2
            ax.plot(cx, cy, 's', color=cmap(norm(q)), markersize=14, alpha=0.85)

# Draw tile outlines
for p in tile_pos:
    rect = plt.Rectangle((p["x"], p["y"]), p["w"], p["h"],
                          fill=False, edgecolor='gray', linewidth=0.5)
    ax.add_patch(rect)
ax.set_aspect("equal")
ax.invert_yaxis()
fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="NCC quality")
ax.set_title(f"Seam quality heatmap (NCC threshold={PARAMS['ncc_threshold']})")
```

[VERIFIED: pattern derived from NCC matrix structure in code lines 386-435 + zarr shape]
[ASSUMED: ncc_scores.json schema — depends on implementation in refine_tile_positions()]

#### Per-tile illumination uniformity (NOTEBOOK-05)

```python
# For each tile, read its bounding box from the substack zarr mid-Z plane
# and compute mean intensity along X (columns) to show illumination gradient

plane_full = z[0, 0, mid_loc, :, :]   # (H, W) uint16, full canvas
canvas_ox = min(p["x"] for p in tile_pos)
canvas_oy = min(p["y"] for p in tile_pos)

fig, ax = plt.subplots(figsize=(14, 5))
for p in tile_pos:
    x0 = p["x"] - canvas_ox
    y0 = p["y"] - canvas_oy
    tile_crop = plane_full[y0:y0+p["h"], x0:x0+p["w"]]
    profile = tile_crop.mean(axis=0)   # mean along Y -> (W,) profile across X
    ax.plot(profile, alpha=0.4, linewidth=0.8)

ax.set_xlabel("X pixel within tile")
ax.set_ylabel("Mean intensity (counts)")
ax.set_title("Per-tile illumination X-profile (ideal = flat line)")
```

[VERIFIED: zarr structure + tile_manifest_corrected.json positions confirmed]

#### best_params output cell (NOTEBOOK-06)

```python
best_params = {
    "z_start":        PARAMS["z_start"],
    "z_end":          PARAMS["z_end"],
    "sigma_frac":     PARAMS["sigma_frac"],
    "taper_px":       PARAMS["taper_px"],
    "ncc_threshold":  PARAMS["ncc_threshold"],
    "fusion_axis":    PARAMS["fusion_axis"],
    "skip_refine":    PARAMS["skip_refine"],
    "z_chunk":        PARAMS.get("z_chunk", 64),
    "workers":        PARAMS.get("workers", 8),
}

print("# Paste into 07_direct_fuse.sh:\n")
for k, v in best_params.items():
    if k not in ("z_start", "z_end"):   # substack-only; full run uses defaults
        flag = k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                print(f"    --{flag} \\")
        else:
            print(f"    --{flag} {v} \\")
```

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Phase cross-correlation | Custom FFT shift estimator | `skimage.registration.phase_cross_correlation` | Sub-pixel accuracy, tested, already imported |
| Sparse graph MST | Custom Prim/Kruskal | `scipy.sparse.csgraph.minimum_spanning_tree` | Already used in script; proven |
| Pyramid TIFF write (Phase 3) | Custom IFD writer | `tifffile.TiffWriter` with `subifds=` | `TiffWriter.write` has `subifds` param; confirmed in env |
| Large array display | Load full canvas to memory | zarr lazy slice at 1/8 downsample | Full mid-Z plane is 178 MB; thumbnail is 3 MB |
| NCC quality heatmap grid | Seaborn heatmap (wrong layout) | Custom matplotlib scatter on canvas coords | Seams are spatial; must be placed at seam midpoints not in a matrix grid |

---

## Common Pitfalls

### Pitfall 1: Absolute vs Local Z Indexing in Main Loop

**What goes wrong:** Using `z0 = chunk_idx * z_chunk` as both the CZI read address AND
the zarr write address. Without `--z-start`, these are the same. With `--z-start=728`,
the first Z-chunk reads CZI planes 728-791 but must write to zarr[0:64] not zarr[728:792].

**Root cause:** `read_tile_zchunk()` takes absolute CZI Z coordinates; zarr indexing is
always 0-based relative to the output shape.

**How to avoid:** Maintain two variables: `z0_abs` (for CZI reads) and `z0_loc` (for zarr
writes). The pattern is:
```python
z0_abs = z_start + chunk_idx * z_chunk
z1_abs = min(z0_abs + z_chunk, z_end)
z0_loc = chunk_idx * z_chunk
out_z[t, c, z0_loc : z0_loc + (z1_abs - z0_abs)] = ...
```

**Warning signs:** Output zarr has `n_z` (1557) planes instead of `n_z_proc` (100); or
`IndexError: index 728 is out of bounds for axis with size 100`.

[VERIFIED: code read of main() loop lines 530-586]

### Pitfall 2: sigma_frac Scope — Global Ramp vs Per-Tile Application

**What goes wrong:** `_gaussian_ramp(size, xp, sigma_frac)` builds a 1-D ramp of `size`
points. The `size` is the tile dimension (1920), not the canvas. `fuse_sides()` currently
receives `axis=2` hardcoded in the `_process_tile` inner function call. Adding `sigma_frac`
as a CLI arg requires threading it through the nested `_process_tile` closure inside `main()`.

**How to avoid:** Pass `sigma_frac` and `fusion_axis` as captured variables in the
`_process_tile` closure, or refactor it out of the nested function. The closure captures
`args` already, so `args.sigma_frac` is accessible — just pass it to `fuse_sides()`.

[VERIFIED: code read of _process_tile lines 544-558 + fuse_sides signature lines 248-279]

### Pitfall 3: CUDA Not Loaded in Subprocess Shell

**What goes wrong:** The notebook subprocess launches `python 07_direct_fuse.py` in a
subshell that does not inherit `module load CUDA/12.1`. CuPy imports fail with
`NVRTC_ERROR_COMPILATION` or silently falls back to NumPy (no GPU acceleration).

**How to avoid:** Use the `bash -c "module load CUDA/12.1 && python ..."` form. Verify
GPU is active by checking `_GPU` status in script output (script prints `Backend: CuPy (GPU)`
or `NumPy (CPU)` at startup). The SLURM wrapper `07_direct_fuse.sh` already does this
correctly — mirror that pattern in subprocess.

**Warning signs:** Script prints `Backend: NumPy (CPU)` in notebook output. Runtime
degrades from ~4 min/chunk to ~30+ min/chunk.

[VERIFIED: 07_direct_fuse.sh lines 44-47 + README CUDA section]

### Pitfall 4: NCC Heatmap Coordinate System Mismatch

**What goes wrong:** Tile positions in `tile_manifest_corrected.json` use absolute canvas
coordinates (e.g., x=16, y=127 for tile 0). The zarr canvas starts at (0,0). If the
notebook uses raw CZI positions (from `czi_layout_cache.json`) instead of refined
positions (from `tile_manifest_corrected.json`), the seam overlay will be shifted.

**How to avoid:** Always read seam visualization positions from `tile_manifest_corrected.json`
(after refinement) or from `ncc_scores.json` (proposed, written by the same run). The
refined canvas origin is `min(all_x0)` computed in `main()`.

[VERIFIED: tile_manifest_corrected.json inspected — tile 0 at x=16,y=127 not x=0,y=0]

### Pitfall 5: ipywidgets Not Activated in Notebook

**What goes wrong:** `ipywidgets` is installed but `%matplotlib widget` or
`interact(...)` shows no output — widget state not committed to frontend.

**How to avoid:** Use the `%matplotlib inline` backend for static plots (simpler, always
works). Reserve `%matplotlib widget` (ipympl) only if interactive pan/zoom on the mosaic
is specifically needed. For parameter sweeps, a manual dict + re-run-cell pattern is more
reliable on HPC than `interact()` callbacks that require a live kernel connection.

[VERIFIED: ipympl 0.9.7 available; ASSUMED that static plots are sufficient for this use case]

### Pitfall 6: zarr Array Memory on Full Plane Read

**What goes wrong:** `z[0, 0, 50, :, :]` loads the entire 10830×8701 plane (188 MB,
uint16). In `spatialdata_env_2`, this is fine for one channel. But looping over all 30
tile crops without releasing intermediate arrays can accumulate to several GB.

**How to avoid:** Use strided indexing for display (`z[0, 0, 50, ::8, ::8]`). For tile
crop illumination profiles, read one tile crop at a time rather than the full canvas then
slicing.

[VERIFIED: zarr shape and dtype confirmed; memory math verified]

---

## Runtime State Inventory

> Greenfield phase (new files added, no renames). No runtime state migration needed.

| Category | Items Found | Action Required |
|----------|-------------|-----------------|
| Stored data | `fused_direct.zarr` at `/vast/scratch/...` — existing broken output, shape (1,2,1557,8701,10830) | No migration; substack writes to separate path `substack_z728_828.zarr` |
| Live service config | None | None |
| OS-registered state | None | None |
| Secrets/env vars | None | None |
| Build artifacts | `czi_layout_cache.json` already at scratch path — used as-is | None |

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| lightsheet_env (conda) | `07_direct_fuse.py` GPU execution | Yes | at `/vast/scratch/users/kriel.j/lightsheet_env` | — |
| cupy-cuda12x | GPU fusion in lightsheet_env | Yes (on GPU node) | 14.0.1 | NumPy CPU fallback (3-5x slower) |
| CUDA/12.1 module | cupy kernel compilation | Yes | CUDA/12.1 available on cluster | CUDA/12.3, 12.4, 12.5, 12.8 also available |
| aicspylibczi | CZI reading | Yes | 3.3.1 (both envs) | — |
| spatialdata_env_2 | Notebook kernel | Yes | at `/vast/projects/BCRL_Multi_Omics/spatialdata_env_2` | — |
| matplotlib | Notebook visualization | Yes | 3.10.7 (spatialdata_env_2) | — |
| ipywidgets | Notebook widgets | Yes | 8.1.7 (spatialdata_env_2) | Static dicts + re-run |
| spatialdata_2025 kernel | JupyterLab kernel | Yes | registered at `~/.local/share/jupyter/kernels/spatialdata_2025/` | Register new kernel from env |
| GPU SLURM node | <30 min substack requirement | Yes | A100 via `--gres=gpu:A100:1` | Longer wall time on CPU (fails the 30 min SLA) |
| KL018 CZI staged | Fast I/O | Yes | `/vast/scratch/users/kriel.j/KL018_lightsheet/KL018_85_D7_CT2AvIII_Overview.czi` | Stornext path (3x slower) |

**Missing dependencies with no fallback:** None — all required components are present.

**Missing dependencies with fallback:**
- ipykernel in lightsheet_env: NOT installed (no fallback needed — notebook uses spatialdata_2025 kernel)

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Fixed axis=2 hardcoded in `fuse_sides()` | Expose as `--fusion-axis` CLI arg | Allows testing Y-axis fusion for datasets where illumination enters from Y |
| Fixed sigma_frac=0.3 hardcoded | Expose as `--sigma-frac` | Wider sigma gives flatter illumination crossover; sweep 0.2–0.6 |
| Fixed NCC threshold=0.05 hardcoded | Expose as `--ncc-threshold` | Raise to 0.1-0.2 to skip weak/noisy overlaps; lower to 0.01 to include all |
| No substack support (processes all 1557 Z) | `--z-start`/`--z-end` | Enables <30 min parameter iteration |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Subprocess `conda run -p env bash -c "module load CUDA && python ..."` pattern works on GPU srun node | Architecture Patterns | Module system may not source correctly from non-login shell; may need absolute CUDA path instead |
| A2 | taper_px=128-144 will visibly reduce seams (22% → 44-50% overlap coverage) | Tile Grid Finding | May need larger values; actual optimal depends on tissue content and stage accuracy |
| A3 | Static matplotlib plots (no ipywidgets interact) are sufficient for NOTEBOOK-02 parameter iteration | Pitfall 5 | If user wants live sliders, ipympl %matplotlib widget mode needs additional setup |
| A4 | `ncc_scores.json` schema as proposed (quality_matrix + tile_positions_refined) matches what refine_tile_positions() can be modified to output | Visualization Patterns | If the dict format is changed, notebook parsing breaks; keep schema simple |
| A5 | The existing `fused_direct.zarr` has visible artifacts at Z=778 representative of the tile seam problem | Project context | True given PROJECT.md description of "broken" output |

---

## Open Questions

1. **Where does the notebook run — GPU SLURM node or login node?**
   - What we know: The notebook itself (matplotlib/zarr visualization) requires no GPU. Only
     the subprocess fusion call needs GPU.
   - What's unclear: Whether the user will launch JupyterLab from a GPU srun node (so
     subprocess inherits GPU access) or from a login node (subprocess must explicitly request
     a GPU via a nested `srun` or `sbatch`).
   - Recommendation: Design the subprocess cell for the GPU-node case (user has done
     `srun --gres=gpu:A100:1 --pty bash` before launching JupyterLab). Add a comment noting
     that login-node notebook use requires wrapping the subprocess in `sbatch` and polling.

2. **Should the notebook re-run NCC or read pre-computed scores?**
   - What we know: NCC takes ~2 min for 30 tiles; it runs as part of `07_direct_fuse.py`.
   - What's unclear: Whether to run NCC once (write JSON, never recompute) or on every
     parameter change.
   - Recommendation: Run NCC once per `--ncc-threshold` change; cache to `ncc_scores.json`;
     parameter changes to `sigma_frac`/`taper_px` reuse the cached NCC scores (add
     `--reuse-ncc-cache` flag or just `--skip-refine` to skip NCC entirely and use cached
     positions from `tile_manifest_corrected.json`).

3. **Should `fuse_sides()` `sigma_frac` parameter be per-call or global?**
   - What we know: It is currently global (one value per run).
   - What's unclear: Whether different tiles might benefit from different sigma values
     (e.g., edge tiles vs. interior tiles).
   - Recommendation: Keep it global for Phase 1 — per-tile sigma is a v2 complexity.

---

## Validation Architecture

> `nyquist_validation: false` in config.json — skip formal test framework setup.

Manual validation gates for this phase:

| Check | How | Pass Criterion |
|-------|-----|----------------|
| SUBSTACK-01: CLI flags accepted | `python 07_direct_fuse.py --help` | All 5 new flags appear in help text |
| SUBSTACK-01: Z-range respected | Check output zarr shape | `z.shape[2] == 100` for `--z-start 728 --z-end 828` |
| SUBSTACK-02: Wall time | SLURM log timestamps | Completion in <30 min |
| NOTEBOOK-01: All cells run | Kernel restart + Run All | No errors |
| NOTEBOOK-03: Mosaic renders | Visual inspection | All 30 tiles visible, no blank regions |
| NOTEBOOK-04: Heatmap renders | Visual inspection | Colored squares at seam locations |
| NOTEBOOK-05: Profiles render | Visual inspection | 30 line profiles with X-axis gradient visible |
| NOTEBOOK-06: Dict output | Cell output | Valid Python dict with all 9 keys |

---

## Sources

### Primary (HIGH confidence)
- `/vast/projects/BCRL_Multi_Omics/scripts/lightsheet_pipeline/07_direct_fuse.py` — full code read
- `/vast/projects/BCRL_Multi_Omics/scripts/lightsheet_pipeline/07_direct_fuse_README.md` — full read
- `/vast/scratch/users/kriel.j/KL018_lightsheet/czi_layout_cache.json` — verified tile positions
- `/vast/scratch/users/kriel.j/KL018_lightsheet/tile_manifest_corrected.json` — verified canvas dims
- `/vast/scratch/users/kriel.j/KL018_lightsheet/fused_direct.zarr` — shape/dtype verified
- `conda run pip list` — both environments verified
- `~/.local/share/jupyter/kernels/spatialdata_2025/kernel.json` — kernel path verified

### Secondary (MEDIUM confidence)
- `07_direct_fuse.sh` — SLURM wrapper read; subprocess CUDA loading pattern derived from it

### Tertiary (LOW confidence)
- taper_px optimal range (128-144) derived from first-principles overlap geometry [ASSUMED]
- Subprocess `module load` behavior in subshell [ASSUMED based on SLURM conventions]

---

## Metadata

**Confidence breakdown:**
- Script modification plan: HIGH — code fully read, every function inspected
- Environment/kernel strategy: HIGH — pip list verified in both envs, kernel.json verified
- Visualization code patterns: HIGH — zarr shape/dtype verified, matplotlib available
- NCC heatmap JSON schema: MEDIUM — schema is proposed, not yet implemented
- taper_px recommendations: LOW — first principles only, not empirically validated

**Research date:** 2026-05-18
**Valid until:** 2026-06-18 (stable: scipy/zarr APIs not changing; env pinned)
