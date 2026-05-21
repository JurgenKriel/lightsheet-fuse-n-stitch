# Light-Sheet Direct-Fuse & Globally-Optimised Stitcher

GPU-accelerated pipeline for fusing dual-side illumination and stitching mosaic light-sheet acquisitions captured on a Zeiss Z.1 (or compatible) into a single OME-Zarr volume.

Two stages, both designed for SLURM-managed HPC:

| Stage | Script | Purpose | Hardware |
|---|---|---|---|
| **07 — Direct Fuse** | `07_direct_fuse.py` | Reads a Zeiss mosaic CZI directly via `aicspylibczi`, GPU-fuses dual-side illumination per tile (`fuse_sides`), and writes a *raw-positioned* OME-Zarr canvas (`fused_direct.zarr`). MST + cosine-taper stitcher (legacy) or stage-position-only output. | 1× A100 (or 2 medium GPUs for stage 1/2 split), ~400 GB RAM |
| **08 — Stitch** | `08_stitch.py` | Replaces the legacy stitcher with `multiview-stitcher` global LSQ registration + weighted-average fusion. Two sub-stages: `register` (Stage 1) computes per-tile corrections from a Z-slab around mid-Z, `blend` (Stage 2) re-applies those corrections and writes the final stitched canvas in parallel SLURM array tasks. | 1× A30 (Stage 1), 8× SLURM array tasks for Stage 2 |

Stage 08 imports the dual-side fusion path (`fuse_sides`, `read_tile_zchunk`) unchanged from 07 — do not delete 07 even if you only run 08.

---

## When to use which

```
┌──────────────────────────────────────────────────────────┐
│ Just need a fused volume at stage-metadata positions?    │
│  → 07_direct_fuse.py (single-job or 2-stage parallel)    │
│                                                          │
│ Need globally-optimised registration across tiles?       │
│  → 08_stitch.py (preflight → production)                 │
│                                                          │
│ Both? 08 will re-use 07's dual-side fusion automatically.│
└──────────────────────────────────────────────────────────┘
```

---

## Hardware & software requirements

- **GPU:** CUDA 12-compatible (CuPy 14.x). A100 for the fastest single-job runs of 07; A30 is sufficient for both stages of 08.
- **Driver/toolkit:** CUDA 12.1 (loaded via `module load CUDA/12.1` in the SLURM scripts — adjust if your HPC uses a different module name).
- **RAM:** ~80 GB per Stage 1/Stage 2 task is typical; 400 GB for single-job `07_direct_fuse.sh` runs on full-stack 30-tile mosaics.
- **Scratch:** Several hundred GB on fast scratch (`/vast/scratch` in our case). The raw CZI for KL018 is ~1.3 TB; the staged fused_direct.zarr at uint16 is ~50–80 GB.
- **Python:** 3.11. **Two separate conda environments** are recommended (see Installation):
  - `lightsheet_env` — for stage 07 (does not need multiview-stitcher).
  - `mvstitch_env` — for stage 08 (needs `multiview-stitcher==0.1.52`, which pins `dask<2025.11.0` and would otherwise downgrade other tools).

---

## Installation

### 1. mvstitch_env (required for stage 08)

```bash
conda env create -f environment.yml -p /path/to/mvstitch_env
# Or, manual:
conda create -n mvstitch_env python=3.11 pip -c conda-forge
conda activate mvstitch_env
pip install \
    "multiview-stitcher==0.1.52" \
    "multiview-stitcher[gpu-cuda12]==0.1.52" \
    "spatial-image==1.2.3" \
    "multiscale-spatial-image==2.0.3" \
    "ngff-zarr>=0.12.2" \
    "aicspylibczi==3.3.1" \
    scipy scikit-image zarr numpy
```

The exact pins matter — `multiview-stitcher` 0.1.52 silently breaks against newer `dask`, and the dual-side fusion path imported from `07_direct_fuse.py` needs `aicspylibczi==3.3.1`.

### 2. lightsheet_env (only if running stage 07 standalone)

```bash
conda create -n lightsheet_env python=3.11 pip -c conda-forge
conda activate lightsheet_env
pip install aicspylibczi==3.3.1 zarr scipy scikit-image cupy-cuda12x
```

---

## Quick start (full pipeline, KL018-style 6×5 mosaic)

```bash
# 0. Stage CZI from slow archive (stornext/object store) to fast scratch.
sbatch scripts/07a_stage_czi.sh

# 1. Direct-fuse 07 — choose ONE of:
#    A) Single A100 job (~3.3 h on a 30-tile, 1.3 TB CZI):
sbatch scripts/07_direct_fuse.sh
#    B) Two-stage parallel (~50 min total):
JID=$(sbatch --parsable scripts/07_direct_fuse_stage1_ncc.sh)
sbatch --dependency=afterok:$JID scripts/07_direct_fuse_stage2_parallel.sh

# 2. Stitcher preflight — 32-plane substack, ~30 min. Gates regressions
#    before committing to the full register/blend cost.
PFR=$(sbatch --parsable scripts/08_stitch_preflight_stage1.sh)
sbatch --dependency=afterok:$PFR scripts/08_stitch_preflight_stage2.sh

# 3. Production stitcher — register (~1 h) then blend (~6 h on 8-task array).
SR=$(sbatch --parsable scripts/08_stitch_stage1_register.sh)
sbatch --dependency=afterok:$SR scripts/08_stitch_stage2_blend.sh
```

---

## What each script does (and what to edit)

### `scripts/07a_stage_czi.sh`
Copies the raw CZI from slow archival storage (`/stornext/...` in our case) to fast scratch (`/vast/scratch/...`). Edit `STORNEXT_CZI` and `STAGED_CZI` at the top to match your storage layout.

### `scripts/07_direct_fuse.py` (Python core)
Reads a Zeiss mosaic CZI directly; GPU-fuses each tile's dual-side illumination using a Gaussian-ramp `fuse_sides`; assembles tiles onto a single canvas with optional NCC-refined offsets propagated along an MST.

Key flags (`python 07_direct_fuse.py --help` for the full list):

| Flag | Default | Notes |
|---|---|---|
| `--czi PATH` | KL018 staged copy | Input CZI |
| `--out PATH` | `fused_direct.zarr` | Output OME-Zarr |
| `--z-chunk N` | 64 | Z planes per processing slab (memory knob) |
| `--workers N` | 16 | Threads for subblock reads |
| `--skip-refine` | off | Use raw stage positions, skip NCC refinement |
| `--taper-px N` | 64 | Cosine taper width at tile edges |
| `--sigma-frac F` | 0.5 | Gaussian sigma fraction for `fuse_sides` |
| `--ncc-threshold F` | 0.5 | Reject pairs below this NCC during MST refinement |
| `--fusion-axis N` | 2 | 2 = X-axis dual-side blend (Z.1 default) |
| `--normalize-tiles` | off | Per-tile median normalization (recommended for KL018) |
| `--z-start N` / `--z-end N` | full | Substack mode — process only `[z_start, z_end)` |

For KL018 the **validated full-stack parameters** (from the 2026-05-18 substack sweep in `08_fusion_dev.ipynb`) are baked into `07_direct_fuse.sh`:

```
--sigma-frac 0.9  --taper-px 288  --ncc-threshold 0.5
--fusion-axis 2   --normalize-tiles
```

### `scripts/07_direct_fuse.sh`
Single-A100 SLURM wrapper (24-hour wall, 400 GB RAM, 48 CPU). Edit:
- `MANIFEST` (only needed if you regenerate per-tile metadata first)
- `STAGED_CZI` / `STORNEXT_CZI` paths
- `SCRIPT_DIR` to point at where you cloned this repo
- `module load CUDA/12.1` if your HPC uses a different module name
- `conda activate /path/to/lightsheet_env` to your env

### `scripts/07_direct_fuse_stage1_ncc.sh` + `scripts/07_direct_fuse_stage2_parallel.sh`
Two-stage parallel variant. Stage 1 computes NCC refinements and pre-allocates the output zarr; Stage 2 is an 8-task SLURM array that writes disjoint Z-slabs in parallel via `mode="r+"`. Total wall time roughly 50 min vs. 3.3 h for the single-job variant on the same hardware.

Submit:
```bash
JID=$(sbatch --parsable scripts/07_direct_fuse_stage1_ncc.sh)
sbatch --dependency=afterok:$JID scripts/07_direct_fuse_stage2_parallel.sh
```

### `scripts/08_stitch.py` (Python core)
The globally-optimised stitcher. Two stages selected by `--stage register|blend`:

**Stage 1 — register**
```
python 08_stitch.py --stage register \
    --czi /path/to/raw.czi \
    --out-dir /path/to/scratch \
    --out-zarr /path/to/output.zarr \
    [--z-slab-half 32] [--reg-z-bin 1] [--pre-reg-pruning-method keep_axis_aligned]
```
Reads a `2 × z-slab-half`-plane Z-slab around mid-Z for each tile, runs `multiview_stitcher.registration.register(..., groupwise_resolution_method="global_optimization")`, and writes:
- `stitch_positions.json` — per-tile canvas-absolute positions in µm + correction deltas
- `stitch_diagnostics.json` — per-pair quality, residuals, canvas extents, groupwise solver state
- `czi_layout_cache.json` — tile bboxes from CZI mosaic metadata (authoritative canvas)

Important knobs:
- `--z-slab-half N` — half-thickness of the registration slab. **Use ≥ 16** (32-plane slab) for stable phase correlation; thinner slabs produce `quality=0` and falsely-converged solutions.
- `--reg-z-bin N` — Z-binning factor passed to mvstitch's `registration_binning`. Keep at 1 unless you understand the trade-off; 2 halves per-pair cost but starves phase correlation of out-of-plane signal on thin slabs.
- `--pre-reg-pruning-method` — `keep_axis_aligned` is the correct choice for a regular grid (deterministic, ~49 pairs for a 6×5 grid). The library default `alternating_pattern` is nondeterministic and may leave diagonal pairs that blow out the runtime.

**Stage 2 — blend**
```
python 08_stitch.py --stage blend \
    --czi /path/to/raw.czi \
    --out-dir /path/to/scratch \
    --out-zarr /path/to/output.zarr \
    --z-start 0 --z-end 256 --z-chunk 64
```
Each SLURM array task gets a disjoint `[z-start, z-end)` and writes into the pre-allocated zarr via `mode="r+"`. Internally:
- Opens the Stage 1 `stitch_positions.json`
- Reads per-tile Z-slabs for the assigned Z range
- Re-applies the dual-side fusion (`fuse_sides`) per tile
- Calls `mvstitch.fusion.fuse(..., transform_key="registered", fusion_func=weighted_average_fusion)`
- Writes the fused slab into the canvas zarr

Blend knobs:
- `--blend-y / --blend-x N` — blending widths (default 144 px ≈ half tile overlap)
- `--blend-z N` — **must be ≥ 1**, even if tiles don't overlap in Z (mvstitch's `get_blending_weights` divides by this; 0 → `ZeroDivisionError`)
- `--z-chunk N` — planes per blend chunk inside this task (default 64)
- `--sigma-frac F` and `--fusion-axis N` — passed through to `fuse_sides` (must match what you used in stage 07)

### `scripts/08_stitch_preflight_stage1.sh` + `scripts/08_stitch_preflight_stage2.sh`
**Run these before kicking off the full register/blend.** Preflight runs stage 1 + stage 2 over a 100-plane substack with the same code path as production, exercising every algorithmic component in ~30 minutes of GPU time. The preflight gate enforces:

- `converged == true`
- `max_residual_px < 5`
- `median pairwise quality > 0.2`  (catches phase-correlation no-ops)
- `n_pairs ≥ 10`
- `canvas_h` / `canvas_w` within **1000 px** of the layout-cache canvas (catches coordinate-frame errors that double the canvas extent)

If any gate fails, do NOT submit production. Inspect the diagnostics first — almost every interesting failure mode caught by this gate corresponds to a real bug in the algorithmic chain.

### `scripts/08_stitch_stage1_register.sh` + `scripts/08_stitch_stage2_blend.sh`
Production wrappers. Stage 1 is a single GPU job (~1 h on A30); Stage 2 is a SLURM array (`--array=0-7%4` by default, 4 concurrent tasks of 200 Z-planes each — adjust based on your canvas Z depth). Both wrappers use `--out-zarr` in `r+` mode to avoid the canvas being truncated by a parallel writer.

Edit each wrapper's header block: `SCRATCH`, `CZI_PATH`, `SCRIPT_DIR`, `ENV_DIR`, and the `module load CUDA/...` line.

---

## What to edit before your first run

Every `.sh` wrapper has a small header block at the top with hardcoded paths. Search-and-replace these three before running anything:

| Placeholder | Replace with |
|---|---|
| `/vast/scratch/users/kriel.j/KL018_lightsheet` | Your fast-scratch working directory |
| `/vast/projects/BCRL_Multi_Omics/scripts/lightsheet_pipeline` | Where you cloned this repo |
| `/vast/scratch/users/kriel.j/{lightsheet,mvstitch}_env` | Your conda env locations |

If your cluster doesn't use the Lmod `module load CUDA/12.1` pattern, swap that line for however you make CUDA 12.1 visible to Python (e.g. `source /opt/cuda-12.1/env.sh`).

The CZI filename `KL018_85_D7_CT2AvIII_Overview.czi` appears in several wrappers — replace with your acquisition's CZI.

---

## Inspecting outputs

```bash
python - <<'PY'
import zarr, numpy as np
z = zarr.open("/path/to/fused_direct.zarr", "r")
print("shape:", z.shape, "dtype:", z.dtype, "chunks:", z.chunks)
# Center patch around the mosaic centre, channel 0, mid-Z:
center = np.asarray(z[0, 0, z.shape[2]//2, z.shape[3]//2-200:z.shape[3]//2+200, z.shape[4]//2-200:z.shape[4]//2+200])
print("center patch:", center.min(), center.mean(), center.max())
PY
```

For Stage 1 diagnostics:

```bash
python - <<'PY'
import json, statistics
d = json.load(open("/path/to/scratch/stitch_diagnostics.json"))
qs = [p["quality"] for p in d["pairwise"]]
print("n_pairs:", len(qs), "median_q:", statistics.median(qs), "min/max:", min(qs), max(qs))
print("converged:", d["groupwise"]["converged"], "max_residual_px:", d["groupwise"]["max_residual_px"])
print("canvas (reg):", d["canvas_summary"]["from_registration_px"])
print("canvas (layout):", d["canvas_summary"]["from_layout_cache_px"])
PY
```

A healthy registration produces median quality > 0.3 (typically 0.5–0.7), max_residual < 1 px, and a registered canvas within ~100 px of the layout canvas.

---

## Common failures and how to diagnose them

| Symptom | Likely cause | Fix |
|---|---|---|
| Stage 1 SLURM job hits time limit | Default mvstitch pruning (`alternating_pattern`) left diagonal pairs in the graph; ~89+ pairs × 30 s each | Pass `--pre-reg-pruning-method keep_axis_aligned` (already the default in `08_stitch_preflight_stage1.sh`) |
| `stitch_diagnostics.json` shows `quality=0.0` on every pair but `converged=true, max_residual=0` | `_flatten_pairwise` is reading the wrong attribute, or `_scalar` is silently catching a `TypeError` on a `(t: 1)` xarray. In a healthy system, mvstitch 0.1.52 wraps quality as a 1-element DataArray | Use this repo's `08_stitch.py` — `_scalar` unwraps 1-element ndarrays via `.ravel()[0]` before `float()`; `_flatten_pairwise` prefers `pairwise_registration.metrics.qualities` |
| Stage 2 aborts: `registered canvas H=2024 W=2039 outside expected range` | `stitch_positions.json` is storing delta-only shifts, not canvas-absolute world positions | Stage 1 must compose layout-cache stage offset + raw correction affine. See `stage_register` in `08_stitch.py:442-510` |
| Stage 2 `ZeroDivisionError` in `weights.get_blending_weights` | `blending_widths["z"] = 0`. mvstitch divides `edt_support_spacing / blending_widths` per dim | Pass `--blend-z 1` (or any positive value — tiles don't overlap in Z in a XY mosaic, so it's geometrically equivalent) |
| Stage 2 `ValueError: could not broadcast input array from shape (X, ~1.78×H, ~1.78×W) into (X, H, W)` | The registered transform is being composed *on top of* the sim's intrinsic origin (which itself contains the stage offset). Net effect: every tile placed at 2× its stage position | Write only the **correction** (delta) to the `registered` transform_key, not the absolute position. The intrinsic origin gives the stage offset; mvstitch composes them. See `stage_blend` in `08_stitch.py:680-690` |
| Quality near zero with a thin Z-slab (`--z-slab-half 8 --reg-z-bin 2`) | Effective Z = 8 planes is too thin for phase correlation to lock | Use `--z-slab-half 16 --reg-z-bin 1` (32 effective Z) — preflight default |

---

## Repository layout

```
.
├── README.md                              ← this file
├── environment.yml                        ← conda env for mvstitch (stage 08)
└── scripts/
    ├── 07a_stage_czi.sh                   ← stage CZI to fast scratch
    ├── 07_direct_fuse.py                  ← stage 07 core (CZI → fused.zarr)
    ├── 07_direct_fuse.sh                  ← single-job SLURM wrapper
    ├── 07_direct_fuse_stage1_ncc.sh       ← 2-stage parallel: NCC + zarr init
    ├── 07_direct_fuse_stage2_parallel.sh  ← 2-stage parallel: 8-task blend array
    ├── 08_stitch.py                       ← stage 08 core (register + blend)
    ├── 08_stitch_preflight_stage1.sh      ← preflight register (~30 min)
    ├── 08_stitch_preflight_stage2.sh      ← preflight blend (~20 min)
    ├── 08_stitch_stage1_register.sh       ← production register (~1 h)
    └── 08_stitch_stage2_blend.sh          ← production blend (SLURM array)
```

---

## Citation / acknowledgement

If this pipeline contributes to your work, please cite:

- `multiview-stitcher` — Marvin Albert, https://github.com/multiview-stitcher/multiview-stitcher
- `aicspylibczi` — Allen Institute for Cell Science, https://github.com/AllenCellModeling/aicspylibczi

Pipeline developed in the Brain Cancer Research Lab (WEHI). Issues / questions: open a GitHub issue on this repo.

---

## License

MIT (or specify your preferred license here before publishing).
