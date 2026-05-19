# Roadmap: Lightsheet Fusion Pipeline Enhancement

## Overview

Three phases take KL018 from a broken 1.3 TB CZI to a clean pyramidal OME-TIFF
ready for QuPath, Napari, and ImageJ. Phase 1 adds substack extraction to the
fusion script and builds an interactive notebook for rapid parameter sweeping on
100 representative slices. Phase 2 runs the validated configuration on the full
1557-plane volume. Phase 3 converts the stitched zarr to a pyramidal OME-TIFF
with correct metadata and verifies viewer compatibility.

## Phases

- [ ] **Phase 1: Substack & Notebook** - Add `--z-start`/`--z-end` to `07_direct_fuse.py` and build `08_fusion_dev.ipynb` for interactive parameter tuning
- [ ] **Phase 2: Full-Stack Fusion** - Run validated parameters on the complete 1557-plane KL018 dataset via SLURM
- [ ] **Phase 2.5: ZarrStitcher Stitching Rework** - Replace ad-hoc NCC+MST + cosine-taper stitching with a PetaKit5D-ZarrStitcher-equivalent globally-optimised stitcher to eliminate residual tile seams (keeps current dual-side fusion as-is)
- [ ] **Phase 3: OME-TIFF Export** - Convert fused zarr to pyramidal OME-TIFF with correct metadata and verify viewer compatibility

## Phase Details

### Phase 1: Substack & Notebook
**Goal**: User can produce and visually evaluate a 100-slice fused substack and arrive at a validated parameter set ready to paste into the SLURM script
**Depends on**: Nothing (first phase)
**Requirements**: SUBSTACK-01, SUBSTACK-02, NOTEBOOK-01, NOTEBOOK-02, NOTEBOOK-03, NOTEBOOK-04, NOTEBOOK-05, NOTEBOOK-06
**Plans**: 3 plans
**Success Criteria** (what must be TRUE):
  1. User can run `07_direct_fuse.py --z-start 728 --z-end 828` and receive a valid zarr output without modifying any other flags
  2. The substack zarr is produced in under 30 minutes on a GPU SLURM node
  3. User can open `08_fusion_dev.ipynb`, run all cells, and see a mid-Z maximum-intensity mosaic of all 30 stitched tiles
  4. Notebook renders a seam-quality heatmap and a per-tile illumination uniformity plot that make fusion artifacts visible
  5. Notebook outputs a `best_params` dict the user can copy directly into `07_direct_fuse.sh` without any editing

Plans:
- [x] 01-01-PLAN.md — Extend 07_direct_fuse.py: add 5 CLI flags, fix z-loop indexing, write ncc_scores.json (2026-05-18)
- [x] 01-02-PLAN.md — Update 07_direct_fuse_README.md and 07_direct_fuse.sh with new flag docs (2026-05-18)
- [ ] 01-03-PLAN.md — Create 08_fusion_dev.ipynb with 6-cell parameter sweep workflow

### Phase 2: Full-Stack Fusion
**Goal**: Full 1557-plane KL018 volume is fused with validated parameters and written as a complete `fused_direct.zarr` as fast as possible by exploiting all available GPU resources via a two-stage SLURM approach
**Depends on**: Phase 1
**Requirements**: FULLSTACK-01, FULLSTACK-02
**Success Criteria** (what must be TRUE):
  1. `07_direct_fuse.sh` contains the parameter values confirmed during Phase 1 notebook evaluation
  2. SLURM job completes without error and `fused_direct.zarr` exists with shape `(1, 2, 1557, H, W)`
  3. Spot-check of mid-volume Z-planes shows no visible tile seams, illumination gradients, or ghosting artifacts
**Plans**: 3 plans

Plans:
- [x] 02-01-PLAN.md — Extend 07_direct_fuse.py: add --load-positions and --zarr-mode flags, conditional write index (2026-05-18)
- [x] 02-02-PLAN.md — Create 07_direct_fuse_stage1_ncc.sh and 07_direct_fuse_stage2_parallel.sh (8-task array) (2026-05-18)
- [x] 02-03-PLAN.md — Update 07_direct_fuse.sh with validated params, workers=12, parallel workflow reference (2026-05-18)

### Phase 2.5: ZarrStitcher Stitching Rework
**Goal**: After Phase 2 ran end-to-end, mid-volume Z-planes still show visible tile seams. Replace the ad-hoc stitcher (NCC + MST + cosine-taper accumulator in `07_direct_fuse.py`) with a globally-optimised stitcher modelled on PetaKit5D `ZarrStitcher` (Ruan et al., *Nature Methods* 2024) so seams are eliminated while keeping the validated dual-side fusion (`fuse_sides`) untouched. The stitcher must read fused per-tile Z-slabs and write a single seamless `fused_direct.zarr`.
**Depends on**: Phase 2
**Requirements**: STITCH-01, STITCH-02, STITCH-03, STITCH-04, STITCH-05, STITCH-06, STITCH-07
**Success Criteria** (what must be TRUE):
  1. Stitcher uses a globally-consistent registration (least-squares / global optimisation over a pairwise constraint graph), not a single-spanning-tree propagation, so per-tile residual translation errors are minimised across the whole canvas
  2. Blending uses feather/sigmoidal (or equivalent globally-normalised) weights in 3D overlap regions, not 2D cosine-taper accumulation, eliminating accumulated brightness ramps at multi-tile junctions
  3. A mid-volume Z-plane (Z=778) at the 8×4 (or full) tile junction shows no visible seam in fused_direct.zarr — measured by `08_fusion_dev.ipynb` seam-heatmap NCC ≥ 0.85 at all neighbour pairs
  4. Per-tile illumination-uniformity plot from `08_fusion_dev.ipynb` shows max-min intensity ratio ≤ 1.15 across each tile (no edge ramps inherited from blending weights)
  5. Stitched output preserves voxel size, dimension order TCZYX, and shape `(1, 2, 1557, H, W)` consistent with downstream Phase 3 OME-TIFF export
  6. Dual-side fusion code path (`fuse_sides`, `_gaussian_ramp`) is unchanged — phase only replaces stitching code (position refinement + canvas accumulation + write)
  7. Existing SLURM two-stage workflow (Stage 1 NCC, Stage 2 parallel slab-write) still works for the new stitcher or is replaced by an explicitly-defined two-stage equivalent
**Plans**: 5 plans

Plans:
- [x] 02.5-01-PLAN.md — Create mvstitch_env conda env + environment_mvstitch.yml (STITCH-07) (2026-05-19)
- [x] 02.5-02-PLAN.md — 08_stitch.py Stage 1 register: global LSQ + stitch_positions.json + stitch_diagnostics.json (STITCH-01, 03, 06, 07) (2026-05-19)
- [x] 02.5-03-PLAN.md — 08_stitch.py Stage 2 blend: weighted_average_fusion + blending_widths, r+ zarr writes at absolute Z indices (STITCH-02, 05, 06) (2026-05-19)
- [ ] 02.5-04-PLAN.md — 08_stitch_stage1_register.sh + 08_stitch_stage2_blend.sh SLURM scripts (STITCH-05)
- [x] 02.5-05-PLAN.md — 08_fusion_dev.ipynb Phase 2.5 diagnostics cell: residual heatmap, NCC ≥ 0.85, illumination ≤ 1.15 (STITCH-04) (2026-05-19)

### Phase 3: OME-TIFF Export
**Goal**: Stitched zarr is exported as a pyramidal OME-TIFF that opens correctly in QuPath, ImageJ, and Napari with accurate pixel-size metadata
**Depends on**: Phase 2.5
**Requirements**: EXPORT-01, EXPORT-02, EXPORT-03
**Success Criteria** (what must be TRUE):
  1. `08_export_ometiff.py` produces a BigTIFF file with at least 3 pyramid resolution levels and uint16 pixel type
  2. OME-XML embedded in the file contains correct pixel sizes (µm), channel names, and dimension order TCZYX
  3. File opens without errors in QuPath and ImageJ, with all pyramid levels visible in the image overview panel
**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Substack & Notebook | 2/3 | In progress | - |
| 2. Full-Stack Fusion | 3/3 | Complete | 2026-05-18 |
| 2.5. ZarrStitcher Stitching Rework | 4/5 | In progress | - |
| 3. OME-TIFF Export | 0/TBD | Not started | - |
