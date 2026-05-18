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

### Phase 3: OME-TIFF Export
**Goal**: Stitched zarr is exported as a pyramidal OME-TIFF that opens correctly in QuPath, ImageJ, and Napari with accurate pixel-size metadata
**Depends on**: Phase 2
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
| 3. OME-TIFF Export | 0/TBD | Not started | - |
