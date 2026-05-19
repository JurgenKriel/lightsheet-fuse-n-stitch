# Requirements — Lightsheet Fusion Pipeline Enhancement

## v1 Requirements

### Substack Extraction

- [ ] **SUBSTACK-01**: User can run `07_direct_fuse.py --z-start 728 --z-end 828` to produce a 100-slice substack without modifying any other parameters
- [ ] **SUBSTACK-02**: Substack run completes in <30 min on a GPU SLURM node and produces a valid zarr at a configurable output path

### Interactive Notebook

- [ ] **NOTEBOOK-01**: User can open `08_fusion_dev.ipynb` and run all cells to produce a fused+stitched substack with default parameters
- [ ] **NOTEBOOK-02**: Notebook exposes interactive widgets (or parameter dicts) for: `sigma_frac`, `taper_px`, `fusion_axis`, `ncc_threshold`, `skip_refine`
- [ ] **NOTEBOOK-03**: Notebook renders a mid-Z maximum-intensity mosaic showing all 30 tiles stitched
- [ ] **NOTEBOOK-04**: Notebook renders a seam-quality heatmap (per-tile NCC scores visualised on canvas)
- [ ] **NOTEBOOK-05**: Notebook renders per-tile illumination uniformity (mean intensity profile across X for each tile)
- [ ] **NOTEBOOK-06**: Notebook outputs a `best_params` dict the user can copy directly into the SLURM submission script

### Full-Stack Run

- [ ] **FULLSTACK-01**: `07_direct_fuse.sh` is updated to use validated parameters from notebook
- [ ] **FULLSTACK-02**: Full 1557-plane run completes successfully on SLURM and produces `fused_direct.zarr` with shape `(1, 2, 1557, H, W)`

### Stitching Rework (Phase 2.5)

- [x] **STITCH-01**: A new module `08_stitch.py` (or refactored region inside `07_direct_fuse.py`) implements a globally-optimised tile registration step that solves for tile positions jointly across all pairwise overlaps (least-squares / global solver), replacing the current MST-propagated NCC shifts (2026-05-19 — 08_stitch.py stage_register uses multiview_stitcher.registration.register with groupwise_resolution_method="global_optimization")
- [x] **STITCH-02**: Stitcher uses feather/sigmoidal (or distance-transform-based) blending weights in 3D overlap regions and produces an output that is mathematically equivalent to a globally normalised weighted average — replacing the current 2D cosine-taper accumulator (2026-05-19 — 08_stitch.py stage_blend calls fusion.fuse with fusion_func=fusion.weighted_average_fusion + blending_widths={"z":0,"y":144,"x":144})
- [x] **STITCH-03**: Per-pair NCC quality scores, residual translation errors, and global-solver convergence diagnostics are written to `stitch_diagnostics.json` for inspection in `08_fusion_dev.ipynb` (2026-05-19 — 08_stitch.py stage_register writes stitch_diagnostics.json with library_version, n_tiles, z_slab_used, voxel_size_um, pairwise [{i,j,quality,shift_um,shift_px,residual_px,accepted}], groupwise {method,converged,rms/max_residual_px,n_variables,n_constraints,solver_iterations} — RESEARCH-defined schema)
- [ ] **STITCH-04**: `08_fusion_dev.ipynb` is extended with a `Phase 2.5` cell that renders the new diagnostics — per-pair residual heatmap, mid-Z seam-quality (NCC ≥ 0.85 target), per-tile illumination uniformity (max/min ≤ 1.15 target)
- [ ] **STITCH-05**: SLURM workflow runs the new stitcher at full scale on KL018 — either by extending the existing Stage 1/Stage 2 scripts or via a clean `07_stitch_stage1_register.sh` + `07_stitch_stage2_blend.sh` pair — and produces `fused_direct.zarr` with shape `(1, 2, 1557, H, W)` and no visible mid-volume seams
- [ ] **STITCH-06**: Dual-side illumination fusion code (`fuse_sides`, `_gaussian_ramp`, `read_tile_zchunk`) is untouched by this phase — `git diff` for the phase MUST NOT modify those functions
- [x] **STITCH-07**: A `.planning/phases/02.5-zarrstitcher-rework/02.5-RESEARCH.md` documents which existing tool was used (PetaKit5D MATLAB direct, Python re-implementation of the same algorithm, or alternative such as BigStitcher/m-stitch) and why, with citations to Ruan et al. *Nature Methods* 2024 and the PetaKit5D source (2026-05-19 — multiview-stitcher 0.1.52 chosen; env installed at /vast/scratch/users/kriel.j/mvstitch_env)

### OME-TIFF Export

- [ ] **EXPORT-01**: `08_export_ometiff.py` converts the stitched zarr to a pyramidal OME-TIFF (BigTIFF, ≥3 resolution levels, uint16)
- [ ] **EXPORT-02**: OME-TIFF includes correct OME-XML metadata: pixel sizes (µm from `voxel_size_um`), channel names, dimension order TCZYX
- [ ] **EXPORT-03**: Output OME-TIFF opens without errors in QuPath and ImageJ (pyramidal levels visible in image overview)

## v2 Requirements (deferred)

- Full deconvolution integration into substack notebook
- GPU-accelerated OME-TIFF pyramid write (tiled parallel write)
- Automated parameter search (grid or Bayesian optimisation of NCC threshold / taper)
- Multi-sample batch runner (extend beyond KL018)

## Out of Scope

- Deskewing — KL018 Z.1 orthogonal geometry, no oblique sheet
- Deconvolution — separate pipeline step (03), not a fusion artifact
- Multi-timepoint support — KL018 is T=1
- Deep-learning-based registration — out of scope for this milestone
- stornext copy automation — handled separately by `06_output.sh`

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| SUBSTACK-01 | Phase 1: Substack & Notebook | Pending |
| SUBSTACK-02 | Phase 1: Substack & Notebook | Pending |
| NOTEBOOK-01 | Phase 1: Substack & Notebook | Pending |
| NOTEBOOK-02 | Phase 1: Substack & Notebook | Pending |
| NOTEBOOK-03 | Phase 1: Substack & Notebook | Pending |
| NOTEBOOK-04 | Phase 1: Substack & Notebook | Pending |
| NOTEBOOK-05 | Phase 1: Substack & Notebook | Pending |
| NOTEBOOK-06 | Phase 1: Substack & Notebook | Pending |
| FULLSTACK-01 | Phase 2: Full-Stack Fusion | Pending |
| FULLSTACK-02 | Phase 2: Full-Stack Fusion | Pending |
| STITCH-01 | Phase 2.5: ZarrStitcher Stitching Rework | Done (2026-05-19) |
| STITCH-02 | Phase 2.5: ZarrStitcher Stitching Rework | Done (2026-05-19) |
| STITCH-03 | Phase 2.5: ZarrStitcher Stitching Rework | Done (2026-05-19) |
| STITCH-04 | Phase 2.5: ZarrStitcher Stitching Rework | Pending |
| STITCH-05 | Phase 2.5: ZarrStitcher Stitching Rework | Pending |
| STITCH-06 | Phase 2.5: ZarrStitcher Stitching Rework | Pending |
| STITCH-07 | Phase 2.5: ZarrStitcher Stitching Rework | Done (2026-05-19) |
| EXPORT-01 | Phase 3: OME-TIFF Export | Pending |
| EXPORT-02 | Phase 3: OME-TIFF Export | Pending |
| EXPORT-03 | Phase 3: OME-TIFF Export | Pending |

**Coverage: 20/20 v1 requirements mapped. No orphans.**
