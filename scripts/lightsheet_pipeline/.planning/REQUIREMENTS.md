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
| EXPORT-01 | Phase 3: OME-TIFF Export | Pending |
| EXPORT-02 | Phase 3: OME-TIFF Export | Pending |
| EXPORT-03 | Phase 3: OME-TIFF Export | Pending |

**Coverage: 13/13 v1 requirements mapped. No orphans.**
