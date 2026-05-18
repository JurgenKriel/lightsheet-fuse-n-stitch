# Lightsheet Fusion Pipeline — Parameter Refinement & Full-Stack Run

## What This Is

A focused enhancement to the KL018 Zeiss lightsheet fusion pipeline to fix
multi-issue fusion artifacts (tile seams, illumination gradients, misalignment),
validate corrected parameters interactively on a 100-slice substack, run the
validated config on the full 1.3 TB dataset, and export the final volume as a
pyramidal OME-TIFF for multi-viewer compatibility.

## Core Value

A single pyramidal OME-TIFF of the KL018 brain volume with no visible fusion
artifacts — usable in QuPath, Napari, Imaris, and ImageJ without further
processing.

## Context

**Codebase:** `scripts/lightsheet_pipeline/` — Python/SLURM pipeline for Zeiss CZI
light sheet data.

**Key script:** `07_direct_fuse.py` — Direct CZI → fused+stitched zarr in one job.
Reads KL018 CZI (1.3 TB), fuses 30 mosaic tiles across 1557 Z-planes (2 channels,
dual-side illumination), uses NCC+MST position refinement.

**Dataset:** KL018_85_D7_CT2AvIII_Overview.czi
- 30 mosaic tiles, each 1920×1920×1557 px
- 2 illumination sides (I dimension), Gaussian-blend fused
- 2 channels, 1 timepoint
- Canvas ~10079×8448 px

**Known problems (all three categories):**
1. Visible seams — taper too narrow or position refinement diverging
2. Illumination gradients — Gaussian blend sigma / axis mismatch
3. Tile misalignment / ghosting — NCC threshold or MST quality failing

**Working paths:**
- Raw CZI: `/vast/scratch/users/kriel.j/KL018_lightsheet/KL018_85_D7_CT2AvIII_Overview.czi`
- Layout cache: `/vast/scratch/users/kriel.j/KL018_lightsheet/czi_layout_cache.json`
- Existing (broken) output: `/vast/scratch/users/kriel.j/KL018_lightsheet/fused_direct.zarr`

## Requirements

### Validated

- ✓ CZI reading via aicspylibczi — working
- ✓ GPU-accelerated Gaussian fusion — working
- ✓ NCC+MST tile position refinement — working (but needs tuning)
- ✓ Z-chunk streaming (stays within GPU/RAM) — working
- ✓ cosine-taper blending — implemented but undertapered
- ✓ Layout cache (`czi_layout_cache.json`) — working

### Active

- [ ] SUBSTACK-01: `07_direct_fuse.py` supports `--z-start`/`--z-end` for substack extraction
- [ ] SUBSTACK-02: 100-slice substack (z=728:828) produced from raw CZI in <30 min on GPU node
- [ ] NOTEBOOK-01: Jupyter notebook runs fusion on substack with adjustable parameters (no code changes needed)
- [ ] NOTEBOOK-02: Notebook visualises fusion quality: mid-Z tile mosaic, seam map, illumination uniformity
- [ ] NOTEBOOK-03: Notebook produces parameter recommendation dict ready to paste into SLURM script
- [ ] FULLSTACK-01: Full 1557-plane run executed with validated parameters via `07_direct_fuse.sh`
- [ ] EXPORT-01: Final stitched volume exported as pyramidal OME-TIFF (tifffile BigTIFF, ≥3 resolution levels)
- [ ] EXPORT-02: OME-TIFF opens correctly in QuPath / ImageJ / Napari

### Out of Scope

- Deskewing — KL018 is a Z.1 orthogonal geometry; `03_deskew_deconv.py` already skips deskew
- Deconvolution — separate concern; pipeline step 03 handles it independently
- Multi-timepoint — KL018 is T=1
- Bayesian/deep-learning registration — out of scope for this milestone

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Substack from raw CZI, not fused_direct.zarr | Tests the actual fusion+stitch code path | — Pending |
| Middle 100 slices (z=728:828) | Most tissue content, representative of seam quality | — Pending |
| Pyramid: zarr → OME-TIFF via tifffile | tifffile is already in environment.yml | — Pending |
| Parameters to sweep: sigma_frac, taper_px, NCC threshold, fusion axis | Cover all three identified artifact classes | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition:**
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After milestone:**
1. Full review of all sections
2. Core Value check — still right priority?
3. Audit Out of Scope — reasons still valid?

---
*Last updated: 2026-05-18 after initialization*
