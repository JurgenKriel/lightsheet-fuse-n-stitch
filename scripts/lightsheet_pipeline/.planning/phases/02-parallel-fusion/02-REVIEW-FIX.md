---
phase: 02-parallel-fusion
fixed_at: 2026-05-18T00:00:00Z
review_path: /vast/projects/BCRL_Multi_Omics/scripts/lightsheet_pipeline/.planning/phases/02-parallel-fusion/02-REVIEW.md
iteration: 1
findings_in_scope: 7
fixed: 7
skipped: 0
status: all_fixed
---

# Phase 02: Code Review Fix Report

**Fixed at:** 2026-05-18
**Source review:** `.planning/phases/02-parallel-fusion/02-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 7 (2 Critical + 5 Warning; Info findings excluded per fix_scope=critical_warning)
- Fixed: 7
- Skipped: 0

## Fixed Issues

### CR-01: Remove `--skip-refine false` from Stage 1 NCC re-run invocation

**Files modified:** `07_direct_fuse_stage1_ncc.sh`
**Commit:** 1a743c7
**Applied fix:** Removed `--skip-refine false` from the threshold-mismatch re-run invocation (lines 60-71). `--skip-refine` is `action='store_true'` and takes no argument — passing `false` caused argparse to exit code 2, leaving the stale ncc_scores.json in place. Also changed `--out "$OUT_ZARR"` to `--out "${SCRATCH}/ncc_probe_rerun.zarr"` in this block (combined with WR-02 intent for the re-run branch).

---

### CR-02: Add `set -euo pipefail` as first executable line in all three .sh scripts

**Files modified:** `07_direct_fuse_stage1_ncc.sh`, `07_direct_fuse_stage2_parallel.sh`, `07_direct_fuse.sh`
**Commit:** 72f628e
**Applied fix:** Added `set -euo pipefail` as the first non-comment, non-SBATCH executable line in all three scripts. Inserted immediately before the first variable assignment in each file so SBATCH directives are unaffected.

---

### WR-01: Canvas dimension validation against pre-allocated zarr shape

**Files modified:** `07_direct_fuse.py`
**Commit:** 039bbf8
**Applied fix:** Added an explicit dimension guard immediately after opening the zarr in `r+` mode (after line 619). Reads `expected_h, expected_w` from `out_z.shape[3]` and `out_z.shape[4]`, then raises a descriptive `ValueError` if `refined_h` or `refined_w` differs. This prevents a cryptic broadcast error mid-slab Z-chunk loop.

---

### WR-02: Use throwaway output path for NCC probe run

**Files modified:** `07_direct_fuse_stage1_ncc.sh`
**Commit:** 4e78ad7
**Applied fix:** Changed `--out "$OUT_ZARR"` to `--out "${SCRATCH}/ncc_probe.zarr"` in the initial probe run branch (when ncc_scores.json is absent, z=728:729). This prevents the initial NCC run from writing to `fused_direct.zarr` with `mode=w`, which would truncate it to shape `(1,2,1,H,W)` and lose the pre-allocated full shape if Step 2 subsequently fails. The re-run branch (threshold mismatch) was handled in the CR-01 commit using `ncc_probe_rerun.zarr`.

---

### WR-03: Validate loaded tile count matches layout n_tiles

**Files modified:** `07_direct_fuse.py`
**Commit:** 885e1c0
**Applied fix:** Added a count check after building `refined_pos` from the JSON. Raises `ValueError` with a descriptive message if `len(refined_pos) != n_tiles`. This converts a cryptic mid-loop `KeyError` into an immediate, actionable error at position-load time.

---

### WR-04: Replace `assert` with explicit if/raise for z_start/z_end validation

**Files modified:** `07_direct_fuse.py`
**Commit:** b82102b
**Applied fix:** Replaced `assert z_start < z_end, ...` with `if z_start >= z_end: raise ValueError(...)`. Assert statements are compiled out with `python -O`; an invalid combination would silently produce n_chunks=0 and an empty zarr. The explicit check fires unconditionally.

---

### WR-05: Replace `json.load(open(...))` with `with` statement in Stage 1 inline Python

**Files modified:** `07_direct_fuse_stage1_ncc.sh`
**Commit:** 4b5b4fe
**Applied fix:** Replaced `man = json.load(open(manifest_path))` with `with open(manifest_path) as f: man = json.load(f)` in the inline Python heredoc (zarr pre-allocation step). The file descriptor is now explicitly closed after reading rather than relying on CPython garbage collection.

---

## Skipped Issues

None — all in-scope findings were successfully fixed.

---

_Fixed: 2026-05-18_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
