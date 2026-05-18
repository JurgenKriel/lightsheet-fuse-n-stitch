---
phase: 02-parallel-fusion
verified: 2026-05-18T08:00:00Z
status: human_needed
score: 7/9 must-haves verified
overrides_applied: 0
human_verification:
  - test: "Submit the two-stage SLURM workflow and confirm fused_direct.zarr is produced with shape (1, 2, 1557, H, W)"
    expected: "Both jobs complete with exit code 0; zarr exists at /vast/scratch/users/kriel.j/KL018_lightsheet/fused_direct.zarr with shape (1, 2, 1557, 8585, 10095)"
    why_human: "Cannot run SLURM jobs in verification — no GPU or HPC cluster available"
  - test: "Visually inspect mid-volume Z-planes (e.g., Z=700-800) for seam artifacts, illumination gradients, and ghosting"
    expected: "No visible tile seams, illumination gradients, or ghosting artifacts in mid-volume planes"
    why_human: "Requires running the full pipeline and visual inspection of rendered zarr slices"
---

# Phase 2: Full-Stack Fusion — Verification Report

**Phase Goal:** Full 1557-plane KL018 volume is fused with validated parameters and written as a complete `fused_direct.zarr` as fast as possible by exploiting all available GPU resources via a two-stage SLURM approach
**Verified:** 2026-05-18
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `python 07_direct_fuse.py --help` shows `--load-positions` and `--zarr-mode` | VERIFIED | Both flags appear in `--help` output (usage + description lines); `--help` exits 0 |
| 2 | Running with `--zarr-mode r+` opens a pre-existing zarr without truncating | VERIFIED | `zarr.open(str(out_path), mode="r+")` in `else` branch at line 619; no shape/chunk args passed — cannot create/truncate |
| 3 | Running with `--load-positions /path/ncc_scores.json` skips `refine_tile_positions()` entirely | VERIFIED | `if args.load_positions:` branch at line 575 reads JSON and populates `refined_pos` directly; `refine_tile_positions()` call is in the `else` branch only |
| 4 | When `zarr_mode=r+`, write index is `z0_abs`; when `mode=w` it is `z0_loc` | VERIFIED | `z_write = z0_abs if args.zarr_mode == "r+" else z0_loc` at line 681, confirmed by grep |
| 5 | `07_direct_fuse_stage1_ncc.sh` exists with `#SBATCH --gres=gpu:A30:1` and validates ncc threshold | VERIFIED | File exists, executable, `bash -n` passes, `--gres=gpu:A30:1` confirmed, threshold validation logic at lines 55-75 confirmed |
| 6 | `07_direct_fuse_stage2_parallel.sh` exists with `#SBATCH --array=1-8`, correct Z_STARTS/Z_ENDS, `--load-positions`, `--zarr-mode r+`, `--workers 12` | VERIFIED | All elements confirmed present: `--array=1-8`, `Z_STARTS=(0 256 448 640 832 1024 1216 1408)`, `Z_ENDS=(256 448 640 832 1024 1216 1408 1557)`, `--load-positions "$NCC_JSON"`, `--zarr-mode r+`, `--workers 12` |
| 7 | `07_direct_fuse.sh` uses `--workers 12` and references the parallel workflow scripts | VERIFIED | Active python call has `--workers 12`; comment block at lines 34-37 references `07_direct_fuse_stage1_ncc.sh` and `07_direct_fuse_stage2_parallel.sh` with ~50 min estimate |
| 8 | SLURM job completes without error and `fused_direct.zarr` exists with shape `(1, 2, 1557, H, W)` | UNCERTAIN (human needed) | Cannot verify without executing on HPC — no GPU cluster available in verification context |
| 9 | Spot-check of mid-volume Z-planes shows no visible tile seams, illumination gradients, or ghosting | UNCERTAIN (human needed) | Requires visual inspection of rendered output; programmatically unverifiable |

**Score:** 7/9 truths verified (2 require human — SLURM runtime)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/lightsheet_pipeline/07_direct_fuse.py` | Extended fusion script with `--load-positions`, `--zarr-mode` flags and absolute zarr write index | VERIFIED | File exists; both flags in `parse_args()`; `load_positions` appears 4 times (parse, if-branch, open, print); `zarr_mode` appears 3 times (parse, if==w, z_write conditional); `z_write` conditional correct; syntax check passes |
| `scripts/lightsheet_pipeline/07_direct_fuse_stage1_ncc.sh` | SLURM Stage 1: NCC validation + zarr pre-allocation | VERIFIED | File exists, executable (`chmod +x` applied), `bash -n` passes, `gpu:A30:1`, zarr init with shape `(1,2,1557,H,W)`, "Stage 1 complete" message, submission hint present |
| `scripts/lightsheet_pipeline/07_direct_fuse_stage2_parallel.sh` | SLURM Stage 2: 8-task array, one A30 per Z-slab | VERIFIED | File exists, executable, `bash -n` passes, `--array=1-8`, correct Z partition arrays, `--load-positions`, `--zarr-mode r+`, `--workers 12`, `CUDA_VISIBLE_DEVICES=0`, zarr pre-check with exit 1 on failure |
| `scripts/lightsheet_pipeline/07_direct_fuse.sh` | Updated single-job fallback with validated params and parallel workflow reference | VERIFIED | File exists, `bash -n` passes, `--workers 12` in active python call, validated params all present in python call, parallel workflow reference comment with stage1/stage2 script names and ~50 min callout |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `parse_args()` | `main() zarr open block` | `args.zarr_mode in ['w', 'r+']` | VERIFIED | `if args.zarr_mode == "w":` at line 608; `else: # r+` at line 617 |
| `main() position loading block` | `refine_tile_positions()` | `if args.load_positions: skip NCC else run NCC` | VERIFIED | `if args.load_positions:` at line 575; `else:` branch calls `refine_tile_positions()` at line 586 |
| `main() zarr write` | `out_z[t, c, ...]` | `z0_abs when zarr_mode=r+, z0_loc when zarr_mode=w` | VERIFIED | `z_write = z0_abs if args.zarr_mode == "r+" else z0_loc` at line 681; write uses `z_write` |
| `07_direct_fuse_stage1_ncc.sh` | `fused_direct.zarr` | `python3 inline zarr.open(mode='w', shape=(1,2,1557,8585,10095))` | VERIFIED | Inline Python block at lines 98-127 creates zarr with shape `(1, 2, 1557, H, W)` |
| `07_direct_fuse_stage2_parallel.sh` | `07_direct_fuse.py` | `--load-positions $NCC_JSON --zarr-mode r+ --z-start $Z_START --z-end $Z_END` | VERIFIED | Python call at lines 99-111 includes all required flags |
| `SLURM_ARRAY_TASK_ID` | `Z_START / Z_END` | `IDX=$((SLURM_ARRAY_TASK_ID - 1)); Z_START=${Z_STARTS[$IDX]}` | VERIFIED | Array lookup at lines 46-48 confirmed |

### Data-Flow Trace (Level 4)

Not applicable — phase delivers SLURM scripts and a CLI tool, not data-rendering components. The data flow at code level is fully traced through key links above. Runtime data flow (actual zarr output with pixel data) requires human verification.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `--help` exits 0 with both flags | `python 07_direct_fuse.py --help` | Exits 0; both `--load-positions` and `--zarr-mode` appear in usage and description | PASS |
| Python syntax clean | `python -c "import ast; ast.parse(open('07_direct_fuse.py').read())"` | `syntax OK` | PASS |
| Stage 1 bash syntax | `bash -n 07_direct_fuse_stage1_ncc.sh` | Exit 0, no output | PASS |
| Stage 2 bash syntax | `bash -n 07_direct_fuse_stage2_parallel.sh` | Exit 0, no output | PASS |
| Single-job script bash syntax | `bash -n 07_direct_fuse.sh` | Exit 0, no output | PASS |
| Z-partition boundaries multiples of 64 | Python boundary check | All 8 start/end boundaries (0,256,...,1408) confirmed multiples of 64 | PASS |
| Stage 2 array task → Z-slab mapping | `grep 'IDX=\|Z_START=\|Z_END='` | Correct bash array index pattern at lines 46-48 | PASS |
| Full-stack fusion executes on SLURM | `sbatch 07_direct_fuse_stage1_ncc.sh` | Cannot run without GPU/HPC | SKIP |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| FULLSTACK-01 | 02-02-PLAN.md, 02-03-PLAN.md | `07_direct_fuse.sh` is updated to use validated parameters from notebook | SATISFIED | `--sigma-frac 0.9`, `--taper-px 288`, `--ncc-threshold 0.5`, `--fusion-axis 2` all in active python call; `--workers 12`; parallel workflow reference comment present |
| FULLSTACK-02 | 02-01-PLAN.md, 02-02-PLAN.md | Full 1557-plane run completes successfully on SLURM and produces `fused_direct.zarr` with shape `(1, 2, 1557, H, W)` | NEEDS HUMAN | Code infrastructure is complete and correctly structured; runtime completion requires SLURM execution on GPU cluster |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `07_direct_fuse_stage1_ncc.sh` | 71 | `--skip-refine false` passed to argparse `store_true` flag — exits with code 2 | WARNING (CR-01, known) | Affects only the threshold-mismatch re-run branch (lines 58-75). The primary NCC-absent branch (lines 76-89) is correct. Normal workflow (ncc_scores.json present with threshold=0.5) takes the `else` path at line 73 — unaffected. Bug noted in 02-REVIEW.md as CR-01; code-review workflow will fix. |
| `07_direct_fuse_stage1_ncc.sh`, `07_direct_fuse_stage2_parallel.sh`, `07_direct_fuse.sh` | all | No `set -e` / `set -euo pipefail` | WARNING (CR-02, known) | Python command failures not automatically fatal. Stage 2 has explicit `$?` check for fusion call. Stage 1 missing it for NCC re-run (compounded by CR-01). Noted in 02-REVIEW.md as CR-02; code-review workflow will fix. |

**Note on CR-01 and CR-02 scope:** Per the verification instructions, these known issues from the code review are noted here but do not alone cause `gaps_found` status. The code-review workflow is the appropriate remediation path. The normal execution path (ncc_scores.json present with correct threshold=0.5 from Phase 1) bypasses the CR-01 bug entirely.

### Human Verification Required

#### 1. Full-Stack SLURM Execution

**Test:** Submit the two-stage workflow:
```bash
JOB1=$(sbatch --parsable scripts/lightsheet_pipeline/07_direct_fuse_stage1_ncc.sh)
sbatch --dependency=afterok:$JOB1 scripts/lightsheet_pipeline/07_direct_fuse_stage2_parallel.sh
```
Monitor with `squeue -u $USER`. After completion, verify:
```python
import zarr
z = zarr.open("/vast/scratch/users/kriel.j/KL018_lightsheet/fused_direct.zarr", "r")
print(z.shape)  # expected: (1, 2, 1557, 8585, 10095)
```

**Expected:** Both jobs exit 0; zarr exists with shape `(1, 2, 1557, 8585, 10095)`; Stage 2 job logs show "Task N: fusion complete for Z=..." for all 8 tasks

**Why human:** Cannot run SLURM jobs in verification — no GPU or HPC cluster available

#### 2. Visual Artifact Inspection

**Test:** Load mid-volume Z-planes (e.g., Z=700-800) from `fused_direct.zarr` into napari or the dev notebook and inspect for seam quality.

**Expected:** No visible tile seams, illumination gradients, or ghosting artifacts. The seam heatmap from the Phase 1 notebook sweep should serve as a reference for what acceptable stitching looks like.

**Why human:** Artifact quality is a subjective visual assessment that cannot be reduced to a programmatic pass/fail check.

---

## Summary

**Phase 2 code infrastructure is complete and correctly structured.** All 7 programmatically verifiable must-haves pass at all three levels (exists, substantive, wired). The two-stage SLURM approach is correctly implemented:

- `07_direct_fuse.py` has working `--load-positions` and `--zarr-mode` flags with correct conditional logic for NCC skipping, zarr open mode, and absolute Z-index write.
- `07_direct_fuse_stage1_ncc.sh` correctly pre-allocates the full-shape zarr with validated NCC threshold checking.
- `07_direct_fuse_stage2_parallel.sh` correctly maps 8 SLURM array tasks to disjoint Z-slabs via chunk-aligned boundaries and calls the Python driver with all required flags.
- `07_direct_fuse.sh` contains the validated parameters and a parallel workflow reference.

**Two known issues from 02-REVIEW.md** (CR-01: `--skip-refine false` in Stage 1 threshold-mismatch branch; CR-02: no `set -e`) are noted but scoped to the code-review remediation workflow per the verification instructions. The normal production path (ncc_scores.json from Phase 1 with threshold=0.5) bypasses CR-01.

**Two ROADMAP success criteria require human verification** (SC2: SLURM job completes + zarr produced; SC3: visual artifact quality). These cannot be verified without a GPU cluster.

---

_Verified: 2026-05-18_
_Verifier: Claude (gsd-verifier)_
