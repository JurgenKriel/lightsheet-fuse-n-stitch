---
phase: 02-parallel-fusion
plan: "03"
subsystem: lightsheet-pipeline
tags:
  - slurm
  - fusion
  - parameters
  - documentation

dependency_graph:
  requires:
    - "02-02-PLAN.md"
  provides:
    - "Updated single-job fallback script with validated params and parallel workflow reference"
  affects:
    - "scripts/lightsheet_pipeline/07_direct_fuse.sh"

tech_stack:
  added: []
  patterns:
    - "Self-documenting SLURM scripts with inline parameter provenance"

key_files:
  created: []
  modified:
    - "scripts/lightsheet_pipeline/07_direct_fuse.sh"

decisions:
  - "--workers bumped from 8 to 12 to match the Stage 2 per-task allocation used in the parallel workflow"
  - "Parallel workflow reference comment placed directly after the validated params comment for discoverability"

metrics:
  duration: "5 min"
  completed_date: "2026-05-18"
  tasks_completed: 1
  tasks_total: 1
---

# Phase 2 Plan 3: Update 07_direct_fuse.sh Validated Params and Parallel Reference Summary

## One-liner

Added parallel workflow reference comment and bumped `--workers` to 12 in the single-job fallback fusion script, making it self-documenting about validated params (sigma_frac=0.9, taper_px=288, ncc_threshold=0.5, fusion_axis=2) and the faster 2-stage alternative.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Update workers to 12 and add parallel workflow reference comment | 5d59c56 | scripts/lightsheet_pipeline/07_direct_fuse.sh |

## Verification Results

```
syntax OK
workers line: --workers            12
stage1_ncc reference: # Stage 1 (NCC + zarr init): sbatch 07_direct_fuse_stage1_ncc.sh
stage2_parallel reference: # Stage 2 (8-task array):    sbatch --dependency=afterok:<job1_id> 07_direct_fuse_stage2_parallel.sh
~50 min reference: # Parallel run: ~50 min vs ~3.3 hours for this single-job script.
All validated params present in python call
All 8 #SBATCH directives unchanged
```

## Deviations from Plan

None - plan executed exactly as written.

The substack example comment block (lines 19-29) retains `--workers 8` as it documents old tuning parameters used during the interactive sweep, not the full-stack validated configuration. The active python call (now line 77) correctly uses `--workers 12`. This is the intended state.

## Known Stubs

None. No placeholder or hardcoded stub values introduced.

## Threat Flags

No new security-relevant surface introduced. The comment block instructs users on sbatch submission commands — no code execution risk. Paths referenced are lab-internal HPC scratch paths already present in the file.

## Self-Check: PASSED

- scripts/lightsheet_pipeline/07_direct_fuse.sh: FOUND (modified)
- Commit 5d59c56: verified present in git log
- bash -n passes: confirmed
- --workers 12 present in active python call: confirmed
- stage1_ncc.sh and stage2_parallel.sh references present: confirmed
- All #SBATCH directives unchanged: confirmed
