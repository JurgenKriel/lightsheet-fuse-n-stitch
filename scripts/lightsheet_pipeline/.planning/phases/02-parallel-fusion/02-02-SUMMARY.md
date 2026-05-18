---
phase: 02-parallel-fusion
plan: "02"
subsystem: infra
tags: [slurm, zarr, gpu, lightsheet, fusion, parallel, cuda]

# Dependency graph
requires:
  - phase: 02-parallel-fusion
    plan: "01"
    provides: "--load-positions and --zarr-mode CLI flags added to 07_direct_fuse.py"
provides:
  - "07_direct_fuse_stage1_ncc.sh: SLURM job that validates ncc_threshold=0.5 and pre-allocates fused_direct.zarr at full shape (1,2,1557,8585,10095)"
  - "07_direct_fuse_stage2_parallel.sh: SLURM array of 8 tasks, each fusing one Z-slab on its own A30 GPU via --load-positions and --zarr-mode r+"
affects:
  - 02-parallel-fusion/03
  - full-stack fusion execution

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two-stage SLURM dependency pattern: sbatch stage1; sbatch --dependency=afterok:$JOB1 stage2"
    - "Zarr pre-allocation then parallel r+ writes — partitioned by Z-chunk boundaries (multiples of 64)"
    - "SLURM array task → Z-slab mapping via bash array index (IDX = SLURM_ARRAY_TASK_ID - 1)"
    - "Belt-and-suspenders GPU isolation: SLURM cgroup + CUDA_VISIBLE_DEVICES=0"

key-files:
  created:
    - scripts/lightsheet_pipeline/07_direct_fuse_stage1_ncc.sh
    - scripts/lightsheet_pipeline/07_direct_fuse_stage2_parallel.sh
  modified: []

key-decisions:
  - "Z partition boundaries (0,256,448,640,832,1024,1216,1408,1557) all multiples of 64 to prevent zarr chunk file collisions between concurrent array tasks"
  - "Stage 1 re-validates ncc_threshold from ncc_scores.json before reuse — guards against stale JSON from a different parameter run"
  - "Stage 2 pre-flight zarr shape check exits with code 1 if zarr absent or wrong shape — mitigates T-02-04 (Stage 2 starts before Stage 1 zarr init)"
  - "CUDA_VISIBLE_DEVICES=0 exported in Stage 2 in addition to SLURM GPU cgroup — belt-and-suspenders, mitigates T-02-06"

patterns-established:
  - "SLURM two-stage dependency: stage1 pre-allocates shared resource; stage2 array uses --dependency=afterok to ensure ordering"
  - "Zarr chunk-boundary Z partitioning: all splits must be multiples of z-chunk size to guarantee disjoint chunk file writes"

requirements-completed:
  - FULLSTACK-01
  - FULLSTACK-02

# Metrics
duration: 2min
completed: 2026-05-18
---

# Phase 02 Plan 02: Parallel Fusion SLURM Scripts Summary

**Two-stage SLURM scripts that decompose the 1557-plane KL018 fusion into a 30-min NCC validation + zarr pre-allocation job followed by an 8-task GPU array that fuses disjoint Z-slabs in parallel, reducing wall time from ~3.3h to ~50min**

## Performance

- **Duration:** 2 min
- **Started:** 2026-05-18T07:08:26Z
- **Completed:** 2026-05-18T07:10:28Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Created `07_direct_fuse_stage1_ncc.sh`: validates ncc_scores.json threshold (0.5), re-runs NCC if mismatch, recreates fused_direct.zarr at exact shape (1,2,1557,8585,10095) with chunks (1,1,64,512,512)
- Created `07_direct_fuse_stage2_parallel.sh`: SLURM array `--array=1-8`, each task maps SLURM_ARRAY_TASK_ID to its Z-slab via Z_STARTS/Z_ENDS lookup arrays, calls 07_direct_fuse.py with `--load-positions`, `--zarr-mode r+`, `--workers 12`
- Both scripts include pre-flight zarr shape verification, CUDA module loading, and detailed logging — mitigate all four STRIDE threats in the plan's threat register

## Task Commits

1. **Task 1: Create 07_direct_fuse_stage1_ncc.sh** - `b1d0773` (feat)
2. **Task 2: Create 07_direct_fuse_stage2_parallel.sh** - `7042edb` (feat)

## Files Created/Modified
- `scripts/lightsheet_pipeline/07_direct_fuse_stage1_ncc.sh` - NCC validation + zarr pre-allocation SLURM job (A30:1, 16 CPUs, 100G, 30min)
- `scripts/lightsheet_pipeline/07_direct_fuse_stage2_parallel.sh` - 8-task parallel fusion SLURM array (A30:1 per task, 12 CPUs, 80G, 2h)

## Decisions Made
- Z partition boundaries (0, 256, 448, 640, 832, 1024, 1216, 1408, 1557) are all multiples of 64 to prevent zarr chunk file collisions between concurrent tasks
- Stage 1 reads and validates ncc_threshold from the JSON before reusing it; mismatch triggers NCC re-run at a single midpoint plane to regenerate a correct JSON
- Stage 2 adds a Python pre-flight check that exits 1 if zarr is absent or wrong shape — ensures the SLURM --dependency ordering is also enforced at the script level
- Task 1 gets larger resources (cpus=16, mem=100G) because NCC is CPU-bound; Task 2 tasks get (cpus=12, mem=80G) tuned for GPU-bound fusion

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Both SLURM scripts are ready for submission. Workflow:
  ```bash
  JOB1=$(sbatch --parsable scripts/lightsheet_pipeline/07_direct_fuse_stage1_ncc.sh)
  sbatch --dependency=afterok:$JOB1 scripts/lightsheet_pipeline/07_direct_fuse_stage2_parallel.sh
  ```
- Plan 02-03 (OME-TIFF export) can proceed once the full-stack zarr is written

## Known Stubs

None - both scripts are fully wired to the validated parameters and paths established in Phase 1.

## Threat Flags

No new security-relevant surface introduced beyond what is documented in the plan's threat register.

## Self-Check: PASSED

- `scripts/lightsheet_pipeline/07_direct_fuse_stage1_ncc.sh` — FOUND, executable, syntax OK
- `scripts/lightsheet_pipeline/07_direct_fuse_stage2_parallel.sh` — FOUND, executable, syntax OK
- Commit `b1d0773` — FOUND
- Commit `7042edb` — FOUND

---
*Phase: 02-parallel-fusion*
*Completed: 2026-05-18*
