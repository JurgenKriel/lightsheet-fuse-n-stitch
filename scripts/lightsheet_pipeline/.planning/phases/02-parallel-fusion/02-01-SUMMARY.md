---
phase: 02-parallel-fusion
plan: 01
subsystem: lightsheet-fusion
tags: [cli, zarr, parallel, slurm, fusion]
dependency_graph:
  requires: []
  provides: [07_direct_fuse.py --load-positions, 07_direct_fuse.py --zarr-mode]
  affects: [02-02-PLAN.md, 02-03-PLAN.md]
tech_stack:
  added: []
  patterns: [conditional-zarr-open, absolute-z-index-write, pre-computed-positions-json]
key_files:
  modified:
    - scripts/lightsheet_pipeline/07_direct_fuse.py
decisions:
  - "Zarr write index uses z0_abs for r+ mode so array tasks map local slab to correct full-stack Z coordinate"
  - "JSON position loading reads tile_positions_refined key matching refine_tile_positions() output schema"
metrics:
  duration: "~8 minutes"
  completed: "2026-05-18T07:06:02Z"
  tasks_completed: 2
  files_modified: 1
---

# Phase 2 Plan 1: CLI Flags for Parallel Zarr Fusion Summary

**One-liner:** Extended 07_direct_fuse.py with --load-positions and --zarr-mode r+ flags enabling 8 concurrent SLURM array tasks to write disjoint Z-slabs into a shared pre-allocated zarr without re-running NCC or truncating the store.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add --load-positions and --zarr-mode CLI flags to parse_args() | d66a85f | 07_direct_fuse.py |
| 2 | Add position loading logic, conditional zarr open, and absolute write index | a5c5303 | 07_direct_fuse.py |

## What Was Built

Three targeted changes to `07_direct_fuse.py` that collectively enable Stage 2 SLURM array parallelism:

**Task 1 — New CLI flags:**
- `--load-positions` (dest=`load_positions`, default=None): accepts path to `ncc_scores.json`
- `--zarr-mode` (dest=`zarr_mode`, choices=['w','r+'], default='w'): controls zarr open mode

**Task 2 — Three logic changes in main():**
- Change A (NCC block): Conditional branch — when `--load-positions` is provided, reads `tile_positions_refined` from JSON and skips `refine_tile_positions()` entirely (~8 min savings per array task)
- Change B (zarr open): When `zarr_mode=='w'`, creates/truncates as before; when `zarr_mode=='r+'`, opens pre-existing zarr without truncating (safe for concurrent writes to disjoint slabs)
- Change C (write index): `z_write = z0_abs if args.zarr_mode == "r+" else z0_loc` — array tasks writing into a full-shape zarr pre-allocated by Stage 1 must use absolute CZI Z coordinates as the zarr index

## Verification Results

All plan success criteria met:
- `python 07_direct_fuse.py --help` exits 0 and shows `--load-positions` and `--zarr-mode`
- `z_write = z0_abs if args.zarr_mode == "r+" else z0_loc` present in file
- `args.load_positions` conditional branch skips `refine_tile_positions()` when path provided
- Zarr open block has two branches: `mode="w"` (creates) and `mode="r+"` (opens existing)
- Python syntax check (`ast.parse`) passes with no errors

## Deviations from Plan

None — plan executed exactly as written.

## Threat Flags

No new security-relevant surface introduced beyond what the plan's threat model covers. The `--load-positions` path is consumed via `json.load()` from researcher-controlled HPC scratch — accepted per T-02-01. The `zarr_mode=r+` on absent zarr raises `zarr.errors.ContainsGroupError` rather than silent failure — accepted per T-02-02 with SLURM dependency enforcement. The `z_write=z0_abs` out-of-bounds write raises `IndexError` per T-02-03.

## Self-Check: PASSED

- [x] `scripts/lightsheet_pipeline/07_direct_fuse.py` exists and modified
- [x] Commit d66a85f exists (Task 1)
- [x] Commit a5c5303 exists (Task 2)
- [x] `python --help` shows both new flags
- [x] `ast.parse` syntax check passes
- [x] `z_write` conditional present in file
- [x] No stubs found in modified file
