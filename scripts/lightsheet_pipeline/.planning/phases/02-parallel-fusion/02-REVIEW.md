---
phase: 02-parallel-fusion
reviewed: 2026-05-18T00:00:00Z
depth: standard
files_reviewed: 4
files_reviewed_list:
  - /vast/projects/BCRL_Multi_Omics/scripts/lightsheet_pipeline/07_direct_fuse.py
  - /vast/projects/BCRL_Multi_Omics/scripts/lightsheet_pipeline/07_direct_fuse_stage1_ncc.sh
  - /vast/projects/BCRL_Multi_Omics/scripts/lightsheet_pipeline/07_direct_fuse_stage2_parallel.sh
  - /vast/projects/BCRL_Multi_Omics/scripts/lightsheet_pipeline/07_direct_fuse.sh
findings:
  critical: 2
  warning: 5
  info: 2
  total: 9
status: issues_found
---

# Phase 02: Code Review Report

**Reviewed:** 2026-05-18
**Depth:** standard
**Files Reviewed:** 4
**Status:** issues_found

## Summary

Four files implementing the two-stage SLURM parallel fusion pipeline were reviewed: the Python fusion driver (`07_direct_fuse.py`) and three SLURM shell scripts (stage1 NCC+prealloc, stage2 8-task array, original single-job script). The partition boundary math is correct — all Z-slab boundaries are multiples of 64 and no zarr chunk is touched by more than one array task. The `z_write` absolute-index logic in `r+` mode is correct for all 8 tasks. The critical issues are both in `07_direct_fuse_stage1_ncc.sh` and interact: a wrong flag argument causes Python to exit with code 2, and the absence of `set -e` means the failure is swallowed, allowing Stage 1 to complete "successfully" while leaving an invalid `ncc_scores.json` in place for Stage 2.

---

## Critical Issues

### CR-01: `--skip-refine false` is not valid argparse syntax — crashes NCC re-run silently

**File:** `07_direct_fuse_stage1_ncc.sh:71`

**Issue:** `--skip-refine` is declared as `action='store_true'` in argparse (line 95–96 of `07_direct_fuse.py`). This flag takes no argument; its presence sets `skip_refine=True`, its absence leaves it `False`. Passing the literal string `false` as the next argument causes argparse to treat it as an unrecognised positional and exit with code 2. The NCC re-run branch (triggered when `ncc_scores.json` exists but has a different threshold than `EXPECTED_THRESHOLD`) therefore always fails without updating positions.

**Verified:** `python3 -c "import argparse; p=argparse.ArgumentParser(); p.add_argument('--skip-refine', action='store_true'); p.parse_args(['--skip-refine','false'])"` exits with code 2 ("unrecognised arguments: false").

**Impact:** The mismatch branch is designed to re-run NCC at the correct threshold before pre-allocating the zarr. Because the re-run fails, the stale `ncc_scores.json` (with the wrong threshold) persists. Stage 1 continues and pre-allocates the zarr. Stage 2 then runs fusion using incorrect tile positions, silently producing a mis-stitched volume with no indication of failure.

**Fix:** Remove `false` from the invocation. `--skip-refine` is simply absent when NCC should run:

```bash
# Remove --skip-refine false (lines 60-72). Correct invocation:
python "${SCRIPT_DIR}/07_direct_fuse.py" \
    --czi            "$CZI_PATH" \
    --out            "$OUT_ZARR" \
    --sigma-frac     0.9 \
    --taper-px       288 \
    --ncc-threshold  0.5 \
    --fusion-axis    2 \
    --z-chunk        64 \
    --workers        16 \
    --z-start        0 \
    --z-end          1
```

---

### CR-02: No `set -e` in any shell script — Python failures are silently swallowed

**File:** `07_direct_fuse_stage1_ncc.sh` (all lines), `07_direct_fuse_stage2_parallel.sh` (all lines), `07_direct_fuse.sh` (all lines)

**Issue:** None of the three shell scripts include `set -e` (or `set -euo pipefail`). Without it, a non-zero exit from any command — including the Python invocations — does not abort the script. This directly enables the bug in CR-01: the failed `python ... --skip-refine false` exits 2, bash logs nothing and continues to the zarr pre-allocation step. Stage 1 exits 0. Stage 2 is submitted and runs with the wrong positions.

The same risk applies to `07_direct_fuse_stage2_parallel.sh` lines 99–112: the fusion Python call exit code IS explicitly checked via `FUSE_EXIT`, but the zarr pre-check Python call (lines 78–94) exits directly with `sys.exit(1)` and the surrounding `if [ $? -ne 0 ]` check on line 91 is correct for that block. However, any Python invocation without an explicit exit-code check (e.g., the post-task spot-check on lines 124–133) will also swallow failures.

**Fix:** Add `set -euo pipefail` as the first non-comment, non-SBATCH line of each script:

```bash
#!/bin/bash
#SBATCH ...

set -euo pipefail
```

For the NCC re-run block in stage1, if you intentionally want to continue on a Python warning exit, check the exit code explicitly rather than relying on the absence of `set -e`.

---

## Warnings

### WR-01: Canvas dimensions recomputed from positions not validated against pre-allocated zarr shape before write

**File:** `07_direct_fuse.py:594-602, 682`

**Issue:** In `r+` mode, `refined_h` and `refined_w` are derived from `min`/`max` of the loaded tile positions (lines 594–601). The zarr was pre-allocated with specific `H` and `W` (from `tile_manifest_corrected.json` or the hardcoded fallback). If the two computations produce different values — for example after a threshold-mismatch NCC re-run that changes refined positions — the write on line 682 will raise `ValueError: could not broadcast input array from shape (nz, refined_h, refined_w) into shape (nz, zarr_H, zarr_W)`. The job fails mid-slab, leaving the zarr partially written.

There is no assertion or early check that `canvas.shape[1] == out_z.shape[3]` and `canvas.shape[2] == out_z.shape[4]` before entering the Z-chunk loop.

**Fix:** Add a shape guard immediately after opening the zarr in `r+` mode (around line 619):

```python
else:  # r+
    out_z = zarr.open(str(out_path), mode="r+")
    expected_h, expected_w = out_z.shape[3], out_z.shape[4]
    if refined_h != expected_h or refined_w != expected_w:
        raise ValueError(
            f"Canvas size mismatch: positions give ({refined_h}, {refined_w}) "
            f"but pre-allocated zarr has ({expected_h}, {expected_w}). "
            "Re-run Stage 1 to reallocate zarr with current positions."
        )
```

---

### WR-02: Stage 1 NCC run with `z-end=1` transiently truncates fused_direct.zarr to shape `(1,2,1,H,W)`

**File:** `07_direct_fuse_stage1_ncc.sh:77-89`

**Issue:** When `ncc_scores.json` is absent, Stage 1 runs fusion for `z=728:729` (1 plane) to generate NCC scores. Because `--zarr-mode` defaults to `w`, this call opens `fused_direct.zarr` with `mode="w"` and creates it with `shape=(1, 2, 1, H, W)`. Step 2 then correctly recreates it with the full shape. However, if Step 2 fails (Python exception, OOM, node failure), the zarr is left with shape `(1, 2, 1, H, W)`. Stage 2 tasks would then pass the shape pre-check (`(1,2,1,H,W) != (1,2,1557,H,W)` → `sys.exit(1)`), so no silent corruption occurs — but recovery requires manually re-running Stage 1 Step 2.

**Fix:** Pass `--zarr-mode r+` to the NCC-only run, or redirect the NCC run output to a throwaway path, so the pre-allocated full-shape zarr is not clobbered. With `set -euo pipefail` (CR-02 fix), a Step 2 failure will abort Stage 1 with a non-zero exit, preventing the Stage 2 dependency from triggering at all.

```bash
# Use a separate throwaway zarr for the NCC probe run:
python "${SCRIPT_DIR}/07_direct_fuse.py" \
    --czi            "$CZI_PATH" \
    --out            "${SCRATCH}/ncc_probe.zarr" \
    --ncc-threshold  0.5 \
    --z-start        728 \
    --z-end          729
```

---

### WR-03: No validation that loaded tile count matches layout `n_tiles`

**File:** `07_direct_fuse.py:579-582`

**Issue:** When `--load-positions` is used, `refined_pos` is populated from the JSON without checking that `len(refined_pos) == layout["n_tiles"]`. If the JSON was written for a different tile count (e.g., an older run, or a truncated file), the subsequent tile loop `for m in range(n_tiles)` at line 656 will attempt `refined_pos[m]` for indices not present in `refined_pos`, raising `KeyError` mid-processing. The error message will point to the dictionary lookup rather than indicating the root cause (tile count mismatch).

**Fix:** Add a count check after loading positions:

```python
refined_pos = {p["M"]: {"x": p["x"], "y": p["y"], "w": p["w"], "h": p["h"]}
               for p in _pos_list}
if len(refined_pos) != n_tiles:
    raise ValueError(
        f"--load-positions: JSON contains {len(refined_pos)} tile positions "
        f"but CZI layout has {n_tiles} tiles."
    )
```

---

### WR-04: `assert` used for CLI argument validation — disabled with `python -O`

**File:** `07_direct_fuse.py:553`

**Issue:** `assert z_start < z_end, ...` is the only guard against an invalid `(--z-start, --z-end)` combination. Python's `assert` statements are compiled out when the interpreter runs with the `-O` (optimise) flag. If invoked as `python -O 07_direct_fuse.py --z-start 1000 --z-end 500`, the assertion is skipped, `n_z_proc` becomes negative, `n_chunks` evaluates to 0, and the Z-chunk loop runs zero iterations — producing an empty zarr with no error.

**Fix:** Replace with an explicit check:

```python
if z_start >= z_end:
    raise ValueError(f"--z-start ({z_start}) must be less than --z-end ({z_end})")
```

---

### WR-05: `open()` without `with` statement leaks file handle in Stage 1 inline Python

**File:** `07_direct_fuse_stage1_ncc.sh:112`

**Issue:** The inline Python block in Stage 1 reads `tile_manifest_corrected.json` via `json.load(open(manifest_path))` without a `with` statement. The file descriptor is not explicitly closed; CPython will close it when the object is garbage collected, but this is an implementation detail and is flagged by linters. In a long-running process with many such patterns, this can exhaust file descriptors.

**Fix:**

```python
if os.path.exists(manifest_path):
    with open(manifest_path) as f:
        man = json.load(f)
    H = man["canvas_shape"]["H"]
    W = man["canvas_shape"]["W"]
```

---

## Info

### IN-01: `_BBox` class defined inside hot inner loop

**File:** `07_direct_fuse.py:663-669`

**Issue:** An anonymous `_BBox` class is created with `class _BBox: pass` inside the `as_completed` loop that runs once per tile per Z-chunk. This creates a new class object on every iteration. It is harmless but unusual; a `SimpleNamespace` or `types.SimpleNamespace` is the idiomatic replacement for a one-off attribute holder.

**Fix:**

```python
from types import SimpleNamespace
# ...
bbox = SimpleNamespace(
    x=bbox_dict["x"], y=bbox_dict["y"],
    w=bbox_dict["w"], h=bbox_dict["h"]
)
```

---

### IN-02: Shell variables embedded in `python3 -c` strings use fragile quoting

**File:** `07_direct_fuse_stage1_ncc.sh:55`, `07_direct_fuse_stage2_parallel.sh:78-90, 124-133`

**Issue:** Shell variables (`$NCC_JSON`, `${OUT_ZARR}`) are interpolated directly into the `-c` Python code string. If these paths ever contain single-quotes or other special characters, the embedded Python will have a syntax error. The paths are currently hardcoded in the scripts so there is no immediate injection risk, but the pattern is fragile. The stage2 pre-check also has the path hardcoded twice (once in the Python string, once in the surrounding shell), creating a maintenance inconsistency risk.

**Fix:** Pass paths as environment variables or command-line arguments to the Python snippet:

```bash
# Stage 1, line 55:
NCC_THRESHOLD=$(python3 - "$NCC_JSON" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(d.get("ncc_threshold", "MISSING"))
PYEOF
)
```

---

_Reviewed: 2026-05-18_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
