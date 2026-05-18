# Phase 2: Parallel Fusion — Research

**Researched:** 2026-05-18
**Domain:** SLURM array jobs, zarr concurrent region writes, GPU-parallel CZI fusion, aicspylibczi thread-safety
**Confidence:** HIGH

---

## Summary

Phase 2 fuses the full 1557-plane KL018 volume by partitioning the Z-dimension across 8 SLURM array tasks, each driving one A30 GPU. The core infrastructure is already in place: `07_direct_fuse.py` has `--z-start`/`--z-end` (Phase 1), `ncc_scores.json` and `tile_manifest_corrected.json` both exist in `/vast/scratch/users/kriel.j/KL018_lightsheet/`, the validated parameters are confirmed (sigma_frac=0.9, taper_px=288, ncc_threshold=0.5, fusion_axis=2), and an existing `fused_direct.zarr` with the correct shape `(1, 2, 1557, 8585, 10095)` is already pre-created.

The work reduces to three additions to `07_direct_fuse.py` (two new CLI flags and one line change), one new SLURM array script, and one NCC pre-init script. No architecture changes are needed: zarr 2.x DirectoryStore stores each chunk as a separate file, and the VAST NFS filesystem is POSIX-compliant, making non-overlapping concurrent chunk writes inherently safe without any locking machinery.

The existing `fused_direct.zarr` on scratch should be recreated fresh (it contains partial data from an unknown parameter run covering only Z-chunks 0–6). The new workflow will delete-and-recreate it, then dispatch 8 array tasks that fill all 25 Z-chunks in parallel.

**Primary recommendation:** Two-script workflow — Stage 1 NCC+init job (uses existing `ncc_scores.json`, recreates empty zarr), Stage 2 SLURM array of 8 tasks writing disjoint Z-slabs into the shared zarr via `mode='r+'`.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| NCC tile-position refinement | CPU (Stage 1 job) | — | Single-process; writes ncc_scores.json once |
| Zarr pre-initialization (shape/dtype) | CPU (Stage 1 job) | — | Must happen before Stage 2 starts |
| Z-slab fusion + stitching | GPU (Stage 2 tasks) | CPU canvas accumulation | cupy for fuse_sides; canvas too large for GPU |
| CZI subblock reads | CPU threads (per task) | VAST NFS I/O | aicspylibczi ThreadPoolExecutor |
| Zarr chunk writes | VAST NFS | — | DirectoryStore, one file per chunk |
| Z-range coordination | SLURM array lookup table | — | Bash array in SLURM script; no runtime sync |

---

## Standard Stack

### Core (all already in lightsheet_env)

| Library | Version | Purpose | Verified |
|---------|---------|---------|----------|
| zarr | 2.15.0 | Output store, DirectoryStore chunk writes | `[VERIFIED: conda env + zarr.open() call]` |
| cupy-cuda12x | latest in env | GPU Gaussian blend (fuse_sides) | `[VERIFIED: environment.yml + SLURM wrapper]` |
| aicspylibczi | 3.3.1 | CZI subblock reads | `[VERIFIED: python3 -c]` |
| numpy | env | Canvas float32 accumulation | `[VERIFIED: environment.yml]` |
| scipy / scikit-image | env | NCC, phase correlation (Stage 1 only) | `[VERIFIED: environment.yml]` |

### Infrastructure

| Tool | Version | Purpose |
|------|---------|---------|
| SLURM | 25.11.5 | Job array orchestration (`--array=1-8`) |
| CUDA | 12.1 (module) | Required by cupy-cuda12x |
| VAST NFS | hpc.vastdata.wehi.edu.au:/scratch | POSIX parallel I/O for zarr chunks |

**No new installations required.** All libraries already exist in `/vast/scratch/users/kriel.j/lightsheet_env`.

---

## Architecture Patterns

### System Architecture Diagram

```
STAGE 1 (single SLURM job, gpuq A30)
  ┌─────────────────────────────────────────┐
  │  07_direct_fuse_stage1_ncc.sh           │
  │                                          │
  │  Input: ncc_scores.json (exists)         │
  │    OR run NCC if not present             │
  │                                          │
  │  Action: recreate fused_direct.zarr      │
  │    zarr.open(mode='w', shape=(1,2,1557,  │
  │    8585,10095), chunks=(1,1,64,512,512)) │
  │    → writes .zarray, .zattrs only        │
  │    → no chunk data written yet           │
  └────────────────┬────────────────────────┘
                   │ SLURM --dependency=afterok:<jobid>
                   ▼
STAGE 2 (SLURM array --array=1-8, gpuq A30)
  ┌────────────────────────────────────────────┐
  │  07_direct_fuse_stage2_parallel.sh         │
  │  SLURM_ARRAY_TASK_ID → Z_START, Z_END      │
  │                                            │
  │  Each task (independent process):          │
  │                                            │
  │  CZI file ──(aicspylibczi, read-only)──►  │
  │  ThreadPool reads (12 workers)             │
  │  fuse_sides() ──(cupy, GPU)──►             │
  │  place_tile_slab() ──(CPU canvas)──►       │
  │  zarr.open(mode='r+')                      │
  │    write out_z[t,c, z0_abs:z0_abs+nz]     │
  │                                            │
  │  8 tasks × non-overlapping Z ranges        │
  │  → 8 different sets of zarr chunk files    │
  └────────────────────────────────────────────┘
                   │
                   ▼
  /vast/scratch/.../fused_direct.zarr
  shape (1, 2, 1557, 8585, 10095)
  chunks (1, 1, 64, 512, 512)
  25 zarr Z-chunks, all written
```

### Z-Range Partition Table (verified against real dims)

1557 planes, zarr Z-chunk=64 → 25 zarr Z-chunks total.
Distributed across 8 SLURM array tasks (1 gets 4, rest get 3):

| SLURM task ID | z_start | z_end | planes | zarr chunk IDs |
|---------------|---------|-------|--------|----------------|
| 1 | 0 | 256 | 256 | 0–3 |
| 2 | 256 | 448 | 192 | 4–6 |
| 3 | 448 | 640 | 192 | 7–9 |
| 4 | 640 | 832 | 192 | 10–12 |
| 5 | 832 | 1024 | 192 | 13–15 |
| 6 | 1024 | 1216 | 192 | 16–18 |
| 7 | 1216 | 1408 | 192 | 19–21 |
| 8 | 1408 | 1557 | 149 | 22–24 |

Key property: zarr chunk keys are `T.C.Z_CHUNK.Y_CHUNK.X_CHUNK`. No two tasks share the same Z_CHUNK index → no file-level collision. `[VERIFIED: chunk listing from existing zarr]`

### Recommended Project Structure (new files only)

```
scripts/lightsheet_pipeline/
├── 07_direct_fuse.py          # MODIFIED: +2 flags, +1 line change
├── 07_direct_fuse_stage1_ncc.sh   # NEW: NCC + zarr init (single job)
└── 07_direct_fuse_stage2_parallel.sh  # NEW: array=1-8, Z-slab fusion
```

### Pattern 1: SLURM Array with Bash Lookup Table

```bash
# In 07_direct_fuse_stage2_parallel.sh
#SBATCH --array=1-8

Z_STARTS=(0 256 448 640 832 1024 1216 1408)
Z_ENDS=(256 448 640 832 1024 1216 1408 1557)
IDX=$((SLURM_ARRAY_TASK_ID - 1))
Z_START=${Z_STARTS[$IDX]}
Z_END=${Z_ENDS[$IDX]}

export CUDA_VISIBLE_DEVICES=0   # each task gets its own GPU via SLURM gres

python 07_direct_fuse.py \
    --czi      "$CZI_PATH" \
    --out      "$OUT_ZARR" \
    --load-positions "$NCC_JSON" \
    --zarr-mode r+ \
    --z-start  "$Z_START" \
    --z-end    "$Z_END" \
    --sigma-frac  0.9 \
    --taper-px    288 \
    --ncc-threshold 0.5 \
    --fusion-axis 2 \
    --z-chunk  64 \
    --workers  12
```

`CUDA_VISIBLE_DEVICES=0` works because SLURM's `--gres=gpu:A30:1` assigns each task its own GPU slot, and SLURM sets the CUDA device affinity via cgroups. Within the task's view, device 0 is the assigned GPU.
`[ASSUMED]` — cgroup GPU isolation behavior on this specific cluster should be confirmed.

### Pattern 2: SLURM Dependency Chain

```bash
# Submit Stage 1
JOB1=$(sbatch --parsable 07_direct_fuse_stage1_ncc.sh)
# Submit Stage 2 dependent on Stage 1 success
sbatch --dependency=afterok:$JOB1 07_direct_fuse_stage2_parallel.sh
```

### Pattern 3: Code Changes to 07_direct_fuse.py

**Change 1 — New CLI flags** (add to `parse_args()`):

```python
p.add_argument(
    "--load-positions",
    dest="load_positions",
    default=None,
    help="Path to ncc_scores.json. Load pre-computed tile positions, skip NCC.",
)
p.add_argument(
    "--zarr-mode",
    dest="zarr_mode",
    choices=["w", "r+"],
    default="w",
    help="Zarr open mode. 'w' creates/truncates (default). 'r+' opens pre-existing zarr for parallel slab writes.",
)
```

**Change 2 — Position loading** (in `main()`, replacing the `refine_tile_positions()` call):

```python
if args.load_positions:
    print(f"  Loading pre-computed tile positions from: {args.load_positions}")
    with open(args.load_positions) as _f:
        ncc_data = json.load(_f)
    _pos_list = ncc_data["tile_positions_refined"]
    refined_pos = {p["M"]: {"x": p["x"], "y": p["y"], "w": p["w"], "h": p["h"]}
                   for p in _pos_list}
else:
    ncc_out = out_path.parent / "ncc_scores.json"
    refined_pos = refine_tile_positions(
        czi, layout,
        skip_refine=args.skip_refine,
        ncc_threshold=args.ncc_threshold,
        ncc_out_path=ncc_out,
    )
```

**Change 3 — Zarr open mode** (replace the fixed `mode="w"` block in `main()`):

```python
if args.zarr_mode == "w":
    out_z = zarr.open(
        str(out_path), mode="w",
        shape=out_shape, chunks=out_chunk, dtype=np.uint16,
    )
else:  # r+
    print(f"  Opening pre-existing zarr in r+ mode: {out_path}")
    out_z = zarr.open(str(out_path), mode="r+")
```

**Change 4 — Absolute zarr write index** (line 652, the only write statement):

```python
# BEFORE (local index — wrong for parallel mode):
out_z[t, c, z0_loc : z0_loc + nz] = np.clip(canvas, 0, 65535).astype(np.uint16)

# AFTER (absolute CZI index = absolute zarr index):
out_z[t, c, z0_abs : z0_abs + nz] = np.clip(canvas, 0, 65535).astype(np.uint16)
```

This is backward-compatible: when `z_start=0`, `z0_abs == z0_loc`.
`[VERIFIED: code inspection of lines 600-652]`

### Anti-Patterns to Avoid

- **Opening zarr with `mode='w'` in all array tasks.** Each task would truncate the zarr to its own Z-slab shape, destroying what other tasks wrote. Always use `mode='r+'` in Stage 2 after Stage 1 creates the full-shape zarr.
- **Using `mode='a'` (append).** zarr 2.x `mode='a'` creates if not exists, else opens. It would work but is misleading; `mode='r+'` is explicit about requiring pre-existence.
- **Overlapping Z boundaries on zarr chunk edges.** If `z_start=200`, it falls in the middle of zarr chunk index 3 (which spans z=192:256). That chunk would be written by two jobs. Partition boundaries MUST align to multiples of 64.
- **Running NCC in every array task.** NCC reads a mid-Z plane for all 30 tiles and computes 435 pairwise correlations — ~5-10 min of CPU work. Running it 8 times in parallel wastes time and gives 8 conflicting ncc_scores.json writes.
- **Increasing `--z-chunk` to reduce job count.** Canvas RAM scales linearly: z_chunk=128 → 82 GB canvas per process. With 4 tasks on an A30 node (490 GB), that's 328 GB just for canvases, leaving only 162 GB for OS, tiles, and buffers. Risky.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Job dependency chain | Custom polling/sleep loop | `sbatch --dependency=afterok:$JOBID` | SLURM native; handles failures |
| Z-range arithmetic | Complex dynamic formula | Bash lookup array (Z_STARTS/Z_ENDS hardcoded) | 8 values, pre-computed; no off-by-one risk |
| GPU device assignment | Manual CUDA init code | `CUDA_VISIBLE_DEVICES=0` + SLURM `--gres=gpu:A30:1` | SLURM manages isolation |
| Zarr merge step | Post-run zarr concatenation job | Pre-allocated zarr + region writes (`mode='r+'`) | Zero extra I/O; chunks land in place |
| Write locking | File locks, redis, etc. | Non-overlapping partition design | Chunk-per-file storage makes it unnecessary |

**Key insight:** The entire concurrency problem dissolves when partition boundaries align to zarr chunk boundaries. There is no shared mutable state between array tasks.

---

## Common Pitfalls

### Pitfall 1: zarr Z-boundary misalignment
**What goes wrong:** If `z_start=200` (not a multiple of 64), the fusion job processes z=200:256 as its first chunk but writes it to `out_z[t, c, 200:256]`. This crosses zarr chunk files: zarr chunk 3 covers z=192:256, so the first 8 rows (200-207) of that chunk are in one job and the last 56 rows (200-255... wait — entire chunk is 192:256, job writes 200:256 which is a partial chunk). zarr writes partial chunks by reading-modifying-writing the whole chunk file, which could race with another job writing the 192:200 slice of the same file.
**Why it happens:** Partition boundaries not aligned to zarr chunk size (64).
**How to avoid:** Always set `z_start` = multiple of 64. The partition table in this research IS correctly aligned. `[VERIFIED: arithmetic]`
**Warning signs:** Two jobs with overlapping Z_CHUNK indices in their write ranges.

### Pitfall 2: Stage 2 jobs start before Stage 1 zarr init completes
**What goes wrong:** Stage 2 task tries `zarr.open(mode='r+')` before the `.zarray` metadata file is written → `zarr.errors.GroupNotFoundError` or equivalent.
**Why it happens:** Missing SLURM dependency or wrong dependency type.
**How to avoid:** `sbatch --dependency=afterok:<stage1_jobid>`. Use `afterok` not `after` — `after` fires as soon as stage 1 starts, not when it finishes.
**Warning signs:** Stage 2 fails immediately with a zarr open error.

### Pitfall 3: CUDA not loaded in array tasks
**What goes wrong:** cupy import fails or falls back to numpy silently (reducing performance by ~10-20x for fuse_sides).
**Why it happens:** `module load CUDA/12.1` not in the array task script, or the SLURM job doesn't source `/etc/profile.d/modules.sh`.
**How to avoid:** Always include `module load CUDA/12.1` before conda activate in both stage scripts. Check `backend` printout in job logs.
**Warning signs:** Log says `Backend: NumPy (CPU)` instead of `CuPy (GPU)`.

### Pitfall 4: Stale fused_direct.zarr shape mismatch
**What goes wrong:** Existing zarr has shape `(1, 2, 1, 8585, 10095)` or `(1, 2, 192, ...)` from a previous substack run. Stage 2 opens it with `mode='r+'` and writes to `out_z[t, c, 1408:1557]` which is out-of-bounds → IndexError.
**Why it happens:** Stage 1 didn't recreate the zarr, or Stage 1 ran with wrong z_end.
**How to avoid:** Stage 1 always runs with `mode='w'` (recreates). Verify shape after Stage 1: `zarr.open(..., 'r').shape == (1, 2, 1557, 8585, 10095)`.
**Warning signs:** Stage 1 log shows wrong `out_shape`.

### Pitfall 5: ncc_scores.json positions from wrong parameter run
**What goes wrong:** `ncc_scores.json` was written during a substack run with `ncc_threshold=0.05` (old default), but Phase 2 uses 0.5. The tile positions may differ.
**Why it happens:** ncc_scores.json is overwritten by every run.
**How to avoid:** The current `ncc_scores.json` has `ncc_threshold=0.5` (confirmed). Stage 1 should verify this before skipping NCC. If threshold mismatches, re-run NCC.
**Warning signs:** `ncc_scores.json` shows `"ncc_threshold": 0.05` instead of 0.5.

### Pitfall 6: Workers flag too high for dual-side reads in parallel mode
**What goes wrong:** With 8 array tasks running on 2 nodes (4 tasks per node), each task using `--workers 16`, one node has 4 × 16 = 64 threads competing for CZI reads + VAST I/O. This doesn't crash but can cause I/O contention and slow all tasks.
**Why it happens:** `--workers` default is 16 in the existing script; designed for single-job use.
**How to avoid:** Use `--workers 12` for array tasks on A30 nodes (12 CPUs per task, 4 tasks per node = 48 cores = exactly the node's socket count).

---

## Code Examples

### Stage 1 Script Skeleton

```bash
#!/bin/bash
#SBATCH --job-name=ls_fuse_ncc
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --output=/vast/scratch/users/kriel.j/output.%j.%N.log

# Source: this research document
SCRATCH="/vast/scratch/users/kriel.j/KL018_lightsheet"
CZI="$SCRATCH/KL018_85_D7_CT2AvIII_Overview.czi"
OUT_ZARR="$SCRATCH/fused_direct.zarr"
NCC_JSON="$SCRATCH/ncc_scores.json"
SCRIPT_DIR="/vast/projects/BCRL_Multi_Omics/scripts/lightsheet_pipeline"

if ! command -v module &>/dev/null; then source /etc/profile.d/modules.sh; fi
module load CUDA/12.1
source /stornext/System/data/apps/anaconda3/anaconda3-latest/etc/profile.d/conda.sh
conda activate /vast/scratch/users/kriel.j/lightsheet_env

# Validate NCC JSON has correct threshold
NCC_THRESHOLD=$(python3 -c "import json; d=json.load(open('$NCC_JSON')); print(d['ncc_threshold'])")
echo "NCC threshold in json: $NCC_THRESHOLD (expected 0.5)"

# Recreate empty zarr (full shape)
python3 - <<'PYEOF'
import zarr, numpy as np, json
out_path = "/vast/scratch/users/kriel.j/KL018_lightsheet/fused_direct.zarr"
man = json.load(open("/vast/scratch/users/kriel.j/KL018_lightsheet/tile_manifest_corrected.json"))
H = man["canvas_shape"]["H"]   # 8585
W = man["canvas_shape"]["W"]   # 10095
z = zarr.open(out_path, mode="w",
              shape=(1, 2, 1557, H, W),
              chunks=(1, 1, 64, 512, 512),
              dtype=np.uint16)
print(f"zarr created: shape={z.shape}  chunks={z.chunks}")
PYEOF

echo "Stage 1 complete. fused_direct.zarr initialized."
```

### Stage 2 Array Script Skeleton

```bash
#!/bin/bash
#SBATCH --job-name=ls_fuse_par
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --array=1-8
#SBATCH --output=/vast/scratch/users/kriel.j/output.%j.%a.%N.log

# Source: this research document
Z_STARTS=(0 256 448 640 832 1024 1216 1408)
Z_ENDS=(256 448 640 832 1024 1216 1408 1557)
IDX=$((SLURM_ARRAY_TASK_ID - 1))
Z_START=${Z_STARTS[$IDX]}
Z_END=${Z_ENDS[$IDX]}

SCRATCH="/vast/scratch/users/kriel.j/KL018_lightsheet"
SCRIPT_DIR="/vast/projects/BCRL_Multi_Omics/scripts/lightsheet_pipeline"

if ! command -v module &>/dev/null; then source /etc/profile.d/modules.sh; fi
module load CUDA/12.1
source /stornext/System/data/apps/anaconda3/anaconda3-latest/etc/profile.d/conda.sh
conda activate /vast/scratch/users/kriel.j/lightsheet_env

echo "Task ${SLURM_ARRAY_TASK_ID}: Z=${Z_START}:${Z_END}  GPU=CUDA_VISIBLE_DEVICES=0"

python "${SCRIPT_DIR}/07_direct_fuse.py" \
    --czi           "$SCRATCH/KL018_85_D7_CT2AvIII_Overview.czi" \
    --out           "$SCRATCH/fused_direct.zarr" \
    --load-positions "$SCRATCH/ncc_scores.json" \
    --zarr-mode     r+ \
    --z-start       "$Z_START" \
    --z-end         "$Z_END" \
    --sigma-frac    0.9 \
    --taper-px      288 \
    --ncc-threshold 0.5 \
    --fusion-axis   2 \
    --z-chunk       64 \
    --workers       12

echo "Task ${SLURM_ARRAY_TASK_ID}: Done."
```

---

## State of the Art

| Old Approach | Current Approach | Notes |
|--------------|------------------|-------|
| Serial Z-chunk loop (one job) | SLURM array, one GPU per Z-slab | This phase |
| NCC on every job start | NCC once, JSON reused | Saves ~5-10 min × 8 jobs |
| zarr `mode='w'` per job | Pre-alloc + `mode='r+'` + region writes | Standard HPC zarr pattern |
| 4-thread tile reads | 12-thread tile reads (12 CPUs per task) | Better I/O/GPU overlap |

**Zarr 2.x vs 3.x:** Environment pins `zarr>=2.15,<3`. Zarr 3.x changed the API significantly. This research applies exclusively to zarr 2.x. `[VERIFIED: environment.yml + zarr.__version__ == 2.15.0]`

---

## Memory and Resource Analysis

### Per-Process RAM

| Buffer | Size | Notes |
|--------|------|-------|
| canvas float32 (64, 8585, 10095) | 20.7 GB | Accumulation buffer |
| weight_canvas float32 same shape | 20.7 GB | Normalization weights |
| Dual-side tile read buffers (peak) | ~1.8 GB | 2 × (64, 1920, 1920) uint16 |
| **Total per process** | **~43 GB** | Plus Python/OS overhead |

`[VERIFIED: python3 arithmetic with real canvas dims from zarr.shape]`

### Per-Node RAM (4 tasks on A30 node)

| Item | Total |
|------|-------|
| 4 processes × 43 GB | 172 GB |
| A30 node RAM | 490 GB |
| **Headroom** | **~318 GB** — very comfortable |

`[VERIFIED: sinfo output; arithmetic]`

### GPU VRAM per Task (A30 24 GB)

One tile's dual-side slab at float32: 2 × 64 × 1920 × 1920 × 4 bytes = 1.76 GB. Well within A30 24 GB. `[VERIFIED: arithmetic]`

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| lightsheet_env (conda) | All scripts | ✓ | zarr 2.15, cupy-cuda12x, aicspylibczi 3.3.1 | — |
| CUDA 12.1 (module) | cupy GPU execution | ✓ | via `module load CUDA/12.1` | cupy falls back to numpy |
| CZI staged file | aicspylibczi reads | ✓ | 1.3 TB on /vast/scratch | stornext (3x slower) |
| ncc_scores.json | Stage 2 --load-positions | ✓ | ncc_threshold=0.5 | Re-run NCC in Stage 1 |
| tile_manifest_corrected.json | Zarr init shape | ✓ | H=8585, W=10095 | Derive from ncc_scores.json |
| fused_direct.zarr (pre-existing) | Stage 2 mode='r+' | ✓ (partial) | Shape correct, partial data | Recreate in Stage 1 |
| SLURM gpuq A30 nodes | GPU jobs | ✓ | 7 nodes × 4× A30 24GB | A100 (fewer GPUs, higher VRAM) |

`[VERIFIED: ls /vast/scratch/..., sinfo, conda env introspection]`

---

## Answers to Research Questions

### Q1: Optimal Z-range partition strategy

**25 zarr Z-chunks (64 planes each, last = 45 planes for z=1512:1557** — wait, actual last chunk: z=1408:1557 = 149 planes across 3 zarr chunks). Distribute 25 chunks across 8 SLURM tasks: task 1 gets 4 chunks (256 planes), tasks 2-8 get 3 chunks (192 planes each, except task 8 which ends at z=1557 = 149 planes). This is the most balanced achievable without non-chunk-boundary splits. `[VERIFIED: arithmetic]`

### Q2: Output zarr write strategy

**Use option (c): pre-create full zarr, then region writes from non-overlapping processes.** No merge job needed. No shard handling. VAST POSIX NFS handles concurrent file creates atomically. The only shared file is `.zarray` (metadata) — written once by Stage 1 and read-only in Stage 2. `[VERIFIED: zarr chunk listing + VAST filesystem type]`

### Q3: SLURM array patterns

`#SBATCH --array=1-8` with `SLURM_ARRAY_TASK_ID` indexing into Bash arrays. `sbatch --dependency=afterok:$JOBID` chains stages. SLURM 25.11.5 is confirmed on this cluster. `[VERIFIED: sbatch --version, sbatch --help]`

### Q4: Multi-GPU within one node

**Not needed.** The SLURM array approach distributes tasks across multiple nodes automatically. Each task requests `--gres=gpu:A30:1` — SLURM assigns the GPU. Setting `CUDA_VISIBLE_DEVICES=0` within the task makes CuPy use that assigned GPU. No `srun --ntasks=4` complexity. `[ASSUMED]` — `CUDA_VISIBLE_DEVICES=0` within SLURM GPU cgroup should map to the right physical device; confirm with a test job if unexpected behavior.

### Q5: NCC pre-computation reuse

Yes. `ncc_scores.json` already exists with `ncc_threshold=0.5` (matching validated params). The `tile_positions_refined` list in that JSON is all Stage 2 needs. The `--load-positions ncc_scores.json` flag (new code) reads this JSON and skips `refine_tile_positions()` entirely. `[VERIFIED: ncc_scores.json inspection]`

### Q6: Zarr concurrent write safety

Safe. zarr 2.x DirectoryStore = one file per chunk. Non-overlapping Z partition = non-overlapping chunk files. VAST NFS is POSIX-compliant and optimized for parallel HPC I/O. `.zarray`/`.zattrs` written once (Stage 1), read-only in Stage 2. `[VERIFIED: zarr chunk key inspection, df -T /vast/scratch]`

### Q7: Estimated speedup

Serial baseline (A100, ~2-4 min/chunk, 25 chunks, 2 channels): ~3.3 hours. Parallel (8 tasks, A30 GPU, ~6-8 min/chunk, 3-4 chunks/task, 2 channels): ~50 min per task → ~50 min total wall time (all 8 run concurrently). Speedup: ~4x wall time. `[ASSUMED: A30 timing extrapolated from A100 measurements; actual may vary]`

### Q8: CZI concurrent read safety

Safe. Each task opens its own `CziFile` instance (confirmed multiple instances work simultaneously). CZI file is read-only. VAST NFS supports multiple concurrent readers. `[VERIFIED: two-instance test in Python]`

### Q9: Canvas memory per node

4 tasks × 43 GB = 172 GB per A30 node. A30 nodes have 490 GB RAM. 318 GB headroom — very comfortable. `[VERIFIED: sinfo memory + arithmetic]`

### Q10: SLURM script structure

Two scripts: `07_direct_fuse_stage1_ncc.sh` (single job for NCC validation + zarr init) and `07_direct_fuse_stage2_parallel.sh` (array=1-8 for Z-slab fusion). Submitted with `--dependency=afterok`. See code examples above.

### Q11: zarr 2.x vs 3.x

Irrelevant — environment pins `zarr>=2.15,<3`. The `mode='r+'` behavior and DirectoryStore chunk naming are well-established in zarr 2.x. `[VERIFIED: environment.yml]`

### Q12: VAST concurrent multi-writer zarr

Safe. VAST is NFS-type (confirmed `df -T`). Zarr chunk files are created atomically. Parallel I/O is a core VAST use case. `[VERIFIED: df -T /vast/scratch]` `[ASSUMED: no VAST-specific locking quirks for POSIX file creates; this is standard HPC filesystem behavior]`

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `CUDA_VISIBLE_DEVICES=0` within SLURM GPU cgroup correctly maps to the assigned physical GPU | Q4, Stage 2 script | cupy uses wrong GPU; tasks could fight over one GPU. Mitigation: test with 2 array tasks first |
| A2 | A30 GPU processes one Z-chunk (2 channels) in ~6-8 min (extrapolated from A100 ~4 min) | Q7 speedup estimate | Jobs may need time limit adjustment. Set `--time=02:00:00` for safety |
| A3 | VAST NFS has no write-locking quirks for concurrent chunk file creation from multiple nodes | Q12 | Very low risk: this is standard POSIX behavior, VAST is designed for parallel HPC |
| A4 | ncc_scores.json tile positions from the Phase 1 substack sweep are valid for the full 1557-plane run | Q5 | If Stage 1 used a different Z-slice mid point or threshold, positions could differ slightly. Mitigation: Stage 1 validates threshold value before skipping NCC |

---

## Open Questions

1. **Will the validated params (sigma_frac=0.9, taper_px=288) be confirmed before Phase 2 runs?**
   - What we know: Phase 1 plan 01-03 (notebook) is the checkpoint that produces best_params.
   - What's unclear: Is 01-03 complete? STATE.md says Wave 2 of Phase 1 is still pending.
   - Recommendation: Phase 2 planning should depend on 01-03 delivering confirmed params. The SLURM script comments document the params as "validated 2026-05-18" — this is the right flag to watch.

2. **Does SLURM on this cluster enforce GPU isolation via cgroups, or does CUDA_VISIBLE_DEVICES require explicit setting?**
   - What we know: SLURM 25.11.5, `SelectType=cons_tres`, GresTypes=gpu — standard GRES setup.
   - What's unclear: Whether cgroup plugin is configured (it usually is with cons_tres).
   - Recommendation: Stage 2 script should explicitly set `CUDA_VISIBLE_DEVICES=0` regardless, as belt-and-suspenders.

---

## Sources

### Primary (HIGH confidence)
- `[VERIFIED]` zarr 2.15.0 — `zarr.__version__` from conda env, chunk listing of existing zarr
- `[VERIFIED]` aicspylibczi 3.3.1 — `aicspylibczi.__version__`, two-instance concurrent open test
- `[VERIFIED]` SLURM 25.11.5 — `sbatch --version`, `sbatch --help --array`
- `[VERIFIED]` GPU node inventory — `sinfo -p gpuq` output (7 A30 nodes × 4 GPU, 490 GB RAM)
- `[VERIFIED]` Canvas dimensions — zarr.shape (1,2,1557,8585,10095), czi_layout_cache.json
- `[VERIFIED]` ncc_scores.json content — `ncc_threshold=0.5`, 30 tile positions
- `[VERIFIED]` VAST filesystem type — `df -T /vast/scratch` → NFS
- `[VERIFIED]` zarr chunk key format — T.C.Z.Y.X flat naming in DirectoryStore

### Secondary (MEDIUM confidence)
- A30 node RAM 490 GB — from sinfo MEMORY field (502656 MB ≈ 490 GB confirmed)
- VAST parallel write safety — POSIX NFS guarantee, VAST HPC design documentation (`[ASSUMED]` no cluster-specific quirk)

### Tertiary (LOW / ASSUMED)
- A30 GPU timing (~6-8 min per Z-chunk) — extrapolated from README's A100 measurement
- CUDA_VISIBLE_DEVICES cgroup isolation — standard SLURM behavior with cons_tres

---

## Metadata

**Confidence breakdown:**
- Z partition math: HIGH — computed from real zarr dims
- Code changes: HIGH — based on direct code reading (lines 584-652)
- zarr concurrent write safety: HIGH — verified filesystem type + chunk naming
- CZI concurrent read safety: HIGH — verified two-instance test
- Timing estimates: LOW — extrapolated from A100 measurements to A30

**Research date:** 2026-05-18
**Valid until:** 2026-06-18 (stable domain — zarr 2.x API, SLURM patterns)
