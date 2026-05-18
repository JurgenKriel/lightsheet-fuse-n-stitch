"""
Direct CZI → fused+stitched zarr pipeline (no raw.zarr intermediate).

Reads a Zeiss mosaic CZI directly via aicspylibczi, fuses dual-side
illumination per-tile on the GPU, and stitches all tiles onto a single
canvas. Processes the output in Z-chunks to stay within GPU/RAM limits.

For the KL018 dataset:
  - 30 mosaic tiles (M dimension), each 1920×1920×1557
  - 2 illumination sides (I dimension), Gaussian-blend fused
  - 2 channels (C dimension)
  - Single timepoint (T=0)

Replaces pipeline steps 02 (bioformats2raw) + 03 (no deskew) + 04 + 05
with one job. Output: fused_direct.zarr — shape (T, C, Z, H, W).

Tile positions are read directly from the CZI bounding boxes (not from
tile_manifest.json, which only captured 1 of the 30 tiles).

Phase correlation refinement is ClearMap-inspired: quality scored via NCC,
global positions propagated along a minimum-spanning tree.

Usage:
    python 07_direct_fuse.py --czi /vast/scratch/.../KL018.czi
    python 07_direct_fuse.py --czi ... --z-chunk 64 --workers 16
    python 07_direct_fuse.py --czi ... --skip-refine
"""

import argparse
import json
import math
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import zarr
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from skimage.registration import phase_cross_correlation

try:
    from aicspylibczi import CziFile
except ImportError as e:
    raise SystemExit(f"aicspylibczi not found — activate lightsheet_env: {e}")

try:
    import cupy as cp
    _GPU = cp.is_available()
except (ImportError, Exception):
    cp = None
    _GPU = False


def _bar(done: int, total: int, width: int = 36) -> str:
    """ASCII progress bar suitable for log files (no ANSI, no carriage return)."""
    frac = done / total if total else 1.0
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {done}/{total} ({frac*100:.0f}%)"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--czi",
        default="/vast/scratch/users/kriel.j/KL018_lightsheet/KL018_85_D7_CT2AvIII_Overview.czi",
        help="Path to CZI file. Use the /vast/scratch staged copy for best throughput.",
    )
    p.add_argument(
        "--out",
        default="/vast/scratch/users/kriel.j/KL018_lightsheet/fused_direct.zarr",
        help="Output OME-zarr path.",
    )
    p.add_argument(
        "--z-chunk",
        type=int,
        default=64,
        help="Z planes per processing slab (default 64 ≈ 21 GB canvas RAM/chunk).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Parallel threads for subblock reads (default 16).",
    )
    p.add_argument(
        "--skip-refine",
        action="store_true",
        help="Skip NCC phase-correlation refinement; use raw stage positions only.",
    )
    p.add_argument(
        "--taper-px",
        type=int,
        default=64,
        help="Cosine taper width in pixels at tile edges for overlap blending (default 64).",
    )
    p.add_argument(
        "--dump-layout",
        action="store_true",
        help="Print tile layout and exit without processing.",
    )
    p.add_argument(
        "--z-start",
        type=int,
        default=0,
        help="First Z plane to process (inclusive). Default=0 (full stack).",
    )
    p.add_argument(
        "--z-end",
        type=int,
        default=None,
        help="Last Z plane to process (exclusive). Default=None means full n_z.",
    )
    p.add_argument(
        "--sigma-frac",
        dest="sigma_frac",
        type=float,
        default=0.3,
        help="Gaussian blend sigma fraction for dual-side fusion (default 0.3). Sweep 0.2-0.6.",
    )
    p.add_argument(
        "--fusion-axis",
        dest="fusion_axis",
        type=int,
        default=2,
        help="Axis along which dual-side illumination varies: 0=Z, 1=Y, 2=X (default 2=X).",
    )
    p.add_argument(
        "--ncc-threshold",
        dest="ncc_threshold",
        type=float,
        default=0.05,
        help="NCC quality threshold for phase-correlation refinement (default 0.05).",
    )
    p.add_argument(
        "--load-positions",
        dest="load_positions",
        default=None,
        help="Path to ncc_scores.json. Load pre-computed tile positions and skip NCC refinement.",
    )
    p.add_argument(
        "--zarr-mode",
        dest="zarr_mode",
        choices=["w", "r+"],
        default="w",
        help="Zarr open mode. 'w' creates/truncates (default). 'r+' opens pre-existing zarr for parallel slab writes.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# CZI introspection
# ---------------------------------------------------------------------------

def extract_voxel_sizes(root: ET.Element) -> dict:
    """Read physical voxel sizes (µm) from CZI XML Scaling block.
    CZI stores scaling in metres; we convert to µm. Returns {'x','y','z'} with
    None for any axis not present.
    """
    sizes = {"x": None, "y": None, "z": None}
    for dist in root.iter("Distance"):
        axis = dist.get("Id", "").lower()
        if axis in sizes:
            val = dist.findtext("Value")
            try:
                sizes[axis] = float(val) * 1e6
            except (TypeError, ValueError):
                pass
    return sizes


def get_czi_layout(czi: CziFile, cache_path=None) -> dict:
    """
    Extract mosaic tile layout and volume dims from CZI metadata.

    Results are cached to `cache_path` (JSON) so re-runs skip the slow CZI
    header parse (~minutes for a 1.3 TB file). Pass cache_path=None to disable.
    The cache stores serialisable data; tile_bboxes are reconstructed as plain dicts.
    """
    if cache_path and cache_path.exists():
        print(f"  Loading layout from cache: {cache_path}")
        with open(cache_path) as f:
            data = json.load(f)
        # Reconstruct tile_bboxes as plain dicts (same interface as BoundingRectangle)
        data["tile_bboxes"] = {
            int(m): type("BB", (), {"x": v["x"], "y": v["y"], "w": v["w"], "h": v["h"]})()
            for m, v in data["tile_bboxes_serialisable"].items()
        }
        # Backwards compat: older caches predate voxel_size_um. Pull from CZI now.
        if "voxel_size_um" not in data:
            data["voxel_size_um"] = extract_voxel_sizes(czi.meta)
        return data

    print("  Parsing CZI layout (slow for large files — result will be cached)...")
    dims = czi.get_dims_shape()[0]
    n_tiles = dims.get("M", (0, 1))[1]
    n_z = dims["Z"][1]
    n_y = dims["Y"][1]
    n_x = dims["X"][1]
    n_c = dims.get("C", (0, 1))[1]
    n_t = dims.get("T", (0, 1))[1]
    n_i = dims.get("I", (0, 1))[1]
    dual = n_i == 2

    mosaic_bbox = czi.get_mosaic_bounding_box()
    voxel_size_um = extract_voxel_sizes(czi.meta)

    tile_bboxes_raw = czi.get_all_mosaic_tile_bounding_boxes()
    tile_bboxes = {}
    tile_bboxes_serial = {}
    for k, v in tile_bboxes_raw.items():
        m_idx = k if isinstance(k, int) else k.m_index
        tile_bboxes[m_idx] = v
        tile_bboxes_serial[str(m_idx)] = {"x": v.x, "y": v.y, "w": v.w, "h": v.h}

    layout = {
        "n_tiles": n_tiles,
        "n_z": n_z,
        "tile_h": n_y,
        "tile_w": n_x,
        "n_c": n_c,
        "n_t": n_t,
        "dual": dual,
        "canvas_x": mosaic_bbox.x,
        "canvas_y": mosaic_bbox.y,
        "canvas_w": mosaic_bbox.w,
        "canvas_h": mosaic_bbox.h,
        "voxel_size_um": voxel_size_um,
        "tile_bboxes": tile_bboxes,
        "tile_bboxes_serialisable": tile_bboxes_serial,
    }

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({k: v for k, v in layout.items() if k != "tile_bboxes"}, f, indent=2)
        print(f"  Layout cached: {cache_path}")

    return layout


# ---------------------------------------------------------------------------
# Subblock reading helpers
# ---------------------------------------------------------------------------

def read_plane(czi: CziFile, M: int, I: int, C: int, T: int, Z: int) -> np.ndarray:
    """Read one (M, I, C, T, Z) plane → 2-D (Y, X) uint16."""
    data, _ = czi.read_image(M=M, I=I, C=C, T=T, Z=Z)
    return data.squeeze()


def read_tile_zchunk(
    czi: CziFile,
    M: int, I: int, C: int, T: int,
    z_start: int, z_end: int,
    tile_h: int, tile_w: int,
    workers: int,
) -> np.ndarray:
    """
    Read Z-slab [z_start, z_end) for one (M, I, C, T) combination.
    Returns (Z_chunk, tile_h, tile_w) uint16.
    Reads are issued in parallel via a thread pool.
    """
    n_z = z_end - z_start
    out = np.empty((n_z, tile_h, tile_w), dtype=np.uint16)

    def _read(z_abs):
        plane = read_plane(czi, M, I, C, T, z_abs)
        return z_abs - z_start, plane

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_read, z) for z in range(z_start, z_end)]
        for fut in as_completed(futs):
            idx, plane = fut.result()
            out[idx] = plane

    return out


# ---------------------------------------------------------------------------
# GPU / CPU Gaussian-blend dual-side fusion
# ---------------------------------------------------------------------------

def _gaussian_ramp(size: int, xp, sigma_frac: float = 0.3):
    x = xp.linspace(-1.0, 1.0, size, dtype=xp.float32)
    return xp.exp(-(x ** 2) / (2 * (sigma_frac * 2) ** 2))


def fuse_sides(side_a: np.ndarray, side_b: np.ndarray, axis: int = 2, sigma_frac: float = 0.3) -> np.ndarray:
    """
    Gaussian-weighted blend of two illumination sides along `axis`.
    Inputs: (Z_chunk, Y, X) uint16.
    Returns: (Z_chunk, Y, X) uint16.
    GPU-accelerated when CuPy is available.
    """
    xp = cp if _GPU else np
    dtype = side_a.dtype
    max_val = np.iinfo(dtype).max

    a = xp.asarray(side_a, dtype=xp.float32)
    b = xp.asarray(side_b, dtype=xp.float32)

    n = a.shape[axis]
    w_a = _gaussian_ramp(n, xp, sigma_frac=sigma_frac)
    w_b = w_a[::-1].copy()

    if axis == 0:
        shape = (-1, 1, 1)
    elif axis == 1:
        shape = (1, -1, 1)
    else:
        shape = (1, 1, -1)

    w_a = w_a.reshape(shape)
    w_b = w_b.reshape(shape)

    fused = (w_a * a + w_b * b) / (w_a + w_b + xp.float32(1e-8))
    fused = xp.clip(fused, 0, max_val)

    return (cp.asnumpy(fused) if _GPU else fused).astype(dtype)


# ---------------------------------------------------------------------------
# Cosine taper blending
# ---------------------------------------------------------------------------

def _cosine_taper(size: int, taper_px: int) -> np.ndarray:
    w = np.ones(size, dtype=np.float32)
    t = min(taper_px, size // 4)
    if t > 0:
        ramp = (1 - np.cos(np.linspace(0, math.pi, t))) / 2
        w[:t] = ramp
        w[-t:] = ramp[::-1]
    return w


def place_tile_slab(
    canvas: np.ndarray,
    weight_canvas: np.ndarray,
    fused_slab: np.ndarray,
    tile_bbox,
    canvas_origin_x: int,
    canvas_origin_y: int,
    taper_px: int,
) -> None:
    """
    Accumulate one fused tile Z-slab onto the canvas with cosine-taper weights.
    canvas / weight_canvas: (Z_chunk, canvas_H, canvas_W) float32
    fused_slab:             (Z_chunk, tile_H, tile_W) uint16
    """
    x0 = tile_bbox.x - canvas_origin_x
    y0 = tile_bbox.y - canvas_origin_y
    tw = tile_bbox.w
    th = tile_bbox.h

    x1 = x0 + tw
    y1 = y0 + th

    # Clip to canvas bounds
    cx0 = max(x0, 0)
    cy0 = max(y0, 0)
    cx1 = min(x1, canvas.shape[2])
    cy1 = min(y1, canvas.shape[1])
    if cx0 >= cx1 or cy0 >= cy1:
        return

    # Corresponding crop in tile space
    tx0 = cx0 - x0
    ty0 = cy0 - y0
    tx1 = tx0 + (cx1 - cx0)
    ty1 = ty0 + (cy1 - cy0)

    tile_crop = fused_slab[:, ty0:ty1, tx0:tx1].astype(np.float32)

    wy = _cosine_taper(th, taper_px)[ty0:ty1]
    wx = _cosine_taper(tw, taper_px)[tx0:tx1]
    w2d = wy[:, None] * wx[None, :]  # (tile_H_crop, tile_W_crop)

    canvas[:, cy0:cy1, cx0:cx1] += tile_crop * w2d[None, :, :]
    weight_canvas[:, cy0:cy1, cx0:cx1] += w2d[None, :, :]


# ---------------------------------------------------------------------------
# ClearMap-inspired NCC + MST position refinement
# ---------------------------------------------------------------------------

def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32) - a.mean()
    b = b.astype(np.float32) - b.mean()
    denom = math.sqrt(float((a ** 2).sum()) * float((b ** 2).sum()))
    return float(np.dot(a.ravel(), b.ravel()) / denom) if denom > 1e-6 else 0.0


def refine_tile_positions(
    czi: CziFile,
    layout: dict,
    skip_refine: bool,
    ncc_threshold: float = 0.05,
    ncc_out_path=None,
) -> dict:
    """
    For adjacent tile pairs (expected overlap > 16 px), compute NCC quality
    scores and optionally sub-pixel phase-correlation shifts.
    Propagate refined positions along the MST of the quality graph.

    Returns {M_idx: BoundingRectangle-like dict with 'x', 'y', 'w', 'h'}.
    """
    n = layout["n_tiles"]
    bboxes = layout["tile_bboxes"]
    ox = layout["canvas_x"]
    oy = layout["canvas_y"]

    # Work in simple dicts for mutability
    pos = {m: {"x": bboxes[m].x, "y": bboxes[m].y,
               "w": bboxes[m].w, "h": bboxes[m].h}
           for m in range(n)}

    if n == 1 or skip_refine:
        return pos

    # Read a single mid-Z plane per tile for quality computation (T=0, C=0, I=0)
    mid_z = layout["n_z"] // 2
    print(f"  Reading mid-Z reference planes for NCC (Z={mid_z})...")
    ref_planes = {}
    for m in range(n):
        plane = read_plane(czi, M=m, I=0, C=0, T=0, Z=mid_z)
        ref_planes[m] = plane

    quality = np.zeros((n, n), dtype=np.float32)
    shifts = {}

    n_pairs = n * (n - 1) // 2
    print_every = max(1, n_pairs // 10)
    print(f"  Computing pairwise NCC for {n} tiles ({n_pairs} pairs)...")
    pair_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            pair_idx += 1
            if pair_idx % print_every == 0 or pair_idx == n_pairs:
                print(f"    NCC  {_bar(pair_idx, n_pairs)}", flush=True)
            pi, pj = pos[i], pos[j]
            # Overlap region in canvas coords
            ov_x0 = max(pi["x"], pj["x"])
            ov_x1 = min(pi["x"] + pi["w"], pj["x"] + pj["w"])
            ov_y0 = max(pi["y"], pj["y"])
            ov_y1 = min(pi["y"] + pi["h"], pj["y"] + pj["h"])

            if ov_x1 - ov_x0 < 16 or ov_y1 - ov_y0 < 16:
                continue

            # Crop each tile to the overlap region
            crop_i = ref_planes[i][
                ov_y0 - pi["y"]: ov_y1 - pi["y"],
                ov_x0 - pi["x"]: ov_x1 - pi["x"],
            ]
            crop_j = ref_planes[j][
                ov_y0 - pj["y"]: ov_y1 - pj["y"],
                ov_x0 - pj["x"]: ov_x1 - pj["x"],
            ]
            q = _ncc(crop_i, crop_j)
            quality[i, j] = quality[j, i] = q

            if q > ncc_threshold:
                shift, _, _ = phase_cross_correlation(
                    crop_i.astype(np.float32),
                    crop_j.astype(np.float32),
                    upsample_factor=10,
                )
                shifts[(i, j)] = (int(round(shift[0])), int(round(shift[1])))
                shifts[(j, i)] = (-int(round(shift[0])), -int(round(shift[1])))

    # MST on negative quality (scipy MST finds minimum, so negate for max quality)
    nonzero = quality > 0
    if not nonzero.any():
        print("  No overlapping tile pairs found — using raw stage positions.")
        return pos

    neg_q = csr_matrix(-quality * nonzero)
    mst = minimum_spanning_tree(neg_q).toarray()
    edges = list(zip(*np.where(mst != 0)))
    print(f"  MST: {len(edges)} connections selected.")

    # BFS from tile 0 to propagate refined shifts
    visited = {0}
    queue = [0]
    while queue:
        src = queue.pop(0)
        for (i, j) in edges:
            nbr = j if i == src else (i if j == src else None)
            if nbr is None or nbr in visited:
                continue
            dy, dx = shifts.get((src, nbr), (0, 0))
            pos[nbr]["x"] += dx
            pos[nbr]["y"] += dy
            visited.add(nbr)
            queue.append(nbr)

    # Write NCC quality matrix to JSON for notebook visualization
    if ncc_out_path is not None:
        ncc_payload = {
            "quality_matrix": quality.tolist(),
            "mid_z": mid_z,
            "ncc_threshold": ncc_threshold,
            "tile_positions_refined": [
                {"M": m, "x": pos[m]["x"], "y": pos[m]["y"],
                 "w": pos[m]["w"], "h": pos[m]["h"]}
                for m in sorted(pos)
            ],
        }
        with open(ncc_out_path, "w") as _f:
            json.dump(ncc_payload, _f, indent=2)
        print(f"  NCC scores written (ncc_scores.json): {ncc_out_path}")

    return pos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    czi_path = Path(args.czi)
    out_path = Path(args.out)

    print(f"Opening CZI: {czi_path} ...")
    czi = CziFile(str(czi_path))
    cache_path = out_path.parent / "czi_layout_cache.json"
    layout = get_czi_layout(czi, cache_path=cache_path)

    n_tiles = layout["n_tiles"]
    n_z = layout["n_z"]
    tile_h = layout["tile_h"]
    tile_w = layout["tile_w"]
    n_c = layout["n_c"]
    n_t = layout["n_t"]
    dual = layout["dual"]
    canvas_h = layout["canvas_h"]
    canvas_w = layout["canvas_w"]
    canvas_ox = layout["canvas_x"]
    canvas_oy = layout["canvas_y"]
    voxel_size_um = layout.get("voxel_size_um", {"x": None, "y": None, "z": None})

    # Substack range — absolute CZI Z coordinates
    z_start = args.z_start
    z_end = args.z_end if args.z_end is not None else n_z
    n_z_proc = z_end - z_start
    assert z_start < z_end, f"--z-start ({z_start}) must be less than --z-end ({z_end})"
    if z_start != 0 or args.z_end is not None:
        print(f"  Substack mode: Z={z_start}:{z_end} ({n_z_proc} planes)")

    backend = "CuPy (GPU)" if _GPU else "NumPy (CPU)"
    print(f"\nLayout:")
    print(f"  Tiles    : {n_tiles} mosaic tiles, each {tile_w}×{tile_h}×{n_z}")
    print(f"  Channels : {n_c}   Timepoints: {n_t}   Dual-side: {dual}")
    print(f"  Canvas   : {canvas_w}×{canvas_h} pixels")
    print(f"  Voxel    : x={voxel_size_um['x']} y={voxel_size_um['y']} z={voxel_size_um['z']} µm")
    print(f"  Output   : {out_path}")
    print(f"  Backend  : {backend}")
    print(f"  Z-chunk  : {args.z_chunk} planes   Workers: {args.workers}")

    if args.dump_layout:
        bboxes = layout["tile_bboxes"]
        for m in sorted(bboxes):
            b = bboxes[m]
            print(f"  M={m:2d}: x={b.x:6d} y={b.y:6d} w={b.w} h={b.h}")
        return

    # Optionally refine tile positions via NCC + MST
    print("\nRefining tile positions...")
    ncc_out = out_path.parent / "ncc_scores.json"
    refined_pos = refine_tile_positions(
        czi, layout,
        skip_refine=args.skip_refine,
        ncc_threshold=args.ncc_threshold,
        ncc_out_path=ncc_out,
    )

    # Recompute canvas extent from refined positions
    all_x1 = [p["x"] + p["w"] for p in refined_pos.values()]
    all_y1 = [p["y"] + p["h"] for p in refined_pos.values()]
    all_x0 = [p["x"] for p in refined_pos.values()]
    all_y0 = [p["y"] for p in refined_pos.values()]
    refined_ox = min(all_x0)
    refined_oy = min(all_y0)
    refined_w = max(all_x1) - refined_ox
    refined_h = max(all_y1) - refined_oy

    print(f"  Refined canvas: {refined_w}×{refined_h} (was {canvas_w}×{canvas_h})")

    # Build output zarr — TCZYX
    out_shape = (n_t, n_c, n_z_proc, refined_h, refined_w)
    out_chunk = (1, 1, args.z_chunk, min(512, refined_h), min(512, refined_w))
    print(f"\nCreating output zarr: shape={out_shape}  chunks={out_chunk}")
    out_z = zarr.open(
        str(out_path),
        mode="w",
        shape=out_shape,
        chunks=out_chunk,
        dtype=np.uint16,
    )

    n_chunks = math.ceil(n_z_proc / args.z_chunk)

    for t in range(n_t):
        for c in range(n_c):
            print(f"\n=== T={t}  C={c} ===")
            for chunk_idx in range(n_chunks):
                z0_loc = chunk_idx * args.z_chunk          # local zarr Z index
                z0_abs = z_start + z0_loc                  # absolute CZI Z
                z1_abs = min(z0_abs + args.z_chunk, z_end) # absolute CZI Z end
                nz = z1_abs - z0_abs
                print(f"  Z-chunk {chunk_idx+1}/{n_chunks}  z={z0_abs}:{z1_abs}  {_bar(chunk_idx+1, n_chunks)}", flush=True)

                canvas = np.zeros((nz, refined_h, refined_w), dtype=np.float32)
                w_canvas = np.zeros_like(canvas)

                def _process_tile(m):
                    """Read, fuse, and return (m, fused_slab) for one tile."""
                    if dual:
                        sa = read_tile_zchunk(
                            czi, m, 0, c, t, z0_abs, z1_abs, tile_h, tile_w, args.workers // 4 + 1
                        )
                        sb = read_tile_zchunk(
                            czi, m, 1, c, t, z0_abs, z1_abs, tile_h, tile_w, args.workers // 4 + 1
                        )
                        fused = fuse_sides(sa, sb, axis=args.fusion_axis, sigma_frac=args.sigma_frac)
                    else:
                        fused = read_tile_zchunk(
                            czi, m, 0, c, t, z0_abs, z1_abs, tile_h, tile_w, args.workers
                        )
                    return m, fused

                # Read + fuse tiles in parallel, then place serially (canvas not thread-safe)
                tiles_done = 0
                with ThreadPoolExecutor(max_workers=min(4, n_tiles)) as pool:
                    futs = {pool.submit(_process_tile, m): m for m in range(n_tiles)}
                    for fut in as_completed(futs):
                        m, fused_slab = fut.result()
                        tiles_done += 1
                        print(f"    tile {_bar(tiles_done, n_tiles, width=28)}", flush=True)
                        bbox_dict = refined_pos[m]

                        class _BBox:
                            pass
                        bbox = _BBox()
                        bbox.x = bbox_dict["x"]
                        bbox.y = bbox_dict["y"]
                        bbox.w = bbox_dict["w"]
                        bbox.h = bbox_dict["h"]

                        place_tile_slab(
                            canvas, w_canvas, fused_slab,
                            bbox, refined_ox, refined_oy, args.taper_px,
                        )

                # Normalize and write
                mask = w_canvas > 0
                canvas[mask] /= w_canvas[mask]
                out_z[t, c, z0_loc : z0_loc + nz] = np.clip(canvas, 0, 65535).astype(np.uint16)

    print(f"\nDone. Output: {out_path}")
    print(f"  Shape (TCZYX): {out_z.shape}")

    # Write a corrected tile manifest alongside the output
    manifest_out = out_path.parent / "tile_manifest_corrected.json"
    corrected = {
        "czi_path": str(czi_path),
        "n_tiles": n_tiles,
        "n_illumination_sides": 2 if dual else 1,
        "dual_side": dual,
        "dims": {"T": n_t, "C": n_c, "Z": n_z_proc, "Y": refined_h, "X": refined_w},
        "canvas_shape": {"H": refined_h, "W": refined_w},
        "voxel_size_um": voxel_size_um,
        "tile_positions_px": [
            {"M": m, "x": refined_pos[m]["x"] - refined_ox,
             "y": refined_pos[m]["y"] - refined_oy,
             "w": refined_pos[m]["w"], "h": refined_pos[m]["h"]}
            for m in sorted(refined_pos)
        ],
    }
    with open(manifest_out, "w") as f:
        json.dump(corrected, f, indent=2)
    print(f"  Corrected manifest: {manifest_out}")


if __name__ == "__main__":
    main()
