"""
08_stitch.py — Globally-optimised stitching for KL018 (Phase 2.5).

Replaces the MST + cosine-taper stitcher path in 07_direct_fuse.py with
multiview-stitcher's pairwise + global-LSQ registration and weighted-average
fusion. The dual-side illumination fusion path (fuse_sides, read_tile_zchunk)
is IMPORTED unchanged from 07_direct_fuse.py — STITCH-06.

Stages
------
Stage 1: --stage register
  Reads a ~64-plane Z-slab around mid-Z for each of 30 tiles, applies the
  validated dual-side fuse_sides path, wraps as spatial_image with stage XY
  translation in µm, and runs registration.register(..., groupwise_resolution_method=
  "global_optimization"). Writes stitch_positions.json + stitch_diagnostics.json.

Stage 2: --stage blend  (implemented in plan 02.5-03)
  Reads per-tile Z-slabs for the assigned [z_start, z_end), reapplies the
  registered transform, and runs fusion.fuse(..., fusion_func=weighted_average_fusion,
  blending_widths=...). Writes into the pre-allocated fused_direct.zarr in r+ mode.

Usage
-----
  # Stage 1 (single CPU+GPU job — GPU only for fuse_sides reference slab):
  python 08_stitch.py --stage register \\
      --czi /vast/scratch/users/kriel.j/KL018_lightsheet/KL018_85_D7_CT2AvIII_Overview.czi \\
      --out-dir /vast/scratch/users/kriel.j/KL018_lightsheet \\
      --out-zarr /vast/scratch/users/kriel.j/KL018_lightsheet/fused_direct.zarr

  # Stage 2 (SLURM array — see plan 02.5-03 + 02.5-04):
  python 08_stitch.py --stage blend \\
      --czi <czi> --out-dir <dir> --out-zarr <zarr> \\
      --z-start 0 --z-end 256 --z-chunk 64

Environment: mvstitch_env (multiview-stitcher==0.1.52, dask<2025.11.0).
"""
from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Import the UNCHANGED dual-side fusion path from 07_direct_fuse.py.
# Module name starts with a digit, so `import 07_direct_fuse` is a SyntaxError.
# We add the script directory to sys.path and use importlib.import_module.
# STITCH-06: this file must not modify fuse_sides, _gaussian_ramp, or
# read_tile_zchunk. We only READ from 07_direct_fuse here.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_direct = import_module("07_direct_fuse")
read_tile_zchunk = _direct.read_tile_zchunk
fuse_sides = _direct.fuse_sides
get_czi_layout = _direct.get_czi_layout

# Third-party (mvstitch_env)
from aicspylibczi import CziFile  # noqa: E402
from multiview_stitcher import (  # noqa: E402
    fusion,
    msi_utils,
    registration,
)
from multiview_stitcher import spatial_image_utils as si_utils  # noqa: E402


# ---------------------------------------------------------------------------
# Build per-tile spatial_image with stage XY translation in µm.
# Used by BOTH stages.
# Stage 2 (plan 02.5-03) will reuse this to set up sims before calling
# fusion.fuse(..., fusion_func=fusion.weighted_average_fusion, ...).
# ---------------------------------------------------------------------------

def build_tile_sims(
    czi: CziFile,
    layout: dict,
    z_slab_start: int,
    z_slab_end: int,
    c: int = 0,
    sigma_frac: float = 0.9,
    fusion_axis: int = 2,
    workers: int = 4,
) -> list[Any]:
    """
    For each of layout["n_tiles"] tiles, read [z_slab_start, z_slab_end) for
    channel c, apply the unchanged fuse_sides dual-side fusion, and wrap as a
    spatial_image with translation set from layout["tile_bboxes"] (the stage-
    position metadata read from the CZI header).

    Returns a list of spatial_image.SpatialImage instances, all in physical
    units (µm) via scale={"z": z_um, "y": y_um, "x": x_um}.
    """
    vs = layout["voxel_size_um"]
    sx, sy, sz = float(vs["x"]), float(vs["y"]), float(vs["z"])
    bboxes = layout["tile_bboxes"]
    sims: list[Any] = []
    for m in range(layout["n_tiles"]):
        if layout["dual"]:
            sa = read_tile_zchunk(
                czi, m, 0, c, 0, z_slab_start, z_slab_end,
                layout["tile_h"], layout["tile_w"], workers,
            )
            sb = read_tile_zchunk(
                czi, m, 1, c, 0, z_slab_start, z_slab_end,
                layout["tile_h"], layout["tile_w"], workers,
            )
            fused = fuse_sides(sa, sb, axis=fusion_axis, sigma_frac=sigma_frac)
        else:
            fused = read_tile_zchunk(
                czi, m, 0, c, 0, z_slab_start, z_slab_end,
                layout["tile_h"], layout["tile_w"], workers,
            )
        bb = bboxes[m]
        tx_px = float(bb.x)
        ty_px = float(bb.y)
        sim = si_utils.get_sim_from_array(
            fused,
            dims=("z", "y", "x"),
            scale={"z": sz, "y": sy, "x": sx},
            translation={"z": 0.0, "y": ty_px * sy, "x": tx_px * sx},
            transform_key="stage_metadata",
        )
        sims.append(sim)
    return sims


# ---------------------------------------------------------------------------
# Helpers for serialising xarray transform results to plain JSON.
# ---------------------------------------------------------------------------

def _extract_translation_um(param_xarray: Any) -> dict:
    """
    multiview-stitcher returns the per-tile transform as an xarray.DataArray
    of an affine matrix. The last column (excluding the homogeneous 1) is the
    translation in physical units (µm).
    """
    arr = np.asarray(param_xarray)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < arr.shape[0]:
        raise ValueError(f"Unexpected param shape: {arr.shape}")
    # affine 4x4 or 3x3 — last column row 0..N-2 is the translation
    n = arr.shape[0] - 1
    t = arr[:n, -1]
    if n == 3:
        return {"z": float(t[0]), "y": float(t[1]), "x": float(t[2])}
    # 2D case (n=2): map to z=0
    return {"z": 0.0, "y": float(t[0]), "x": float(t[1])}


def _to_pixels(t_um: dict, layout: dict) -> dict:
    vs = layout["voxel_size_um"]
    return {
        "z": float(t_um["z"]) / float(vs["z"]),
        "y": float(t_um["y"]) / float(vs["y"]),
        "x": float(t_um["x"]) / float(vs["x"]),
    }


def _flatten_pairwise(pairwise: Any, layout: dict) -> list[dict]:
    """
    Convert multiview-stitcher pairwise_registration_results into a flat list
    of dicts matching the diagnostics JSON schema.
    The exact return shape from registration.register depends on lib version;
    we handle both list-of-dicts and dict-of-dicts shapes defensively.
    """
    out: list[dict] = []
    items = pairwise.items() if isinstance(pairwise, dict) else enumerate(pairwise)
    for _, edge in items:
        # edge is a dict-like with keys: pair, quality, shift, success, ...
        if isinstance(edge, dict):
            i_j = edge.get("pair") or edge.get("indices") or (
                edge.get("i"), edge.get("j"),
            )
            quality = float(edge.get("quality", edge.get("ncc", 0.0)) or 0.0)
            shift = edge.get("shift") or edge.get("translation") or {}
            accepted = bool(edge.get("accepted", edge.get("success", True)))
            residual = float(edge.get("residual_px", 0.0) or 0.0)
        else:
            # tuple/object — fall back to attribute access
            i_j = getattr(edge, "pair", (None, None))
            quality = float(getattr(edge, "quality", 0.0))
            shift = getattr(edge, "shift", {})
            accepted = bool(getattr(edge, "accepted", True))
            residual = float(getattr(edge, "residual_px", 0.0))
        try:
            i_val, j_val = int(i_j[0]), int(i_j[1])
        except (TypeError, ValueError):
            continue
        # Coerce shift to dict of floats; treat as µm if keys present
        if isinstance(shift, dict):
            shift_um = {k: float(v) for k, v in shift.items()
                        if k in ("z", "y", "x")}
        else:
            try:
                arr = np.asarray(shift).ravel()
                shift_um = {"z": float(arr[0]), "y": float(arr[1]), "x": float(arr[2])}
            except Exception:
                shift_um = {"z": 0.0, "y": 0.0, "x": 0.0}
        shift_px = _to_pixels(
            {**{"z": 0.0, "y": 0.0, "x": 0.0}, **shift_um}, layout,
        )
        out.append({
            "i": i_val, "j": j_val,
            "quality": quality,
            "shift_um": shift_um,
            "shift_px": shift_px,
            "residual_px": residual,
            "accepted": accepted,
        })
    return out


def _flatten_groupwise(groupwise: Any) -> dict:
    """Coerce groupwise_resolution_info into a fixed-schema dict."""
    if not isinstance(groupwise, dict):
        groupwise = getattr(groupwise, "__dict__", {})
    return {
        "method": str(groupwise.get("method", "global_optimization")),
        "converged": bool(groupwise.get("converged", True)),
        "rms_residual_px": float(groupwise.get("rms_residual_px", 0.0) or 0.0),
        "max_residual_px": float(groupwise.get("max_residual_px", 0.0) or 0.0),
        "n_variables": int(groupwise.get("n_variables", 0) or 0),
        "n_constraints": int(groupwise.get("n_constraints", 0) or 0),
        "solver_iterations": int(groupwise.get("solver_iterations", 0) or 0),
    }


# ---------------------------------------------------------------------------
# Affine reattach helpers (Stage 2 input from Stage 1 JSON)
# ---------------------------------------------------------------------------

def _build_affine_um(translation_um: dict) -> np.ndarray:
    """
    Build a 4x4 pure-translation affine matrix in physical units (µm).
    multiview-stitcher uses (z, y, x) convention; matrix rows correspond to
    that order. Homogeneous identity for rotation/scale.
    """
    m = np.eye(4, dtype=np.float64)
    m[0, 3] = float(translation_um.get("z", 0.0))
    m[1, 3] = float(translation_um.get("y", 0.0))
    m[2, 3] = float(translation_um.get("x", 0.0))
    return m


def _apply_registered_transform(sim: Any, translation_um: dict) -> None:
    """
    Reattach the Stage-1-computed 'registered' transform onto `sim`.

    multiview-stitcher 0.1.52 exposes set_sim_affine and (in newer minor
    versions) set_sim_affine_from_array. We use whichever exists, falling
    back to a manual xarray.DataArray construction with the lib's dim names
    ("x_in", "x_out") that match the affine xarrays returned by register().
    """
    affine_np = _build_affine_um(translation_um)
    setter = getattr(si_utils, "set_sim_affine_from_array", None)
    if setter is not None:
        setter(sim, affine_np, transform_key="registered")
        return
    # Manual xarray build — same shape register() returns
    import xarray as xr
    da = xr.DataArray(
        affine_np,
        dims=("x_in", "x_out"),
        coords={"x_in": ["z", "y", "x", "1"], "x_out": ["z", "y", "x", "1"]},
    )
    si_utils.set_sim_affine(sim, da, transform_key="registered")


# ---------------------------------------------------------------------------
# Stage 1: REGISTER
# ---------------------------------------------------------------------------

def stage_register(args: argparse.Namespace, czi: CziFile, layout: dict) -> None:
    """
    Register all tiles using a Z-slab around mid-Z. Writes stitch_positions.json
    and stitch_diagnostics.json to args.out_dir.

    STITCH-01: groupwise_resolution_method='global_optimization' replaces the
    MST + BFS propagation in 07_direct_fuse.refine_tile_positions.
    STITCH-03: pairwise NCC/residuals + groupwise solver flag are persisted.
    """
    n_z = int(layout["n_z"])
    z_mid = args.z_mid if args.z_mid is not None else n_z // 2
    half = max(1, args.z_slab_half)
    z0 = max(0, z_mid - half)
    z1 = min(n_z, z_mid + half)
    print(f"[Stage 1] Registering using Z-slab [{z0}:{z1}) (mid={z_mid}, half={half})")
    print(f"[Stage 1] n_tiles={layout['n_tiles']} voxel_um={layout['voxel_size_um']}")

    sims = build_tile_sims(
        czi, layout, z0, z1,
        c=args.reg_channel,
        sigma_frac=args.sigma_frac,
        fusion_axis=args.fusion_axis,
        workers=args.workers,
    )
    msims = [msi_utils.get_msim_from_sim(s) for s in sims]

    print("[Stage 1] Calling multiview_stitcher.registration.register("
          "groupwise_resolution_method='global_optimization')...")
    params = registration.register(
        msims,
        reg_channel_index=0,
        transform_key="stage_metadata",
        new_transform_key="registered",
        pairwise_reg_func=registration.phase_correlation_registration,
        groupwise_resolution_method="global_optimization",
        post_registration_do_quality_filter=True,
        post_registration_quality_threshold=args.quality_threshold,
        return_dict=True,
    )

    # Persist transforms ----------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    positions: dict[str, dict] = {}
    for m, p in params["params"].items():
        t_um = _extract_translation_um(p)
        positions[str(m)] = {
            "translation_um": t_um,
            "translation_px": _to_pixels(t_um, layout),
        }
    with open(out_dir / "stitch_positions.json", "w") as f:
        json.dump(positions, f, indent=2)
    print(f"[Stage 1] Wrote stitch_positions.json ({len(positions)} tiles)")

    # Persist diagnostics ---------------------------------------------------
    pairwise_raw = params.get("pairwise_registration_results", [])
    groupwise_raw = params.get("groupwise_resolution_info", {})
    diagnostics = {
        "library_version": "multiview-stitcher==0.1.52",
        "n_tiles": int(layout["n_tiles"]),
        "z_slab_used": [int(z0), int(z1)],
        "voxel_size_um": {k: float(v) for k, v in layout["voxel_size_um"].items()},
        "pairwise": _flatten_pairwise(pairwise_raw, layout),
        "groupwise": _flatten_groupwise(groupwise_raw),
    }
    with open(out_dir / "stitch_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)
    n_pairs = len(diagnostics["pairwise"])
    print(f"[Stage 1] Wrote stitch_diagnostics.json ({n_pairs} pairs, "
          f"converged={diagnostics['groupwise']['converged']}, "
          f"max_residual_px={diagnostics['groupwise']['max_residual_px']:.2f})")


# ---------------------------------------------------------------------------
# Stage 2: BLEND
# ---------------------------------------------------------------------------

def stage_blend(args: argparse.Namespace, czi: CziFile, layout: dict) -> None:
    """
    Blend the assigned Z-slab [args.z_start, args.z_end) into the pre-allocated
    output zarr at absolute Z indices, using multiview-stitcher's
    weighted_average_fusion with N-D blending_widths.

    STITCH-02: blending_widths produces smoothly-normalised weights across
    tile overlaps in N-D, replacing the broken 2-D cosine-taper accumulator
    that caused brightness ramps at multi-tile junctions.
    STITCH-05: opens output zarr in mode='r+' so parallel SLURM array tasks
    can write disjoint Z-slabs without truncating the shared canvas
    (matches Phase 2 plan 02-01 contract).
    """
    import zarr

    # ---- 1. Load Stage 1 registered positions --------------------------
    pos_path = Path(args.out_dir) / "stitch_positions.json"
    if not pos_path.exists():
        raise FileNotFoundError(
            f"stitch_positions.json not found at {pos_path}. "
            "Run Stage 1 (--stage register) first."
        )
    with open(pos_path) as f:
        positions = json.load(f)
    print(f"[Stage 2] Loaded registered positions for {len(positions)} tiles")

    # ---- 2. Open output zarr in r+ (NEVER mode='w' in parallel tasks) ---
    out_z = zarr.open(args.out_zarr, mode="r+")
    print(f"[Stage 2] Output zarr: shape={out_z.shape}  chunks={out_z.chunks}")

    # ---- 3. Iterate (T, C, Z-chunk) -------------------------------------
    n_t = int(layout.get("n_t", 1))
    n_c = int(layout.get("n_c", 2))
    z_chunk = max(1, int(args.z_chunk))
    blending = {"z": 0, "y": int(args.blend_y), "x": int(args.blend_x)}
    print(f"[Stage 2] Range Z=[{args.z_start}:{args.z_end}) chunk={z_chunk} "
          f"blending_widths={blending}")
    print(f"[Stage 2] Tiles: {layout['n_tiles']} channels: {n_c}")

    for t in range(n_t):
        for c in range(n_c):
            for chunk_z0 in range(args.z_start, args.z_end, z_chunk):
                chunk_z1 = min(chunk_z0 + z_chunk, args.z_end)
                nz = chunk_z1 - chunk_z0
                print(f"[Stage 2]  t={t} c={c} Z=[{chunk_z0}:{chunk_z1}) "
                      f"({nz} planes)")

                # Build sims with stage_metadata translation (initial guess)
                sims = build_tile_sims(
                    czi, layout, chunk_z0, chunk_z1, c=c,
                    sigma_frac=args.sigma_frac,
                    fusion_axis=args.fusion_axis,
                    workers=args.workers,
                )
                # Reattach Stage 1 'registered' transform on each sim
                for m, sim in enumerate(sims):
                    key = str(m)
                    if key not in positions:
                        raise KeyError(
                            f"stitch_positions.json missing tile {m}; "
                            "Stage 1 output is incomplete."
                        )
                    _apply_registered_transform(
                        sim, positions[key]["translation_um"]
                    )

                # Fuse with feather/blending widths in N-D (STITCH-02)
                fused = fusion.fuse(
                    sims=sims,
                    transform_key="registered",
                    fusion_func=fusion.weighted_average_fusion,
                    blending_widths=blending,
                    output_stack_mode="union",
                    output_chunksize={"z": nz, "y": 512, "x": 512},
                )
                # Materialise the dask result and clip to uint16 range
                arr = np.asarray(fused.compute())
                arr = np.clip(arr, 0, 65535).astype(np.uint16)

                # Defensive: trim/pad the fused canvas to the zarr H/W if
                # output_stack_mode='union' produced a slightly different
                # extent (off-by-one is possible at sub-pixel translations).
                H, W = int(out_z.shape[-2]), int(out_z.shape[-1])
                if arr.shape[-2] != H or arr.shape[-1] != W:
                    fitted = np.zeros((nz, H, W), dtype=np.uint16)
                    hh = min(H, arr.shape[-2])
                    ww = min(W, arr.shape[-1])
                    fitted[:, :hh, :ww] = arr[:, :hh, :ww]
                    arr = fitted

                # Write at ABSOLUTE Z indices into the pre-allocated zarr
                out_z[t, c, chunk_z0:chunk_z1] = arr
                print(f"[Stage 2]    wrote out_z[{t},{c},{chunk_z0}:{chunk_z1}] "
                      f"= {arr.shape} uint16")

    print(f"[Stage 2] Done range Z=[{args.z_start}:{args.z_end}).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2.5 globally-optimised stitcher")
    p.add_argument("--stage", choices=["register", "blend"], required=True,
                   help="register=Stage 1 (write positions+diagnostics); "
                        "blend=Stage 2 (write fused zarr slabs)")
    p.add_argument("--czi", required=True,
                   help="Path to the Zeiss CZI mosaic file")
    p.add_argument("--out-dir", required=True,
                   help="Directory for stitch_positions.json, "
                        "stitch_diagnostics.json, and czi_layout_cache.json")
    p.add_argument("--out-zarr", required=True,
                   help="Path to the output fused_direct.zarr (created by "
                        "Stage 1 final block, populated by Stage 2)")
    # Stage 1 knobs
    p.add_argument("--z-mid", type=int, default=None,
                   help="Reference Z-plane (default: layout['n_z']//2)")
    p.add_argument("--z-slab-half", type=int, default=32,
                   help="Half-width of Z-slab used for registration (default: 32 → 64-plane window)")
    p.add_argument("--reg-channel", type=int, default=0,
                   help="Channel index used for registration (default: 0)")
    p.add_argument("--quality-threshold", type=float, default=0.2,
                   help="post_registration_quality_threshold for register() (default: 0.2)")
    # Stage 2 knobs (consumed in plan 02.5-03)
    p.add_argument("--z-start", type=int, default=0)
    p.add_argument("--z-end", type=int, default=None)
    p.add_argument("--z-chunk", type=int, default=64)
    p.add_argument("--blend-y", type=int, default=144,
                   help="blending_widths Y in pixels (default: 144 ≈ half tile overlap)")
    p.add_argument("--blend-x", type=int, default=144,
                   help="blending_widths X in pixels (default: 144 ≈ half tile overlap)")
    # Shared
    p.add_argument("--sigma-frac", type=float, default=0.9,
                   help="fuse_sides Gaussian sigma fraction (validated: 0.9)")
    p.add_argument("--fusion-axis", type=int, default=2,
                   help="fuse_sides axis (validated: 2 = X)")
    p.add_argument("--workers", type=int, default=4,
                   help="ThreadPool workers for read_tile_zchunk")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    czi = CziFile(args.czi)
    cache_path = Path(args.out_dir) / "czi_layout_cache.json"
    layout = get_czi_layout(czi, cache_path=cache_path)
    if args.stage == "register":
        stage_register(args, czi, layout)
    else:
        if args.z_end is None:
            args.z_end = int(layout["n_z"])
        stage_blend(args, czi, layout)


if __name__ == "__main__":
    main()
