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

  # Stage 1 preflight (fast: 32-plane slab, no z-binning, axis-aligned neighbors only):
  python 08_stitch.py --stage register \\
      --czi <czi> --out-dir <preflight_dir> --out-zarr <placeholder> \\
      --z-slab-half 16 --reg-z-bin 1

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

def _scalar(v: Any, default: float = 0.0) -> float:
    """Coerce xarray DataArray / numpy scalar / plain value to a Python float.

    multiview-stitcher 0.1.52 stores per-edge `quality` as a (t: 1) DataArray,
    not a true scalar. Modern numpy refuses `float(arr_1d_len1)` with
    `TypeError: only 0-dimensional arrays can be converted to Python scalars`,
    so we unwrap any 1-element ndarray via `.item()` before float().
    """
    if v is None:
        return default
    if hasattr(v, "values"):  # xarray DataArray
        v = v.values
    if hasattr(v, "size") and hasattr(v, "ndim") and v.ndim >= 1:
        try:
            if v.size == 1:
                v = v.ravel()[0]
        except Exception:
            pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _extract_translation_um(param_xarray: Any) -> dict:
    """
    multiview-stitcher returns the per-tile transform as an xarray.DataArray
    of an affine matrix. The last column (excluding the homogeneous 1) is the
    translation in physical units (µm).
    """
    arr = np.asarray(param_xarray)
    # multiview-stitcher 0.1.52 wraps the affine in a leading time dim → (1, 4, 4)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
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


def _stage_translation_um(layout: dict, m: int) -> dict:
    """
    Return the canvas-absolute stage translation for tile m, in µm.
    Mirrors the {z:0, y:bb.y*sy, x:bb.x*sx} convention used in build_tile_sims.
    """
    vs = layout["voxel_size_um"]
    sx, sy = float(vs["x"]), float(vs["y"])
    bb = layout["tile_bboxes"][m]
    return {"z": 0.0, "y": float(bb.y) * sy, "x": float(bb.x) * sx}


def _flatten_pairwise(
    g_reg_computed: Any,
    layout: dict,
    metrics_qualities: dict | None = None,
) -> list[dict]:
    """
    Convert multiview-stitcher's pairwise registration graph into a flat list
    of dicts matching the diagnostics JSON schema.

    The networkx edge attributes set by mvstitch (registration.py:1449-1452) are:
      - "transform"  : xr.DataArray, affine in physical (µm) coords
      - "quality"    : xr.DataArray, scalar NCC-like score
      - "bbox"       : xr.DataArray, overlap-region corners in physical coords

    `metrics_qualities` (when provided) is mvstitch's canonical
    `nx.get_edge_attributes(g_reg_computed, "quality")` dict from
    params["pairwise_registration"]["metrics"]["qualities"] — preferred over
    walking edge attrs because mvstitch sometimes assigns the quality as a
    dask-wrapped xarray whose `.values` reads as 0 from a fresh edge attr
    walk in dependent libraries.

    Older versions used "shift" / "translation" keys — we handle both for
    backwards compatibility, but the 0.1.52 path is the canonical case.
    """
    out: list[dict] = []

    # Accept either a networkx graph (mvstitch 0.1.52) or a pre-built list/dict
    # of edges. Convert to a uniform list of (i, j, attrs) tuples.
    edges_iter = None
    if hasattr(g_reg_computed, "edges"):
        try:
            edges_iter = [
                (i, j, data) for i, j, data in g_reg_computed.edges(data=True)
            ]
        except TypeError:
            edges_iter = None
    if edges_iter is None:
        if isinstance(g_reg_computed, dict):
            edges_iter = [(d.get("i"), d.get("j"), d)
                          for d in g_reg_computed.values()]
        elif isinstance(g_reg_computed, list):
            edges_iter = [(d.get("i"), d.get("j"), d)
                          for d in g_reg_computed if isinstance(d, dict)]
        else:
            edges_iter = []

    metrics_qualities = metrics_qualities or {}

    first_logged = False
    for i_raw, j_raw, attrs in edges_iter:
        try:
            i_val = int(attrs.get("i", i_raw))
            j_val = int(attrs.get("j", j_raw))
        except (TypeError, ValueError):
            continue

        if not first_logged:
            try:
                raw_q = attrs.get("quality")
                print(
                    f"[Stage 1 DEBUG] first edge ({i_raw},{j_raw}) "
                    f"attrs_keys={list(attrs.keys())} "
                    f"quality_type={type(raw_q).__name__} "
                    f"quality_repr={raw_q!r} "
                    f"metrics_quality_present={(i_raw, j_raw) in metrics_qualities or (j_raw, i_raw) in metrics_qualities}",
                    file=sys.stderr,
                )
            except Exception as exc:
                print(f"[Stage 1 DEBUG] first-edge log failed: {exc}", file=sys.stderr)
            first_logged = True

        # Prefer the canonical metrics dict (mvstitch's own qualities export);
        # fall back to the edge attr. Try both edge orderings — networkx
        # undirected graphs may key either way.
        q_source = (
            metrics_qualities.get((i_raw, j_raw))
            or metrics_qualities.get((j_raw, i_raw))
            or attrs.get("quality")
        )
        quality = _scalar(q_source)

        # Extract shift in µm. Preferred source: the "transform" affine. Fall
        # back to legacy "shift" / "translation" keys if present.
        transform_xf = attrs.get("transform")
        if transform_xf is not None:
            try:
                shift_um = _extract_translation_um(transform_xf)
            except (ValueError, TypeError):
                shift_um = {"z": 0.0, "y": 0.0, "x": 0.0}
        else:
            shift_legacy = attrs.get("shift") or attrs.get("translation") or {}
            if isinstance(shift_legacy, dict):
                shift_um = {
                    "z": _scalar(shift_legacy.get("z")),
                    "y": _scalar(shift_legacy.get("y")),
                    "x": _scalar(shift_legacy.get("x")),
                }
            else:
                try:
                    arr = np.asarray(shift_legacy).ravel()
                    shift_um = {
                        "z": float(arr[0]),
                        "y": float(arr[1]),
                        "x": float(arr[2]),
                    }
                except Exception:
                    shift_um = {"z": 0.0, "y": 0.0, "x": 0.0}

        shift_px = _to_pixels(shift_um, layout)
        accepted = bool(attrs.get("accepted", attrs.get("success", True)))
        residual = _scalar(attrs.get("residual_px"))

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
    """Coerce groupwise_resolution_info into a fixed-schema dict.

    multiview-stitcher 0.1.52 returns:
      {
        "metrics": pd.DataFrame[mean_residual, max_residual, iteration, ...] | None,
        "edge_residuals": {it_index: {edge_tuple: np.ndarray, ...}, ...},
        "used_edges": {it_index: [edge_tuple, ...], ...}    # ← keyed by timepoint
      }
    Older or user-supplied dicts may carry flat keys (converged, rms_residual_px, …).
    We handle both shapes so the gate-check numbers are always real.
    """
    if not isinstance(groupwise, dict):
        groupwise = getattr(groupwise, "__dict__", {})

    # ── flat schema (user-supplied or legacy) ──────────────────────────────
    if "converged" in groupwise:
        return {
            "method": str(groupwise.get("method", "global_optimization")),
            "converged": bool(groupwise["converged"]),
            "rms_residual_px": float(groupwise.get("rms_residual_px", 0.0) or 0.0),
            "max_residual_px": float(groupwise.get("max_residual_px", 0.0) or 0.0),
            "n_variables": int(groupwise.get("n_variables", 0) or 0),
            "n_constraints": int(groupwise.get("n_constraints", 0) or 0),
            "solver_iterations": int(groupwise.get("solver_iterations", 0) or 0),
        }

    # ── library shape: {"metrics": DataFrame-or-None, "used_edges": {...}} ─
    df = groupwise.get("metrics")
    rms_residual = 0.0
    max_residual = 0.0
    n_iters = 0
    converged = True  # identity-transform fallback (empty graph) is "converged"
    if df is not None and hasattr(df, "iloc") and len(df) > 0:
        last = df.iloc[-1]
        rms_residual = float(last["mean_residual"]) if "mean_residual" in df.columns else 0.0
        max_residual = float(last["max_residual"]) if "max_residual" in df.columns else 0.0
        n_iters = len(df)
        converged = True  # optimiser ran to completion (abs_tol met or edge exhausted)

    # used_edges is dict-keyed-by-timepoint-index of lists of edge tuples.
    # Total constraints = union of edges across timepoints. For a single-T
    # registration this equals the number of pairwise edges that survived
    # the global solver's outer loop.
    used_edges_raw = groupwise.get("used_edges", {})
    if isinstance(used_edges_raw, dict):
        n_constraints = len({
            tuple(sorted(e))
            for edges in used_edges_raw.values()
            for e in (edges if isinstance(edges, (list, tuple, set)) else [])
        })
    elif isinstance(used_edges_raw, (list, tuple, set)):
        n_constraints = len(used_edges_raw)
    else:
        n_constraints = 0

    return {
        "method": "global_optimization",
        "converged": converged,
        "rms_residual_px": rms_residual,
        "max_residual_px": max_residual,
        "n_variables": 0,
        "n_constraints": n_constraints,
        "solver_iterations": n_iters,
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

    # Build registration_binning dict — None means no binning (production default).
    # Preflight may pass --reg-z-bin 2 to halve the per-pair Z cost without
    # affecting XY translation accuracy (stitching only needs XY shifts).
    reg_binning = None
    if args.reg_z_bin > 1:
        reg_binning = {"z": int(args.reg_z_bin)}

    pruning_method = args.pre_reg_pruning_method  # default: keep_axis_aligned

    print(
        f"[Stage 1] Calling multiview_stitcher.registration.register("
        f"groupwise_resolution_method='global_optimization', "
        f"pre_registration_pruning_method='{pruning_method}', "
        f"registration_binning={reg_binning})..."
    )
    params = registration.register(
        msims,
        reg_channel_index=0,
        transform_key="stage_metadata",
        new_transform_key="registered",
        pairwise_reg_func=registration.phase_correlation_registration,
        groupwise_resolution_method="global_optimization",
        pre_registration_pruning_method=pruning_method,
        registration_binning=reg_binning,
        post_registration_do_quality_filter=True,
        post_registration_quality_threshold=args.quality_threshold,
        return_dict=True,
    )

    # Persist transforms ----------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compose canvas-absolute world positions explicitly.
    # ----------------------------------------------------
    # In multiview-stitcher 0.1.52, registration.register() returns a dict with:
    #   "params" : list of per-view raw correction affines (one per tile, in
    #              the order of sorted(g.nodes())). The global_optimization
    #              solver anchors the reference tile to identity; other tiles
    #              get a small correction translation. These corrections are
    #              what we want to add on top of the stage-metadata grid.
    # The library *also* writes a rebased "registered" transform onto each
    # msim via set_affine_transform(..., base_transform_key='stage_metadata').
    # In theory that rebased transform equals stage @ correction (absolute world),
    # but empirically (mvstitch 0.1.52) the values come back as if the global
    # solver re-anchored the entire frame at the reference tile — making the
    # rebased "registered" transform unsafe as a canvas-absolute placement.
    # Instead, we read the raw correction from params["params"] and combine
    # it with the bbox-derived stage offset (the same value we passed into
    # build_tile_sims), producing an unambiguous absolute_um for each tile.
    raw_params = params.get("params") or []
    if not isinstance(raw_params, (list, tuple)) or len(raw_params) != layout["n_tiles"]:
        raise RuntimeError(
            f"[Stage 1] Expected params['params'] to be a list of "
            f"{layout['n_tiles']} per-view affines; got {type(raw_params).__name__} "
            f"of length {len(raw_params) if hasattr(raw_params, '__len__') else '?'}."
        )

    positions: dict[str, dict] = {}
    abs_x_px: list[float] = []
    abs_y_px: list[float] = []
    for m in range(layout["n_tiles"]):
        stage_um = _stage_translation_um(layout, m)
        correction_um = _extract_translation_um(raw_params[m])
        absolute_um = {
            "z": stage_um["z"] + correction_um["z"],
            "y": stage_um["y"] + correction_um["y"],
            "x": stage_um["x"] + correction_um["x"],
        }
        absolute_px = _to_pixels(absolute_um, layout)
        positions[str(m)] = {
            "translation_um": absolute_um,
            "translation_px": absolute_px,
            # Keep the components for traceability / debugging
            "stage_um": stage_um,
            "stage_px": _to_pixels(stage_um, layout),
            "correction_um": correction_um,
            "correction_px": _to_pixels(correction_um, layout),
        }
        abs_x_px.append(absolute_px["x"])
        abs_y_px.append(absolute_px["y"])

    with open(out_dir / "stitch_positions.json", "w") as f:
        json.dump(positions, f, indent=2)
    print(f"[Stage 1] Wrote stitch_positions.json ({len(positions)} tiles)")

    # One-line canvas summary — easy visual cross-check against layout cache.
    canvas_h = int(round(max(abs_y_px) - min(abs_y_px) + layout["tile_h"]))
    canvas_w = int(round(max(abs_x_px) - min(abs_x_px) + layout["tile_w"]))
    print(
        f"[Stage 1] Canvas from absolute positions: H={canvas_h} W={canvas_w} "
        f"(layout cache: H={layout['canvas_h']} W={layout['canvas_w']})"
    )
    # Sanity check — registration corrections are nominally a few pixels; if
    # the absolute canvas differs from the layout canvas by more than half a
    # tile (1920 px / 2 ≈ 1000 px) something has gone badly wrong.
    if abs(canvas_h - layout["canvas_h"]) > 1000 or abs(canvas_w - layout["canvas_w"]) > 1000:
        print(
            f"[Stage 1] WARNING: canvas extent from registration disagrees with "
            f"layout cache by >1000 px. Inspect stitch_positions.json corrections.",
            file=sys.stderr,
        )

    # Persist diagnostics ---------------------------------------------------
    # return_dict keys: "pairwise_registration" → {"graph": nx.Graph, ...}
    #                   "groupwise_resolution"  → {"metrics": dict, ...}
    pairwise_dict = params.get("pairwise_registration", {}) or {}
    g_reg = pairwise_dict.get("graph")
    metrics_qualities = (pairwise_dict.get("metrics") or {}).get("qualities") or {}
    groupwise_raw = params.get("groupwise_resolution", {}).get("metrics", {})
    print(
        f"[Stage 1 DEBUG] pairwise.metrics.qualities: type={type(metrics_qualities).__name__} "
        f"n_entries={len(metrics_qualities) if hasattr(metrics_qualities, '__len__') else '?'}",
        file=sys.stderr,
    )
    diagnostics = {
        "library_version": "multiview-stitcher==0.1.52",
        "n_tiles": int(layout["n_tiles"]),
        "z_slab_used": [int(z0), int(z1)],
        "voxel_size_um": {k: float(v) for k, v in layout["voxel_size_um"].items()},
        "canvas_summary": {
            "from_registration_px": {"h": canvas_h, "w": canvas_w},
            "from_layout_cache_px": {
                "h": int(layout["canvas_h"]),
                "w": int(layout["canvas_w"]),
            },
        },
        "pairwise": _flatten_pairwise(g_reg, layout, metrics_qualities) if g_reg is not None else [],
        "groupwise": _flatten_groupwise(groupwise_raw),
    }
    with open(out_dir / "stitch_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)
    n_pairs = len(diagnostics["pairwise"])
    qualities = [p["quality"] for p in diagnostics["pairwise"]]
    median_q = float(np.median(qualities)) if qualities else 0.0
    print(f"[Stage 1] Wrote stitch_diagnostics.json ({n_pairs} pairs, "
          f"median_quality={median_q:.3f}, "
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
    # mvstitch's weights.get_blending_weights divides edt_support_spacing by
    # blending_widths[dim] for every spatial dim, so 0 causes ZeroDivisionError.
    # Tiles in this dataset don't overlap in Z (XY grid only), so any positive
    # value is geometrically equivalent — clamp to 1 px to keep the math safe.
    blend_z = max(1, int(args.blend_z))
    blending = {"z": blend_z, "y": int(args.blend_y), "x": int(args.blend_x)}
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
                    # The sim's intrinsic origin (set in build_tile_sims via
                    # `translation={..., y: ty*sy, x: tx*sx}`) already encodes
                    # the stage offset. mvstitch's fusion.fuse(transform_key=
                    # "registered", ...) composes the registered transform ON
                    # TOP of that origin. So `registered` must be the
                    # CORRECTION ONLY (delta) -- writing the absolute placement
                    # here double-counts the stage offset and inflates the
                    # canvas by ~2x. correction_um is exactly the per-tile
                    # delta returned by Stage 1's mvstitch register() call.
                    _apply_registered_transform(
                        sim, positions[key].get("correction_um")
                              or {"z": 0.0, "y": 0.0, "x": 0.0}
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
    p.add_argument("--pre-reg-pruning-method", default="keep_axis_aligned",
                   help="pre_registration_pruning_method passed to registration.register(). "
                        "Use 'keep_axis_aligned' for regular grid layouts (default). "
                        "Other options: 'alternating_pattern', 'otsu_threshold_on_overlap', "
                        "'shortest_paths_overlap_weighted', or 'None' (no pruning).")
    p.add_argument("--reg-z-bin", type=int, default=1,
                   help="Z binning factor during phase-correlation registration (default: 1 = no "
                        "binning). Set to 2 for preflight: halves per-pair cost without affecting "
                        "XY translation accuracy (stitching only uses XY shifts).")
    # Stage 2 knobs (consumed in plan 02.5-03)
    p.add_argument("--z-start", type=int, default=0)
    p.add_argument("--z-end", type=int, default=None)
    p.add_argument("--z-chunk", type=int, default=64)
    p.add_argument("--blend-y", type=int, default=144,
                   help="blending_widths Y in pixels (default: 144 ≈ half tile overlap)")
    p.add_argument("--blend-x", type=int, default=144,
                   help="blending_widths X in pixels (default: 144 ≈ half tile overlap)")
    p.add_argument("--blend-z", type=int, default=1,
                   help="blending_widths Z in pixels (default: 1; tiles don't overlap "
                        "in Z so any positive value is geometrically equivalent; "
                        "must be >0 to avoid ZeroDivisionError in mvstitch weights)")
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
    # Allow passing --pre-reg-pruning-method None (string) to disable pruning
    if hasattr(args, "pre_reg_pruning_method"):
        if args.pre_reg_pruning_method in ("None", "none", ""):
            args.pre_reg_pruning_method = None
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
