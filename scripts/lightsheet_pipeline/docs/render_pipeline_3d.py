"""
Render a schematic of the fusion -> stitch -> output_ome_zarr pipeline as a
three-panel 3D figure. Run with the spatialdata_env_2 conda env (matplotlib +
numpy are enough). Output PNG: scripts/lightsheet_pipeline/docs/pipeline_3d.png

Reproduce:
    conda activate /vast/projects/BCRL_Multi_Omics/spatialdata_env_2
    python scripts/lightsheet_pipeline/docs/render_pipeline_3d.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ---- Tile / grid constants (KL018) ----
TILE_X, TILE_Y, TILE_Z = 1920, 1920, 1557
SPACING = 1632  # px, axis-aligned stage spacing
N_COLS, N_ROWS = 6, 5

# Display scaling (so axes are readable; visuals only)
DX, DY, DZ = 1.0, 1.0, 0.15


def cuboid_faces(x: float, y: float, z: float, dx: float, dy: float, dz: float):
    """Return the six face polygons of a cuboid for Poly3DCollection."""
    p = np.array(
        [
            [x, y, z],
            [x + dx, y, z],
            [x + dx, y + dy, z],
            [x, y + dy, z],
            [x, y, z + dz],
            [x + dx, y, z + dz],
            [x + dx, y + dy, z + dz],
            [x, y + dy, z + dz],
        ]
    )
    return [
        [p[0], p[1], p[2], p[3]],  # bottom
        [p[4], p[5], p[6], p[7]],  # top
        [p[0], p[1], p[5], p[4]],  # front
        [p[2], p[3], p[7], p[6]],  # back
        [p[1], p[2], p[6], p[5]],  # right
        [p[0], p[3], p[7], p[4]],  # left
    ]


def add_cuboid(ax, x, y, z, dx, dy, dz, face="#7fb3d5", edge="#1f5582", alpha=0.55, lw=0.6):
    coll = Poly3DCollection(
        cuboid_faces(x, y, z, dx, dy, dz),
        facecolor=face,
        edgecolor=edge,
        alpha=alpha,
        linewidths=lw,
    )
    ax.add_collection3d(coll)


def style_axes(ax, xlim, ylim, zlim, title):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, pad=4)


# ---------------------------------------------------------------------------
# Panel 1 — fusion: one tile, dual-side illumination merged
# ---------------------------------------------------------------------------
def render_fusion(ax):
    tw = TILE_X * DX
    th = TILE_Y * DY
    tz = TILE_Z * DZ

    # Left-illumination view (slightly offset / lighter)
    add_cuboid(ax, -tw * 0.18, 0, 0, tw, th, tz, face="#f5b041", edge="#b9770e", alpha=0.35)
    ax.text(-tw * 0.18 + tw / 2, th / 2, tz + 30, "view L", ha="center", fontsize=8, color="#b9770e")

    # Right-illumination view
    add_cuboid(ax, tw * 0.18, 0, 0, tw, th, tz, face="#a3e4d7", edge="#117864", alpha=0.35)
    ax.text(tw * 0.18 + tw / 2, th / 2, tz + 30, "view R", ha="center", fontsize=8, color="#117864")

    # Fused tile (centre, opaque)
    add_cuboid(ax, 0, -th * 1.5, 0, tw, th, tz, face="#5dade2", edge="#1f5582", alpha=0.85)
    ax.text(tw / 2, -th * 1.5 + th / 2, tz + 30, "fused tile",
            ha="center", fontsize=9, color="#1f5582", weight="bold")

    # Arrow: views → fused
    ax.quiver(tw * 0.5, th * 0.1, tz / 2, 0, -th * 1.3, 0,
              color="#34495e", arrow_length_ratio=0.12, linewidth=1.4)

    style_axes(
        ax,
        xlim=(-tw * 0.5, tw * 1.5),
        ylim=(-th * 2.2, th * 1.3),
        zlim=(0, tz * 1.4),
        title="07_direct_fuse  ·  dual-side illumination → one fused tile",
    )
    ax.view_init(elev=20, azim=-55)


# ---------------------------------------------------------------------------
# Panel 2 — stitch: 6×5 grid of fused tiles with overlap zones
# ---------------------------------------------------------------------------
def render_stitch(ax):
    tw = TILE_X * DX
    th = TILE_Y * DY
    tz = TILE_Z * DZ
    sp = SPACING * DX

    # Background canvas outline (the layout-cache canvas)
    canvas_w = (N_COLS - 1) * sp + tw
    canvas_h = (N_ROWS - 1) * sp + th
    add_cuboid(ax, -50, -50, -50, canvas_w + 100, canvas_h + 100, tz + 100,
               face="#ecf0f1", edge="#566573", alpha=0.10, lw=0.4)

    # 6×5 tiles
    for row in range(N_ROWS):
        for col in range(N_COLS):
            x = col * sp
            y = row * sp
            # Alternate slight colour shift so neighbours are distinguishable
            face = "#5dade2" if (row + col) % 2 == 0 else "#85c1e2"
            add_cuboid(ax, x, y, 0, tw, th, tz, face=face, edge="#1f5582", alpha=0.65)

    # Highlight a single overlap zone (X overlap between two horizontally adjacent tiles)
    # in red so the reader sees what "15% overlap" means visually
    ov_x0 = sp
    ov_x1 = tw
    ov_y0 = 0
    ov_y1 = th
    add_cuboid(
        ax,
        ov_x0, ov_y0, 0, ov_x1 - ov_x0, ov_y1 - ov_y0, tz,
        face="#e74c3c", edge="#922b21", alpha=0.45, lw=0.5,
    )

    # Annotate
    ax.text(canvas_w / 2, canvas_h + 220, tz + 80,
            f"{N_COLS}×{N_ROWS} = 30 tiles\n~15% XY overlap (red)",
            ha="center", fontsize=9, color="#1f5582")

    style_axes(
        ax,
        xlim=(-100, canvas_w + 250),
        ylim=(-100, canvas_h + 600),
        zlim=(-80, tz + 200),
        title="08_stitch  ·  global registration + Z-array blend",
    )
    ax.view_init(elev=22, azim=-58)


# ---------------------------------------------------------------------------
# Panel 3 — output: single merged canvas + pyramid levels
# ---------------------------------------------------------------------------
def render_output(ax):
    canvas_w = ((N_COLS - 1) * SPACING + TILE_X) * DX
    canvas_h = ((N_ROWS - 1) * SPACING + TILE_Y) * DY
    canvas_z = TILE_Z * DZ

    # Level 0 (full res) — bottom, large
    add_cuboid(ax, 0, 0, 0, canvas_w, canvas_h, canvas_z,
               face="#5dade2", edge="#1f5582", alpha=0.85)
    ax.text(canvas_w / 2, canvas_h / 2, canvas_z + 30,
            "level 0", ha="center", fontsize=8, color="#1f5582")

    # Pyramid levels: stack smaller copies in front (offset along -Y)
    factor = 1
    y_off = canvas_h + 400
    z_lift = canvas_z + 200
    for lvl in range(1, 5):
        factor *= 2
        sw = canvas_w / factor
        sh = canvas_h / factor
        sz = canvas_z / factor
        x = (canvas_w - sw) / 2
        y = y_off
        z = z_lift
        add_cuboid(ax, x, y, z, sw, sh, sz,
                   face="#85c1e2", edge="#1f5582", alpha=0.7)
        ax.text(x + sw / 2, y + sh / 2, z + sz + 20,
                f"L{lvl}", ha="center", fontsize=7, color="#1f5582")
        y_off += sh + 200

    style_axes(
        ax,
        xlim=(-200, canvas_w + 200),
        ylim=(-200, y_off + 200),
        zlim=(-100, z_lift + canvas_z * 0.7),
        title="09_output_ome_zarr  ·  5-level OME-Zarr pyramid",
    )
    ax.view_init(elev=22, azim=-58)


# ---------------------------------------------------------------------------
def main():
    fig = plt.figure(figsize=(16.5, 5.5))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    render_fusion(ax1)
    render_stitch(ax2)
    render_output(ax3)

    fig.suptitle(
        "Light-sheet pipeline: fusion → stitch → OME-Zarr pyramid",
        fontsize=13,
        y=1.02,
    )
    out_png = "/vast/projects/BCRL_Multi_Omics/scripts/lightsheet_pipeline/docs/pipeline_3d.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
