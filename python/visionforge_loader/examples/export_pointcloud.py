"""Export a world-space point cloud from a VisionForge depth + RGB frame as ASCII PLY.

Usage (from python/visionforge_loader with PYTHONPATH=.):
    python3 examples/export_pointcloud.py --dataset-root /path/to/dataset --out scene.ply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from visionforge_loader.dataset import VisionForgeDataset


def backproject_depth_batch(
    depth: np.ndarray,
    c2w: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    eps: float = 1e-3,
) -> np.ndarray:
    """
    Vectorised back-projection of all valid depth pixels to world-space XYZ.

    depth  : [H, W] float32/64  — linear ray-distance from camera origin (EXR convention)
    c2w    : [4, 4] float64     — camera-to-world (P_w = M @ P_c, row-major)
    Returns: [N, 3] float64 xyz for valid pixels (finite and > eps).

    Camera convention: +X right, +Y down, +Z forward (OpenCV / VisionForge meta_pose).
    """
    H, W = depth.shape

    # Pixel coordinate grids: jj = row index, ii = column index
    jj, ii = np.meshgrid(
        np.arange(H, dtype=np.float64),
        np.arange(W, dtype=np.float64),
        indexing="ij",
    )

    valid = np.isfinite(depth) & (depth > eps)
    depth_flat = depth[valid].astype(np.float64)
    ii_flat = ii[valid]
    jj_flat = jj[valid]

    # Unproject to unit direction in camera space (matches unproject_pixel_ray_direction_cam)
    dcx = (ii_flat - cx) / fx        # column → +X
    dcy = -(jj_flat - cy) / fy       # row    → +Y (negated: row increases downward = +Y)
    d_cam = np.stack([dcx, dcy, np.ones_like(dcx)], axis=-1)  # [N, 3]
    d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True)

    R = c2w[:3, :3].astype(np.float64, copy=False)
    t = c2w[:3, 3].astype(np.float64, copy=False)
    d_world = d_cam @ R.T            # [N, 3] — rotate ray into world space
    pts = t + d_world * depth_flat[:, None]

    return pts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Back-project a VisionForge depth frame to a world-space point cloud (ASCII PLY)."
    )
    parser.add_argument("--dataset-root", required=True, help="Path to forge export root (contains train/ val/)")
    parser.add_argument("--split", default="train", choices=["train", "val"], help="Dataset split (default: train)")
    parser.add_argument("--frame-index", type=int, default=0, help="Index into VisionForgeDataset (default: 0)")
    parser.add_argument("--max-points", type=int, default=50_000, help="Maximum points to write; random subsample if exceeded (default: 50000)")
    parser.add_argument("--out", default="pointcloud.ply", help="Output PLY path (default: pointcloud.ply)")
    args = parser.parse_args()

    dataset = VisionForgeDataset(args.dataset_root, split=args.split)
    item = dataset[args.frame_index]

    depth_np = item["depth"].numpy()                      # [H, W] float32
    rgb_np = item["rgb"].numpy()                          # [3, H, W] float32 in [0, 1]
    meta = item["meta"]
    c2w_np = meta.c2w.numpy().astype(np.float64)          # [4, 4]
    cam_origin = c2w_np[:3, 3]

    pts = backproject_depth_batch(depth_np, c2w_np, meta.fx, meta.fy, meta.cx, meta.cy)

    # Gather RGB for the same valid pixels (same mask used inside backproject_depth_batch)
    valid = np.isfinite(depth_np) & (depth_np > 1e-3)
    rgb_hwc = rgb_np.transpose(1, 2, 0)   # [H, W, 3]
    rgb_valid = rgb_hwc[valid]             # [N, 3] float32 in [0, 1]

    # Subsample if needed
    N = len(pts)
    if N > args.max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(N, size=args.max_points, replace=False)
        pts = pts[idx]
        rgb_valid = rgb_valid[idx]
        N = args.max_points

    # Write ASCII PLY
    out_path = Path(args.out)
    rgb_u8 = np.clip(rgb_valid * 255.0, 0, 255).astype(np.uint8)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            x, y, z = pts[i]
            r, g, b = rgb_u8[i]
            f.write(f"{x} {y} {z} {r} {g} {b}\n")

    print(f"wrote {N} points to {args.out}")

    valid_depths = depth_np[valid]
    print(
        f"frame={meta.stem}  "
        f"origin=({cam_origin[0]:.3f}, {cam_origin[1]:.3f}, {cam_origin[2]:.3f})  "
        f"depth_range=[{float(valid_depths.min()):.3f}, {float(valid_depths.max()):.3f}]",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
