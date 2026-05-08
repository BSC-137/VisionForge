"""Camera / projection math matching VisionForge meta_pose + README convention."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def pinhole_intrinsics_match_renderer(width: int, height: int, vfov_deg: float) -> Tuple[float, float, float, float]:
    """Match vf::meta_pose::pinhole_intrinsics_match_renderer (C++)."""
    aspect = (width / height) if height else 1.0
    vfov = math.radians(vfov_deg)
    vh = 2.0 * math.tan(0.5 * vfov)
    vw = aspect * vh
    wm = float(width - 1) if width > 1 else 1.0
    hm = float(height - 1) if height > 1 else 1.0
    fx = wm / vw
    fy = hm / vh
    cx = 0.5 * wm
    cy = 0.5 * hm
    return fx, fy, cx, cy


def c2w_from_row_major_list(
    camera_extrinsics: list[float] | tuple[float, ...],
) -> np.ndarray:
    """Reshape 16 floats (row-major 4x4) to (4,4) float64; P_w = M @ P_c (homogeneous column)."""
    a = np.asarray(camera_extrinsics, dtype=np.float64).reshape(4, 4)
    return a


def unproject_pixel_ray_direction_cam(
    i: float, j: float, fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    """Unit direction in OpenCV camera coordinates (+X right, +Y down, +Z forward)."""
    dcx = (i - cx) / fx
    dcy = -(j - cy) / fy
    d = np.array([dcx, dcy, 1.0], dtype=np.float64)
    n = np.linalg.norm(d)
    if n <= 0:
        raise ValueError("degenerate ray direction")
    return d / n


def backproject_depth_to_world(
    cam_origin: np.ndarray,
    c2w: np.ndarray,
    i: float,
    j: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_m: float,
) -> np.ndarray:
    """
    Inverse of pinhole + linear radial depth stored in EXR (see passes.hpp).

    P_world = origin + depth * normalize(Rwc @ d_cam) with Rwc = c2w[:3,:3].
    """
    if depth_m <= 0 or not math.isfinite(depth_m):
        raise ValueError("depth must be positive finite")
    o = np.asarray(cam_origin, dtype=np.float64).reshape(3)
    R = c2w[:3, :3].astype(np.float64, copy=False)
    d_cam = unproject_pixel_ray_direction_cam(i, j, fx, fy, cx, cy)
    d_w = R @ d_cam
    d_w /= np.linalg.norm(d_w)
    return o + depth_m * d_w


def w2c_from_c2w(c2w: np.ndarray) -> np.ndarray:
    return np.linalg.inv(c2w.astype(np.float64))


def project_world_to_pixel(
    w2c: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    p_world: np.ndarray,
    *,
    z_eps: float = 1e-9,
) -> Optional[Tuple[float, float]]:
    """
    Continuous pixel (u, v) with v increasing downward, matching VisionForge `*_meta.json` /
    meta_pose + `Camera::get_ray` (same as tests/test_meta_pose.cpp: v uses negated Yc/Zc term).
    """
    pw = np.asarray(p_world, dtype=np.float64).reshape(3)
    h = np.array([pw[0], pw[1], pw[2], 1.0], dtype=np.float64)
    pc = w2c @ h
    X, Y, Z = float(pc[0]), float(pc[1]), float(pc[2])
    if Z <= z_eps:
        return None
    u = fx * (X / Z) + cx
    v = -fy * (Y / Z) + cy
    return u, v
