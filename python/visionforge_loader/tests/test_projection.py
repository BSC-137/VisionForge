from __future__ import annotations

import math

import numpy as np
import pytest

from visionforge_loader.geometry import (
    backproject_depth_to_world,
    c2w_from_row_major_list,
    pinhole_intrinsics_match_renderer,
    project_world_to_pixel,
    unproject_pixel_ray_direction_cam,
    w2c_from_c2w,
)

pytest.importorskip("torch")


def test_a_principal_ray_roundtrip_float64():
    """Synthetic: point along central ray at distance d projects back to (cx, cy)."""
    W, H = 64, 36
    vfov = 35.0
    fx, fy, cx, cy = pinhole_intrinsics_match_renderer(W, H, vfov)
    # Camera at origin looking toward +Z, y down, x right — simple pose:
    # Use c2w = I with camera convention +Z forward (OpenCV): actually identity means
    # world equals camera iff we use standard OpenCV. Here emulate README: origin (0,0,0),
    # forward +Z, x right, y down:
    c2w = np.eye(4, dtype=np.float64)
    w2c = w2c_from_c2w(c2w)
    origin = np.array([0.0, 0.0, 0.0])
    ic, jc = cx, cy
    d = 3.7
    pw = backproject_depth_to_world(origin, c2w, ic, jc, fx, fy, cx, cy, d)
    uv = project_world_to_pixel(w2c, fx, fy, cx, cy, pw)
    assert uv is not None
    err = math.hypot(uv[0] - ic, uv[1] - jc)
    assert err < 0.5


def test_ray_direction_matches_meta_pose_snippet():
    """Match test_meta_pose.cpp reconstruction for arbitrary c2w (translation only)."""
    W, H = 64, 36
    vfov = 35.0
    fx, fy, cx, cy = pinhole_intrinsics_match_renderer(W, H, vfov)
    # Identity rotation, camera at (11,6,18) — only test unproject direction vs row-major multiply
    c2w = np.array(
        [[1, 0, 0, 11.0], [0, 1, 0, 6.0], [0, 0, 1, 18.0], [0, 0, 0, 1.0]],
        dtype=np.float64,
    )
    s, t = 0.12, 0.11
    i = s * float(W - 1)
    j = t * float(H - 1)
    dcx = (i - cx) / fx
    dcy = -(j - cy) / fy
    dcz = 1.0
    wx = c2w[0, 0] * dcx + c2w[0, 1] * dcy + c2w[0, 2] * dcz
    wy = c2w[1, 0] * dcx + c2w[1, 1] * dcy + c2w[1, 2] * dcz
    wz = c2w[2, 0] * dcx + c2w[2, 1] * dcy + c2w[2, 2] * dcz
    d1 = np.array([wx, wy, wz], dtype=np.float64)
    d1 /= np.linalg.norm(d1)
    d2_cam = unproject_pixel_ray_direction_cam(i, j, fx, fy, cx, cy)
    d2 = c2w[:3, :3] @ d2_cam
    d2 /= np.linalg.norm(d2)
    assert np.dot(d1, d2) > 1.0 - 1e-9


def test_b_depth_buffer_roundtrip():
    W, H = 32, 24
    vfov = 40.0
    fx, fy, cx, cy = pinhole_intrinsics_match_renderer(W, H, vfov)
    rng = np.random.default_rng(1)
    A = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(A)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = Q
    c2w[:3, 3] = rng.standard_normal(3)
    w2c = w2c_from_c2w(c2w)
    origin = c2w[:3, 3].copy()
    for (ii, jj) in [(5, 7), (20, 12), (28, 3)]:
        dep = 6.5
        pw = backproject_depth_to_world(origin, c2w, float(ii), float(jj), fx, fy, cx, cy, dep)
        uv = project_world_to_pixel(w2c, fx, fy, cx, cy, pw)
        assert uv is not None
        err = math.hypot(uv[0] - ii, uv[1] - jj)
        assert err < 1.0


def test_c_instance_id_float32_roundtrip():
    """EXR stores InstanceID as float32; values up to 2**24-1 round-trip exactly."""
    for v in [1.0, 12345.0, 1048576.0, 16777215.0]:
        f = np.float32(v)
        assert int(round(float(f))) == int(v)


def test_meta_c2w_list_parse():
    flat = [
        0.93611686,
        -0.11797813,
        -0.32849213,
        2.1,
        0.35160725,
        0.31408869,
        0.88233389,
        1.7,
        0.00346020,
        -0.94244645,
        0.33437231,
        4.2,
        0,
        0,
        0,
        1,
    ]
    M = c2w_from_row_major_list(flat)
    assert M.shape == (4, 4)
    assert M[0, 3] == 2.1
