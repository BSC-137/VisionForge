"""Tests for backproject_depth_batch — no real EXR files required."""

from __future__ import annotations

import numpy as np
import pytest

from examples.export_pointcloud import backproject_depth_batch


def _identity_c2w() -> np.ndarray:
    """Camera at world origin looking along +Z (identity pose)."""
    return np.eye(4, dtype=np.float64)


def test_output_shape_nx3() -> None:
    """pts shape must be (N, 3) for any valid depth array."""
    H, W = 4, 4
    depth = np.ones((H, W), dtype=np.float32)
    pts = backproject_depth_batch(depth, _identity_c2w(), fx=2.0, fy=2.0, cx=1.5, cy=1.5)
    assert pts.ndim == 2
    assert pts.shape[1] == 3
    assert pts.shape[0] <= H * W


def test_point_count_at_most_hw() -> None:
    """Number of returned points must never exceed H*W."""
    H, W = 8, 6
    depth = np.random.default_rng(42).uniform(0.5, 5.0, (H, W)).astype(np.float32)
    pts = backproject_depth_batch(depth, _identity_c2w(), fx=4.0, fy=4.0, cx=2.5, cy=3.5)
    assert pts.shape[0] <= H * W


def test_all_invalid_depth_returns_empty() -> None:
    """All-negative depth produces an empty (0, 3) array."""
    H, W = 4, 4
    depth = np.full((H, W), -1.0, dtype=np.float32)
    pts = backproject_depth_batch(depth, _identity_c2w(), fx=2.0, fy=2.0, cx=1.5, cy=1.5)
    assert pts.shape == (0, 3)


def test_center_pixel_identity_c2w() -> None:
    """Center pixel (i=cx, j=cy) with identity c2w back-projects to (0, 0, depth)."""
    H, W = 5, 5   # odd so (2, 2) is an exact integer center
    depth_val = 3.7
    depth = np.full((H, W), depth_val, dtype=np.float32)
    cx = (W - 1) / 2.0   # 2.0
    cy = (H - 1) / 2.0   # 2.0
    fx = fy = 4.0

    pts = backproject_depth_batch(depth, _identity_c2w(), fx=fx, fy=fy, cx=cx, cy=cy)

    # Identify the point from row=2, col=2 (center) by reconstructing which index it is.
    # With identity c2w and center pixel, d_cam = [0, 0, 1] → world_pt = (0, 0, depth_val).
    # Find any point close to (0, 0, depth_val):
    dists = np.linalg.norm(pts - np.array([0.0, 0.0, depth_val]), axis=-1)
    assert dists.min() < 1e-6, (
        f"No point near (0, 0, {depth_val}); closest distance={dists.min():.2e}"
    )


def test_eps_filtering() -> None:
    """Pixels at or below eps must be excluded from the output."""
    H, W = 4, 4
    depth = np.full((H, W), 1e-4, dtype=np.float32)  # below default eps=1e-3
    pts = backproject_depth_batch(depth, _identity_c2w(), fx=2.0, fy=2.0, cx=1.5, cy=1.5)
    assert pts.shape[0] == 0

    depth_above = np.full((H, W), 2e-3, dtype=np.float32)  # above eps
    pts_above = backproject_depth_batch(depth_above, _identity_c2w(), fx=2.0, fy=2.0, cx=1.5, cy=1.5)
    assert pts_above.shape[0] == H * W


def test_non_identity_translation() -> None:
    """Camera translated to (1, 2, 3): center ray result should be offset by that translation."""
    H, W = 5, 5
    depth_val = 4.0
    depth = np.full((H, W), depth_val, dtype=np.float32)
    cx, cy, fx, fy = 2.0, 2.0, 3.0, 3.0

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 3] = [1.0, 2.0, 3.0]

    pts = backproject_depth_batch(depth, c2w, fx=fx, fy=fy, cx=cx, cy=cy)

    # Center pixel: d_cam=[0,0,1] → d_world=[0,0,1] → pt = (1, 2, 3) + 4*(0,0,1) = (1, 2, 7)
    dists = np.linalg.norm(pts - np.array([1.0, 2.0, 7.0]), axis=-1)
    assert dists.min() < 1e-6, (
        f"Expected center point at (1, 2, 7); closest dist={dists.min():.2e}"
    )
