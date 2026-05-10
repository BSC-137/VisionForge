"""EXR loader channel wiring without writing a real OpenEXR file on disk."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("torch")


def _f32_be_bytes2d(arr2d: np.ndarray) -> bytes:
    flat = arr2d.astype(np.float32).copy().tobytes(order="C")
    return flat


def test_read_spatial_exr_channel_names_and_shapes() -> None:
    """Depth / InstanceID / Normal.X|Y|Z map to [H,W] and [3,H,W] with Z last axis."""
    import visionforge_loader.exr_io as exr_io

    if exr_io.OpenEXR is None:
        pytest.skip("OpenEXR not installed")

    H, W = 3, 4
    depth = np.arange(H * W, dtype=np.float32).reshape(H, W)
    inst = np.ones((H, W), dtype=np.float32) * 7.0
    nx = np.zeros((H, W), dtype=np.float32)
    ny = np.zeros((H, W), dtype=np.float32)
    nz = np.ones((H, W), dtype=np.float32)

    header = {
        "dataWindow": MagicMock(min=MagicMock(x=0, y=0), max=MagicMock(x=W - 1, y=H - 1)),
        "channels": {"Depth": None, "InstanceID": None, "Normal.X": None, "Normal.Y": None, "Normal.Z": None},
    }

    def channel(name: str, _pt) -> bytes:
        if name == "Depth":
            return _f32_be_bytes2d(depth)
        if name == "InstanceID":
            return _f32_be_bytes2d(inst)
        if name == "Normal.X":
            return _f32_be_bytes2d(nx)
        if name == "Normal.Y":
            return _f32_be_bytes2d(ny)
        if name == "Normal.Z":
            return _f32_be_bytes2d(nz)
        raise AssertionError(name)

    fake_exr = MagicMock()
    fake_exr.header = MagicMock(return_value=header)
    fake_exr.channel = MagicMock(side_effect=channel)

    with patch.object(exr_io, "OpenEXR") as mock_mod:
        mock_mod.InputFile = MagicMock(return_value=fake_exr)
        out = exr_io.read_spatial_exr("/ignored/path.exr")

    assert out.depth.shape == (H, W)
    assert out.instance_id.shape == (H, W)
    assert out.normal.shape == (3, H, W)
    assert np.allclose(out.depth, depth)
    assert np.allclose(out.normal[2], nz)
    assert np.allclose(out.normal[0], nx)


def test_pinhole_matches_meta_pose_cpp_numeric() -> None:
    """Cross-check one W,H,vfov tuple against C++ test_meta_pose conventions (Python geometry)."""
    from visionforge_loader.geometry import pinhole_intrinsics_match_renderer

    W, H = 64, 36
    vfov = 35.0
    fx, fy, cx, cy = pinhole_intrinsics_match_renderer(W, H, vfov)
    aspect = float(W) / float(H)
    vh = 2.0 * np.tan(0.5 * np.radians(vfov))
    vw = aspect * vh
    assert pytest.approx(float(fx), rel=0, abs=1e-6) == float(W - 1) / vw
    assert pytest.approx(float(fy), rel=0, abs=1e-6) == float(H - 1) / vh
    assert pytest.approx(cx, abs=1e-6) == 0.5 * float(W - 1)
    assert pytest.approx(cy, abs=1e-6) == 0.5 * float(H - 1)
