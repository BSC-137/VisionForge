"""EXR loader channel wiring without writing a real OpenEXR file on disk."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("torch")


def _f32_be_bytes2d(arr2d: np.ndarray) -> bytes:
    flat = arr2d.astype(np.float32).copy().tobytes(order="C")
    return flat


def _make_fake_exr(H: int, W: int, include_flow: bool = False):
    """Return (fake_exr_obj, arrays_dict) for mocking OpenEXR.InputFile."""
    import visionforge_loader.exr_io as exr_io  # noqa: F401 — ensure module loaded

    depth = np.arange(H * W, dtype=np.float32).reshape(H, W)
    inst = np.ones((H, W), dtype=np.float32) * 7.0
    nx = np.zeros((H, W), dtype=np.float32)
    ny = np.zeros((H, W), dtype=np.float32)
    nz = np.ones((H, W), dtype=np.float32)
    fx_arr = np.full((H, W), 2.5, dtype=np.float32)
    fy_arr = np.full((H, W), -1.0, dtype=np.float32)

    channels_hdr: dict = {
        "Depth": None,
        "InstanceID": None,
        "Normal.X": None,
        "Normal.Y": None,
        "Normal.Z": None,
    }
    if include_flow:
        channels_hdr["flow.x"] = None
        channels_hdr["flow.y"] = None

    header = {
        "dataWindow": MagicMock(min=MagicMock(x=0, y=0), max=MagicMock(x=W - 1, y=H - 1)),
        "channels": channels_hdr,
    }

    data_map = {
        "Depth": _f32_be_bytes2d(depth),
        "InstanceID": _f32_be_bytes2d(inst),
        "Normal.X": _f32_be_bytes2d(nx),
        "Normal.Y": _f32_be_bytes2d(ny),
        "Normal.Z": _f32_be_bytes2d(nz),
        "flow.x": _f32_be_bytes2d(fx_arr),
        "flow.y": _f32_be_bytes2d(fy_arr),
    }

    def channel(name: str, _pt) -> bytes:
        if name in data_map:
            return data_map[name]
        raise AssertionError(f"Unexpected channel: {name}")

    fake_exr = MagicMock()
    fake_exr.header = MagicMock(return_value=header)
    fake_exr.channel = MagicMock(side_effect=channel)

    arrays = {
        "depth": depth, "inst": inst,
        "nx": nx, "ny": ny, "nz": nz,
        "fx": fx_arr, "fy": fy_arr,
    }
    return fake_exr, arrays


def test_read_spatial_exr_channel_names_and_shapes() -> None:
    """Depth / InstanceID / Normal.X|Y|Z map to [H,W] and [3,H,W] with Z last axis."""
    import visionforge_loader.exr_io as exr_io

    if exr_io.OpenEXR is None:
        pytest.skip("OpenEXR not installed")

    H, W = 3, 4
    fake_exr, arrays = _make_fake_exr(H, W, include_flow=False)

    with patch.object(exr_io, "OpenEXR") as mock_mod:
        mock_mod.InputFile = MagicMock(return_value=fake_exr)
        out = exr_io.read_spatial_exr("/ignored/path.exr")

    assert out.depth.shape == (H, W)
    assert out.instance_id.shape == (H, W)
    assert out.normal.shape == (3, H, W)
    assert np.allclose(out.depth, arrays["depth"])
    assert np.allclose(out.normal[2], arrays["nz"])
    assert np.allclose(out.normal[0], arrays["nx"])
    # flow should fall back to zeros when channels absent
    assert out.flow.shape == (2, H, W)
    assert np.allclose(out.flow, 0.0)


def test_spatial_frame_has_flow_field() -> None:
    """SpatialFrame must expose a `flow` field with shape (2, H, W)."""
    import visionforge_loader.exr_io as exr_io

    if exr_io.OpenEXR is None:
        pytest.skip("OpenEXR not installed")

    H, W = 5, 7
    fake_exr, arrays = _make_fake_exr(H, W, include_flow=True)

    with patch.object(exr_io, "OpenEXR") as mock_mod:
        mock_mod.InputFile = MagicMock(return_value=fake_exr)
        frame = exr_io.read_spatial_exr("/ignored/path.exr")

    # Type must be SpatialFrame (not just the alias)
    assert isinstance(frame, exr_io.SpatialFrame)
    assert hasattr(frame, "flow"), "SpatialFrame must have a 'flow' attribute"
    assert frame.flow.shape == (2, H, W), f"Expected flow shape (2,{H},{W}), got {frame.flow.shape}"
    assert frame.flow.dtype == np.float32

    # Values must match what we injected
    assert np.allclose(frame.flow[0], arrays["fx"]), "flow.x channel mismatch"
    assert np.allclose(frame.flow[1], arrays["fy"]), "flow.y channel mismatch"


def test_spatial_exr_alias_still_works() -> None:
    """SpatialExr backward-compat alias must resolve to the same class as SpatialFrame."""
    import visionforge_loader.exr_io as exr_io

    assert exr_io.SpatialExr is exr_io.SpatialFrame, (
        "SpatialExr must remain a backward-compatible alias for SpatialFrame"
    )


def test_flow_zeros_when_channels_absent() -> None:
    """When flow channels are missing, flow must be an all-zero (2,H,W) array."""
    import visionforge_loader.exr_io as exr_io

    if exr_io.OpenEXR is None:
        pytest.skip("OpenEXR not installed")

    H, W = 4, 6
    fake_exr, _ = _make_fake_exr(H, W, include_flow=False)

    with patch.object(exr_io, "OpenEXR") as mock_mod:
        mock_mod.InputFile = MagicMock(return_value=fake_exr)
        frame = exr_io.read_spatial_exr("/ignored/path.exr")

    assert frame.flow.shape == (2, H, W)
    assert np.allclose(frame.flow, 0.0)


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
