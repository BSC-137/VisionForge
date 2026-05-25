"""Tests for visionforge_loader.webdataset_export.to_webdataset."""

from __future__ import annotations

import io
import json
import math
import tarfile
from pathlib import Path

import numpy as np
import pytest

# Skip the entire module if webdataset is not installed.
wds = pytest.importorskip("webdataset", reason="webdataset not installed; pip install webdataset")

from visionforge_loader.webdataset_export import to_webdataset  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers to build a minimal fake dataset
# ---------------------------------------------------------------------------

def _make_tiny_png_bytes(w: int = 4, h: int = 4) -> bytes:
    """Return raw PNG bytes for a small solid-colour image."""
    from PIL import Image  # type: ignore[import]

    buf = io.BytesIO()
    img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_stub_exr_bytes() -> bytes:
    """Return minimal stub bytes that stand in for a spatial EXR in the tar test."""
    # We only need non-empty bytes; the export function does a raw byte copy.
    return b"\x76\x2f\x31\x01" + b"\x00" * 28  # EXR magic + padding


def _make_stub_json(frame_id: int, split: str, w: int = 4, h: int = 4) -> bytes:
    """Return UTF-8 encoded VisionForge meta JSON bytes."""
    import math

    vfov_deg = 35.0
    aspect = float(w) / float(h)
    vh = 2.0 * math.tan(0.5 * math.radians(vfov_deg))
    vw = aspect * vh
    wm = float(w - 1) if w > 1 else 1.0
    hm = float(h - 1) if h > 1 else 1.0
    fx = wm / vw
    fy = hm / vh
    cx = 0.5 * wm
    cy = 0.5 * hm

    meta = {
        "frame_id": frame_id,
        "split": split,
        "image_width": w,
        "image_height": h,
        "camera_intrinsics": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "vfov_deg": vfov_deg,
            "skew": 0,
            "pixel_ray": "s=i/max(W-1,1),t=j/max(H-1,1)_via_Camera::get_ray",
        },
        "camera_extrinsics": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 10, 0, 0, 0, 1],
    }
    return json.dumps(meta).encode("utf-8")


def _build_fake_dataset(root: Path, frames: list[tuple[str, int]]) -> None:
    """Write fake dataset files.

    Parameters
    ----------
    frames:
        List of (split, frame_index) tuples to create.
    """
    png_bytes = _make_tiny_png_bytes()
    exr_bytes = _make_stub_exr_bytes()

    for split_name, idx in frames:
        d = root / split_name
        d.mkdir(parents=True, exist_ok=True)
        stem = f"frame_{idx:04d}"
        (d / f"{stem}.png").write_bytes(png_bytes)
        (d / f"{stem}_spatial.exr").write_bytes(exr_bytes)
        (d / f"{stem}_meta.json").write_bytes(_make_stub_json(idx, split_name))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_shard_count(tmp_path: Path) -> None:
    """3 frames with frames_per_shard=2 must produce exactly 2 shards."""
    fake_root = tmp_path / "dataset"
    _build_fake_dataset(fake_root, [("train", 0), ("train", 1), ("val", 2)])

    shards = to_webdataset(
        fake_root,
        tmp_path / "shards",
        split="both",
        frames_per_shard=2,
    )

    assert len(shards) == 2, f"Expected 2 shards, got {len(shards)}: {shards}"


def test_shard_files_exist(tmp_path: Path) -> None:
    """Written shard paths must refer to existing files."""
    fake_root = tmp_path / "dataset"
    _build_fake_dataset(fake_root, [("train", 0), ("train", 1), ("val", 2)])

    shards = to_webdataset(
        fake_root, tmp_path / "shards", frames_per_shard=2
    )

    for shard in shards:
        assert Path(shard).is_file(), f"Shard file not found: {shard}"


def test_shard_tar_contains_png_and_json(tmp_path: Path) -> None:
    """Every shard must contain at least one .png and one .json member."""
    fake_root = tmp_path / "dataset"
    _build_fake_dataset(fake_root, [("train", 0), ("train", 1), ("val", 2)])

    shards = to_webdataset(
        fake_root, tmp_path / "shards", frames_per_shard=2
    )

    for shard_path in shards:
        with tarfile.open(shard_path, "r") as tf:
            names = [m.name for m in tf.getmembers()]
        assert any(n.endswith(".png") for n in names), (
            f"No .png in shard {shard_path}: {names}"
        )
        assert any(n.endswith(".json") for n in names), (
            f"No .json in shard {shard_path}: {names}"
        )


def test_shard_tar_contains_exr(tmp_path: Path) -> None:
    """Every shard must contain at least one .exr member."""
    fake_root = tmp_path / "dataset"
    _build_fake_dataset(fake_root, [("train", 0), ("val", 1)])

    shards = to_webdataset(
        fake_root, tmp_path / "shards", frames_per_shard=10
    )

    assert len(shards) == 1
    with tarfile.open(shards[0], "r") as tf:
        names = [m.name for m in tf.getmembers()]
    assert any(n.endswith(".exr") for n in names), f"No .exr in shard: {names}"


def test_shard_frame_keys_match_stems(tmp_path: Path) -> None:
    """Tar member stems must match the expected frame stems."""
    fake_root = tmp_path / "dataset"
    _build_fake_dataset(fake_root, [("train", 3), ("train", 7)])

    shards = to_webdataset(
        fake_root, tmp_path / "shards", frames_per_shard=10
    )

    assert len(shards) == 1
    with tarfile.open(shards[0], "r") as tf:
        names = {m.name for m in tf.getmembers()}

    assert "frame_0003.png" in names, f"frame_0003.png missing: {names}"
    assert "frame_0007.json" in names, f"frame_0007.json missing: {names}"


def test_train_only_split(tmp_path: Path) -> None:
    """split='train' packs only train frames; val frames are skipped."""
    fake_root = tmp_path / "dataset"
    _build_fake_dataset(fake_root, [("train", 0), ("val", 1)])

    shards = to_webdataset(
        fake_root, tmp_path / "shards", split="train", frames_per_shard=10
    )

    assert len(shards) == 1
    with tarfile.open(shards[0], "r") as tf:
        names = {m.name for m in tf.getmembers()}

    assert any("frame_0000" in n for n in names), f"train frame missing: {names}"
    assert not any("frame_0001" in n for n in names), f"val frame leaked: {names}"


def test_empty_dataset_returns_empty_list(tmp_path: Path) -> None:
    """An empty or missing dataset returns an empty shard list without error."""
    fake_root = tmp_path / "empty_dataset"
    fake_root.mkdir()

    shards = to_webdataset(fake_root, tmp_path / "shards")
    assert shards == []


def test_optional_txt_included_when_present(tmp_path: Path) -> None:
    """When a YOLO .txt file exists, it must appear in the shard."""
    fake_root = tmp_path / "dataset"
    _build_fake_dataset(fake_root, [("train", 0)])
    # Write a stub YOLO label next to the frame.
    (fake_root / "train" / "frame_0000.txt").write_text("2 0.5 0.5 0.25 0.25\n")

    shards = to_webdataset(fake_root, tmp_path / "shards", frames_per_shard=10)

    with tarfile.open(shards[0], "r") as tf:
        names = {m.name for m in tf.getmembers()}
    assert "frame_0000.txt" in names, f".txt not packed: {names}"


def test_missing_webdataset_raises_import_error(tmp_path: Path, monkeypatch) -> None:
    """ImportError with 'pip install webdataset' hint when webdataset absent."""
    import sys
    import visionforge_loader.webdataset_export as mod

    original = sys.modules.get("webdataset")
    monkeypatch.setitem(sys.modules, "webdataset", None)  # type: ignore[arg-type]
    try:
        with pytest.raises(ImportError, match="pip install webdataset"):
            mod.to_webdataset(tmp_path, tmp_path / "shards")
    finally:
        if original is None:
            sys.modules.pop("webdataset", None)
        else:
            sys.modules["webdataset"] = original
