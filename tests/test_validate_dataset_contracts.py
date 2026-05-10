"""
Contract tests for ``scripts/validate_dataset.py`` (synthetic tree under ``tmp_path``).

Run from repo root with pytest installed::

    python3 -m pytest tests/test_validate_dataset_contracts.py -q
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import validate_dataset as vd  # noqa: E402

# 1×1 PNG (IHDR parseable by validate_dataset.read_png_dimensions)
_PNG_1x1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _identity_c2w_flat() -> list[float]:
    return [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def _minimal_meta(w: int, h: int) -> dict:
    return {
        "frame_id": 0,
        "split": "train",
        "image_width": w,
        "image_height": h,
        "camera_extrinsics": _identity_c2w_flat(),
        "camera_intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        "convention": "opencv",
        "render": {},
        "camera": {},
        "sun": {},
        "objects": [
            {
                "instance_id": 1,
                "local_to_world_row_major": _identity_c2w_flat(),
                "position": [0.0, 0.0, 0.0],
                "rotation_deg": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            }
        ],
    }


def _write_good_train_frame(train: Path, stem: str, meta: dict) -> None:
    png = train / f"{stem}.png"
    png.write_bytes(_PNG_1x1)
    # Full-bleed box so pixel-area and fractional checks succeed on a 1×1 PNG.
    (train / f"{stem}.txt").write_text("2 0.5 0.5 1.0 1.0\n", encoding="utf-8")
    with (train / f"{stem}_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")


def _layout_good_flat(tmp: Path) -> None:
    """Single-frame forge-style layout under train/, COCO at root."""
    root = tmp / "ds"
    root.mkdir()
    train = root / "train"
    train.mkdir()
    meta = _minimal_meta(1, 1)
    _write_good_train_frame(train, "frame_0000", meta)

    coco = {
        "images": [{"id": 1, "file_name": "frame_0000.png", "width": 1, "height": 1}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 2,
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "area": 1.0,
            }
        ],
        "categories": [{"id": 2, "name": "t"}],
    }
    (root / "annotations_coco.json").write_text(json.dumps(coco), encoding="utf-8")


def test_run_validation_cli_passes_minimal_good(tmp_path: Path) -> None:
    _layout_good_flat(tmp_path)
    root = tmp_path / "ds"
    report, code = vd.run_validation_cli(["--dataset-root", str(root), "--split", "all"])
    assert code == 0
    errs = [i for i in report.issues if i.severity == "error"]
    assert errs == []


def test_missing_coco_fails(tmp_path: Path) -> None:
    _layout_good_flat(tmp_path)
    root = tmp_path / "ds"
    (root / "annotations_coco.json").unlink()
    report, code = vd.run_validation_cli(["--dataset-root", str(root), "--split", "all"])
    assert code == 1
    codes = [i.code for i in report.issues if i.severity == "error"]
    assert "coco_not_found" in codes


def test_check_meta_rejects_bad_json(tmp_path: Path) -> None:
    _layout_good_flat(tmp_path)
    root = tmp_path / "ds"
    bad = {"image_width": 1}
    with (root / "train" / "frame_0000_meta.json").open("w", encoding="utf-8") as f:
        json.dump(bad, f)

    report, code = vd.run_validation_cli(["--dataset-root", str(root), "--split", "all", "--check-meta"])
    assert code == 1
    assert any(i.code == "meta_schema" for i in report.issues if i.severity == "error")


def test_yolo_bad_line_errors(tmp_path: Path) -> None:
    _layout_good_flat(tmp_path)
    root = tmp_path / "ds"
    (root / "train" / "frame_0000.txt").write_text("not five tokens here\n", encoding="utf-8")

    report, code = vd.run_validation_cli(["--dataset-root", str(root), "--split", "all"])
    assert code == 1
    assert any(i.code == "yolo_bad_line" for i in report.issues if i.severity == "error")


def test_extra_png_warns_and_strict_fails(tmp_path: Path) -> None:
    """On-disk RGB not listed in COCO → catalog/disks disagree; ``--strict`` promotes to failure."""
    _layout_good_flat(tmp_path)
    root = tmp_path / "ds"
    train = root / "train"
    (train / "frame_0001.png").write_bytes(_PNG_1x1)
    (train / "frame_0001.txt").write_text("", encoding="utf-8")

    report, code = vd.run_validation_cli(["--dataset-root", str(root), "--split", "all"])
    assert code == 0
    assert any(i.code == "png_not_in_coco" for i in report.issues)

    _, code_s = vd.run_validation_cli(["--dataset-root", str(root), "--split", "all", "--strict"])
    assert code_s == 1
