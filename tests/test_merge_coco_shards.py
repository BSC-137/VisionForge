"""
Contract tests for ``scripts/merge_coco_shards.py`` (synthetic JSON under ``tmp_path``).

Run from repo root::

    python3 -m pytest tests/test_merge_coco_shards.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import merge_coco_shards as mcs  # noqa: E402


def _write_shard(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_merge_two_shards_image_count(tmp_path: Path) -> None:
    shard0 = _write_shard(
        tmp_path / "shard0.json",
        {
            "images": [{"id": 1, "file_name": "frame_0000.png", "width": 320, "height": 180}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 2,
                    "bbox": [10, 10, 50, 50],
                    "area": 2500,
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": 2, "name": "test_cube"}],
        },
    )
    shard1 = _write_shard(
        tmp_path / "shard1.json",
        {
            "images": [{"id": 1, "file_name": "frame_0002.png", "width": 320, "height": 180}],
            "annotations": [],
            "categories": [{"id": 2, "name": "test_cube"}],
        },
    )

    merged = mcs.merge_coco_shards([shard0, shard1])

    assert len(merged["images"]) == 2
    assert len(merged["annotations"]) == 1
    assert len(merged["categories"]) == 1
    assert merged["images"][0]["id"] != merged["images"][1]["id"]


def test_merge_renumbers_annotation_image_ids(tmp_path: Path) -> None:
    """Both shards have image id=1; after merge, ids must be distinct and annotations remapped."""
    ann_shard = {
        "images": [{"id": 1, "file_name": "frame_0000.png", "width": 320, "height": 180}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 2, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0}
        ],
        "categories": [{"id": 2, "name": "obj"}],
    }
    shard0 = _write_shard(tmp_path / "s0.json", ann_shard)
    shard1 = _write_shard(tmp_path / "s1.json", ann_shard)

    merged = mcs.merge_coco_shards([shard0, shard1])

    img_ids = [img["id"] for img in merged["images"]]
    assert len(set(img_ids)) == 2, "image ids must be unique after renumbering"

    ann_img_ids = {ann["image_id"] for ann in merged["annotations"]}
    assert ann_img_ids <= set(img_ids), "every annotation image_id must reference a merged image id"
    # Each shard had one annotation → each must map to its own image
    assert len(merged["annotations"]) == 2
    assert merged["annotations"][0]["image_id"] != merged["annotations"][1]["image_id"]


def test_merge_empty_shard(tmp_path: Path) -> None:
    """One populated shard + one empty shard: result equals the populated shard's data."""
    populated = _write_shard(
        tmp_path / "pop.json",
        {
            "images": [{"id": 5, "file_name": "frame_0005.png", "width": 64, "height": 64}],
            "annotations": [
                {"id": 3, "image_id": 5, "category_id": 1, "bbox": [0, 0, 32, 32], "area": 1024, "iscrowd": 0}
            ],
            "categories": [{"id": 1, "name": "box"}],
        },
    )
    empty = _write_shard(
        tmp_path / "empty.json",
        {"images": [], "annotations": [], "categories": []},
    )

    merged = mcs.merge_coco_shards([populated, empty])

    assert len(merged["images"]) == 1
    assert len(merged["annotations"]) == 1
    assert merged["images"][0]["file_name"] == "frame_0005.png"
    assert merged["annotations"][0]["image_id"] == merged["images"][0]["id"]
