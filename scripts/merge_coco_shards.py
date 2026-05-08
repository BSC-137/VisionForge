#!/usr/bin/env python3
"""Merge VisionForge per-shard COCO JSON files into one dataset (renumber image/annotation ids)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def merge_coco_shards(inputs: list[Path]) -> dict[str, Any]:
    next_image_id = 1
    next_ann_id = 1
    all_images: list[dict[str, Any]] = []
    all_anns: list[dict[str, Any]] = []
    categories: dict[int, dict[str, Any]] = {}

    for path in inputs:
        doc = json.loads(path.read_text(encoding="utf-8"))
        old_to_new_img: dict[int, int] = {}
        for im in doc.get("images", []):
            old_id = int(im["id"])
            im = dict(im)
            im["id"] = next_image_id
            old_to_new_img[old_id] = next_image_id
            next_image_id += 1
            all_images.append(im)
        for ann in doc.get("annotations", []):
            ann = dict(ann)
            ann["id"] = next_ann_id
            next_ann_id += 1
            ann["image_id"] = old_to_new_img[int(ann["image_id"])]
            all_anns.append(ann)
        for cat in doc.get("categories", []):
            cid = int(cat["id"])
            if cid not in categories:
                categories[cid] = dict(cat)

    cats_sorted = sorted(categories.values(), key=lambda c: int(c["id"]))
    return {"images": all_images, "annotations": all_anns, "categories": cats_sorted}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("shards", nargs="+", type=Path)
    args = p.parse_args()
    merged = merge_coco_shards(args.shards)
    args.output.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
