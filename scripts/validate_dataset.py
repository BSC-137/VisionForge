#!/usr/bin/env python3
"""
VisionForge dataset directory validator (post-render quality gate).

Requires Python 3.10+. Uses stdlib only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# VisionForge aggregates (repo README): forge vs scenario
DEFAULT_COCO_CANDIDATES = ("annotations_coco.json", "scenario_coco.json")

# RGB stems produced by forge (frame_XXXX) or scenario (sfrm_XXXX)
PNG_PATTERN = re.compile(r"^(?:frame|sfrm)_\d{4}\.png$")


@dataclass
class Issue:
    severity: str  # "error" | "warning"
    code: str
    message: str
    path: str | None = None


@dataclass
class ValidationReport:
    dataset_root: str
    split_filter: str
    coco_path: str | None
    manifest_path: str | None
    scan_directories: list[str]
    issues: list[Issue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def add(self, issue: Issue) -> None:
        self.issues.append(issue)


def _is_integral_number(x: Any) -> bool:
    if isinstance(x, bool):  # bool is int subclass
        return False
    if isinstance(x, int):
        return True
    if isinstance(x, float):
        return math.isfinite(x) and float(int(x)) == x
    return False


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    epilog = """
COCO discovery:
  If neither --coco nor an unambiguous default applies, validation exits with an error.
  Default search under DATASET_ROOT for exactly one of:
    annotations_coco.json   (forge aggregate)
    scenario_coco.json      (scenario aggregate)
  If both exist, pass --coco explicitly.

YOLO pairing:
  For each RGB stem matching frame_XXXX.png or sfrm_XXXX.png, VisionForge writes YOLO labels as stem.txt .
  An empty .txt file means zero detected boxes for that frame (allowed).

BBox math:
  COCO annotations use xywh in pixels; area_px = width * height.
  YOLO lines store normalized cx,cy,w,h in [0,1]; fractional image coverage is area_frac = w * h
  (independent of image dimensions when comparing to --max-bbox-area-frac).
"""
    p = argparse.ArgumentParser(
        description="Validate VisionForge dataset layout (COCO, YOLO, PNG sidecars).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    p.add_argument("--dataset-root", required=True, type=Path, help="Exported dataset root directory.")
    p.add_argument(
        "--split",
        choices=("train", "val", "all"),
        default="all",
        help="Restrict scanned folders when train/ and val/ exist (default: all). Ignored for flat layouts.",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="manifest.json path (default: <dataset-root>/manifest.json). Parsed when present.",
    )
    p.add_argument(
        "--coco",
        type=Path,
        default=None,
        help="Explicit COCO JSON path. Overrides automatic discovery.",
    )
    p.add_argument(
        "--max-bbox-area-frac",
        type=float,
        default=0.95,
        help="Flag boxes covering >= this fraction of the image (COCO: (w*h)/(W*H); YOLO: w*h norms).",
    )
    p.add_argument(
        "--min-bbox-area-pixels",
        type=int,
        default=1,
        help="Minimum COCO w*h (pixels^2) and minimum YOLO pixel-area estimate (w_norm*h_norm*W*H).",
    )
    p.add_argument(
        "--min-image-bytes",
        type=int,
        default=1,
        help="Minimum PNG file size on disk (bytes).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures (non-zero exit).",
    )
    p.add_argument("--check-meta", action="store_true", help="Require frame/sfrm *_meta.json sidecars + minimal schema.")
    p.add_argument("--json-report", type=Path, default=None, help="Write structured JSON summary for CI artifacts.")
    return p.parse_args(argv)


def discover_scan_directories(root: Path, split: str) -> tuple[list[Path], list[Issue]]:
    """Directories containing frame PNGs / labels."""
    issues: list[Issue] = []
    train = root / "train"
    val = root / "val"
    has_train = train.is_dir()
    has_val = val.is_dir()

    if has_train or has_val:
        if split == "all":
            dirs = [d for d in (train, val) if d.is_dir()]
        elif split == "train":
            if not has_train:
                issues.append(
                    Issue(
                        "error",
                        "split_missing",
                        "--split train requested but train/ does not exist under dataset root.",
                        str(train),
                    )
                )
                return [], issues
            dirs = [train]
        else:  # val
            if not has_val:
                issues.append(
                    Issue(
                        "error",
                        "split_missing",
                        "--split val requested but val/ does not exist under dataset root.",
                        str(val),
                    )
                )
                return [], issues
            dirs = [val]
        return dirs, issues

    # Flat layout
    return [root], issues


def discover_coco_path(root: Path, explicit: Path | None) -> tuple[Path | None, list[Issue]]:
    issues: list[Issue] = []
    if explicit is not None:
        p = explicit
        if not p.is_file():
            issues.append(Issue("error", "coco_missing", "Explicit --coco path is not a file.", str(p)))
            return None, issues
        return p.resolve(), issues

    hits = []
    for name in DEFAULT_COCO_CANDIDATES:
        cand = root / name
        if cand.is_file():
            hits.append(cand.resolve())

    if len(hits) == 0:
        issues.append(
            Issue(
                "error",
                "coco_not_found",
                "No default COCO aggregate found. Expected exactly one of: "
                + ", ".join(DEFAULT_COCO_CANDIDATES)
                + " under dataset root, or pass --coco PATH.",
                str(root),
            )
        )
        return None, issues
    if len(hits) > 1:
        names = [h.name for h in hits]
        issues.append(
            Issue(
                "error",
                "coco_ambiguous",
                "Multiple COCO aggregates exist (" + ", ".join(names) + "). Pass --coco PATH explicitly.",
                str(root),
            )
        )
        return None, issues
    return hits[0], issues


def read_png_dimensions(path: Path) -> tuple[int, int] | None:
    """Parse IHDR width/height without Pillow."""
    try:
        with path.open("rb") as f:
            sig = f.read(8)
            if sig != b"\x89PNG\r\n\x1a\n":
                return None
            while True:
                chunk_len_bytes = f.read(4)
                if len(chunk_len_bytes) < 4:
                    return None
                chunk_len = int.from_bytes(chunk_len_bytes, "big")
                ctype = f.read(4)
                if len(ctype) < 4:
                    return None
                data = f.read(chunk_len)
                f.read(4)  # CRC
                if ctype == b"IHDR":
                    if len(data) < 8:
                        return None
                    w = int.from_bytes(data[0:4], "big")
                    h = int.from_bytes(data[4:8], "big")
                    return w, h
                if ctype == b"IEND":
                    return None
    except OSError:
        return None


def normalize_root_compare(path_a: str, path_b: str) -> bool:
    try:
        return os.path.abspath(os.path.realpath(path_a)) == os.path.abspath(os.path.realpath(path_b))
    except OSError:
        return False


def extract_manifest_roots(manifest: dict[str, Any]) -> list[str]:
    roots: list[str] = []
    dl = manifest.get("dataset_layout")
    if isinstance(dl, dict):
        dr = dl.get("dataset_root")
        if isinstance(dr, str) and dr:
            roots.append(dr)
    ca = manifest.get("config_audit")
    if isinstance(ca, dict):
        rc = ca.get("resolved_config")
        if isinstance(rc, dict):
            ds = rc.get("dataset")
            if isinstance(ds, dict):
                r = ds.get("root")
                if isinstance(r, str) and r:
                    roots.append(r)
    return roots


def validate_manifest_paths(dataset_root: Path, manifest_path: Path, strict: bool) -> list[Issue]:
    issues: list[Issue] = []
    if not manifest_path.is_file():
        return issues

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            man = json.load(f)
    except (json.JSONDecodeError, OSError) as ex:
        issues.append(Issue("error", "manifest_io", f"Could not read manifest: {ex}", str(manifest_path)))
        return issues

    for hinted in extract_manifest_roots(man):
        if not hinted.strip():
            continue
        if not normalize_root_compare(str(dataset_root), hinted):
            msg = (
                f"manifest records dataset_root/root '{hinted}' which differs from --dataset-root "
                f"'{dataset_root}' (best-effort realpath comparison)."
            )
            sev = "error" if strict else "warning"
            issues.append(Issue(sev, "manifest_root_mismatch", msg, str(manifest_path)))

    return issues


def validate_meta_json(path: Path) -> list[Issue]:
    issues: list[Issue] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except (json.JSONDecodeError, OSError) as ex:
        issues.append(Issue("error", "meta_json", f"Invalid meta JSON: {ex}", str(path)))
        return issues

    if not isinstance(obj, dict):
        issues.append(Issue("error", "meta_schema", "Meta JSON root must be an object.", str(path)))
        return issues

    for key in ("frame_id", "split", "image_width", "image_height"):
        if key not in obj:
            issues.append(Issue("error", "meta_schema", f"Missing required field '{key}'.", str(path)))

    if "frame_id" in obj and not _is_integral_number(obj["frame_id"]):
        issues.append(Issue("error", "meta_schema", "frame_id must be an integral number.", str(path)))
    if "split" in obj and not isinstance(obj["split"], str):
        issues.append(Issue("error", "meta_schema", "split must be a string.", str(path)))
    for ik in ("image_width", "image_height"):
        if ik in obj and not _is_integral_number(obj[ik]):
            issues.append(Issue("error", "meta_schema", f"{ik} must be an integral number.", str(path)))

    for nk, nt in (("render", dict), ("camera", dict), ("sun", dict)):
        if nk not in obj:
            issues.append(Issue("error", "meta_schema", f"Missing required field '{nk}'.", str(path)))
        elif not isinstance(obj[nk], nt):
            issues.append(Issue("error", "meta_schema", f"'{nk}' must be an object.", str(path)))

    if "objects" not in obj:
        issues.append(Issue("error", "meta_schema", "Missing required field 'objects'.", str(path)))
    elif not isinstance(obj["objects"], list):
        issues.append(Issue("error", "meta_schema", "objects must be an array.", str(path)))

    return issues


def validate_coco_bbox(
    bbox_xywh: list[float],
    img_w: int,
    img_h: int,
    *,
    max_area_frac: float,
    min_area_px: int,
    ann_desc: str,
) -> list[Issue]:
    issues: list[Issue] = []
    if len(bbox_xywh) != 4:
        issues.append(Issue("error", "coco_bbox_len", f"COCO bbox must have length 4 ({ann_desc})."))
        return issues

    x, y, w, h = bbox_xywh
    for lab, v in (("x", x), ("y", y), ("w", w), ("h", h)):
        if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            issues.append(Issue("error", "coco_bbox_nan", f"Non-finite bbox component {lab} ({ann_desc})."))

    if issues:
        return issues

    x, y, w, h = float(x), float(y), float(w), float(h)
    area_px = w * h
    if area_px < float(min_area_px):
        issues.append(
            Issue(
                "error",
                "coco_bbox_small_area",
                f"COCO box pixel area {area_px:g} < min_bbox_area_pixels={min_area_px} ({ann_desc}).",
            )
        )

    if w <= 0 or h <= 0:
        issues.append(Issue("error", "coco_bbox_nonpos_wh", f"Non-positive COCO w/h ({ann_desc})."))

    denom = float(img_w * img_h)
    if denom <= 0:
        issues.append(Issue("error", "coco_image_dim", f"Invalid image dims W,H={img_w},{img_h} ({ann_desc})."))
        return issues

    frac = area_px / denom
    if frac >= max_area_frac:
        issues.append(
            Issue(
                "warning",
                "coco_bbox_huge_frac",
                f"COCO box fraction {(frac * 100):.4g}% of image pixels ≥ max_bbox_area_frac={max_area_frac} ({ann_desc}).",
            )
        )

    xmax = x + w
    ymax = y + h
    eps = 1e-6
    if x < -eps or y < -eps or xmax > img_w + eps or ymax > img_h + eps:
        issues.append(
            Issue(
                "warning",
                "coco_bbox_oob",
                f"COCO box exceeds image bounds: xywh=({x},{y},{w},{h}) vs W,H=({img_w},{img_h}) ({ann_desc}).",
            )
        )

    return issues


def validate_yolo_line(
    tokens: list[str],
    *,
    img_w: int,
    img_h: int,
    max_area_frac: float,
    min_area_px: int,
    desc: str,
) -> list[Issue]:
    issues: list[Issue] = []
    if len(tokens) != 5:
        issues.append(Issue("error", "yolo_tokens", f"YOLO line must have 5 fields ({desc})."))
        return issues
    try:
        _cls = int(tokens[0])
        cx = float(tokens[1])
        cy = float(tokens[2])
        wn = float(tokens[3])
        hn = float(tokens[4])
    except ValueError:
        issues.append(Issue("error", "yolo_parse", f"Could not parse YOLO floats/int ({desc})."))
        return issues

    for lab, v in (("cx", cx), ("cy", cy), ("wn", wn), ("hn", hn)):
        if not math.isfinite(v):
            issues.append(Issue("error", "yolo_nan", f"Non-finite {lab} ({desc})."))

    if issues:
        return issues

    if wn <= 0 or hn <= 0:
        issues.append(Issue("error", "yolo_nonpos_wh", f"Non-positive normalized w/h ({desc})."))

    area_frac = wn * hn  # fractional coverage of image (peer convention)

    if area_frac >= max_area_frac:
        issues.append(
            Issue(
                "warning",
                "yolo_huge_frac",
                f"YOLO box fractional area {area_frac:.6g} ≥ max_bbox_area_frac={max_area_frac} ({desc}).",
            )
        )

    pixel_area = area_frac * float(img_w * img_h)
    if pixel_area + 1e-12 < float(min_area_px):
        issues.append(
            Issue(
                "error",
                "yolo_small_area",
                f"YOLO pixel-area estimate {pixel_area:g} < min_bbox_area_pixels={min_area_px} ({desc}).",
            )
        )

    xmin_n = cx - wn / 2.0
    ymin_n = cy - hn / 2.0
    xmax_n = cx + wn / 2.0
    ymax_n = cy + hn / 2.0
    eps = 1e-6
    if xmin_n < -eps or ymin_n < -eps or xmax_n > 1.0 + eps or ymax_n > 1.0 + eps:
        issues.append(
            Issue(
                "warning",
                "yolo_oob_norm",
                f"YOLO box exceeds normalized bounds: cxcywh=({cx},{cy},{wn},{hn}) ({desc}).",
            )
        )

    return issues


def run_validation(args: argparse.Namespace) -> ValidationReport:
    root = args.dataset_root.resolve()
    report = ValidationReport(
        dataset_root=str(root),
        split_filter=args.split,
        coco_path=None,
        manifest_path=None,
        scan_directories=[],
    )

    if not root.is_dir():
        report.add(Issue("error", "bad_root", "dataset-root is not a directory.", str(root)))
        return report

    scan_dirs, sd_issues = discover_scan_directories(root, args.split)
    report.scan_directories = [str(p) for p in scan_dirs]
    report.issues.extend(sd_issues)
    if not scan_dirs and sd_issues:
        return report

    manifest_path = args.manifest.resolve() if args.manifest is not None else (root / "manifest.json")
    report.manifest_path = str(manifest_path)
    report.issues.extend(validate_manifest_paths(root, manifest_path, args.strict))

    coco_path, coco_issues = discover_coco_path(root, args.coco)
    report.issues.extend(coco_issues)
    if coco_path is None:
        return report
    report.coco_path = str(coco_path)

    try:
        with coco_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)
    except (json.JSONDecodeError, OSError) as ex:
        report.add(Issue("error", "coco_io", f"Could not read COCO JSON: {ex}", str(coco_path)))
        return report

    images = coco.get("images")
    annotations = coco.get("annotations")
    categories = coco.get("categories")
    if not isinstance(images, list) or not isinstance(annotations, list):
        report.add(Issue("error", "coco_schema", "COCO document must contain images[] and annotations[] arrays.", str(coco_path)))
        return report

    cat_names: dict[int, str] = {}
    if isinstance(categories, list):
        for c in categories:
            if isinstance(c, dict) and isinstance(c.get("id"), int) and isinstance(c.get("name"), str):
                cat_names[c["id"]] = c["name"]

    id_to_image: dict[int, dict[str, Any]] = {}
    file_lookup: dict[str, dict[str, Any]] = {}
    for im in images:
        if not isinstance(im, dict) or not isinstance(im.get("id"), int):
            report.add(Issue("error", "coco_image_schema", "Malformed images[] entry (need dict with int id).", str(coco_path)))
            continue
        iid = im["id"]
        fn = im.get("file_name")
        if not isinstance(fn, str) or not fn:
            report.add(Issue("error", "coco_image_fname", f"Image id {iid} missing file_name.", str(coco_path)))
            continue
        id_to_image[iid] = im
        prev = file_lookup.get(fn)
        if prev is not None and prev.get("id") != iid:
            report.add(
                Issue(
                    "warning",
                    "coco_duplicate_fname",
                    f"Duplicate COCO file_name {fn!r} on distinct image ids ({prev.get('id')} vs {iid}).",
                    str(coco_path),
                )
            )
        file_lookup[fn] = im

    allowed_roots = [p.resolve() for p in scan_dirs]

    def resolve_image_path(file_name: str) -> Path | None:
        hits: list[Path] = []
        for base in allowed_roots:
            cand = (base / file_name).resolve()
            if cand.is_file():
                hits.append(cand)
        if len(hits) == 1:
            return hits[0]
        if len(hits) == 0:
            return None
        report.add(
            Issue(
                "error",
                "path_ambiguous",
                f"Multiple on-disk paths for COCO file_name={file_name!r}: {[str(h) for h in hits]}",
            )
        )
        return None

    for ann in annotations:
        if not isinstance(ann, dict):
            report.add(Issue("warning", "coco_ann_dtype", "Non-object annotation entry skipped.", str(coco_path)))
            continue
        iid_ref = ann.get("image_id")
        if not isinstance(iid_ref, int) or iid_ref not in id_to_image:
            aid = ann.get("id")
            report.add(
                Issue(
                    "error",
                    "coco_orphan_ann",
                    f"Annotation references unknown image_id={iid_ref!r} (ann id={aid!r}).",
                    str(coco_path),
                )
            )

    combined_hist: Counter[int] = Counter()

    images_checked = 0
    for fn, im in file_lookup.items():
        path = resolve_image_path(fn)
        desc = f"image file_name={fn}"
        if path is None:
            report.add(Issue("error", "coco_image_missing", f"COCO references missing file ({desc})."))
            continue

        try:
            sz = path.stat().st_size
        except OSError as ex:
            report.add(Issue("error", "image_stat", f"{ex}", str(path)))
            continue

        if sz < args.min_image_bytes:
            report.add(
                Issue(
                    "error",
                    "image_small",
                    f"PNG size {sz} bytes < min-image-bytes={args.min_image_bytes} ({desc}).",
                    str(path),
                )
            )

        iid = im["id"]
        default_w = im.get("width")
        default_h = im.get("height")
        iw = ih = None
        if isinstance(default_w, int) and isinstance(default_h, int):
            iw, ih = default_w, default_h
        else:
            sniff = read_png_dimensions(path)
            if sniff is None:
                report.add(
                    Issue(
                        "warning",
                        "png_ihdr",
                        "Could not read PNG dimensions from IHDR; skipping COCO bbox geometry checks for this image.",
                        str(path),
                    )
                )
            else:
                iw, ih = sniff

        images_checked += 1

        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            if ann.get("image_id") != iid:
                continue
            cat = ann.get("category_id")
            if isinstance(cat, int):
                combined_hist[cat] += 1

            bbox = ann.get("bbox")
            if not isinstance(bbox, list):
                report.add(Issue("error", "coco_bbox_type", f"annotation bbox must be list ({desc}).", str(coco_path)))
                continue

            if iw is None or ih is None:
                continue

            try:
                bbox_f = [float(x) for x in bbox]
            except (TypeError, ValueError):
                report.add(Issue("error", "coco_bbox_values", f"Non-numeric bbox ({desc}).", str(coco_path)))
                continue

            ann_tag = f"image_id={iid} ann_id={ann.get('id')}"
            report.issues.extend(
                validate_coco_bbox(
                    bbox_f,
                    iw,
                    ih,
                    max_area_frac=args.max_bbox_area_frac,
                    min_area_px=args.min_bbox_area_pixels,
                    ann_desc=ann_tag,
                )
            )

    report.stats["coco_images_referenced"] = len(id_to_image)
    report.stats["coco_images_checked_existing"] = images_checked

    # YOLO pairing + histogram from labels
    for d in scan_dirs:
        for png in sorted(d.iterdir()):
            if not png.is_file() or not PNG_PATTERN.match(png.name):
                continue
            if png.name not in file_lookup:
                report.add(
                    Issue(
                        "warning",
                        "png_not_in_coco",
                        f"RGB file {png.name} not referenced by COCO images[].file_name entries.",
                        str(png),
                    )
                )

            stem = png.stem  # frame_0000 or sfrm_0000
            yolo = d / f"{stem}.txt"
            if not yolo.is_file():
                report.add(Issue("error", "yolo_missing", f"Missing YOLO label for {png.name}.", str(yolo)))
                continue

            meta_path = d / f"{stem}_meta.json"
            if args.check_meta:
                if not meta_path.is_file():
                    report.add(Issue("error", "meta_missing", f"Missing meta JSON for {png.name}.", str(meta_path)))
                else:
                    report.issues.extend(validate_meta_json(meta_path))

            sniff = read_png_dimensions(png)
            if sniff is None:
                report.add(Issue("warning", "png_ihdr", f"Could not read PNG IHDR for YOLO audit.", str(png)))
                continue
            iw, ih = sniff

            try:
                raw = yolo.read_text(encoding="utf-8")
            except OSError as ex:
                report.add(Issue("error", "yolo_io", str(ex), str(yolo)))
                continue

            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            for li, line in enumerate(lines):
                parts = line.split()
                if len(parts) != 5:
                    report.add(
                        Issue(
                            "error",
                            "yolo_bad_line",
                            f"Expected 5 whitespace-separated tokens (line {li + 1}).",
                            str(yolo),
                        )
                    )
                    continue
                try:
                    cid = int(parts[0])
                    combined_hist[cid] += 1
                except ValueError:
                    report.add(Issue("error", "yolo_class", f"Bad class id on line {li + 1}.", str(yolo)))
                    continue

                report.issues.extend(
                    validate_yolo_line(
                        parts,
                        img_w=iw,
                        img_h=ih,
                        max_area_frac=args.max_bbox_area_frac,
                        min_area_px=args.min_bbox_area_pixels,
                        desc=f"{yolo.name} line {li + 1}",
                    )
                )

    report.stats["class_histogram"] = {str(k): combined_hist[k] for k in sorted(combined_hist.keys())}
    report.stats["category_names"] = {str(k): v for k, v in sorted(cat_names.items())}

    return report


def exit_code_for(report: ValidationReport, strict: bool) -> int:
    errs = [i for i in report.issues if i.severity == "error"]
    warns = [i for i in report.issues if i.severity == "warning"]
    if errs:
        return 1
    if strict and warns:
        return 1
    return 0


def dump_human(report: ValidationReport, strict: bool) -> None:
    code = exit_code_for(report, strict)
    print(f"VisionForge dataset validation — exit {'PASS' if code == 0 else 'FAIL'} ({code})", file=sys.stderr)
    print(f"  dataset_root: {report.dataset_root}", file=sys.stderr)
    print(f"  split: {report.split_filter}", file=sys.stderr)
    print(f"  scan_directories:", file=sys.stderr)
    for sd in report.scan_directories:
        print(f"    - {sd}", file=sys.stderr)
    if report.coco_path:
        print(f"  coco: {report.coco_path}", file=sys.stderr)
    if report.manifest_path:
        print(f"  manifest path: {report.manifest_path}", file=sys.stderr)

    if report.issues:
        print("\nIssues:", file=sys.stderr)
        for it in report.issues:
            prefix = it.severity.upper()
            loc = f" ({it.path})" if it.path else ""
            print(f"  [{prefix}] {it.code}: {it.message}{loc}", file=sys.stderr)
    else:
        print("\nNo issues.", file=sys.stderr)

    hist = report.stats.get("class_histogram") or {}
    names = report.stats.get("category_names") or {}
    if hist:

        def _cid_sort_key(s: str):
            try:
                return (0, int(s))
            except ValueError:
                return (1, s)

        print("\nPer-class counts (COCO annotations + YOLO lines):", file=sys.stdout)
        for cid_s in sorted(hist.keys(), key=_cid_sort_key):
            cnt = hist[cid_s]
            nm = names.get(cid_s, "")
            suffix = f"  ({nm})" if nm else ""
            print(f"  id={cid_s}: {cnt}{suffix}", file=sys.stdout)


def write_json_report(path: Path, report: ValidationReport, strict: bool) -> None:
    payload = {
        "dataset_root": report.dataset_root,
        "split": report.split_filter,
        "coco_path": report.coco_path,
        "manifest_path": report.manifest_path,
        "scan_directories": report.scan_directories,
        "exit_code": exit_code_for(report, strict),
        "issues": [
            {"severity": i.severity, "code": i.code, "message": i.message, "path": i.path} for i in report.issues
        ],
        "stats": report.stats,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.dataset_root
    if not root.exists():
        print(f"error: dataset-root does not exist: {root}", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(f"error: dataset-root is not a directory: {root}", file=sys.stderr)
        return 2

    try:
        report = run_validation(args)
    except OSError as ex:
        print(f"error: I/O failure during validation: {ex}", file=sys.stderr)
        return 2

    dump_human(report, args.strict)

    if args.json_report is not None:
        try:
            write_json_report(args.json_report, report, args.strict)
        except OSError as ex:
            print(f"error: could not write json-report: {ex}", file=sys.stderr)
            return 2

    return exit_code_for(report, args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
