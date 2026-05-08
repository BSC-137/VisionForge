"""PyTorch Dataset for VisionForge exports."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from visionforge_loader.exr_io import read_spatial_exr
from visionforge_loader.geometry import c2w_from_row_major_list, pinhole_intrinsics_match_renderer


StemPattern = Literal["frame", "sfrm", "auto"]


@dataclass
class VisionForgeFrameMeta:
    frame_id: int
    stem: str
    split: str  # train | val
    image_width: int
    image_height: int
    c2w: torch.Tensor  # [4,4] float32
    K: torch.Tensor  # [3,3] float32
    fx: float
    fy: float
    cx: float
    cy: float
    vfov_deg: float
    json_raw: dict


def _load_meta(path: Path) -> VisionForgeFrameMeta:
    with path.open("r", encoding="utf-8") as f:
        j = json.load(f)
    w = int(j["image_width"])
    h = int(j["image_height"])
    intr = j["camera_intrinsics"]
    vfov = float(intr["vfov_deg"])
    fx, fy, cx, cy = (
        float(intr["fx"]),
        float(intr["fy"]),
        float(intr["cx"]),
        float(intr["cy"]),
    )
    # Sanity-check intrinsics vs vfov (meta should be self-consistent).
    fx_e, fy_e, cx_e, cy_e = pinhole_intrinsics_match_renderer(w, h, vfov)
    if abs(fx - fx_e) > 1e-3 or abs(fy - fy_e) > 1e-3:
        raise ValueError(f"meta intrinsics mismatch pinhole model for {path}: {fx} {fy} vs expected {fx_e} {fy_e}")
    if abs(cx - cx_e) > 1e-3 or abs(cy - cy_e) > 1e-3:
        raise ValueError(f"meta principal point mismatch for {path}")
    c2w = torch.from_numpy(c2w_from_row_major_list(j["camera_extrinsics"]).astype(np.float32))
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)
    stem = path.name.removesuffix("_meta.json")
    return VisionForgeFrameMeta(
        frame_id=int(j.get("frame_id", -1)),
        stem=stem,
        split=str(j.get("split", "")),
        image_width=w,
        image_height=h,
        c2w=c2w,
        K=K,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        vfov_deg=vfov,
        json_raw=j,
    )


_STEM_RE = re.compile(r"^(frame|sfrm)_(\d{4})_meta\.json$")


def _discover_frames(split_dir: Path, stem_pat: StemPattern) -> list[Tuple[str, int, Path]]:
    out: list[Tuple[str, int, Path]] = []
    for p in sorted(split_dir.glob("*_meta.json")):
        m = _STEM_RE.match(p.name)
        if not m:
            continue
        kind, idx_s = m.group(1), m.group(2)
        if stem_pat != "auto" and kind != stem_pat:
            continue
        out.append((kind, int(idx_s), p))
    out.sort(key=lambda t: t[1])
    return out


def auto_detect_stem(dataset_root: Path) -> StemPattern:
    for sub in ("train", "val"):
        d = dataset_root / sub
        if not d.is_dir():
            continue
        has_frame = any(d.glob("frame_*_meta.json"))
        has_sfrm = any(d.glob("sfrm_*_meta.json"))
        if has_frame and not has_sfrm:
            return "frame"
        if has_sfrm and not has_frame:
            return "sfrm"
    return "frame"


class VisionForgeDataset(torch.utils.data.Dataset):
    """
    Indexes frames by sorted global stem id under train/ and/or val/.

    Returns a dict with rgb float [3,H,W] in [0,1], spatial tensors, and meta dataclass.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: Literal["train", "val", "both"] = "both",
        stem_pattern: StemPattern = "auto",
        transform: Optional[Callable[[dict], dict]] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        if stem_pattern == "auto":
            stem_pattern = auto_detect_stem(self.root)
        self.stem_pattern: Literal["frame", "sfrm"] = stem_pattern  # type: ignore[assignment]

        records: list[Tuple[str, Path, Path]] = []
        splits_to_scan = ("train", "val") if split == "both" else (split,)
        for sp_name in splits_to_scan:
            d = self.root / sp_name
            if not d.is_dir():
                continue
            for _kind, idx, meta_path in _discover_frames(d, self.stem_pattern):
                png = meta_path.with_name(meta_path.name.replace("_meta.json", ".png"))
                records.append((sp_name, idx, meta_path, png))

        records.sort(key=lambda r: (r[0], r[1]))
        self._records = [(a, c, d) for a, _, c, d in records]

    def __len__(self) -> int:
        return len(self._records)

    def paths_for_index(self, index: int) -> Tuple[str, Path, Path]:
        return self._records[index]

    def __getitem__(self, index: int) -> dict:
        sp_name, meta_path, png_path = self._records[index]
        meta = _load_meta(meta_path)
        exr_path = meta_path.with_name(meta.stem + "_spatial.exr")
        if not png_path.is_file():
            raise FileNotFoundError(png_path)
        if not exr_path.is_file():
            raise FileNotFoundError(exr_path)

        pil = Image.open(png_path).convert("RGB")
        rgb_u8 = torch.from_numpy(np.asarray(pil, dtype=np.uint8)).permute(2, 0, 1)
        rgb = rgb_u8.float() / 255.0

        sp = read_spatial_exr(str(exr_path))
        depth = torch.from_numpy(sp.depth)
        inst = torch.from_numpy(sp.instance_id)
        normal = torch.from_numpy(sp.normal)

        item = {
            "rgb": rgb,
            "depth": depth,
            "instance_id": inst,
            "normal": normal,
            "meta": meta,
            "split": sp_name,
            "paths": {"png": str(png_path), "exr": str(exr_path), "meta": str(meta_path)},
        }
        if self.transform:
            item = self.transform(item)
        return item
