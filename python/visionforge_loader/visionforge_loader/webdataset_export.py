"""Re-pack a VisionForge dataset into WebDataset .tar shards for streaming training."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

# Matches frame_0042_meta.json and sfrm_0042_meta.json (4+ digit indices).
_STEM_RE = re.compile(r"^(frame|sfrm)_(\d{4,})_meta\.json$")


def _collect_frames(split_dir: Path) -> list[tuple[str, Path]]:
    """Return (stem, meta_path) pairs from *split_dir*, sorted by frame index."""
    out: list[tuple[str, int, Path]] = []
    for p in split_dir.glob("*_meta.json"):
        m = _STEM_RE.match(p.name)
        if not m:
            continue
        stem = f"{m.group(1)}_{m.group(2)}"
        out.append((stem, int(m.group(2)), p))
    out.sort(key=lambda t: t[1])
    return [(stem, meta) for stem, _, meta in out]


def to_webdataset(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    split: Literal["train", "val", "both"] = "both",
    frames_per_shard: int = 1000,
    compress: bool = False,
) -> list[str]:
    """Re-pack a VisionForge dataset into WebDataset ``.tar`` shards.

    Each sample's key is the frame stem (e.g. ``frame_0042``).
    Per-sample entries written into every shard:

    * ``{stem}.png``  — RGB image
    * ``{stem}.exr``  — spatial G-Buffer EXR (depth, normals, optical flow)
    * ``{stem}.json`` — frame meta JSON (verbatim byte copy)
    * ``{stem}.txt``  — YOLO label file (included only when present on disk)

    Parameters
    ----------
    dataset_root:
        Root directory of the VisionForge export (must contain ``train/``
        and/or ``val/`` sub-directories).
    output_dir:
        Destination directory for shard files.  Created when absent.
        Shard files are named ``shard-000000.tar`` (or ``.tar.gz``).
    split:
        Which split(s) to pack.  ``"both"`` writes all train frames first,
        then all val frames, into a single shard sequence.
    frames_per_shard:
        Maximum number of samples per shard.  Use ``1000`` for GPU training
        (keeps shard size roughly 200–500 MB depending on resolution) and
        ``100`` for easier local debugging.
    compress:
        When ``True``, shards are written as gzip-compressed ``.tar.gz``
        files.  Reduces storage at the cost of slightly higher CPU on the
        data-loader side.

    Returns
    -------
    list[str]
        Absolute paths of every written shard file, sorted lexicographically.

    Raises
    ------
    ImportError
        If the ``webdataset`` package is not installed.
    """
    try:
        import webdataset as wds  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The webdataset package is required for shard export.\n"
            "Install it with:  pip install webdataset"
        ) from exc

    root = Path(dataset_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect frames from the requested split(s), train before val.
    splits_to_pack: tuple[str, ...] = ("train", "val") if split == "both" else (split,)
    all_frames: list[tuple[str, str, Path]] = []  # (split_name, stem, meta_path)
    for sp in splits_to_pack:
        d = root / sp
        if not d.is_dir():
            continue
        for stem, meta_path in _collect_frames(d):
            all_frames.append((sp, stem, meta_path))

    if not all_frames:
        return []

    # Extension controls compression; ShardWriter infers format from filename.
    ext = ".tar.gz" if compress else ".tar"
    shard_pattern = str(out_dir / f"shard-%06d{ext}")

    # Snapshot existing shard files so we return only what *this* call created.
    existing = {str(p) for p in out_dir.glob(f"shard-*{ext}")}

    with wds.ShardWriter(shard_pattern, maxcount=frames_per_shard, verbose=0) as sink:
        for _sp_name, stem, meta_path in all_frames:
            sample: dict = {"__key__": stem}

            # PNG — primary RGB render
            png_path = meta_path.with_name(stem + ".png")
            if png_path.is_file():
                sample["png"] = png_path.read_bytes()

            # Spatial EXR — depth / normals / flow
            exr_path = meta_path.with_name(stem + "_spatial.exr")
            if exr_path.is_file():
                sample["exr"] = exr_path.read_bytes()

            # Meta JSON — verbatim byte copy
            sample["json"] = meta_path.read_bytes()

            # YOLO label — optional
            txt_path = meta_path.with_name(stem + ".txt")
            if txt_path.is_file():
                sample["txt"] = txt_path.read_bytes()

            sink.write(sample)

    after = {str(p) for p in out_dir.glob(f"shard-*{ext}")}
    return sorted(after - existing)
