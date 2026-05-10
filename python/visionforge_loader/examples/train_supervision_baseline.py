#!/usr/bin/env python3
"""
Minimal RGB → depth / world-normal supervision baseline for VisionForge exports.

**Valid-pixel rule (sky / ray misses):** depth can be non-finite, ~0, or a large finite sentinel
when the ray misses (e.g. ~1e30). We only supervise where
``torch.isfinite(depth) & (depth > DEPTH_EPS) & (depth < DEPTH_MAX)``.
The same mask is used for depth and normal losses. Averaging is over masked pixels only; batches
with zero valid pixels are skipped.
"""

from __future__ import annotations

import argparse
import random
import sys
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from visionforge_loader.dataset import VisionForgeDataset

# Depth mask: ignore sky, inf, non-positive, and huge sentinels (miss pixels are often ~1e30)
DEPTH_EPS = 1e-6
DEPTH_MAX = 1.0e6


def _init_conv(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def build_model(out_channels: int) -> nn.Module:
    """Small encoder–decoder CNN (no torchvision). Input [B,3,H,W]; output [B,C,H,W]."""
    c1, c2, c3, c4 = 32, 64, 128, 256

    class BaselineUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc0 = nn.Conv2d(3, c1, 3, padding=1)
            self.enc1 = nn.Conv2d(c1, c2, 3, padding=1)
            self.enc2 = nn.Conv2d(c2, c3, 3, padding=1)
            self.enc3 = nn.Conv2d(c3, c4, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.dec3 = nn.Conv2d(c4 + c3, c3, 3, padding=1)
            self.dec2 = nn.Conv2d(c3 + c2, c2, 3, padding=1)
            self.dec1 = nn.Conv2d(c2 + c1, c1, 3, padding=1)
            self.head = nn.Conv2d(c1, out_channels, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            s0 = F.relu(self.enc0(x))
            s1 = F.relu(self.enc1(self.pool(s0)))
            s2 = F.relu(self.enc2(self.pool(s1)))
            s3 = F.relu(self.enc3(self.pool(s2)))

            u2 = torch.cat(
                [F.interpolate(s3, size=s2.shape[2:], mode="bilinear", align_corners=False), s2], dim=1
            )
            u2 = F.relu(self.dec3(u2))
            u1 = torch.cat(
                [F.interpolate(u2, size=s1.shape[2:], mode="bilinear", align_corners=False), s1], dim=1
            )
            u1 = F.relu(self.dec2(u1))
            u0 = torch.cat(
                [F.interpolate(u1, size=s0.shape[2:], mode="bilinear", align_corners=False), s0], dim=1
            )
            u0 = F.relu(self.dec1(u0))
            return self.head(u0)

    net = BaselineUNet()
    net.apply(_init_conv)
    return net


def depth_valid_mask(depth: torch.Tensor) -> torch.Tensor:
    """[B,H,W] bool mask."""
    return torch.isfinite(depth) & (depth > DEPTH_EPS) & (depth < DEPTH_MAX)


def masked_depth_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked L1 on [B,1,H,W] pred/target; mask [B,H,W]."""
    if pred.shape[1] != 1:
        raise ValueError("masked_depth_loss expects pred with C=1")
    diff = (pred[:, 0] - target).abs()
    m = mask.float()
    denom = m.sum().clamp(min=1e-8)
    return (diff * m).sum() / denom


def masked_normal_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Masked 1 - cosine similarity per pixel. Predictions are L2-normalized per pixel (channel dim)
    before the loss; targets are normalized the same way for stable cosine in dead zones.
    Shapes: pred/target [B,3,H,W], mask [B,H,W].
    """
    pred_n = F.normalize(pred, dim=1, eps=1e-6)
    tgt_n = F.normalize(target, dim=1, eps=1e-6)
    cos = (pred_n * tgt_n).sum(dim=1).clamp(-1.0, 1.0)
    one_minus = 1.0 - cos
    m = mask.float()
    denom = m.sum().clamp(min=1e-8)
    return (one_minus * m).sum() / denom


def collate_supervision_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    rgb = torch.stack([b["rgb"] for b in batch], dim=0)
    depth = torch.stack([b["depth"] for b in batch], dim=0)
    normal = torch.stack([b["normal"] for b in batch], dim=0)
    meta_list = [b["meta"] for b in batch]
    return {"rgb": rgb, "depth": depth, "normal": normal, "meta": meta_list}


def assert_batch_single_resolution(batch: dict[str, Any]) -> None:
    meta_list: list = batch["meta"]
    if not meta_list:
        return
    w0 = int(meta_list[0].image_width)
    h0 = int(meta_list[0].image_height)
    for m in meta_list[1:]:
        if int(m.image_width) != w0 or int(m.image_height) != h0:
            raise RuntimeError(
                "Batch has mixed resolutions. This baseline expects a single H×W per dataset "
                "(one forge run). Use a single-resolution export or batch_size=1."
            )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(
    dataset_root: str,
    split: Literal["train", "val", "both"],
) -> Dataset:
    return VisionForgeDataset(root=dataset_root, split=split)


def train(
    *,
    dataset_root: str | None,
    split: Literal["train", "val", "both"],
    target: Literal["depth", "normal"],
    epochs: int,
    batch_size: int,
    lr: float,
    max_samples: int | None,
    device: torch.device,
    num_workers: int,
    seed: int,
    dataset: Dataset | None = None,
) -> None:
    set_seed(seed)

    ds: Dataset
    if dataset is not None:
        ds = dataset
    else:
        if not dataset_root:
            raise ValueError("dataset_root is required when dataset is None.")
        ds = build_dataset(dataset_root, split)

    if max_samples is not None:
        n = min(max_samples, len(ds))
        ds = Subset(ds, list(range(n)))

    if len(ds) == 0:
        raise RuntimeError("Dataset is empty (check --dataset-root and --split).")

    out_ch = 1 if target == "depth" else 3
    model = build_model(out_ch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_supervision_batch,
    )

    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        for batch in loader:
            assert_batch_single_resolution(batch)
            rgb = batch["rgb"].to(device)
            depth_tgt = batch["depth"].to(device)
            normal_tgt = batch["normal"].to(device)
            mask = depth_valid_mask(depth_tgt)
            if mask.sum() == 0:
                continue

            if target == "depth":
                pred = model(rgb)
                pred = F.relu(pred) + DEPTH_EPS
                loss = masked_depth_loss(pred, depth_tgt, mask)
            else:
                pred = model(rgb)
                loss = masked_normal_loss(pred, normal_tgt, mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"epoch {epoch} mean_loss={mean_loss:.6f}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RGB → depth / normals baseline for VisionForge datasets.")
    p.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="VisionForge dataset root (directory with train/ and/or val/).",
    )
    p.add_argument(
        "--split",
        choices=("train", "val", "both"),
        default="train",
        help="Subset of splits to load (default: train).",
    )
    p.add_argument("--target", choices=("depth", "normal"), default="depth")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-samples", type=int, default=None, help="Cap dataset size for quick runs.")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    train(
        dataset_root=args.dataset_root,
        split=args.split,  # type: ignore[arg-type]
        target=args.target,  # type: ignore[arg-type]
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
        device=device,
        num_workers=args.num_workers,
        seed=args.seed,
        dataset=None,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
