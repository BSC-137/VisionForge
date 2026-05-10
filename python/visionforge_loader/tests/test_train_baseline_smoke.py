"""CPU smoke tests for training utilities (no OpenEXR / real exports required)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.train_supervision_baseline import (
    build_model,
    depth_valid_mask,
    masked_depth_loss,
    masked_normal_loss,
    train,
)


class _FakeSupervisionDataset(torch.utils.data.Dataset):
    """Returns dicts matching VisionForgeDataset tensor shapes."""

    def __init__(self, n: int, h: int, w: int) -> None:
        self._n = n
        self._h = h
        self._w = w

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict:
        rng = torch.Generator().manual_seed(idx + self._h * 1000 + self._w)
        rgb = torch.rand(3, self._h, self._w, generator=rng)
        depth = torch.rand(self._h, self._w, generator=rng) * 8.0 + 0.5
        nrm = torch.randn(3, self._h, self._w, generator=rng)
        nrm = nrm / nrm.norm(dim=0, keepdim=True).clamp(min=1e-6)
        meta = SimpleNamespace(image_width=self._w, image_height=self._h)
        return {"rgb": rgb, "depth": depth, "normal": nrm, "meta": meta}


def test_masked_depth_loss_basic() -> None:
    b, h, w = 2, 8, 8
    pred = torch.ones(b, 1, h, w) * 2.0
    tgt = torch.ones(b, h, w) * 3.0
    mask = torch.ones(b, h, w, dtype=torch.bool)
    loss = masked_depth_loss(pred, tgt, mask)
    assert loss.shape == torch.Size([])
    assert float(loss) == pytest.approx(1.0)


def test_masked_depth_loss_zero_mask_fallback() -> None:
    pred = torch.zeros(1, 1, 4, 4)
    tgt = torch.zeros(1, 4, 4)
    mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    loss = masked_depth_loss(pred, tgt, mask)
    assert float(loss) == pytest.approx(0.0)


def test_masked_normal_loss_aligned_vectors() -> None:
    b, h, w = 1, 4, 4
    v = torch.randn(b, 3, h, w)
    v = torch.nn.functional.normalize(v, dim=1)
    loss = masked_normal_loss(v, v, torch.ones(b, h, w, dtype=torch.bool))
    assert float(loss) < 1e-5


def test_model_forward_depth_and_normal() -> None:
    m1 = build_model(1)
    m3 = build_model(3)
    x = torch.randn(2, 3, 16, 24)
    assert m1(x).shape == (2, 1, 16, 24)
    assert m3(x).shape == (2, 3, 16, 24)


def test_train_smoke_cpu() -> None:
    """One epoch on tiny fake tensors (normals path: unit tests + 3-ch forward)."""
    import io
    from contextlib import redirect_stdout

    torch.set_num_threads(1)
    ds = _FakeSupervisionDataset(4, 8, 8)
    buf = io.StringIO()
    with redirect_stdout(buf):
        train(
            dataset_root=None,
            split="train",
            target="depth",
            epochs=1,
            batch_size=4,
            lr=1e-3,
            max_samples=4,
            device=torch.device("cpu"),
            num_workers=0,
            seed=0,
            dataset=ds,
        )
    line = buf.getvalue().strip().splitlines()[-1]
    assert "mean_loss=" in line
    assert np.isfinite(float(line.split("mean_loss=")[1].strip()))


def test_train_normal_backward_once() -> None:
    """Single backward on normals without a full on-disk dataset."""
    torch.set_num_threads(1)
    ds = _FakeSupervisionDataset(2, 8, 8)
    b = ds[0]
    model = build_model(3)
    rgb = b["rgb"].unsqueeze(0)
    tgt = b["normal"].unsqueeze(0)
    depth_b = b["depth"].unsqueeze(0)
    mask = depth_valid_mask(depth_b)
    pred = model(rgb)
    loss = masked_normal_loss(pred, tgt, mask)
    loss.backward()
    assert loss.shape == torch.Size([])
