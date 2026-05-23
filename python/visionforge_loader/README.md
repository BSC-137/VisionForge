# visionforge-loader

PyTorch-first helpers to read VisionForge `forge` / `scenario` datasets (`train/` + `val/`, `*_meta.json`, `*_spatial.exr`) and verify that `camera_extrinsics` (c2w) and `camera_intrinsics` agree with the documented pinhole + `Camera::get_ray` mapping.

## Install

From the repo root:

```bash
pip install ./python/visionforge_loader
# dev
pip install ./python/visionforge_loader[dev]
```

Requires a working OpenEXR Python build (wheels are available on PyPI for common platforms).

## Examples

`examples/train_supervision_baseline.py` trains a small encoder–decoder CNN to predict **depth** (default) or **world normals** from RGB, using `VisionForgeDataset` only (same tensors as production: `rgb`, `depth`, `normal`, `meta`). Loss is masked so sky / miss pixels (including large finite depth sentinels) do not dominate the objective—see the module docstring.

For a full end-to-end verification walkthrough including validation and projection checks, see **Verifying a rendered dataset** in the root README.

Then run:

```bash
cd python/visionforge_loader
PYTHONPATH=. python3 examples/train_supervision_baseline.py \
  --dataset-root /path/to/your/forge/export \
  --epochs 1 --max-samples 16 --batch-size 2 --device cpu
```

If you use `pip install -e ./python/visionforge_loader`, you still execute the script from the checkout path shown above, or set `PYTHONPATH` to the `python/visionforge_loader` directory that contains both `examples/` and the `visionforge_loader` package.

## Tests

```bash
cd python/visionforge_loader
pytest -q
```

## CLI smoke

After rendering a dataset:

```bash
python -m visionforge_loader.cli_projection_smoke --dataset-root /path/to/dataset --max-frames 5
```

## Camera contract

Matches the main VisionForge README section **Meta / pose convention** (`*_meta.json`):

- `camera_extrinsics`: row-major \(4 \times 4\) **camera-to-world** \(P_w = M \, P_c\) (homogeneous column vectors).
- Pixel \((i,j)\) uses \(s = i/\max(W-1,1)\), \(t = j/\max(H-1,1)\) for `get_ray`.
- OpenCV-style camera axes in the rotation part of \(M\): **+X** right, **+Y** down, **+Z** forward.
- G-buffer **Depth** is linear **distance in metres from `Camera::origin`** along the pinhole ray through the pixel (not \(Z_c\) alone).

Projection implements the same **negated vertical** term as `tests/test_meta_pose.cpp`: with \(P_c = \mathrm{w2c}\,P_w\), \(u = f_x X_c/Z_c + c_x\), \(v = -f_y Y_c/Z_c + c_y\) (v increases downward, consistent with `get_ray`'s \((t-0.5)\) on `Camera::vertical`).

## Sharding

Distributed renders use global frame indices in filenames (`frame_0123` → \(g=123\)). Merge COCO with:

```bash
python ../../scripts/merge_coco_shards.py out.json shard0.json shard1.json ...
```
