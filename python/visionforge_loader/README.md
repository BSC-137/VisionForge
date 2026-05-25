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

`examples/export_pointcloud.py` back-projects the G-buffer depth into a world-space point cloud and writes an ASCII PLY file:

```bash
PYTHONPATH=. python3 examples/export_pointcloud.py \
  --dataset-root /path/to/dataset --frame-index 0 --out scene.ply
```

Produces an ASCII PLY viewable in MeshLab or CloudCompare.

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

Use `instance_id_to_class_id(instance_id_arr, meta.json_raw)` to map the `InstanceID` EXR channel to per-pixel semantic class without any C++ changes.

## Streaming Export

Re-pack a rendered dataset into [WebDataset](https://github.com/webdataset/webdataset) `.tar` shards for fast sequential reads on NFS, S3, or distributed training clusters.

### Install the optional dependency

```bash
pip install "visionforge-loader[streaming]"
# or directly:
pip install webdataset
```

### Pack a dataset

```python
from visionforge_loader import to_webdataset

shards = to_webdataset(
    "/path/to/forge/export",   # directory containing train/ and val/
    "/path/to/output/shards",  # destination; created if absent
    split="both",              # "train", "val", or "both"
    frames_per_shard=1000,     # 1000 for GPU training; 100 for debugging
)
print(f"Wrote {len(shards)} shards: {shards[:3]} …")
```

Each shard is a standard `.tar` file.  Shard names follow the pattern `shard-000000.tar`, `shard-000001.tar`, … Each sample contains:

| Key | Content |
|-----|---------|
| `{stem}.png` | RGB rendered image |
| `{stem}.exr` | Spatial G-Buffer (depth, normals, optical flow) |
| `{stem}.json` | Frame meta JSON (camera intrinsics, extrinsics, objects) |
| `{stem}.txt` | YOLO label (only when present on disk) |

Use `compress=True` to write `.tar.gz` shards (saves ~30–50 % storage, small CPU overhead at load time).

### Load shards with WebDataset

```python
import webdataset as wds

dataset = (
    wds.WebDataset("shards/shard-{000000..000009}.tar")
    .decode("rgb8")            # decode PNG → uint8 numpy [H,W,3]
    .to_tuple("png", "json")   # select per-sample keys
)

for img, meta_bytes in dataset:
    import json
    meta = json.loads(meta_bytes)
    print(img.shape, meta["frame_id"])
```

### Shard sizing guidance

| `frames_per_shard` | Use case |
|--------------------|----------|
| `1000` | GPU / multi-node training (shards ~200–500 MB at 1280×720) |
| `100` | Local debugging, fast iteration |
| `50` | CI smoke tests |

---

## Sharding

Distributed renders use global frame indices in filenames (`frame_0123` → \(g=123\)). Merge COCO with:

```bash
python ../../scripts/merge_coco_shards.py out.json shard0.json shard1.json ...
```
