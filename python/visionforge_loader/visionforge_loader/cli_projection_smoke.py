"""Smoke: load a few frames and verify projection round-trips."""

from __future__ import annotations

import argparse
import math

import numpy as np

from visionforge_loader.dataset import VisionForgeDataset
from visionforge_loader.geometry import (
    backproject_depth_to_world,
    pinhole_intrinsics_match_renderer,
    project_world_to_pixel,
    w2c_from_c2w,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="VisionForge projection smoke test on a dataset root.")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--max-frames", type=int, default=4)
    args = p.parse_args(argv)

    ds = VisionForgeDataset(args.dataset_root, split="both", stem_pattern="auto")
    n = min(len(ds), args.max_frames)
    if n == 0:
        print("No frames found.")
        return 2

    for i in range(n):
        item = ds[i]
        meta = item["meta"]
        depth = item["depth"].numpy()
        inst = item["instance_id"].numpy()
        H, W = depth.shape
        c2w = meta.c2w.numpy().astype(np.float64)
        w2c = w2c_from_c2w(c2w)
        cam_json = meta.json_raw["camera"]
        origin = np.array(cam_json["lookfrom"], dtype=np.float64)

        # Test A: principal center pixel round-trip
        ic, jc = meta.cx, meta.cy
        fx, fy, cx, cy = pinhole_intrinsics_match_renderer(W, H, meta.vfov_deg)
        assert abs(fx - meta.fx) < 1e-2
        d = 5.0
        p_w = backproject_depth_to_world(origin, c2w, ic, jc, meta.fx, meta.fy, meta.cx, meta.cy, d)
        uv = project_world_to_pixel(w2c, meta.fx, meta.fy, meta.cx, meta.cy, p_w)
        assert uv is not None
        err = math.hypot(uv[0] - ic, uv[1] - jc)
        if err > 0.5:
            raise SystemExit(f"center ray round-trip error {err} px on frame {meta.stem}")

        # Test B: random foreground pixels with valid depth
        rng = np.random.default_rng(0)
        tries = 0
        ok = 0
        while ok < 8 and tries < 200:
            tries += 1
            ii = int(rng.integers(0, W))
            jj = int(rng.integers(0, H))
            dep = float(depth[jj, ii])
            ins = float(inst[jj, ii])
            if ins <= 0 or dep <= 0 or not math.isfinite(dep):
                continue
            p_world = backproject_depth_to_world(
                origin, c2w, float(ii), float(jj), meta.fx, meta.fy, meta.cx, meta.cy, dep
            )
            uv2 = project_world_to_pixel(w2c, meta.fx, meta.fy, meta.cx, meta.cy, p_world)
            if uv2 is None:
                continue
            err2 = math.hypot(uv2[0] - ii, uv2[1] - jj)
            if err2 > 1.0:
                raise SystemExit(f"depth round-trip error {err2} at ({ii},{jj}) on {meta.stem}")
            ok += 1
        if ok < 1:
            print(f"warn: no valid foreground pixels for depth test on {meta.stem}")
        else:
            print(f"ok depth round-trip ({ok} samples) {meta.stem}")

        print(f"ok principal-ray {meta.stem}")

    print(f"smoke passed ({n} frames).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
