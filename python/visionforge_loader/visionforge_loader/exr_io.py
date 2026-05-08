"""Read VisionForge *_spatial.exr (TinyEXR channel names)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

try:
    import Imath
    import OpenEXR
except ImportError:  # pragma: no cover
    OpenEXR = None
    Imath = None


@dataclass
class SpatialExr:
    depth: np.ndarray  # [H,W] float32
    instance_id: np.ndarray  # [H,W] float32 (round for uint semantics)
    normal: np.ndarray  # [3,H,W] float32 (X,Y,Z channels)


def read_spatial_exr(path: str) -> SpatialExr:
    if OpenEXR is None:
        raise ImportError("OpenEXR and Imath are required; pip install OpenEXR")

    exr = OpenEXR.InputFile(path)
    dw = exr.header()["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    def ch(name: str) -> np.ndarray:
        s = exr.channel(name, pt)
        arr = np.frombuffer(s, dtype=np.float32).reshape(h, w)
        return arr.copy()

    # Alphabetic storage order in file; loader maps by channel name.
    depth = ch("Depth")
    inst = ch("InstanceID")
    nx = ch("Normal.X")
    ny = ch("Normal.Y")
    nz = ch("Normal.Z")
    normal = np.stack([nx, ny, nz], axis=0)
    return SpatialExr(depth=depth, instance_id=inst, normal=normal)


def channel_shapes(path: str) -> Dict[str, Tuple[int, int]]:
    """Debug helper: channel name -> (width, height)."""
    if OpenEXR is None:
        raise ImportError("OpenEXR is required")
    exr = OpenEXR.InputFile(path)
    dw = exr.header()["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    return {name: (w, h) for name in exr.header()["channels"].keys()}
