"""Read VisionForge *_spatial.exr (TinyEXR channel names)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

try:
    import Imath
    import OpenEXR
except ImportError:  # pragma: no cover
    OpenEXR = None
    Imath = None


@dataclass
class SpatialFrame:
    depth: np.ndarray       # [H,W] float32 — linear distance from camera (metres)
    instance_id: np.ndarray  # [H,W] float32 (round for uint semantics)
    normal: np.ndarray      # [3,H,W] float32 (X,Y,Z world-space channels)
    # Screen-space optical flow (prev_frame − curr_frame) in pixels.
    # Shape [2,H,W] where channel 0 = flow.x (horizontal) and 1 = flow.y (vertical).
    # Filled with zeros if the channel is absent (backward-compatible with old EXR files).
    flow: np.ndarray = field(default_factory=lambda: np.zeros((2, 1, 1), dtype=np.float32))


# Backward-compatible alias — existing code using SpatialExr continues to work.
SpatialExr = SpatialFrame


def read_spatial_exr(path: str) -> SpatialFrame:
    if OpenEXR is None:
        raise ImportError("OpenEXR and Imath are required; pip install OpenEXR")

    exr = OpenEXR.InputFile(path)
    dw = exr.header()["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    available = set(exr.header()["channels"].keys())

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

    # Optical flow channels — optional for backward compatibility with old EXR files.
    if "flow.x" in available and "flow.y" in available:
        fx_arr = ch("flow.x")
        fy_arr = ch("flow.y")
        flow = np.stack([fx_arr, fy_arr], axis=0)
    else:
        flow = np.zeros((2, h, w), dtype=np.float32)

    return SpatialFrame(depth=depth, instance_id=inst, normal=normal, flow=flow)


def channel_shapes(path: str) -> Dict[str, Tuple[int, int]]:
    """Debug helper: channel name -> (width, height)."""
    if OpenEXR is None:
        raise ImportError("OpenEXR is required")
    exr = OpenEXR.InputFile(path)
    dw = exr.header()["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    return {name: (w, h) for name in exr.header()["channels"].keys()}
