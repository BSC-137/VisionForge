"""
End-to-end optical flow regression test.

Spawns the visionforge binary in scenario mode with a 2-frame camera
trajectory, then verifies that at least 10% of non-sky pixels on frame 1
carry non-zero optical flow — catching silent regressions where the EXR
flow channels are written as all-zeros.

Skipped automatically when:
  - the binary has not been built (build/visionforge missing), or
  - test_cube.obj is not present at the repo root, or
  - the visionforge_loader package itself is unavailable.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Repo-relative paths resolved from this file's location
#   tests/test_flow_e2e.py
#     → tests/
#       → visionforge_loader/
#         → python/
#           → VisionForge/  ← repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BINARY    = _REPO_ROOT / "build" / "visionforge"
_OBJ_PATH  = _REPO_ROOT / "test_cube.obj"

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------
_needs_binary = pytest.mark.skipif(
    not _BINARY.is_file(),
    reason=f"visionforge binary not built: {_BINARY}",
)
_needs_obj = pytest.mark.skipif(
    not _OBJ_PATH.is_file(),
    reason=f"test_cube.obj not found at repo root: {_OBJ_PATH}",
)


def _make_flow_scenario_config(dataset_root: str) -> dict:
    """Minimal 2-keyframe trajectory scenario for flow testing."""
    return {
        "assets": [
            {
                "name": "cube",
                "path": "test_cube.obj",
                "weight": 1.0,
                "scale": 1.2,
                "color": "white",
                "label": "cube",
                "class_id": 1,
                "y_offset": 0.6,
                "roughness": {"min": 0.3, "max": 0.7},
                "metallic":  {"min": 0.0, "max": 0.4},
            }
        ],
        "terrain": {
            "amp": 0.0, "scale": 0.0,
            "nx": 16, "nz": 16,
            "xmin": -12.0, "xmax": 12.0,
            "zmin": -12.0, "zmax": 12.0,
        },
        "lighting": {
            "sun_azimuth_deg": 270.0,
            "sun_elevation_deg": 25.0,
        },
        "dataset": {"root": dataset_root, "train_split": 1.0},
        "scenarios": [
            {
                "name": "FlowE2E",
                "camera": {
                    "lookat": [0.0, 0.5, 0.0],
                    "up":     [0.0, 1.0, 0.0],
                    "fov_deg": 35.0,
                    "trajectory": [
                        {"t": 0.0, "pos": [18.0,  8.0, 24.0]},
                        {"t": 1.0, "pos": [-18.0, 8.0, 24.0]},
                    ],
                },
                "root_nodes": [
                    {
                        "name": "FlowCube",
                        "asset": "cube",
                        "position": [0.0, 0.0, 0.0],
                        "rotation": [0.0, 30.0, 0.0],
                        "scale":    [1.0, 1.0, 1.0],
                        "grounding_constraint": True,
                    }
                ],
            }
        ],
    }


@_needs_binary
@_needs_obj
def test_flow_nonzero_after_camera_move(tmp_path: Path) -> None:
    """
    Frame 1 of a 2-frame camera-trajectory render must have >=10% of
    non-sky pixels with |flow| > 0.1 px.  Frame 0 always has zero flow
    (no previous frame), so frame 1 is the first meaningful signal.
    """
    from visionforge_loader.exr_io import read_spatial_exr  # noqa: PLC0415

    dataset_out = str(tmp_path / "dataset")
    config = _make_flow_scenario_config(dataset_out)

    config_path = tmp_path / "flow_e2e.json"
    config_path.write_text(json.dumps(config, indent=2))

    result = subprocess.run(
        [
            str(_BINARY), "scenario",
            "--config", str(config_path),
            "--name",   "FlowE2E",
            "--frames", "2",
            "--width",  "64",
            "--height", "64",
            "--spp",    "2",
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(
            f"visionforge exited {result.returncode}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    # With train_split=1.0 and 2 frames, both frames land in train/
    exr_path = Path(dataset_out) / "train" / "sfrm_0001_spatial.exr"
    assert exr_path.is_file(), (
        f"Expected EXR not found: {exr_path}\n"
        f"Listing {Path(dataset_out)}: {list(Path(dataset_out).rglob('*.exr'))}"
    )

    frame = read_spatial_exr(str(exr_path))

    # Sky pixels have effectively infinite depth — use 1e28 as the threshold
    sky_mask  = frame.depth >= 1e28
    non_sky   = ~sky_mask
    total_non_sky = int(non_sky.sum())

    if total_non_sky == 0:
        pytest.skip("All pixels are sky — scene geometry did not land in frame")

    flow_mag    = np.sqrt(frame.flow[0] ** 2 + frame.flow[1] ** 2)
    nonzero     = (flow_mag > 0.1) & non_sky
    fraction    = float(nonzero.sum()) / total_non_sky

    assert fraction >= 0.10, (
        f"Only {fraction:.1%} of non-sky pixels ({nonzero.sum()}/{total_non_sky}) "
        f"have |flow| > 0.1 px on frame 1.  "
        f"Flow range: [{flow_mag.min():.4f}, {flow_mag.max():.4f}]"
    )
