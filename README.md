# VisionForge

**VisionForge** is a high-performance Spatial AI engine and synthetic data factory built on a physically-based CPU path tracer in modern C++20. It renders procedural desert scenes with PBR materials and exports ground-truth data -- RGB images, depth maps, surface normals, instance masks, bounding boxes, and YOLO/COCO labels -- for training computer vision and spatial perception models. Designed for teams building object detection, depth estimation, surface normal prediction, or 6-DoF pose pipelines that need large, photorealistic, annotated datasets without manual labeling.

---

## Highlights

* **Cook-Torrance PBR** with GGX microfacet distribution, Fresnel-Schlick approximation, and importance-sampled specular lobes
* **Forge data factory**: `visionforge forge` generates thousands of labeled frames with full domain randomization from a single JSON config
* **Async I/O pipeline**: dedicated background thread writes PNG, EXR, YOLO, and COCO while the next frame renders
* **Thread-local xoshiro256+ PRNG**: zero-contention random number generation across OpenMP threads
* **G-Buffer export**: per-pixel depth, **InstanceID**, world-space normals, and **screen-space optical flow** in one multi-channel OpenEXR (`Depth`, `InstanceID`, `Normal.X/Y/Z`, `flow.x`, `flow.y`; all float32—instance ids round-trip exactly when ≤ 16,777,215)
* **One-pass labeling**: bounding boxes computed from the G-Buffer in a single scan, feeding YOLO `.txt`, COCO `.json`, CSV, and JSON simultaneously
* **BVH acceleration** with longest-extent axis splitting over all scene primitives
* **Phase 8 textures without Phase 7 slowdown**: UVs (and normal-map TBN basis) are computed lazily *only when* a material actually has textures bound
* **Procedural terrain**: ridged fractal heightfield with domain-warped Perlin noise and micro-bump normals
* **Semantic grounding**: terrain-aware gravity snapping + slope alignment, so cubes and assets sit flush on dunes with a controllable “sink” depth
* **Analytic sky + HDR IBL**: built-in sky model with sun + optional `.hdr` environment maps via `stb_image.h` (equirectangular projection)
* **Triplanar terrain texturing**: “no-UV” ground material that blends a single texture along XYZ, avoiding stretching on steep slopes
* **Deterministic Scenario Engine**: Handcrafted node-based scenes configured through JSON with kinematic hierarchy (`ScenarioNode`), while retaining domain randomization limits (e.g. lighting/PBR parameters).
* **Geometry**: OBJ mesh loading (via fast_obj), triangles, axis-aligned rectangles, procedural heightfields
* **Materials**: PBR (metallic/roughness), Lambertian diffuse, dielectric, metal, emissive area lights
* **Adaptive sampling** with Welford variance and 95% confidence interval early termination
* **OpenMP parallelization** across all render paths
* **`manifest.json` (schema `2`)**: one file per dataset (or per shard with `--num-shards`) recording the exact CLI args, seed, RNG contract, resolved config SHA-256, train/val split, and a stable `dataset_id` fingerprint — everything needed to reproduce or audit a run.

---

## Project Layout

The C++ engine lives under `apps/` and `include/visionforge/`. The Python consumption layer (`visionforge_loader`) and data-factory scripts (`scripts/`) are fully independent and do not require rebuilding the engine after changes.

```
VisionForge/
├─ apps/visionforge/main.cpp       # CLI application (manual + forge paths)
├─ include/visionforge/            # Engine headers
│  ├─ pbr_material.hpp             #   PBR material (Cook-Torrance, GGX sampling)
│  ├─ world_config.hpp             #   Forge JSON config parser (render, camera, lighting, terrain, assets[])
│  ├─ dataset_manifest.hpp         #   Dataset export manifest (`manifest.json`) helpers + schema version constant
│  ├─ config_keys.hpp              #   Canonical JSON key names for configs
│  ├─ mesh.hpp                     #   OBJ mesh loader (per-mesh BVH)
│  ├─ bvh.hpp                      #   BVH node (longest-extent axis split)
│  ├─ passes.hpp                   #   GBuffer (depth, normals, instance IDs)
│  ├─ sky.hpp                      #   Analytic sky model
│  ├─ hdr_sky.hpp                  #   HDR environment sky (equirectangular .hdr)
│  ├─ image_texture.hpp            #   Shared float textures with bilinear sampling
│  ├─ triplanar_material.hpp       #   Triplanar terrain material using ImageTexture
│  ├─ asset_manager.hpp            #   Multi-asset loader + weighted selection
│  ├─ terrain.hpp                  #   HeightField + world-space height/normal queries
│  ├─ placement.hpp                #   SlopeAlign + snap_y grounding helpers
│  ├─ vec3.hpp                     #   Vec3 + thread-local xoshiro256+ PRNG
│  ├─ meta_pose.hpp               #   Camera c2w + intrinsics + logical object TRS helpers for dataset meta
│  ├─ exr_writer.hpp               #   EXR export
│  ├─ png_writer.hpp               #   PNG export (zlib)
│  ├─ scene_graph.hpp              #   Hierarchical transforms and Node architecture
│  └─ ...                          #   materials, camera, labels, transforms
├─ src/io/
│  ├─ dataset_manifest.cpp         # Atomic manifest writer + resolved-config JSON + SHA-256
│  ├─ exr_writer.cpp               # TinyEXR implementation
│  ├─ png_writer.cpp               # Minimal PNG writer
│  ├─ fast_obj.cpp                 # fast_obj implementation unit
│  └─ stb_image_impl.cpp           # stb_image implementation unit (HDR + LDR)
├─ third_party/
│  ├─ tinyexr.h                    # Single-header OpenEXR (MIT)
│  ├─ fast_obj.h                   # Single-header OBJ loader (MIT)
│  ├─ nlohmann/                    # JSON for Modern C++ (MIT)
│  └─ stb_image.h                  # Single-header image loader (MIT/public domain)
├─ scripts/                        # validate_dataset.py, merge_coco_shards.py, dev_smoke.sh
├─ python/visionforge_loader/      # PyTorch loader, examples/, tests/
├─ world.json                      # Example forge config with DR ranges + asset library
├─ test_cube.obj                   # Minimal test geometry
├─ CMakeLists.txt
├─ LICENSE                         # BSD-3-Clause
└─ README.md
```

---

## Prerequisites

**Ubuntu / Debian**

```bash
sudo apt update
sudo apt install -y build-essential cmake zlib1g-dev
```

A compiler with **OpenMP** support (GCC, Clang, or MSVC) enables multithreaded rendering.

**Python (optional — loader, examples, validation scripts)**

```bash
pip install ./python/visionforge_loader        # PyTorch loader
pip install ./python/visionforge_loader[dev]   # + pytest for tests
```

---

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVISIONFORGE_OMP=ON
cmake --build build -j
```

The executable is written to `./build/visionforge`.

Optional parser tests:

```bash
ctest --test-dir build -R world_config
```

## Verification

Optional post-render QA for exported dataset directories (Python 3.10+, stdlib only):

```bash
./scripts/validate_dataset.py --dataset-root <outdir>
```

See `./scripts/validate_dataset.py --help` for `--split`, `--strict`, `--json-report`, COCO discovery overrides, bbox thresholds, and `--check-meta` (pose/camera sidecars).

---

## Forge: Synthetic Data Factory

The `forge` subcommand generates labeled training data at scale with full domain randomization. A single JSON config (`world.json`) defines randomization ranges for camera position, sun angles, object placement, and PBR material properties.

**Generate 1,000 frames:**

```bash
./build/visionforge forge --config world.json --frames 1000
```

### Per-frame outputs (train/val split)

| File | Description |
|------|-------------|
| `frame_XXXX.png` | PBR-rendered RGB |
| `frame_XXXX_spatial.exr` | G-Buffer EXR: `Depth`, `InstanceID`, `Normal.X`, `Normal.Y`, `Normal.Z`, `flow.x`, `flow.y` (all float32) |
| `frame_XXXX_meta.json` | Frame metadata: `camera_extrinsics` (16 floats, **c2w** row-major), `camera_intrinsics` (`fx`,`fy`,`cx`,`cy`, vertical `vfov_deg`), legacy `camera` block, sun, and `objects[]` with full `rotation_deg` / `scale`, `local_to_world_row_major`, `grounding_constraint`, `terrain_normal`, `transform_supervision` |

The `scenario` subcommand uses the same sidecar layout with stems `sfrm_XXXX` (instead of `frame_XXXX`) and writes `scenario_coco.json` at the dataset root (or `scenario_coco_shard_K.json` when sharding).

### G-Buffer channels

| EXR channel | Type | Description |
|-------------|------|-------------|
| `Depth` | float32 | Linear distance from camera origin to hit point (metres); `1e30` for sky/miss |
| `InstanceID` | float32 | Per-pixel instance id (integer round-trips exactly for ids ≤ 16 777 215) |
| `Normal.X` | float32 | World-space surface normal X |
| `Normal.Y` | float32 | World-space surface normal Y |
| `Normal.Z` | float32 | World-space surface normal Z |
| `flow.x` | float32 | **Optical flow** — horizontal displacement in pixels: \(u_{\text{prev}} - u_{\text{curr}}\) |
| `flow.y` | float32 | **Optical flow** — vertical displacement in pixels: \(v_{\text{prev}} - v_{\text{curr}}\) |

**Optical flow sign convention**: positive `flow.x` means the scene moved right between the previous and current frame (i.e. the camera moved left). Zero for sky/miss pixels and for the first frame in a sequence. Computed from a post-render pinhole reprojection of each depth pixel through the previous frame's camera.

### Meta / pose convention (`*_meta.json`)

- **`camera_extrinsics`**: row-major \(4 \times 4\) **camera-to-world** transform \(P_{\mathrm{world}} = M \, P_{\mathrm{cam}}\) (homogeneous column). Rotation columns in world space follow an OpenCV-style right-handed camera: **+X** = renderer `Camera::u` (right), **+Y** = \(-\) `Camera::v` (image \(t\) increases along \(+v\), so “down” in pixels aligns with \(+v\) in world), **+Z** = \(-\) `Camera::w` (forward into the scene; `w = normalize(lookfrom - lookat)`). Translation is `Camera::origin` (`lookfrom`). Derived from the same `Camera` instance used for `get_ray`, not a recomputed look-at.
- **`camera_intrinsics`**: pinhole **\(f_x, f_y, c_x, c_y\)** chosen so pixel \((i,j)\) maps to `get_ray`’s \((s,t) = (i/\max(W-1,1),\, j/\max(H-1,1))\), with \(c_x=(W-1)/2\), \(c_y=(H-1)/2\). `fov_deg` in `camera` is **vertical** FOV (same as `vfov_deg` in intrinsics). Use `skew: 0`. When projecting a 3D point with \(\mathbf{P}_c=\mathrm{w2c}\,\mathbf{P}_w\), use \(u = f_x X_c/Z_c + c_x\) and \(v = -f_y Y_c/Z_c + c_y\) (v grows downward, matching the `Camera::get_ray` / `meta_pose` tests).
- **Objects**: `instance_id` matches G-buffer / EXR **`InstanceID`**. `local_to_world_row_major` encodes **logical** pose \(R_z R_y R_x \,\mathrm{diag}(s)\) plus translation from `SceneNode::world_transform`, matching Euler order in `flat_pack` **before** terrain **slope alignment** and **vertex snap** on grounded assets. When `grounding_constraint` is true, `transform_supervision` is `logical_excludes_slope_and_vertex_snap`; use rasterized labels / EXR for exact silhouette supervision.
- **`validate_dataset.py --check-meta`** expects the fields above. Older datasets without `camera_extrinsics` will fail this check until re-rendered. Run validation **after** merging shard outputs if you split COCO across `annotations_coco_shard_*.json` (see **Deterministic sharding**).

### Python loader (`python/visionforge_loader`)

A small **PyTorch**-first package (`pip install ./python/visionforge_loader`) loads `train/` / `val/` frames, parses `*_meta.json` + `*_spatial.exr`, and includes pytest checks that **world ↔ image projection** matches this README. See `python/visionforge_loader/README.md` and `python -m visionforge_loader.cli_projection_smoke`. `examples/train_supervision_baseline.py` is a minimal RGB→depth / RGB→normal supervision demo using only the public `VisionForgeDataset` API (details and run commands in that README). See also `export_pointcloud.py` in the loader examples for 3D back-projection of the G-buffer depth into world-space point clouds.

### Domain randomization parameters (`world.json`)

Minimal example using the new asset library and optional HDR sky:

```json
{
  "render": {
    "width": 320,
    "height": 180,
    "spp": 4,
    "max_depth": 6,
    "seed": 123
  },
  "camera": {
    "lookat": [0.0, 1.2, 0.0],
    "up": [0.0, 1.0, 0.0],
    "lookfrom": {
      "min": [14.0, 7.0, 20.0],
      "max": [20.0, 10.0, 28.0]
    },
    "fov_deg": { "min": 30.0, "max": 40.0 }
  },
  "lighting": {
    "sun_azimuth_deg": { "min": 220.0, "max": 320.0 },
    "sun_elevation_deg": { "min": 8.0, "max": 28.0 }
  },
  "terrain": {
    "amp": 1.8,
    "scale": 0.14,
    "nx": 64,
    "nz": 64,
    "bounds": {
      "xmin": -22.0,
      "xmax": 22.0,
      "zmin": -22.0,
      "zmax": 22.0
    }
  },
  "assets": [
    {
      "name": "rock_01",
      "path": "models/rock.obj",
      "weight": 0.8,
      "scale": 1.0,
      "color": "sand",
      "label": "rock",
      "class_id": 2,
      "y_offset": 0.0,
      "roughness": { "min": 0.3, "max": 0.8 },
      "metallic": { "min": 0.0, "max": 0.2 }
    },
    {
      "name": "rover",
      "path": "models/sirius.obj",
      "weight": 0.2,
      "scale": 1.4,
      "color": "white",
      "label": "rover",
      "class_id": 3,
      "y_offset": 0.15,
      "roughness": { "min": 0.05, "max": 0.5 },
      "metallic": { "min": 0.2, "max": 1.0 }
    }
  ],
  "placement": {
    "x": { "min": -12.0, "max": 12.0 },
    "z": { "min": -12.0, "max": 12.0 },
    "yaw_deg": { "min": 0.0, "max": 360.0 }
  },
  "dataset": {
    "root": "dataset",
    "train_split": 0.8
  },
  "hdr": {
    "path": "mars.hdr",
    "intensity": 1.0
  }
}
```

### Config schema notes (canonical keys & validation)

- **Stable identifiers**: Root sections include `render`, `camera`, `lighting`, `terrain`, `assets` (or legacy singular `asset`), optional `hdr`, `placement`, `dataset`, and optional `scenarios`. Canonical mesh paths live under **`path`**. Canonical asset textures use **`albedo_map`**, **`normal_map`**, **`roughness_map`**, **`metallic_map`** (see `include/visionforge/config_keys.hpp`).
- **Strict parsing**: `visionforge forge` and `visionforge scenario` load configs with **strict validation** (`unknown keys` are rejected; deprecated aliases fail with an actionable message). Programmatic callers may pass `{ .strict = false }` to `load_world_config` when migrating legacy files (warnings print to stderr; aliases map onto canonical fields).
- **Deprecation**: **`albedo_path`**, **`normal_path`**, **`roughness_path`**, **`metallic_path`** → use `*_map` keys instead. **`obj`** as a mesh path field → use **`path`**. Deprecated texture aliases emit `[world.json warning]` when non-strict; strict mode rejects them.
- **`terrain` bounds**: Use **`terrain.bounds`** `{ xmin, xmax, zmin, zmax }` and/or **flat** `xmin`/`xmax`/`zmin`/`zmax` on `terrain` (flat keys override `bounds` if both are set; a warning is printed when non-strict).
- **`hdr`**: Either a **string** path at `"hdr"`, or an **object** `{ "path": "...", "intensity": 1.0 }`.
- **Scenarios**: Each scenario requires a non-empty **`name`**. **`root_nodes[].asset`** must reference an **`assets[].name`** defined in the same file.

### Performance

At 320x180, forge renders **~12ms per frame** (20 OMP threads) with I/O fully overlapped. 1,000 frames complete in under 30 seconds.

---

## Scenario: Deterministic Scene Graphs

The `scenario` subcommand supports crafting distinct environments explicitly mapped by a JSON `SceneNode` hierarchy, ensuring layout invariants while still supporting Domain Randomization for generalized lighting and materials.

**Render a scenario:**

```bash
./build/visionforge scenario --config world.json --name "MyScenario" --frames 10
```

Where a `world.json` might define a `Scenario` explicitly structuring parent-child transform inheritance and physical grounding properties:

```json
{
  "scenarios": [
    {
      "name": "MyScenario",
      "root_nodes": [
        {
          "name": "BaseRover",
          "asset": "rover",
          "position": [0, 0, 0],
          "rotation": [0, 45, 0],
          "scale": [1, 1, 1],
          "grounding_constraint": true,
          "children": [
            {
              "name": "CargoBox",
              "asset": "box",
              "position": [1.5, 2.0, 0],
              "scale": [0.5, 0.5, 0.5]
            }
          ]
        }
      ]
    }
  ]
}
```

The underlying pipeline traces nested `local_to_world` conversions globally into perfectly registered YOLO / COCO JSON matrices for complex geometries.

---

## Camera Trajectories

By default every scenario frame is rendered from the fixed `lookfrom` position. Adding a `trajectory` array to a scenario's `camera` block enables **per-frame camera movement**, making it possible to generate optical-flow ground truth and multi-view datasets in a single pass.

### How it works

Each keyframe has a normalized time `t ∈ [0, 1]` and a `pos` (world-space camera origin). The engine linearly interpolates between keyframes using `alpha = g / (frames - 1)` for frame `g`. When no `trajectory` is provided the existing `lookfrom` behaviour is preserved — fully backwards compatible.

### Example: linear dolly over 100 frames

```json
{
  "assets": [{"path": "assets/rover.obj", "name": "rover"}],
  "scenarios": [
    {
      "name": "DollyShot",
      "camera": {
        "lookat": [0.0, 1.2, 0.0],
        "fov_deg": 35,
        "trajectory": [
          {"t": 0.0, "pos": [ 18.0, 8.0, 24.0]},
          {"t": 1.0, "pos": [-18.0, 8.0, 24.0]}
        ]
      },
      "root_nodes": [
        {"name": "rover", "asset": "rover", "position": [0, 0, 0], "grounding_constraint": true}
      ]
    }
  ]
}
```

Render with:

```bash
./build/visionforge scenario --config world.json --name DollyShot --frames 100
```

This sweeps the camera from `(18, 8, 24)` to `(-18, 8, 24)` in 100 evenly-spaced steps, writing a `meta.json` per frame that records the exact camera origin — ready for optical-flow or NeRF training pipelines.

### Multi-stop trajectory

More than two keyframes are supported. Keyframes are automatically sorted by `t` and the correct pair is selected for each frame:

```json
"trajectory": [
  {"t": 0.00, "pos": [ 18.0,  8.0, 24.0]},
  {"t": 0.50, "pos": [  0.0, 14.0,  0.0]},
  {"t": 1.00, "pos": [-18.0,  8.0, 24.0]}
]
```

---

## Manual Render

The default path renders a single scene with labeled cubes and optional OBJ meshes.

**Quick render (default 320x180, 1 spp):**

```bash
./build/visionforge --out out
```

**High-quality render with OBJ mesh:**

```bash
./build/visionforge --out out --spp 128 --width 640 --height 360 \
  --obj model.obj --obj-pos "0,2,0" --obj-scale 3.0 --obj-color blue --exr
```

**Single-file output** (no labels, just the image):

```bash
./build/visionforge --out hero.png --spp 64
```

### I/O gating

| `--out` value | Behavior |
|---------------|----------|
| `file.ppm` or `file.png` | Writes only that image file |
| `directory/` | Writes full labeled dataset (see table below) |

### Directory output files

| File | Description |
|------|-------------|
| `image.ppm`, `image.png` | Tonemapped RGB (ACES + sqrt gamma) |
| `gbuffer.exr` | 32-bit float EXR: `Depth`, `InstanceID`, `Normal.X`, `Normal.Y`, `Normal.Z`, `flow.x`, `flow.y` (with `--exr`) |
| `inst.pgm` | Per-pixel instance mask (8-bit; ids above 255 clamped—use EXR `InstanceID` for full precision) |
| `labels_yolo.txt` | YOLO-format labels |
| `labels_coco.json` | COCO annotations |
| `bboxes.csv`, `bboxes.json` | Bounding boxes from G-Buffer |
| `manifest.json` | Render metadata (spp, timing, seed) |

**Fast depth-only pass:**

```bash
./build/visionforge --out out --depth-only
```

---

## CLI Reference

### Render

| Flag | Default | Description |
|------|---------|-------------|
| `--out` | `out` | Output path (file or directory) |
| `--width`, `--height` | `320`, `180` | Image resolution |
| `--spp` | `1` | Samples per pixel |
| `--max-depth` | `6` | Maximum bounce depth |
| `--preview` | off | Fast preset (12 spp, depth 6) |
| `--seed` | `1337` | RNG seed |
| `--exposure` | `6.5` | Exposure multiplier |
| `--sky-gain` | `45` | Sky brightness |
| `--exr` | off | Write `gbuffer.exr` (`Depth`, `InstanceID`, `Normal.X/Y/Z`, `flow.x`, `flow.y`) |
| `--depth-only` | off | Primary-ray pass only (implies `--exr`) |

### Camera

| Flag | Default | Description |
|------|---------|-------------|
| `--lookfrom` | `18,8,24` | Camera position |
| `--lookat` | `0,1.2,0` | Look-at target |
| `--fov` | `35` | Vertical FOV (degrees) |

### Terrain

| Flag | Default | Description |
|------|---------|-------------|
| `--terrain-amp` | `1.8` | Dune height amplitude |
| `--terrain-scale` | `0.14` | Feature size |
| `--terrain-nx`, `--terrain-nz` | `96` | Heightfield tessellation |
| `--sand-bump-freq` | `5.0` | Micro-bump frequency |
| `--sand-bump-scale` | `0.22` | Micro-bump strength |
| `--world-bounds` | `-22,22,-22,22` | World extents |

### Cubes

| Flag | Default | Description |
|------|---------|-------------|
| `--cubes` | `4` | Number of cubes |
| `--cube-edge-min`, `--cube-edge-max` | `1.4`, `1.9` | Edge length range |
| `--cube-colors` | `red,green,blue,white` | Colors (cycled) |
| `--placement` | `grid` | `grid` or `random` |
| `--min-spacing` | `3.5` | Min separation (random mode) |
| `--tilt-abs` | `12` | Max random tilt (degrees) |

### OBJ Mesh

| Flag | Default | Description |
|------|---------|-------------|
| `--obj` | | OBJ file to load |
| `--obj-pos` | `0,0,0` | World-space translation |
| `--obj-scale` | `1.0` | Uniform scale |
| `--obj-color` | `white` | Material color (name, `#hex`, or `r,g,b`) |

### Lighting

| Flag | Default | Description |
|------|---------|-------------|
| `--light-x` | `-30` | Area light X position |
| `--light-y` | `2,14` | Vertical span |
| `--light-z` | `-25,25` | Depth span |
| `--light-intensity` | `8000` | Radiance multiplier |
| `--light-color` | `1,0.98,0.92` | Emissive color |
| `--show-light` | `false` | Show rect light to camera |
| `--match-sky` | `true` | Align sun to light normal |
| `--sun-az`, `--sun-el` | `300`, `12` | Sun azimuth / elevation |

### Color syntax

Colors can be: names (`red`, `blue`, `white`, `sand`, `yellow`, `magenta`, `cyan`, `gray`), hex (`#ffaa00`), or RGB triples (`0.1,0.2,0.9`). Separate multiple entries with commas; use semicolons when mixing with raw RGB (`"red;#3cb371;0.1,0.2,0.9"`).

---

## Forge CLI (Bun-style overrides)

Forge has a focused CLI that layers on top of `world.json`. JSON provides the defaults; **CLI flags always win**, similar to Bun:

```bash
./build/visionforge forge --config world.json --frames 100 \
  --width 640 --height 360 --spp 128 \
  --hdr mars.hdr --hdr-intensity 1.5 \
  --ground-tex sand.png --ground-scale 3.0 \
  --verbose
```

| Flag | Description |
|------|-------------|
| `--config` | Path to `world.json` |
| `--frames` | Number of frames to render |
| `--width`, `--height` | Override `render.width` / `render.height` |
| `--spp` | Override `render.spp` (samples per pixel) |
| `--hdr` | Override `hdr.path` (HDR environment map, equirectangular) |
| `--hdr-intensity` | Override `hdr.intensity` |
| `--ground-tex` | Override triplanar ground texture path |
| `--ground-scale` | Override triplanar UV scale |
| `--verbose` | Print per-frame grounding logs (asset label, snap-Y, normal, sink) |
| `--shard-id` | Distributed sharding: this worker renders global frame `g` only when `g % num_shards == shard_id` (default `0`) |
| `--num-shards` | Shard count `W ≥ 1`. Each process still iterates all `g ∈ [0, frames)` so domain-randomization `std::mt19937` matches a single run; path tracer uses `vf_rng` seed `render.seed + g`. With `W>1`, writes `manifest_shard_{id}.json` and `annotations_coco_shard_{id}.json`. Merge COCO with `scripts/merge_coco_shards.py`. |

### Deterministic sharding

Use the same `world.json`, `--frames F`, `--shard-id K`, and `--num-shards W` on each worker. **Global** indices appear in filenames (`frame_0007` is always frame 7). Train/val split is based on `g`, not a per-shard counter. Pixel-identical output for frame `g` is reproduced whether rendered in one process (`W=1`) or on the shard that owns `g`.

- Uses **semantic grounding** (heightfield raycast, slope-aligned up vector, and randomized “sink” depth) for whichever asset is sampled.
- Draws terrain with either the analytic sand material or the triplanar ground texture.
- Uses the analytic sky or an HDR environment, depending on `hdr` / `--hdr`.

---

## Performance

| Scenario | Time |
|----------|------|
| Forge frame (320x180, 1 spp, PBR) | ~12ms |
| Forge 1,000 frames (full labels) | ~30s |
| Manual render (320x180, 1 spp) | ~0.2s |
| Manual render (320x180, 128 spp) | ~3s |
| Manual render (640x360, 128 spp) | ~12s |

Notes:

- **Lazy texture work**: `MeshTriangle::hit()` returns only geometry; `PBRMaterial::scatter()` computes UVs from barycentrics only on the textured branch.
- **Scene graph packing cache**: `SceneNode::flat_pack()` caches the baked BVH when the node’s object + world transform are unchanged, avoiding per-frame BVH rebuilds in `forge` / `scenario`.

Measured on 20-thread Xeon (WSL2). Build with `-DCMAKE_BUILD_TYPE=Release -DVISIONFORGE_OMP=ON`.

---

## Troubleshooting

* **Dark image** -- raise `--exposure` (e.g. 8-10).
* **Sky too bright/dim** -- adjust `--sky-gain`.
* **Noisy** -- increase `--spp` or lower `--sand-bump-scale`.
* **OBJ won't load** -- check the path and that the file contains at least one face.

---

## Testing

Run a **Release** build with OpenMP, then **CTest** for C++ contracts (world JSON parser, dataset manifest writer, meta/pose intrinsics), the **visionforge** CLI smoke checks (`--help` / unknown flag), **loader** pytest (geometry, projection, EXR wiring, training smoke), **dataset validator** pytest on synthetic trees, and **`dev_smoke.sh`** for forge → validate → loader tests on a throwaway dataset. From the repo root (with pytest installed for Python tests):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVISIONFORGE_OMP=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
cd python/visionforge_loader && PYTHONPATH=. python3 -m pytest -q
cd ../.. && python3 -m pytest tests/test_validate_dataset_contracts.py -q
python3 -m pytest tests/test_merge_coco_shards.py -q
python3 scripts/validate_dataset.py --dataset-root /path/to/dataset --split all --check-meta
./scripts/dev_smoke.sh
# For scenario end-to-end testing:
VF_SMOKE_SCENARIO=1 ./scripts/dev_smoke.sh
```

GitHub Actions runs all of the above on every push and pull request to main.

`validate_dataset.py` supports `--strict` (warnings fail) and `--json-report PATH` for machine-readable CI output; `--check-meta` enforces per-frame `*_meta.json` schema in addition to COCO/YOLO pairing.

## Developer tooling

### Repository smoke (`scripts/dev_smoke.sh`)

`./scripts/dev_smoke.sh` incrementally builds the C++ target (running CMake configure only when `build/visionforge` is absent), renders three `forge` frames into a **temporary** dataset whose `dataset.root` is injected via a generated JSON (the tree is removed on exit), runs `scripts/validate_dataset.py --check-meta` on that export, and runs `python/visionforge_loader` pytest with `PYTHONPATH=.` so the package imports without an editable install. Loader tests require the dependencies in `python/visionforge_loader/README.md`, including `pytest` from the `[dev]` extra (`pip install ./python/visionforge_loader[dev]`).

```bash
./scripts/dev_smoke.sh
```

### Verifying a rendered dataset

```bash
# 1. Generate a representative dataset from repo root:
./build/visionforge forge --config world.json --frames 100

# 2. Validate disk layout, COCO/YOLO pairing, and per-frame meta schema:
python3 scripts/validate_dataset.py \
  --dataset-root dataset --split all \
  --check-meta --strict \
  --json-report dataset/validation_report.json

# 3. Verify camera projection math on real EXR + meta files:
python3 -m visionforge_loader.cli_projection_smoke \
  --dataset-root dataset --max-frames 5

# 4. Run the training baseline on CPU (from python/visionforge_loader):
PYTHONPATH=. python3 examples/train_supervision_baseline.py \
  --dataset-root ../../dataset \
  --epochs 1 --max-samples 16 --batch-size 2 --device cpu
```

Step 1 produces the synthetic export; step 2 checks disk layout and label contracts; step 3 confirms pose math round-trips through the pinhole model; step 4 proves the tensors are valid trainable ML inputs.

Use `--frames 3` for a quick sanity check; use 500–1000 for a real training dataset.

> **Note:** `world.json` ships a `test_cube.obj` asset. Swap in your own OBJ under `assets[]` for production datasets (see **Forge CLI** section above).

### Baseline supervision (optional)

For a tiny PyTorch CNN that predicts depth or world normals from RGB using existing loader tensors, see **Examples** in `python/visionforge_loader/README.md` (`examples/train_supervision_baseline.py`).

## License

BSD-3-Clause. See `LICENSE`.
