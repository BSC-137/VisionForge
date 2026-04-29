# VisionForge

**VisionForge** is a high-performance Spatial AI engine and synthetic data factory built on a physically-based CPU path tracer in modern C++20. It renders procedural desert scenes with PBR materials and exports ground-truth data -- RGB images, depth maps, surface normals, instance masks, bounding boxes, and YOLO/COCO labels -- for training computer vision and spatial perception models.

---

## Highlights

* **Cook-Torrance PBR** with GGX microfacet distribution, Fresnel-Schlick approximation, and importance-sampled specular lobes
* **Forge data factory**: `visionforge forge` generates thousands of labeled frames with full domain randomization from a single JSON config
* **Async I/O pipeline**: dedicated background thread writes PNG, EXR, YOLO, and COCO while the next frame renders
* **Thread-local xoshiro256+ PRNG**: zero-contention random number generation across OpenMP threads
* **G-Buffer export**: per-pixel depth, world-space normals, and instance IDs as 32-bit float OpenEXR
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

---

## Project Layout

```
VisionForge/
├─ apps/visionforge/main.cpp       # CLI application (manual + forge paths)
├─ include/visionforge/            # Engine headers
│  ├─ pbr_material.hpp             #   PBR material (Cook-Torrance, GGX sampling)
│  ├─ world_config.hpp             #   Forge JSON config parser (render, camera, lighting, terrain, assets[])
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
│  ├─ exr_writer.hpp               #   EXR export
│  ├─ png_writer.hpp               #   PNG export (zlib)
│  ├─ scene_graph.hpp              #   Hierarchical transforms and Node architecture
│  └─ ...                          #   materials, camera, labels, transforms
├─ src/io/
│  ├─ exr_writer.cpp               # TinyEXR implementation
│  ├─ png_writer.cpp               # Minimal PNG writer
│  ├─ fast_obj.cpp                 # fast_obj implementation unit
│  └─ stb_image_impl.cpp           # stb_image implementation unit (HDR + LDR)
├─ third_party/
│  ├─ tinyexr.h                    # Single-header OpenEXR (MIT)
│  ├─ fast_obj.h                   # Single-header OBJ loader (MIT)
│  ├─ nlohmann/                    # JSON for Modern C++ (MIT)
│  └─ stb_image.h                  # Single-header image loader (MIT/public domain)
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
| `frame_XXXX_spatial.exr` | G-Buffer: depth, normals, instance IDs |
| `frame_XXXX_meta.json` | Full randomized state (camera, sun, materials, transforms) |
| `frame_XXXX.txt` | YOLO labels (class, cx, cy, w, h — normalized) |

| Global file | Description |
|-------------|-------------|
| `annotations_coco.json` | COCO-format annotations for the entire dataset |

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
| `gbuffer.exr` | 32-bit float EXR: Depth + Normal.X/Y/Z (with `--exr`) |
| `inst.pgm` | Per-pixel instance mask |
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
| `--exr` | off | Write G-Buffer EXR |
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

Every forged frame:

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

## License

BSD-3-Clause. See `LICENSE`.
