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
* **Procedural terrain**: ridged fractal heightfield with domain-warped Perlin noise and micro-bump normals
* **Analytic sky** with sun disc, warm halo, and Rayleigh scatter
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
│  ├─ world_config.hpp             #   Forge JSON config parser
│  ├─ mesh.hpp                     #   OBJ mesh loader (per-mesh BVH)
│  ├─ bvh.hpp                      #   BVH node (longest-extent axis split)
│  ├─ passes.hpp                   #   GBuffer (depth, normals, instance IDs)
│  ├─ sky.hpp                      #   Analytic sky model
│  ├─ vec3.hpp                     #   Vec3 + thread-local xoshiro256+ PRNG
│  ├─ exr_writer.hpp               #   EXR export
│  ├─ png_writer.hpp               #   PNG export (zlib)
│  └─ ...                          #   materials, camera, labels, transforms
├─ src/io/
│  ├─ exr_writer.cpp               # TinyEXR implementation
│  ├─ png_writer.cpp               # Minimal PNG writer
│  └─ fast_obj.cpp                 # fast_obj implementation unit
├─ third_party/
│  ├─ tinyexr.h                    # Single-header OpenEXR (MIT)
│  ├─ fast_obj.h                   # Single-header OBJ loader (MIT)
│  └─ nlohmann/                    # JSON for Modern C++ (MIT)
├─ world.json                      # Example forge config with DR ranges
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

```json
{
  "render": { "width": 320, "height": 180, "spp": 4, "max_depth": 6, "seed": 123 },
  "camera": {
    "lookfrom": { "min": [14, 7, 20], "max": [20, 10, 28] },
    "fov_deg": { "min": 30, "max": 40 }
  },
  "lighting": {
    "sun_azimuth_deg": { "min": 220, "max": 320 },
    "sun_elevation_deg": { "min": 8, "max": 28 }
  },
  "asset": {
    "obj": "test_cube.obj", "scale": 1.5,
    "roughness": { "min": 0.05, "max": 0.9 },
    "metallic": { "min": 0.0, "max": 1.0 }
  },
  "placement": {
    "x": { "min": -12, "max": 12 },
    "z": { "min": -12, "max": 12 },
    "yaw_deg": { "min": 0, "max": 360 }
  },
  "dataset": { "root": "dataset", "train_split": 0.8 }
}
```

### Performance

At 320x180, forge renders **~12ms per frame** (20 OMP threads) with I/O fully overlapped. 1,000 frames complete in under 30 seconds.

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

## Performance

| Scenario | Time |
|----------|------|
| Forge frame (320x180, 1 spp, PBR) | ~12ms |
| Forge 1,000 frames (full labels) | ~30s |
| Manual render (320x180, 1 spp) | ~0.2s |
| Manual render (320x180, 128 spp) | ~3s |
| Manual render (640x360, 128 spp) | ~12s |

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
