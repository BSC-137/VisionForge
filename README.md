# VisionForge

**VisionForge** is a high-performance Spatial AI engine built on a physically-based CPU path tracer in modern C++20. It renders synthetic scenes and exports ground-truth data -- RGB images, depth maps, surface normals, instance masks, and object labels -- for training and evaluating spatial perception models.

---

## Highlights

* **Unbiased path tracing** (Monte Carlo) with Russian roulette and adaptive sampling
* **Materials**: Lambertian diffuse, dielectric, metal, emissive area lights
* **Geometry**: OBJ mesh loading (via fast\_obj), triangles, AABB rectangles, spheres, procedural heightfield terrain
* **Acceleration**: BVH (bounding volume hierarchy) over all scene primitives
* **Sky**: Analytic daytime sky with visible sun disc and warm halo
* **Mesh pipeline**: `Mesh` class loads `.obj` files, builds a per-mesh BVH of `MeshTriangle` primitives that reference shared vertex buffers for memory efficiency
* **G-Buffer export**: Per-pixel linear depth and world-space normals exported as 32-bit float OpenEXR
* **Fast depth-only mode**: Single primary-ray pass (`--depth-only`) for ground-truth spatial data at interactive speeds
* **OpenMP parallelization** across all render paths
* **Dataset outputs out of the box**:
  * RGB image (PPM)
  * Multi-channel EXR (`gbuffer.exr`) with `Depth`, `Normal.X`, `Normal.Y`, `Normal.Z`
  * Per-pixel instance mask (`inst.pgm`)
  * 2D bounding boxes from mask (`labels_from_mask.csv/json`)
  * YOLO labels (`labels_yolo.txt`)
  * COCO annotations (`labels_coco.json`)
  * Projected boxes (`bboxes.csv/json`)
  * Render metadata (`manifest.json`)

---

## Project Layout

```
VisionForge/
├─ apps/visionforge/main.cpp       # CLI application
├─ include/visionforge/            # Engine headers
│  ├─ mesh.hpp                     #   OBJ mesh loader (Hittable, internal BVH)
│  ├─ mesh_triangle.hpp            #   MeshData + MeshTriangle (shared vertex refs)
│  ├─ triangle.hpp                 #   Standalone triangle (terrain)
│  ├─ bvh.hpp                      #   BVH node
│  ├─ passes.hpp                   #   GBuffer (depth, normals, instance IDs)
│  ├─ exr_writer.hpp               #   EXR export declarations
│  └─ ...                          #   materials, camera, sky, labels, transforms
├─ src/io/
│  ├─ exr_writer.cpp               # TinyEXR implementation (RGB, float, G-Buffer EXR)
│  └─ fast_obj.cpp                 # fast_obj implementation unit
├─ third_party/
│  ├─ tinyexr.h                    # Single-header OpenEXR (MIT)
│  ├─ fast_obj.h                   # Single-header OBJ loader (MIT)
│  └─ CLI11/                       # CLI11 (vendored)
├─ build/                          # CMake build tree (generated)
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

Optional: **ImageMagick** (`imagemagick`) to convert PPM output to PNG.

A compiler with **OpenMP** support (GCC, Clang, or MSVC) enables multithreaded rendering.

**Windows**: Build via CMake + MSVC or use WSL.

---

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVISIONFORGE_OMP=ON
cmake --build build -j
```

The executable is written to `./build/visionforge`.

### OpenMP tuning (optional)

```bash
export OMP_NUM_THREADS="$(nproc)"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
```

---

## Quick Start

**Render with default scene (720p, 24 spp):**

```bash
./build/visionforge --out out --spp 24 --max-depth 10 --seed 42
```

**Load an OBJ mesh into the scene:**

```bash
./build/visionforge --out out --spp 24 \
  --obj model.obj --obj-pos "0,2,0" --obj-scale 3.0 --obj-color blue
```

**Export G-Buffer EXR alongside the render:**

```bash
./build/visionforge --out out --spp 24 --exr
```

**Fast depth-only pass (no shading, 1 spp):**

```bash
./build/visionforge --out out --depth-only
```

Outputs are written into the `--out` directory:

| File | Description |
|------|-------------|
| `image.ppm` | Tonemapped RGB (ACES + sqrt gamma) |
| `gbuffer.exr` | 32-bit float EXR: Depth + Normal.X/Y/Z channels (with `--exr` or `--depth-only`) |
| `inst.pgm` | Per-pixel instance mask |
| `labels_from_mask.csv/json` | Tight bounding boxes per object |
| `labels_yolo.txt` | YOLO-format labels |
| `labels_coco.json` | COCO annotations |
| `bboxes.csv/json` | Projected cube boxes (analytic) |
| `manifest.json` | Render metadata (spp, timing, seed) |

When `--depth-only` is used, only `gbuffer.exr` and `manifest.json` are written.

Convert PPM to PNG (optional):

```bash
magick convert out/image.ppm out/image.png
```

---

## CLI Reference

### Render

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--out` | string | `out` | Output directory |
| `--width`, `--height` | int | `1280`, `720` | Image resolution |
| `--spp` | int | `96` | Samples per pixel |
| `--max-depth` | int | `16` | Maximum bounce depth |
| `--preview` | flag | off | Fast preset (spp 12, depth 6, fewer light samples) |
| `--seed` | int | `1337` | RNG seed |
| `--exposure` | float | `6.5` | Exposure multiplier before tonemapping |
| `--sky-gain` | float | `45` | Sky brightness |
| `--exr` | flag | off | Write `gbuffer.exr` (depth + normals) |
| `--depth-only` | flag | off | Primary-ray-only pass; implies `--exr`, skips shading and label outputs |

### Camera

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--lookfrom` | `x,y,z` | `18,8,24` | Camera position |
| `--lookat` | `x,y,z` | `0,1.2,0` | Look-at target |
| `--fov` | deg | `35` | Vertical field of view |

### Terrain & sand

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--terrain-amp` | float | `1.8` | Dune height amplitude |
| `--terrain-scale` | float | `0.14` | Feature size (smaller = busier dunes) |
| `--terrain-nx`, `--terrain-nz` | int | `96` | Heightfield tessellation |
| `--sand-bump-freq` | float | `5.0` | Micro-bump frequency |
| `--sand-bump-scale` | float | `0.22` | Micro-bump strength |
| `--world-bounds` | `xmin,xmax,zmin,zmax` | `-22,22,-22,22` | World extents |

### Cubes

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--cubes` | int | `4` | Number of cubes |
| `--cube-edge-min`, `--cube-edge-max` | float | `1.4`, `1.9` | Edge length range |
| `--cube-colors` | list | `red,green,blue,white` | Colors, cycled across cubes |
| `--placement` | enum | `grid` | `grid` or `random` |
| `--min-spacing` | float | `3.5` | Min separation (random mode) |
| `--tilt-abs` | deg | `12` | Max random tilt |

### OBJ Mesh Loading

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--obj` | path | | OBJ file to load |
| `--obj-pos` | `x,y,z` | `0,0,0` | World-space translation |
| `--obj-scale` | float | `1.0` | Uniform scale factor |
| `--obj-color` | color | `white` | Material color (name, `#hex`, or `r,g,b`) |

### Lighting & sky

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--light-x` | float | `-30` | X position of area light |
| `--light-y` | `y0,y1` | `2,14` | Vertical span |
| `--light-z` | `z0,z1` | `-25,25` | Depth span |
| `--light-intensity` | float | `8000` | Radiance multiplier |
| `--light-color` | color | `1,0.98,0.92` | Emissive color |
| `--show-light` | bool | `false` | Show rect light to camera |
| `--match-sky` | bool | `true` | Align sun direction to light normal |
| `--sun-az`, `--sun-el` | deg | `300`, `12` | Sun azimuth / elevation |
| `--turbidity` | float | `3.5` | Sky haze |

### Color syntax

Colors can be specified as:
* Name: `red`, `blue`, `white`, `sand`, `yellow`, `magenta`, `cyan`, `gray`
* Hex: `#ffaa00`
* RGB triple: `0.1,0.2,0.9`
* Multiple entries: comma-separate names/hex, or use semicolons when mixing with raw RGB triples (`"red;#3cb371;0.1,0.2,0.9"`)

---

## Recipes

**Fast preview while iterating:**

```bash
./build/visionforge --out out --preview --width 960 --height 540 --seed $RANDOM
```

**6 cubes, random placement, custom camera:**

```bash
./build/visionforge --out out --spp 24 --max-depth 10 --seed 4242 \
  --lookfrom "12,10,-26" --lookat "0,2,0" --fov 33 \
  --cubes 6 --placement random --cube-colors red,blue \
  --terrain-nx 80 --terrain-nz 80
```

**OBJ mesh with depth export:**

```bash
./build/visionforge --out out --spp 32 --exr \
  --obj model.obj --obj-pos "0,2,0" --obj-scale 2.5 --obj-color "#3cb371"
```

**Batch depth-only for dataset generation:**

```bash
for seed in $(seq 1 100); do
  ./build/visionforge --depth-only --seed $seed --out "data/frame_${seed}" \
    --cubes 6 --placement random
done
```

---

## Performance Tips

* Build in **Release** mode.
* Use `--preview` or smaller resolution while iterating on scene composition.
* `--depth-only` is roughly 25x faster than a full render -- use it when you only need spatial ground truth.
* Lower `--terrain-nx/nz` (e.g. 64-80) for faster scene builds.
* Lower `--sand-bump-scale` reduces variance at low SPP.

---

## Troubleshooting

* **Dark image** -- raise `--exposure` (e.g. 8-10).
* **Sky too bright/dim** -- adjust `--sky-gain`.
* **Noisy** -- increase `--spp`, lower `--sand-bump-scale`, or reduce light intensity contrast.
* **"Bad color token" warnings** -- use color names or hex codes; separate RGB triples with semicolons.
* **OBJ won't load** -- check the path is correct and the file contains at least one face.

---

## License

BSD-3-Clause. See `LICENSE`.
