# VisionForge

**VisionForge** is a physically‑based CPU path tracer written in modern C++. It’s built for both **learning** and **synthetic dataset generation**. You can preview quickly with lightweight settings, or render high‑quality outputs with YOLO/COCO labels and per‑pixel masks.

The demo scene features a procedural sand dune, color‑coded cubes, a soft rectangular sun light, and a visible sky with a sun disc aligned to the key light.

---

## Highlights

* **Unbiased path tracing** (Monte Carlo) with Russian roulette
* **Materials**: Lambertian (diffuse) + emissive area light
* **Geometry**: AABB rectangles, cubes (from rects), triangles, procedural heightfield terrain
* **Acceleration**: BVH for fast ray–scene intersections
* **Sky**: Analytic daytime sky with visible sun disc + warm halo
* **Area lights**: Rectangular emitter for soft shadows
* **Adaptive sampling** per pixel for efficient noise reduction
* **Dataset/labels out of the box**

  * RGB (PPM)
  * Per‑pixel **instance mask** (`inst.pgm`)
  * 2D bounding boxes from mask (`labels_from_mask.csv/json`)
  * YOLO labels (`labels_yolo.txt`)
  * COCO annotations (`labels_coco.json`)
  * Projected boxes (`bboxes.csv/json`) for reference
  * Render metadata (`manifest.json`)

---

## Project Layout

```
VisionForge/
├─ apps/visionforge/           # CLI app (main.cpp)
├─ include/visionforge/        # Public headers (math, geometry, sky, etc.)
├─ src/io/                     # Output helpers (EXR scaffold)
├─ third_party/                # External libs
├─ build/                      # CMake build tree (generated)
├─ out/                        # Render outputs (generated)
├─ CMakeLists.txt
└─ README.md
```

---

## Prerequisites

**Ubuntu / Debian**

```bash
sudo apt update
sudo apt install -y build-essential cmake zlib1g-dev imagemagick
```

Optional: a compiler with **OpenMP** support (GCC/Clang/MSVC) for multithreaded sampling.

**Windows**: Build via CMake + MSVC or use WSL (recommended).

---

## Build

```bash
# From the repo root
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVISIONFORGE_OMP=ON
cmake --build build -j
```

The executable is written to: `./build/visionforge`

> The CMake option `VISIONFORGE_OMP=ON` enables OpenMP if your compiler supports it and defines `VF_USE_OMP` for the code path that parallelizes sampling.

### Use all CPU cores at runtime (optional)

```bash
export OMP_NUM_THREADS="$(nproc)"   # Linux/WSL/macOS
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
```

---

## Quick Start

Minimal render (720p, low SPP, reasonable depth):

```bash
./build/visionforge --out out \
  --width 1280 --height 720 \
  --spp 24 --max-depth 10 --seed 42
```

Outputs are written into `out/` and overwritten each run:

* `image.ppm` – tonemapped RGB image (ACES + sqrt gamma)
* `inst.pgm` – per‑pixel instance mask (uint32 IDs)
* `labels_from_mask.csv/json` – tight bounding boxes per object
* `labels_yolo.txt` – YOLO labels
* `labels_coco.json` – COCO annotations
* `bboxes.csv/json` – projected cube boxes (analytic; comparison only)
* `manifest.json` – render metadata (spp, timing, seed, etc.)

Convert PPM → PNG (optional):

```bash
magick convert out/image.ppm out/image.png
```

---

## Using the CLI

The `visionforge` CLI lets you control **camera**, **terrain & sand**, **cubes (count/colors/placement)**, and **lighting/sky**. All parameters are optional; sensible defaults are provided.

### Syntax & Conventions

* **Booleans**: `--flag` (present = true) or `--flag true|false`
* **CSV triples**: `"x,y,z"` (quote if the argument contains commas)
* **Ranges**: `"a,b"` (e.g., `--light-y "2,14"`)
* **Colors** (any of):

  * name: `red`, `blue`, `white`
  * hex: `#ffaa00`
  * rgb triple: `0.1,0.2,0.9`
  * multiple entries: comma‑separate names/hex; if using raw RGB triples use **semicolons** to separate entries, e.g. `"red;#3cb371;0.1,0.2,0.9"` (requires the semicolon‑aware color parser).

> To **overwrite** outputs each run, keep `--out out` (the default) or use the same folder. To start fresh: `rm -f out/*` before rendering.

### Parameters (what each does)

#### Global render

| Flag                  | Type   | Meaning                                             | Default      |
| --------------------- | ------ | --------------------------------------------------- | ------------ |
| `--out`               | string | Output directory (PPM, masks, labels).              | `out`        |
| `--width`, `--height` | int    | Image resolution in pixels.                         | `1280 × 720` |
| `--spp`               | int    | Target samples per pixel (noise ≈ 1/√SPP).          | `96`         |
| `--max-depth`         | int    | Maximum bounce depth.                               | `16`         |
| `--preview`           | flag   | Fast preset (spp 12, depth 6, fewer light samples). | off          |
| `--seed`              | int    | RNG seed (change to reroll layouts).                | `1337`       |
| `--exposure`          | float  | Exposure multiplier before tone‑map.                | `6.5`        |
| `--sky-gain`          | float  | Visible sky brightness (background only).           | `45`         |

#### Camera

| Flag         | Type    | Meaning                      | Default   |
| ------------ | ------- | ---------------------------- | --------- |
| `--lookfrom` | `x,y,z` | Camera position.             | `18,8,24` |
| `--lookat`   | `x,y,z` | Target point in world space. | `0,1.2,0` |
| `--fov`      | deg     | Vertical field of view.      | `35`      |

#### Terrain & sand

| Flag                           | Type  | Meaning                                | Default          |
| ------------------------------ | ----- | -------------------------------------- | ---------------- |
| `--terrain-amp`                | float | Dune height amplitude.                 | `1.8`            |
| `--terrain-scale`              | float | Feature size (smaller → busier dunes). | `0.14`           |
| `--terrain-nx`, `--terrain-nz` | int   | Tessellation of the heightfield grid.  | `96`, `96`       |
| `--sand-bump-freq`             | float | Micro‑bump frequency; `0` disables.    | `5.0` (example)  |
| `--sand-bump-scale`            | float | Micro‑bump strength; `0` disables.     | `0.22` (example) |

> **Speed tip**: For quick iterations use `--terrain-nx 80 --terrain-nz 80` and `--sand-bump-scale 0.10–0.15`.

#### Cubes (objects)

| Flag                                 | Type  | Meaning                                                                                        | Default                |
| ------------------------------------ | ----- | ---------------------------------------------------------------------------------------------- | ---------------------- |
| `--cubes`                            | int   | Number of cubes.                                                                               | `4`                    |
| `--cube-edge-min`, `--cube-edge-max` | float | Edge length range.                                                                             | `1.4, 1.9`             |
| `--cube-colors`                      | list  | Colors used cyclically for cubes. Names/hex; raw RGB triples allowed when semicolon‑separated. | `red,green,blue,white` |
| `--placement`                        | enum  | `grid` or `random`.                                                                            | `grid`                 |
| `--min-spacing`                      | float | Minimum separation for random placement.                                                       | (sensible default)     |
| `--tilt-abs`                         | deg   | Max random tilt about X/Z (±).                                                                 | `12`                   |

#### Lighting & sky

| Flag                   | Type    | Meaning                                                               | Default       |
| ---------------------- | ------- | --------------------------------------------------------------------- | ------------- |
| `--light-x`            | float   | X position of the wall light (YZ rect at x=k).                        | `-30`         |
| `--light-y`            | `y0,y1` | Vertical span of the rect.                                            | `2,14`        |
| `--light-z`            | `z0,z1` | Depth span of the rect.                                               | `-25,25`      |
| `--light-intensity`    | float   | Radiance multiplier of the rect light.                                | `8000`        |
| `--light-color`        | color   | Emissive color (name / `#hex` / `r,g,b`).                             | `1,0.98,0.92` |
| `--show-light`         | bool    | If true, show the rect to the camera (otherwise hidden so sky shows). | `false`       |
| `--match-sky`          | bool    | Align the visible sun direction to the rect’s normal.                 | `true`        |
| `--sun-az`, `--sun-el` | deg     | Sun azimuth / elevation for the visible sky.                          | `300, 12`     |
| `--turbidity`          | float   | Haze of the sky model (higher = hazier/warmer).                       | `3.5`         |

#### Advanced variance controls *(optional, if enabled in your build)*

| Flag              | Type  | Meaning                                                      |
| ----------------- | ----- | ------------------------------------------------------------ |
| `--noise-target`  | float | Relative 95% CI target per pixel (e.g., `0.04` = 4%).        |
| `--min-spp`       | int   | Minimum spp before early‑stop can trigger.                   |
| `--light-samples` | int   | Shadow rays per hit for direct lighting (quality vs. speed). |

---

## Recipes (copy‑paste)

**A) “Fast‑but‑pretty” baseline (720p, spp 24, depth 10):**

```bash
./build/visionforge --out out \
  --width 1280 --height 720 \
  --spp 24 --max-depth 10 --seed 42 \
  --exposure 6.2 --sky-gain 45 \
  --lookfrom "-28,8,6" --lookat "12,3,0" --fov 35
```

**B) Two colors, 6 cubes, random placement, new camera & light:**

```bash
./build/visionforge --out out \
  --width 1280 --height 720 --spp 24 --max-depth 10 --seed 4242 \
  --exposure 6.2 --sky-gain 45 \
  --lookfrom "12,10,-26" --lookat "0,2,0" --fov 33 \
  --terrain-amp 1.8 --terrain-scale 0.14 --terrain-nx 80 --terrain-nz 80 \
  --sand-bump-freq 3.5 --sand-bump-scale 0.12 \
  --cubes 6 --placement random --min-spacing 4.0 \
  --cube-edge-min 1.6 --cube-edge-max 1.9 \
  --cube-colors red,blue \
  --tilt-abs 12 \
  --sun-az 235 --sun-el 10 --turbidity 4.0 \
  --light-x -26 --light-y "3,13" --light-z "-18,18" \
  --light-color "#ffd7a0" --light-intensity 5000 \
  --match-sky true --show-light false
```

**C) 4 cubes (2 red, 2 blue), grid placement, different look:**

```bash
./build/visionforge --out out \
  --width 1280 --height 720 --spp 24 --max-depth 10 --seed 202 \
  --exposure 6.2 --sky-gain 45 \
  --lookfrom "-20,9,-16" --lookat "0,2,0" --fov 32 \
  --terrain-amp 1.8 --terrain-scale 0.14 --terrain-nx 80 --terrain-nz 80 \
  --sand-bump-freq 3.5 --sand-bump-scale 0.12 \
  --cubes 4 --placement grid \
  --cube-edge-min 1.6 --cube-edge-max 1.9 \
  --cube-colors red,blue,red,blue \
  --tilt-abs 10 \
  --sun-az 255 --sun-el 9 --turbidity 4.5 \
  --light-x -28 --light-y "2,12" --light-z "-20,20" \
  --light-color "#ffd7a0" --light-intensity 5200 \
  --match-sky true --show-light false
```

**D) Super‑fast preview while iterating:**

```bash
./build/visionforge --out out \
  --preview --width 960 --height 540 --seed $RANDOM \
  --cube-colors red,blue --cubes 6 --placement random
```

---

## Performance Tips

* Build in **Release**.
* Use `--preview` and/or a smaller resolution while composing.
* For speed, try: `--spp 24–48`, `--max-depth 8–12`, `--light-samples 2–3` (if supported).
* Sand micro‑bump increases variance; lower `--sand-bump-scale` if you need cleaner images at low SPP.
* Terrain tessellation (`--terrain-nx/nz`) trades triangle count for detail; 64–96 is a good range.

---

## Troubleshooting

* **Dark image** → raise `--exposure` (e.g., 8–10).
* **Sky too bright/dim** → adjust `--sky-gain`.
* **Noisy output** → increase `--spp`, or lower bump scale, or reduce `--light-intensity` contrast.
* **"Bad color token" warnings** → use color names/hex codes, or separate RGB triples with semicolons.
* **Nothing in `out/`** → double‑check `--out out` and that the app had write permissions.

---

## License

BSD‑3‑Clause. See `LICENSE`.
