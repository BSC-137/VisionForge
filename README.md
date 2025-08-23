# VisionForge

**VisionForge** is a physically‑based CPU path tracer written in modern C++. It’s built for both **learning** and **synthetic dataset generation**. You can preview quickly with lightweight settings, or render high‑quality outputs with YOLO/COCO labels and per‑pixel masks.

The demo scene features a procedural sand dune, color‑coded cubes, a soft rectangular sun light, and a visible sky with a sun disc aligned to the key light.

---

## Highlights

* **Unbiased path tracing** (Monte Carlo) with Russian roulette.
* **Materials**: Lambertian (diffuse), emission lights.
* **Geometry**: axis‑aligned rectangles, cubes (via rects), triangles, procedural heightfield terrain.
* **Acceleration**: BVH for fast ray–scene intersections.
* **Sky**: analytic daytime sky with visible sun disc + glow.
* **Area lights**: rectangular emitter for soft shadows.
* **Adaptive sampling** per pixel for efficient noise reduction.
* **Outputs**:

  * RGB (PPM)
  * Per‑pixel **instance mask** (`inst.pgm`)
  * 2D bounding boxes from mask (`labels_from_mask.csv/json`)
  * YOLO labels (`labels_yolo.txt`)
  * COCO annotations (`labels_coco.json`)
  * Projected boxes (`bboxes.csv/json`) for comparison
  * Render metadata (`manifest.json`)

---

## Project Layout

```
VisionForge/
├─ apps/visionforge/           # CLI app (main.cpp)
├─ include/visionforge/        # Public headers (math, geometry, sky, etc.)
├─ src/io/                     # Output helpers (EXR scaffold)
├─ third_party/                # External libs (e.g., tinyexr)
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
sudo apt install -y build-essential cmake zlib1g-dev libpng-dev
```

Optional: a compiler with **OpenMP** support (GCC/Clang) for multithreaded sampling.

---

## Build

```bash
# From the repo root
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Executable: `./build/visionforge`

---

## Quick Start

Minimal render:

```bash
./build/visionforge --out out --width 1280 --height 720 --spp 24 --max-depth 10 --seed 42
```

Outputs in `out/`:

* `image.ppm` – tonemapped RGB image (ACES + sqrt gamma)
* `inst.pgm` – per‑pixel instance mask (IDs)
* `labels_from_mask.csv/json` – tight bounding boxes per object
* `labels_yolo.txt` – YOLO labels
* `labels_coco.json` – COCO annotations
* `bboxes.csv/json` – projected cube boxes (legacy)
* `manifest.json` – render metadata

Convert PPM → PNG:

```bash
convert out/image.ppm out/image.png
```

---

## CLI Options

| Flag          | Type   | What it does                                             | Default   |
| ------------- | ------ | -------------------------------------------------------- | --------- |
| `--out`       | string | Output directory.                                        | `out`     |
| `--width`     | int    | Image width in pixels.                                   | `1280`    |
| `--height`    | int    | Image height in pixels.                                  | `720`     |
| `--spp`       | int    | Target samples per pixel.                                | `96`      |
| `--max-depth` | int    | Maximum path depth.                                      | `16`      |
| `--seed`      | int    | RNG seed.                                                | `1337`    |
| `--preview`   | flag   | Fast preview mode (lower spp/depth/light samples).       | off       |
| `--exposure`  | float  | Exposure multiplier before tonemapping.                  | `6.5`     |
| `--sky-gain`  | float  | Multiplier for visible sky brightness (background only). | `45`      |
| `--lookfrom`  | csv    | Camera position (x,y,z).                                 | `18,8,24` |
| `--lookat`    | csv    | Camera target (x,y,z).                                   | `0,1.2,0` |
| `--fov`       | deg    | Vertical field of view.                                  | `35`      |

**Presets:**

* **Preview** (`--preview`): spp=12, max‑depth=6, light samples=2.
* **Final**: user‑set spp/depth, light samples=6.

---

## Lighting & Sky

* Large **rectangular area light** (key sun) for soft shadows.
* **Analytic sky** with sun disc + halo, aligned to area light direction.
* Sky background intensity (`--sky-gain`) is independent of shading.

---

## Performance Tips

* Build in **Release**.
* Use `--preview` and small resolutions while iterating.
* Increase `--spp` gradually (noise decreases \~ √SPP).
* Keep `--max-depth` around 8–16 for typical outdoor scenes.

---

## Roadmap

* **Dataset mode** (`--dataset N --name <run>`): render N randomized frames into `datasets/<name>/...`.
* **EXR output** toggle for HDR.
* More materials (metal, dielectric, textured).
* MIS direct lighting for better convergence.
* Optional Intel OIDN denoiser for previews.

---

## Troubleshooting

* **Dark image** → increase `--exposure`.
* **Sky too bright/dim** → adjust `--sky-gain`.
* **Noisy output** → raise `--spp`, `--max-depth`.
* **Build errors** → ensure CMake ≥ 3.16, C++17 compiler, deps installed.

---

## License

BSD‑3‑Clause. See `LICENSE`.
