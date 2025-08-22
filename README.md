# VisionForge

**VisionForge** is a physically‑based CPU path tracer written in modern C++. It’s built for learning and experimentation—fast previewing while you iterate, with knobs for quality when you’re ready to output. The demo scene features a procedural sand dune, color‑coded cubes, a soft rectangular sun light, and a visible sky with a sun disc that aligns to the key light.

---

## Highlights

* **Unbiased path tracing** (Monte Carlo) with Russian roulette.
* **Materials**: Lambertian (diffuse), emission lights.
* **Geometry**: axis‑aligned rectangles, cubes (via rects), triangles, procedural heightfield terrain.
* **Acceleration**: BVH for fast ray–scene intersections.
* **Sky**: analytic daytime sky with visible sun disc + glow.
* **Area lights**: rectangular emitter for soft shadows.
* **Adaptive sampling** per pixel for efficient noise reduction.
* **Output**: RGB (PPM), 2D bounding boxes (CSV/JSON), manifest (JSON). EXR writer scaffold included (static lib), ready to hook up.

> **Note**: The demo intentionally keeps the light transport simple so it’s easy to read/modify. Add metals, dielectrics, MIS, textures, etc., as you grow the project.

---

## Project Layout

```
VisionForge/
├─ apps/visionforge/           # CLI app (main.cpp)
├─ include/visionforge/        # Public headers (math, geometry, sky, etc.)
├─ src/io/                     # Output helpers (EXR writer scaffold)
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

Optional but recommended: a compiler with **OpenMP** support for multi‑threaded sampling (GCC/Clang).

---

## Build

```bash
# From the repo root
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The executable is produced at:

```
./build/visionforge
```

> If your generator nests targets (e.g., `apps/visionforge/visionforge`), CMake prints the final path at link time.

---

## Quick Start (Mini Manual)

### Minimal render

```bash
./build/visionforge --out out --width 1280 --height 720 --spp 24 --max-depth 10 --seed 42
```

Outputs inside `out/`:

* `image.ppm` – tonemapped RGB image (ACES + sqrt)
* `bboxes.csv` – 2D boxes for the cubes (label,xmin,ymin,xmax,ymax,width,height)
* `bboxes.json` – same, JSON
* `manifest.json` – render metadata (resolution, spp, time, etc.)

Convert PPM to PNG (ImageMagick):

```bash
convert out/image.ppm out/image.png
```

### CLI options

| Flag          | Type   | What it does                                                             | Default |
| ------------- | ------ | ------------------------------------------------------------------------ | ------- |
| `--out`       | string | Output directory (created if missing).                                   | `out`   |
| `--width`     | int    | Image width in pixels.                                                   | `1280`  |
| `--height`    | int    | Image height in pixels.                                                  | `720`   |
| `--spp`       | int    | Target samples per pixel (adaptive early‑out per pixel may stop sooner). | `96`    |
| `--max-depth` | int    | Maximum path length (bounces).                                           | `16`    |
| `--seed`      | int    | RNG seed (affects scene jitter, footprints, sampling).                   | `1337`  |
| `--preview`   | flag   | Use lighter sampling/depth for fast iteration.                           | off     |
| `--exposure`  | float  | Exposure compensation multiplier applied before tonemapping.             | `6.5`   |
| `--sky-gain`  | float  | Extra multiplier for *visible* sky brightness (background only).         | `45`    |

> **Camera**: The demo scene uses a fixed camera in code. CLI camera controls (`--lookfrom`, `--lookat`, `--vfov`) are planned; see the *Roadmap* below.

### Example: brighter sky, faster preview

```bash
./build/visionforge \
  --out out/preview --width 960 --height 540 --spp 12 --max-depth 8 --preview \
  --exposure 6.5 --sky-gain 45
```

### Example: higher quality

```bash
./build/visionforge \
  --out out/hq --width 1920 --height 1080 --spp 256 --max-depth 16 \
  --exposure 6.5 --sky-gain 40
```

---

## About the Lighting & Sky

* A **rectangular area light** to camera‑left stands in for the sun. It produces **soft, directional** illumination and shadows.
* The **visible sky** (background) is analytic with a **sun disc** and warm halo. Its sun direction is **aligned to the area light’s normal**, so the disc appears where the lighting says it should.
* The sky background **does not** light the scene (by design for the demo), so you can tweak `--sky-gain` without affecting shading—useful for composition.

**Tuning knobs** (edit in `apps/visionforge/main.cpp`):

* Area light size/position → softer or sharper shadows.
* Sky turbidity/strength → richer or flatter sky.
* Exposure → overall scene brightness before tonemapping.

---

## Adaptive Sampling

Each pixel accumulates samples until either it reaches `--spp`, or the **95% confidence interval** for luminance falls below a relative threshold. This helps keep total render time down while focusing effort on noisy regions.

---

## Performance Tips

* Use `--preview` and smaller resolution while iterating on code/parameters.
* Keep `--max-depth` between 8–16 for typical outdoor scenes.
* Increase SPP in steps; double SPP ≈ √2 noise reduction (rough rule).
* Build with `Release` and a recent compiler. Enable OpenMP if available.

---

## Output Files

```
out/
├─ image.ppm          # Final tonemapped frame
├─ bboxes.csv         # 2D boxes (for ML/data tasks)
├─ bboxes.json        # Same in JSON
└─ manifest.json      # Render metadata
```

**EXR**: An EXR writer is present under `src/io/` (static library built as `vf_io`). Hook it into the app as needed to write HDR in addition to PPM.

---

## Roadmap (Short‑Term)

* **CLI Camera controls**: `--lookfrom ax,ay,az`, `--lookat bx,by,bz`, `--vfov deg`.
* **Output formats**: optional `--exr out/image.exr` (HDR) alongside PPM.
* **More materials**: metal, dielectric, textured albedo; MIS for light sampling.
* **Denoising**: optional Intel OIDN pass for faster clean previews.

---

## Troubleshooting

* **Image looks dark** → raise `--exposure` (e.g., 8–10).
* **Sky too dim/bright** → adjust `--sky-gain` (background only).
* **Noisy** → increase `--spp` and/or `--max-depth`. Make sure you’re building in `Release`.
* **Build errors** → verify CMake ≥ 3.16 and a C++17 compiler; ensure third‑party headers are present.

---

## Contributing

PRs welcome! Please keep changes readable and isolated (materials, integrator, geometry). Add a short note to the README for new flags or outputs.

---

## License

BSD‑3‑Clause. See `LICENSE`.
