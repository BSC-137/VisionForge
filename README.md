# VisionForge

**VisionForge** is a physically-based CPU ray tracer written in modern C++. It is built for physically accurate light transport simulation using path tracing, supporting a variety of materials, scene primitives, and render passes. The project is designed for both learning and practical experimentation in physically-based rendering.

---

## Features

* **Physically-Based Rendering (PBR)** with unbiased path tracing.
* **Material Support**:

  * Lambertian (diffuse)
  * Metal (reflective)
  * Dielectric (transparent/refractive)
  * Emissive (light sources)
* **Acceleration Structure**:

  * Bounding Volume Hierarchy (BVH) for efficient ray-scene intersections.
* **Scene Elements**:

  * Procedural terrain generation with variable roughness.
  * Geometric primitives such as spheres, rectangles, and cubes.
  * HDR and sky backgrounds.
* **Multiple Render Passes**:

  * `rgb` – Final rendered image.
  * `albedo` – Base surface colors without lighting.
  * `normal` – Surface normal visualization.
  * `depth` – Encoded depth values.
* **High Dynamic Range (HDR)** output via OpenEXR.
* **Configurable Command-Line Interface** for full control over rendering parameters.

---

## Project Structure

```
VisionForge/
├── apps/
│   └── visionforge/        # Application entry point
├── include/visionforge/    # Public header files
├── src/                    # Core source code
│   └── io/                 # Output writers (EXR, PPM)
├── third_party/            # External libraries (CLI11, tinyexr, etc.)
├── out/                    # Output images (generated after rendering)
├── CMakeLists.txt          # Build configuration
├── README.md               # This file
└── .gitignore              # Git ignore rules
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/BSC-137/VisionForge.git
cd VisionForge
```

### 2. Install dependencies (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install build-essential cmake zlib1g-dev libpng-dev
```

> For EXR output, ensure `tinyexr` is included in `third_party`.

---

## Build Instructions

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j$(nproc)
```

The compiled binary will be located at:

```
VisionForge/build/apps/visionforge/visionforge
```

---

## Usage

Run the renderer with custom parameters:

```bash
./visionforge --out ../out --width 800 --height 600 --spp 256 --max-depth 8 --seed 42
```

**Available Options**:

| Option        | Description                     | Default |
| ------------- | ------------------------------- | ------- |
| `--out`       | Output folder                   | `./out` |
| `--width`     | Image width in pixels           | `800`   |
| `--height`    | Image height in pixels          | `600`   |
| `--spp`       | Samples per pixel               | `256`   |
| `--max-depth` | Max path tracing depth          | `8`     |
| `--seed`      | Random seed for reproducibility | `0`     |

---

## Current Output Examples

Recent builds have produced:

* **Procedural terrain** with adjustable roughness and fine-tuned detail levels.
* **Cubes** positioned on the terrain with distinct materials (e.g., red diffuse, green reflective/diffuse hybrid).
* **Lighting and shading** consistent with physically-based principles, producing realistic shadows and soft global illumination.
* **Intermediate quality** between fast previews and full production renders, balancing noise reduction with render time.

The renderer can output both **quick preview renders** (low SPP, low max-depth) for iteration, and **high-quality outputs** (high SPP, BVH acceleration) for final scenes.

---

## Performance Tips

* Use **low SPP (16–64)** for previews, then increase for final renders.
* Keep `--max-depth` reasonable (8–16) to balance quality and performance.
* BVH acceleration significantly reduces render time for large scenes.
* Reduce terrain complexity if quick iterations are needed.

---

## Viewing Output

* **PPM** files can be opened in GIMP, Photoshop, or converted with ImageMagick:

```bash
convert rgb.ppm rgb.png
```

* **EXR** files can be viewed with:

  * GIMP
  * Krita
  * Blender
  * `exrdisplay` from OpenEXR tools

---

## Troubleshooting

* **Image is noisy** → Increase SPP or apply a denoiser like Intel OpenImageDenoise.
* **Build errors** → Verify all dependencies are installed and your compiler supports C++17 or newer.

---

## License

This project is licensed under the BSD-3-Clause License.
