# VisionForge

**VisionForge** is a physically-based CPU ray tracer written in modern C++.
It supports high dynamic range output (EXR), physically accurate materials, multiple render passes, and is designed for learning, experimentation, and extension.

---

## ✨ Features

* **Physically-Based Rendering (PBR)** with path tracing.
* **Material Support**:

  * Lambertian (diffuse)
  * Metal (reflective)
  * Dielectric (transparent/refractive)
  * Emissive (light sources)
* **Acceleration Structure**: Bounding Volume Hierarchy (BVH) for faster ray-scene intersections.
* **Multiple Render Passes**:

  * `rgb` – final image
  * `albedo` – base surface color
  * `normal` – surface normals
  * `depth` – depth information
* **High Dynamic Range (HDR)** support via OpenEXR.
* **Configurable Command-Line Interface** for render parameters.

---

## 📂 Project Structure

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

## 🛠 Installation

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

> **Note**: If you want EXR output, ensure `tinyexr` is included in `third_party`.

---

## ⚙️ Build Instructions

```bash
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)
```

The compiled binary will be located at:

```
VisionForge/build/apps/visionforge/visionforge
```

---

## 🚀 Usage

Run the renderer with custom parameters:

```bash
./visionforge --out ../out --width 800 --height 600 --spp 1024 --max-depth 16 --seed 42
```

**Available Options**:

| Option        | Description                     | Default |
| ------------- | ------------------------------- | ------- |
| `--out`       | Output folder                   | `./out` |
| `--width`     | Image width in pixels           | `800`   |
| `--height`    | Image height in pixels          | `600`   |
| `--spp`       | Samples per pixel (quality)     | `256`   |
| `--max-depth` | Max path tracing depth          | `8`     |
| `--seed`      | Random seed for reproducibility | `0`     |

---

## 📊 Render Passes

When rendering, VisionForge outputs multiple files:

* **rgb.exr / rgb.ppm** → Final render
* **albedo.exr** → Base surface color without lighting
* **normal.exr** → Encoded surface normals
* **depth.exr** → Depth information for compositing

---

## 🖼 Viewing Output

* **PPM** files can be opened in GIMP, Photoshop, or converted with ImageMagick:

  ```bash
  convert rgb.ppm rgb.png
  ```
* **EXR** files can be viewed with:

  * [GIMP](https://www.gimp.org/)
  * [Krita](https://krita.org/)
  * [Blender](https://www.blender.org/)
  * `exrdisplay` from OpenEXR tools

---

## 📈 Performance Tips

* Increase `--spp` for better quality (more samples per pixel).
* Raise `--max-depth` for more accurate global illumination.
* For quick previews, use low SPP (16–64).
* BVH acceleration greatly reduces render time for complex scenes.

---

## 🐛 Troubleshooting

**Image is noisy** → Increase SPP or use a denoiser like [OpenImageDenoise](https://www.openimagedenoise.org/).
**Build errors** → Check that all third-party libraries are cloned and your compiler supports C++17 or higher.

---

## 🤝 Contributing

1. Fork the repository
2. Create a new feature branch
3. Commit your changes
4. Open a pull request

---

## 📜 License

This project is licensed under the **BSD-3-Clause License**.
