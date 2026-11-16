# BPSC — Bit-Plane Spline & Procedural Texture Codec

**BPSC** is an experimental, asymmetric, GPU-oriented texture compression format designed to pack high-resolution game textures, masks, and UI assets into a fraction of their original size while enabling **infinite zoom (resolution independence)** and **zero-overhead hardware mipmapping**.

> **Status:** Early R&D / Architecture & Python Reference Prototype  
> **Looking for:** Graphics Programmers (WebGPU / Vulkan / GLSL / HLSL) & Compression Enthusiasts to build the first GPU compute/mesh shader rasterizer.

---

## 💡 Core Idea

Modern games weigh 100–200+ GB largely due to raster textures (BC7/ASTC) and pre-calculated mipmap chains (+33% memory overhead). **BPSC** replaces dense pixel grids with a hybrid vector-procedural representation:

1. **Anisotropic Smoothing (Surface Blur):** Strips high-frequency sensor noise while preserving sharp object boundaries.
2. **Gray-Code Bit-Plane Slicing:** Decomposes Luma (and compressed 1D/2D Chroma) into 8 bit-planes without binary boundary artifacts.
3. **Catmull-Rom Splines + Shared Vertex Buffer:** Vectorizes bit-plane contours using smooth splines without extra control points. Overlapping vertices across different bit-planes are deduplicated into a single indexed vertex stack, preventing quantization halos between layers.
4. **Iterative Procedural Noise (Matching Pursuit):** Reconstructs natural micro-textures (skin, asphalt, foliage, grain) via parametric noise generators (Perlin, Simplex, Voronoi) masked by splines, minimizing residual entropy.
5. **Zero-Cost Hardware Mipmapping:** Mip-levels are generated on the fly with **0 bytes of extra storage** by bit-shifting vertex coordinates (`coord >> mip_level`), dropping lower bit-planes, and culling zero-area micro-splines directly in the vertex/mesh shader.

---

## 🏗️ Encoding Pipeline

```text
[Original Image] 
       │
       ├──► Iterative Procedural Noise Extraction (Matching Pursuit) ──► [Noise Params & Masks]
       ▼
[Surface Blur (Bilateral Filter)]
       │
       ├──► Multi-Scale Edge Analysis (0.25x / 1x / 4x Pyramid)
       ▼
[Gray Code Conversion] ──► [8 Bit-Planes]
       │
       ▼
[Catmull-Rom Spline Vectorization] ──► [Shared Vertex Buffer] ──► [.bpsc Binary]
```

---

## 🚀 Quick Start (Python Prototype)

The repository includes a reference CPU prototype (`prototype.py`) demonstrating the core pipeline: Surface Blur, Gray-code slicing, Catmull-Rom spline fitting, Shared Vertex Buffer deduplication, and bit-shift mipmapping.

### Run Locally
```bash
pip install opencv-python numpy
python prototype.py
```
*(If no `test.jpg` is provided in the folder, the script automatically generates a synthetic test pattern and outputs size benchmarks alongside a visual comparison sheet).*

---

## 🤝 How to Contribute / Open Tasks

The mathematical architecture and binary container format are specified in [`SPECIFICATION.md`](SPECIFICATION.md). We are currently looking for contributors to tackle the next milestones:

- [ ] **WebGPU / WebGL2 Demo:** A shader-based decoder rendering `.bpsc` binary buffers directly on the GPU with real-time 50x zoom.
- [ ] **Sharp Corner Detection:** Adding 1-bit vertex curvature flags (`smooth` vs `sharp`) to the Shared Vertex Buffer.
- [ ] **Procedural Noise Matcher:** Implementing RDO-guided Perlin/Simplex noise subtraction in the encoder.
- [ ] **C++ / Rust Fast Encoder:** Porting the contour extraction and RDO optimization from Python.

If you are interested in graphics programming, demoscene compression, or GPU compute shaders — feel free to open an **Issue** or start a **Discussion**!
