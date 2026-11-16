# BPSC (Bit-Plane Spline & Procedural Codec) — Technical Specification

**Version:** 1.0-Draft  
**Category:** Asymmetric Hybrid Vector-Procedural Texture & Image Compression Format  
**Target Hardware:** GPU-native decompression (Compute / Mesh / Fragment Shaders) & CPU reference decoders

---

## 1. Purpose and Architectural Goals

**BPSC** is designed for ultra-compact storage of high-resolution game textures, terrain/material masks, and UI assets while providing:
1. **Resolution Independence (Infinite Zoom):** Smooth vector boundaries without pixelation at arbitrary magnification.
2. **Zero-Overhead Mipmapping:** Mip-levels are generated on the fly with **0 bytes** of extra file storage via coordinate bit-shifting and lower bit-plane culling.
3. **Direct GPU Decompression:** Designed to be rasterized or evaluated directly on the GPU, drastically reducing VRAM footprint and memory bandwidth bottlenecks.
4. **Asymmetric Pipeline:** Heavy RDO (Rate-Distortion Optimization) and Matching Pursuit during encoding; fast, deterministic $O(N)$ reconstruction during decoding.

---

## 2. Encoding Pipeline

### Step 1: 2-Channel Color Decomposition (Luma + Chroma)
Input RGB data is decoupled into two logical channels:
* **Channel Y (Luma):** 8-bit luminance ($0\dots255$) preserved at full base coordinate resolution.
* **Channel C (Chroma):** Compact chrominance representation (either a 1D space-filling curve projection such as Peano/Hilbert across the CbCr plane, or a subsampled 2D chroma map with a coarser spline coordinate grid).

### Step 2: Iterative Procedural Noise Extraction (Matching Pursuit)
Before vectorizing base shapes, stochastic noise and regular micro-textures are extracted:
1. The encoder evaluates a dictionary of parametric procedural noise generators $G_k(\text{seed}, \text{freq}, \text{amp}, \text{octaves})$ (Gaussian, Perlin, Simplex, Voronoi/Cellular).
2. For each pass, the encoder selects a generator $G_k$ and fits a closed **spline region mask** $M_k$ such that subtracting $G_k$ inside $M_k$ maximizes entropy reduction (compressed bitstream size savings) under Rate-Distortion Optimization (RDO).
3. This process repeats iteratively ($2\dots4$ passes) until the marginal bitrate cost of storing the spline mask and noise parameters exceeds the entropy savings of the residual.
4. Remaining non-stochastic micro-details are either discarded below a perceptual threshold, passed to the vector layer, or stored in a local fallback tile if error exceeds target bounds.

### Step 3: Anisotropic Smoothing (Surface Blur)
The residual base image is filtered via **Surface Blur (Bilateral Filtering)** to eliminate high-frequency bit jitter in lower bit-planes while strictly preserving sharp structural edges.

### Step 4: Multi-Scale Edge Analysis (Scale-Space Anti-Aliasing)
To prevent pixel staircase artifacts (aliasing) from being vectorized as jagged polygons, the encoder constructs a 3-tier scale pyramid:
* $I_{0.25\times}$ (downscaled $4\times$)
* $I_{1.0\times}$ (native resolution)
* $I_{4.0\times}$ (sub-pixel interpolated upscale $4\times$)

Gradients are compared across scales: if a staircase edge at $I_{1.0\times}$ resolves into a continuous linear or smooth gradient at $I_{4.0\times}$, it is fitted as a single smooth spline segment.

### Step 5: Gray-Code Conversion & Bit-Plane Slicing
1. Pixel intensities $B$ are converted into **Reflected Binary Code (Gray Code)** to prevent simultaneous multi-bit transitions across smooth gradients:
   $$G = B \oplus (B \gg 1)$$
2. The image is sliced into 8 binary bit-planes $P_0 \dots P_7$ (from LSB $0$ to MSB $7$).

### Step 6: Spline Vectorization & Vertex Deduplication
1. Closed contours with nesting hierarchy (outer fill vs. inner hole) are extracted for each bit-plane $P_i$.
2. Contours are approximated using **closed cubic Catmull-Rom splines** with adaptive grid precision $\epsilon_i$ (driven by AI saliency segmentation or per-layer precision presets).
3. Each control vertex is assigned a 1-bit curvature flag (`0` = smooth Catmull-Rom node, `1` = sharp corner).
4. **Shared Vertex Buffer Construction:** Control points $(X, Y)$ across all bit-planes are deduplicated into a single global indexed vertex stack. Vertices are sorted by topological significance (curvature weight) to optimize progressive LOD culling.

---

## 3. Binary Container Format (`.bpsc`)

**Endianness:** Little-Endian.  
All payload blocks following the 16-byte header are compressed using an entropy coder (Zstandard / Arithmetic Coding).

### 3.1. File Header (`16 bytes`)

| Offset | Size | Type | Field | Description |
| :--- | :--- | :--- | :--- | :--- |
| `0x00` | 4 B | `char[4]` | `Magic` | Magic bytes: `'B', 'P', 'S', 'C'` |
| `0x04` | 1 B | `uint8` | `Version` | Format version (`0x01`) |
| `0x05` | 1 B | `uint8` | `Flags` | Bit `0`: Has Chroma; Bit `1`: Has Noise Layers; Bit `2`: Bit-Planar Vertex Stream |
| `0x06` | 2 B | `uint16` | `Width` | Base canvas width ($1\dots65535$) |
| `0x08` | 2 B | `uint16` | `Height` | Base canvas height ($1\dots65535$) |
| `0x0A` | 4 B | `uint32` | `NumVertices` | Total unique vertices in `Shared Vertex Buffer` |
| `0x0E` | 2 B | `uint16` | `NumNoiseLayers` | Number of procedural noise passes |

### 3.2. Shared Vertex Buffer
To support bandwidth-efficient hardware mipmapping, vertex coordinates may be stored in **Bit-Planar** order (from MSB to LSB of coordinates) or delta-encoded blocks:
* `X_coords[NumVertices]` (`uint16[]`): X coordinates aligned to base grid.
* `Y_coords[NumVertices]` (`uint16[]`): Y coordinates aligned to base grid.
* `VertexFlags[ceil(NumVertices / 8)]` (`uint8[]`): Packed 1-bit corner flags (`1` = sharp corner, `0` = smooth spline).

### 3.3. Bit-Plane Vector Layers (`8 Luma layers` + `Chroma layers`)
For each layer $b \in [0\dots7]$:
* `NumSplines` (`uint16`): Number of closed splines in layer $b$.
* For each spline:
  * `FlagsAndCount` (`uint16`):
    * Bit `15` (MSB): `IsHole` (`0` = additive fill `1`, `1` = subtractive hole `0`).
    * Bit `14`: `HasDynamicFeather` (optional edge anti-aliasing/feather profile).
    * Bits `0..13`: `PointCount` (number of control vertices in spline, up to $16383$).
  * `VertexIndices[PointCount]` (`uint16[]` or `uint32[]` if `NumVertices > 65535`): Indices into `Shared Vertex Buffer`.
  * *(Optional)* `FeatherProfile`: Edge falloff parameters if `HasDynamicFeather == 1`.

### 3.4. Procedural Noise Layers (`NumNoiseLayers` entries)
For each noise pass $k$:
* `GeneratorID` (`uint8`): `0x01` = Gaussian, `0x02` = Perlin, `0x03` = Simplex, `0x04` = Voronoi/Cellular.
* `Seed` (`uint32`): Deterministic PRNG seed.
* `Frequency` (`float16`), `Amplitude` (`float16`), `Octaves` (`uint8`), `BlendMode` (`uint8`).
* `MaskSplineCount` (`uint16`): Number of closed spline masks defining the active region (referencing indices from `Shared Vertex Buffer`).

---

## 4. Decoding & GPU Rendering Pipeline

### 4.1. Zero-Cost Hardware Mipmapping (LOD $M$)
When rendering an asset at mip-level $M$ (resolution scaled by $2^{-M}$):
1. **Memory Pruning:** The decoder skips loading the $M$ lowest bit-planes and skips reading the $M$ least significant coordinate bits from the Bit-Planar Vertex Stream.
2. **Coordinate Bit-Shift:** Vertex coordinates are projected onto the mip grid via right bit-shift:
   $$X' = X \gg M, \quad Y' = Y \gg M$$
3. **Zero-Area Degenerate Culling:** Consecutive spline vertices collapsing to identical coordinates $(X'_i == X'_{i+1} \land Y'_i == Y'_{i+1})$ are deduplicated in the vertex/mesh shader. Splines with $< 3$ unique vertices are culled prior to rasterization.

### 4.2. GPU Execution Stages
1. **Spline Tessellation:** A Mesh or Tessellation shader evaluates Catmull-Rom segments into screen-space triangles with adaptive subdivision based on projected screen length.
2. **Bitwise Rasterization:** Closed contours are rasterized into an integer target (`R8_UINT`) using bitwise operations (`target |= (1 << bit)` for fills, `target &= ~(1 << bit)` for holes).
3. **Gray-Code Reconstruction:** A single Compute Shader pass converts the Gray-coded integer buffer back to linear luminance and merges chrominance:
   $$B = G \oplus (G \gg 1) \oplus (G \gg 2) \oplus \dots \oplus (G \gg 7)$$
4. **Procedural Synthesis:** A Fragment/Compute shader evaluates procedural noise functions $G_k$ in UV/screen space (clamping octaves at the Nyquist limit for mip-level $M$) and blends them within active spline masks.
