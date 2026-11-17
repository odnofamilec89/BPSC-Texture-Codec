# BPSC 1.1 — Bit-Plane Spline & Procedural Codec

An asymmetric vector-procedural texture compression format: minimal bytes on disk, evaluation on the fly on the GPU, and zero-cost mipmaps via bit-plane pruning.

## Status

Specification version: **1.1-Draft (0x02)**.
This is a stable specification intended for reference implementation: 32-byte header, container size aligned to 512 KiB, deterministic pipeline. The prototype in `prototype.py` belongs to the earlier architecture (16-byte header, Hilbert-curve chroma, noise before filtering) and is **not a reference for 1.1**.

---

## Why It Matters

- **VRAM savings.** The texture is not stored as a pixel array: it is evaluated at render time (mesh/fragment/compute shaders).
- **Resolution independence.** Boundaries are splines, gradients are analytic profiles (reserved for 1.2). Zoom without pixelation.
- **Zero-cost mipmaps.** LOD levels are produced by coordinate shifting and dropping lower bit-planes — 0 bytes of additional storage.
- **Asymmetry.** The encoder may be arbitrarily heavy (RDO, Matching Pursuit); the decoder is deterministic and linear in the number of splines.

---

## What to Encode — and What Comes with Caveats

**Recommended:** albedo, masks, UI atlases, AO, roughness, metallic.
**With caveats (Data mode):** octahedral normals — allowed as an experimental input, but perceptual and physical accuracy is not guaranteed. Artifacts are possible (banding, lighting streaks).
**Not intended:** height/displacement maps and any data where subpixel gradient accuracy is critical — the format is not designed for such scenarios.

---

## Key Features of Version 1.1

- **32-byte header.** Contains mode flags, dimensions, PayloadLength, CRC32, ChannelFlags (signed channels), and 512 KiB alignment.
- **Bit-planes (8 per channel).** Gray code + isolines with a step of `2^(7−b)`. The lowest bit holds fine gradations, the highest holds coarse contrast.
- **Content modes (`ContentClass`):**
  - **Artistic:** the full pipeline (bilateral filter, noise layers, adaptive precision).
  - **Data:** no bilateral filter, hard spline epsilon floor (`ε ≤ 0.5`), noise layers optional. For AO/roughness/metallic and octahedral normals.
- **Signed channels (`ChannelFlags`).** Octahedral normals are stored as offset-binary (`value + 128`) and go through the standard pipeline; the decoder subtracts 128 after Gray-code reconstruction.
- **Encoding pipeline (strictly in order):**
  1. Channel decomposition (Y, Cb, Cr, A).
  2. Bilateral filter (skipped in Data mode).
  3. Residual = original − filtered.
  4. Matching Pursuit on the residual (all frequencies, global masks, decision made by RDO).
  5. Multi-scale edge analysis (0.25× / 1× / 4×).
  6. Gray code → 8 bit-planes per channel.
  7. Spline vectorization (Catmull-Rom splines), vertex deduplication, Shared Vertex Buffer.
- **Procedural noise layers.** Applied after Gray-code reconstruction; ordered by frequency; masked contribution; LOD amplitude falloff via `AmplitudeFalloff`.
- **Determinism.** Reference arithmetic is fixed-point Q24.8; generators are integer-only, no floating point; results must match bit for bit.
- **Reserved extensions (1.2):** gradient fills (analytic profile between guide curves), tiling, HDR (>8 bits).

---

## File Format (.bpsc)

- **Header:** 32 bytes (little-endian), Magic `BPSC`, Version `0x02`, flags, dimensions, NumVertices, NumNoiseLayers, CompressionMethod, ChannelFlags, PayloadLength, CRC32.
- **Alignment:** file size ≥ 512 KiB and a multiple of 512 KiB; the tail is zero-padded; decoders read only up to `PayloadLength`.
- **Payload:** Bit-Planar Vertex Stream, VertexFlags, bit-plane blocks (Y, Cb, Cr, A), procedural noise layers.
- **Compression:** applied to the payload as a single block (Zstandard or raw).

---

## Usage

### For Engines / Renderers

- Support both modes (Artistic/Data) via the `ContentClass` flag.
- Implement deterministic rasterization: fills-then-holes rule, pixel-center sampling, Q24.8 arithmetic.
- Mipmapping: prune the lowest bit-planes and shift coordinates (truncation or rounding — per the `CoordinateRounding` flag).
- At `M ≥ 8`: use the constant value 128 (0 for signed channels) and apply noise layers if present.

### For Encoders

- Strictly follow the pipeline order and bit-plane semantics.
- Use RDO to choose between splines and procedural layers.
- Ensure determinism: identical inputs and parameters must produce an identical bitstream.
- For Data mode: disable (or minimize) the bilateral filter, keep `ε ≤ 0.5`, avoid redundant noise layers.

---

## Limitations & Disclaimers

- Normal maps are an experimental zone: lighting artifacts are possible. Do not rely on this format for physically accurate PBR maps.
- Bit 13 of `FlagsAndCount` is reserved for gradient fills: 1.1 decoders must reject files with this bit set.
- Canvas size: up to 65535 per axis.
- Points per spline: up to 8191.

---

## Roadmap

1. **Gradient fills (1.2).** Analytic transition profile between guide curves — to eliminate banding on smooth gradients.
2. **Tiling and progressive loading.** Partial decoding of very large textures.
3. **HDR (>8 bits).** Header extension and plane semantics.
4. **Generator test vectors and conformance tests.** Golden files for verifying determinism.

---

## Resources

- **SPECIFICATION.md** — the full format specification (binary format, pipeline, rasterization, determinism).
- **Reference implementation (planned):** a Python encoder/decoder for 1.1 with the 32-byte header, Bit-Planar stream, CRC32, and deterministic generators.
