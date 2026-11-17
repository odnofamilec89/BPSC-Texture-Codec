# BPSC 1.1 — Format Specification

**BPSC — Bit-Plane Spline & Procedural Codec**
Version: 1.1-Draft (0x02). Category: asymmetric hybrid vector-procedural texture & image compression format. Target platform: GPU-native decompression (compute / mesh / fragment shaders) and a CPU reference decoder.

## 1. Purpose and Architectural Goals

BPSC is designed for ultra-compact storage of high-resolution game textures, masks, and UI assets while providing:

- **VRAM savings.** The texture is not stored as a pixel array — it is evaluated procedurally at render time, every frame. The canonical consumption mode is streaming GPU evaluation (mesh / fragment / compute shaders). A CPU decoder exists only as a testing and debugging reference (Section 8).
- **Resolution independence.** Boundaries are defined by splines; gradient transitions by analytic profiles (reserved extension, Section 9.1). Arbitrary magnification without pixelation.
- **Zero-overhead mipmapping.** Mip levels are generated on the fly via coordinate bit-shifting and dropping lower bit-planes — 0 bytes of extra storage.
- **Asymmetry.** The encoder may be arbitrarily heavy (RDO, Matching Pursuit); the decoder is deterministic and linear in the number of splines.

**Scope.** The format targets color and gradient data: albedo, masks, UI atlases. Physically meaningful maps (normal, roughness, metallic, height) may be encoded only in `Data` mode (Section 2.1) with conscious acceptance of possible defects: perceptual and physical accuracy are not guaranteed. Octahedral normals are a permitted experimental input (Section 2.2).

## 2. Channels, Content Modes, and Sign Handling

### 2.1 Content modes

The header carries a `ContentClass` flag (bit 3 of Flags):

- **Artistic (0).** Full pipeline: bilateral filter, noise layers, adaptive spline precision. Intended for visual textures.
- **Data (1).** Bilateral filtering is disabled (or its sigma is bounded to be negligible by the encoder); spline approximation epsilon has a hard floor of `eps <= 0.5` pixel; noise layers are not applied by default. Intended for AO, roughness, metallic, octahedral normals. Expected size growth is 2–4x versus Artistic on the same image.

### 2.2 Signed channels (octahedral normals)

The `ChannelFlags` header byte marks channels as signed: bit 0 — Y, bit 1 — Cb, bit 2 — Cr, bit 3 — A. A signed channel stores values −128…127 in **offset-binary** encoding: `byte = value + 128`. This preserves value monotonicity, so Gray-code conversion and bit-plane slicing apply unchanged. After reconstruction from Gray code, the decoder subtracts 128.

An octahedral normal is encoded as two signed channels (X, Y); Z is reconstructed outside the format as `sqrt(1 − X² − Y²)`. Channels X and Y pass through the standard pipeline in Data mode.

### 2.3 Channels and their resolution

| Channel | Content | Resolution |
|---|---|---|
| Y | luma (Rec.709, Section 5.4) | full |
| Cb, Cr | chroma | full (`ChromaFactorLog2 = 0`) |
| A | alpha, when `HasAlpha` | full |

All channels are processed identically through one code path and share a single Shared Vertex Buffer. The `ChromaFactorLog2` field is reserved for future subsampling (e.g., terrain datasets); in 1.1 it is always 0.

## 3. Semantic Model of Bit-Planes

Each channel, after conversion to Gray code, is represented by eight binary planes `P_0 … P_7` (LSB → MSB). Semantics: **a plane is a set of value isolines whose spacing doubles as bit significance decreases**:

- `P_7` (MSB) — a single isoline at level 128: a coarse "above/below the middle of the range" split;
- `P_6` — isolines at 64 and 192;
- `P_b` — isolines spaced `2^(7−b)` apart.

Gray code (`G = B ⊕ (B >> 1)`) guarantees that neighboring values differ in exactly one bit: transitions never toggle multiple planes simultaneously, and plane boundaries are smooth isolines suitable for vectorization.

Mipmapping inherits this model: at LOD `M`, the `M` lowest planes are dropped — i.e., gradation steps finer than `2^M`. Subtle differences disappear first: fine noise and thin gradients. A value itself may be anything; it is not "weak values" that are dropped but **fine gradation steps**.

## 4. Encoding Pipeline

The step order is mandatory:

```
[Channel decomposition (2.3)]
   → [Bilateral filter (except Data mode)]
   → [Residual = original − filtered]
   → [Matching Pursuit on residual] ──► [noise layers (Section 7), optional]
   → [Multi-scale edge analysis on the cleaned base]
   → [Gray code → 8 planes per channel]
   → [Spline vectorization → Shared Vertex Buffer → sorting → packing]
```

1. **Bilateral filter.** Removes high-frequency jitter of the lower bit-planes while strictly preserving structural edges. Parameters are left to encoder RDO.
2. **Procedural noise extraction (Matching Pursuit).** Operates across the **entire frequency range**, not only micro-texture: low-frequency Perlin/Voronoi clouds can cheaply encode large-scale variation (dirt, wear, lighting variation), replacing hundreds of splines with a single layer record. Layers in the file are ordered by increasing frequency. 2…4 iterations; an iteration stops when the cost of storing the mask and parameters exceeds the residual's entropy savings. Recommendation: replace full RDO enumeration with greedy selection — pick the generator by local variance of the residual and confirm the choice with a real measurement of stream size. The decision "write this layer to the file" belongs to RDO by bitrate: if splines are cheaper, the layer is not written. A file with `NumNoiseLayers = 0` is valid.
3. **Multi-scale edge analysis.** A 3-tier pyramid `0.25x / 1x / 4x`. If a staircase edge at `1x` resolves into a continuous gradient at `4x`, the boundary is fitted as a single smooth spline rather than a chain of steps. (Perspective: regions with smooth gradients may be transferred to gradient fills — Section 9.1.)
4. **Gray code and slicing.** Per channel.
5. **Spline vectorization.** Closed cubic Catmull-Rom splines with adaptive precision `eps_b` (increasing from lower planes to higher; in Data mode, with the floor from 2.1). Each vertex receives a curvature flag. Control points of all channels and planes are **snapped to a common grid** (recommendation: 1–2 px step for the higher planes — this sharply increases deduplication share) and deduplicated into a single Shared Vertex Buffer.
6. **Sorting and packing.** Vertices are sorted by topological significance (curvature weight) — the basis for future progressive loading. Packing per Section 6.

## 5. Rasterization Semantics and Determinism

This section is normative: any implementation claiming correctness must reproduce the reference decoder output bit-for-bit.

**5.1 Overlap rule.** Within a channel's plane, all fills (`IsHole = 0`) are rasterized first in arbitrary order, then all holes (`IsHole = 1`) in arbitrary order: `target |= (1 << b)` for fills, `target &= ~(1 << b)` for holes. There is no order dependency. The encoder must classify contours correctly (nesting via contour hierarchy).

**5.2 Sampling point.** Pixel `(x, y)` is filled if its center `(x + 0.5, y + 0.5)` lies inside the polygon. A boundary passing through the center counts as filled (the top-left rule is not applied).

**5.3 Arithmetic.** The reference uses fixed-point Q24.8.

- Vertex coordinates are integers on the luma grid.
- Catmull-Rom spline for a closed contour of `n >= 3` points: for segment `P_0 = P[i−1], P_1 = P[i], P_2 = P[i+1], P_3 = P[i+2]` (indices modulo n): tangents `m_1 = (T/2)·(P_2 − P_0)`, `m_2 = (T/2)·(P_3 − P_1)`; Hermite basis `h00 = 2t³ − 3t² + 1`, `h10 = t³ − 2t² + t`, `h01 = −2t³ + 3t²`, `h11 = t³ − t²`; point: `h00·P_1 + h10·m_1 + h01·P_2 + h11·m_2`. Default `T = 1.0`; a "sharp corner" flag on a vertex zeroes its contribution to the tangents of adjacent segments.
- **Segment subdivision: `S = 8` points, `t = i/S` — fixed and not stored in the file.** `S` being a power of two guarantees subdivision-point agreement across mip levels.
- Intermediate quantities — Q24.8, rounded to nearest.

**5.4 Color.** Luma: `Y = 0.2126·R + 0.7152·G + 0.0722·B` (BT.709). Chroma: `Cb = 0.5·(B − Y)/(1 − 0.0722) + 128`, `Cr = 0.5·(R − Y)/(1 − 0.2126) + 128`. Inverse conversion uses the same coefficients; arithmetic is Q24.8. Values are interpreted as sRGB-encoded; procedural noise is applied in the encoded domain (a conscious assumption).

**5.5 Gray-code reconstruction.** `B = G ⊕ (G >> 1) ⊕ (G >> 2) ⊕ … ⊕ (G >> 7)` — prefix fold, one GPU compute pass.

**5.6 Determinism of procedural generators.** Mandatory (Section 7.2); implementation-dependent operations are forbidden: `rand()`, library-dependent floating-point trigonometry. On divergence, the CPU reference implementation is the arbiter.

## 6. Binary Container Format (.bpsc)

Byte order: little-endian. All blocks after the header form a single payload, compressed as a whole by the selected method.

### 6.1 Header (32 bytes)

| Offset | Size | Type | Field | Description |
|---|---|---|---|---|
| 0x00 | 4 | char[4] | Magic | `'B','P','S','C'` |
| 0x04 | 1 | uint8 | Version | `0x02` |
| 0x05 | 1 | uint8 | Flags | bit 0: HasChroma; bit 1: HasNoiseLayers; bit 2: HasAlpha; bit 3: ContentClass (0 = Artistic, 1 = Data); bit 4: CoordinateRounding (0 = truncation, 1 = round-to-nearest); bits 5–7: reserved (0) |
| 0x06 | 2 | uint16 | Width | 1…65535 |
| 0x08 | 2 | uint16 | Height | 1…65535 |
| 0x0A | 4 | uint32 | NumVertices | unique vertices in the Shared Vertex Buffer |
| 0x0E | 2 | uint16 | NumNoiseLayers | number of procedural passes |
| 0x10 | 1 | uint8 | CompressionMethod | 0 = raw, 1 = Zstandard |
| 0x11 | 1 | uint8 | ChromaFactorLog2 | always 0 in 1.1; reserved |
| 0x12 | 1 | uint8 | ChannelFlags | bits 0–3: Y/Cb/Cr/A signed; bits 4–7: reserved (0) |
| 0x13 | 1 | uint8 | Reserved | 0 |
| 0x14 | 4 | uint32 | PayloadLength | actual size of the compressed payload |
| 0x18 | 4 | uint32 | CRC32 | CRC-32 (IEEE, poly 0xEDB88320, init 0xFFFFFFFF) of the uncompressed payload |
| 0x1C | 4 | uint32 | Reserved | 0 |

### 6.2 Container Alignment

File size: `max(512 KiB, ceil(PayloadLength / 512 KiB) × 512 KiB)`. The tail after the payload is zero-padded; the decoder ignores everything beyond `PayloadLength`. Block-aligned size simplifies disk streaming and GPU memory mapping.

### 6.3 Payload (uncompressed form); block order is strict

1. **Bit-Planar Vertex Stream.** `X` coordinates of all vertices, then `Y`. Each coordinate is stored as 16 bit-planes from MSB to LSB; plane `b` occupies `ceil(NumVertices / 8)` bytes of packed bits (bit 0 of the first byte = bit `b` of vertex 0). This allows the decoder at LOD `M` to read only the high bits without unpacking the low ones. The only coordinate storage method (delta encoding was removed as incompatible with pruning).
2. **VertexFlags.** `ceil(NumVertices / 8)` bytes: 1 = sharp corner, 0 = smooth Catmull-Rom node.
3. **Bit-plane blocks** in order: Y (planes 0…7), then Cb (0…7), Cr (0…7) if HasChroma, then A (0…7) if HasAlpha. Each block:
   - `NumSplines` (uint16);
   - per spline: `FlagsAndCount` (uint16) — bit 15: IsHole; bit 14: HasDynamicFeather; bit 13: IsGradient (reserved, Section 9.1; a 1.1 decoder must reject the file if set); bits 0…12: `PointCount` <= 8191;
     - `VertexIndices` — uint16 if `NumVertices <= 65535`, otherwise uint32;
     - if bit 14: **FeatherProfile** (4 bytes): `FeatherWidth` (uint8, Q4.4, in base-level pixels), `FalloffType` (uint8: 0 = linear, 1 = smoothstep, 2 = gaussian), 2 reserved bytes (0).
   - Within a block, splines must appear first with all `IsHole = 0`, then all `IsHole = 1`.
4. **Procedural Noise Layers** if HasNoiseLayers, `NumNoiseLayers` records ordered by increasing frequency:
   - `GeneratorID` (uint8): 0x01 Gaussian, 0x02 Perlin, 0x03 Simplex, 0x04 Voronoi/Cellular;
   - `Seed` (uint32);
   - `Frequency` (float16) — cycles per base pixel;
   - `Amplitude` (float16) — in units of 1/255;
   - `Octaves` (uint8);
   - `BlendMode` (uint8): 0 = Add, 1 = Subtract;
   - `AmplitudeFalloff` (uint8) — per-LOD amplitude falloff; 0 = hard cutoff at Nyquist, other values reserved;
   - `MaskSplineCount` (uint16); **0 = global mask** — the layer applies to the entire canvas;
   - per mask: `FlagsAndCount` (uint16, same semantics as splines; bit 15 = mask hole) + `VertexIndices`. Masks are specified in luma-grid coordinates.

Compression is applied to blocks 1–4 as a single array.

### 6.4 Decoder Validation

The decoder must reject a file on: invalid Magic/Version; CRC32 mismatch; indices out of range of `NumVertices`; fill/hole order violation within a block; IsGradient bit set; `ChromaFactorLog2 != 0`; unknown GeneratorID/BlendMode/FalloffType; `PayloadLength` inconsistent with the actual file size.

## 7. Procedural Noise Layers

### 7.1 Application Semantics

Noise is applied **after** Gray-code reconstruction, in the channel value range 0…255:

- Add: `value' = clamp(value + A_k(uv), 0, 255)`
- Subtract: `value' = clamp(value − A_k(uv), 0, 255)`

where `A_k(uv)` is the generator's evaluation with masks applied: full contribution inside masks with `IsHole = 0` and outside holes, zero otherwise. With `MaskSplineCount = 0` — full contribution everywhere. Layer application order is the file order.

### 7.2 Generator Determinism

- Lattice hash — integer-based (baseline: splitmix64 reduced to u32).
- Perlin permutation table — Fisher-Yates shuffle with a PCG32 PRNG seeded from `Seed`.
- Smoothing function — quintic `t³(t(6t − 15) + 10)`, Q16.16.
- Gaussian — deterministic white noise from an integer pixel hash (the only way to guarantee reproducibility without floating point).
- Simplex and Voronoi — specified by a mandatory reference implementation in the repository plus test vectors; on divergence, the CPU implementation is the arbiter.

### 7.3 Nyquist Limit and Amplitude Falloff

Octave frequency `f·2^k` must not exceed half the sampling frequency of the current mip level; higher octaves are dropped. When `AmplitudeFalloff != 0`, a gradual per-LOD falloff of octave contribution is permitted (the specific curve is a quality-control matter, to be fixed by benchmark); otherwise the contribution is cut off.

## 8. Decoding and GPU Pipeline

### 8.1 Mipmapping (LOD `M`)

1. **Memory pruning.** The `M` lowest bit-planes of each channel and the `M` lowest bits of coordinates in the Bit-Planar Vertex Stream are skipped.
2. **Coordinate shift.** With `CoordinateRounding = 0`: `X' = X >> M`. With `CoordinateRounding = 1`: `X' = (X + (1 << (M−1))) >> M` for `M >= 1`, otherwise `X' = X`. The mode is set by the encoder flag (6.1).
3. **Spline evaluation on the mip grid.** Subdivision (`S = 8`) is performed in the mip-level grid coordinates.
4. **Degenerate culling.** Consecutive vertices coinciding after the shift are deduplicated; splines with fewer than 3 unique vertices are dropped; splines with a bounding box outside the mip grid are dropped.
5. **Degenerate case `M >= 8`.** All channel planes are dropped. The channel takes the constant value 128 (0 for signed channels); noise layers are applied if present. This is a fixed fallback, not a file error.

### 8.2 GPU Stages

1. **Spline tessellation** (mesh/tessellation): Catmull-Rom evaluation in screen space with adaptive subdivision by projected segment length. The reference norm is `S = 8` (5.3); adaptive tessellation is permitted as an approximation if the result on the integer grid matches the reference.
2. **Bitwise rasterization** into an `R8_UINT` target per rule 5.1.
3. **Gray-code reconstruction** in a single compute pass (5.5), sign correction per ChannelFlags, channel merge (5.4).
4. **Procedural synthesis**: generators evaluated in UV/screen space, octave clamping at the level-`M` Nyquist limit, blending by masks (Section 7).

## 9. Reserved Extensions

### 9.1 Gradient Fills (planned for 1.2)

Bit 13 of `FlagsAndCount` is reserved for `IsGradient`: a pair of guide splines defines a strip of smooth transition; the cross-profile is an analytic function `f(t)`, `t ∈ [0,1]` from guide A to guide B. The value inside the strip: `value = A_val + f(t)·(B_val − A_val)`. The profile is stored as N stops (uint8 value + t position in Q8) + interpolation type between them (linear / smoothstep / with break points). The decoder rasterizes the strip as a ribbon: vertices on the A boundary carry `t = 0`, on the B boundary `t = 1`; `t` interpolation is performed by the GPU for free.

Properties: the gradation **within** the transition is infinite — the value is an analytic function of coordinates, not a discrete sample; zoom and mip-pruning trim only the guide geometry, the cross-profile does not degrade. The position of the transition axis is limited by encoder precision (the guides' epsilon).

Encoder requirements (when implemented): guide strips must not self-intersect; the cross-strip mapping must be injective. A 1.1 decoder must reject files with bit 13 set.

### 9.2 Other Directions

- **Tiling** for partial decoding of very large textures: 256×256 tiles, a tile table — a separate format extension.
- **Progressive loading**: vertex sorting by significance is already in the pipeline (Step 6); the partial-read protocol is a future spec.
- **HDR / >8 bit**: will require a plane-count field in a header extension.

## 10. Format Limits

| Parameter | Value |
|---|---|
| Canvas size | 1…65535 per axis |
| NumVertices | <= 2³²−1; indices uint16 when <= 65535, otherwise uint32 |
| Points per spline | <= 8191 |
| Splines per plane | <= 65535 |
| Channels | 1 (Y) … 4 (Y, Cb, Cr, A) |
| Planes per channel | 8 (fixed) |
| LOD range | 0…∞; for M >= 8 — fallback 8.1(5) |
| File size | >= 512 KiB, multiple of 512 KiB |

## 11. Encoder Requirements

- Produce a stream conforming to Sections 5 and 6: contour classification, fill/hole ordering, coordinate snapping, determinism.
- Determinism: identical input and parameters produce byte-identical output. RDO heuristics are permitted but must be deterministic.
- RDO optimization, filter/generator/snapping parameter selection are outside the format: the spec normalizes the stream, not the encoder strategy.

## 12. Open Questions (non-blocking for 1.1)

1. Per-LOD noise amplitude falloff curve (7.3) — to be fixed by benchmark results.
2. Gradient fill implementation (9.1) — after the first encoder benchmark: the measured share of gradient regions in real textures will set the priority.
3. Noise generator test vectors and golden conformance files — the first 1.1 commit.
4. Bilinear chroma upscale instead of nearest neighbor — a quality matter, not a format matter.