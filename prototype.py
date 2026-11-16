import os
import struct
import zlib
import cv2
import numpy as np

# ==========================================
# НАСТРОЙКИ КОДЕКА BPSC (Bit-Plane Spline Codec)
# ==========================================
IMAGE_PATH = "test.jpg"  # Положите свою текстуру/картинку рядом или оставьте так для автогенерации

# 1. Surface Blur (удаление сенсорного шума при сохранении резких границ)
BLUR_DIAMETER = 9
BLUR_SIGMA_COLOR = 35
BLUR_SIGMA_SPACE = 35

# 2. Адаптивная точность опорных узлов сплайна по битовым слоям (от 0-го младшего к 7-му старшему)
LAYER_EPSILON = [5.0, 4.5, 4.0, 3.0, 2.5, 2.0, 1.5, 1.2]
MIN_CONTOUR_AREA = [40, 35, 30, 25, 20, 15, 10, 5]

# 3. Параметры кривых сплайнов Catmull-Rom
SPLINE_TENSION = 0.5
SPLINE_SUBDIVISIONS = 8


def create_synthetic_texture(width=512, height=512):
    """Генерирует тестовую игровую текстуру/декаль с градиентами, четкими формами и шумом."""
    y, x = np.mgrid[0:height, 0:width]
    img = (x * 0.25 + y * 0.25).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Геометрические элементы (имитация элементов HUD/текстуры)
    cv2.circle(img, (170, 170), 95, 225, -1)
    cv2.circle(img, (170, 170), 50, 60, -1)
    cv2.rectangle(img, (270, 250), (455, 435), 35, -1)
    cv2.ellipse(img, (350, 145), (85, 42), 25, 0, 360, 175, -1)

    # Добавляем шум матрицы/зерно
    np.random.seed(42)
    noise = np.random.normal(0, 5.5, img.shape)
    noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_img


def to_gray_code(img):
    return img ^ (img >> 1)


def from_gray_code(gray):
    mask = gray >> 1
    while np.any(mask):
        gray ^= mask
        mask >>= 1
    return gray


def evaluate_closed_catmull_rom(points, subdivisions=8, tension=0.5):
    """Строит гладкую замкнутую кривую сплайна Catmull-Rom по опорным узлам из общего стека."""
    pts = np.array(points, dtype=np.float32)
    n = len(pts)
    if n < 3:
        return pts.astype(np.int32)

    curve_points = []
    t = np.linspace(0, 1, subdivisions, endpoint=False)[:, None]
    t2 = t * t
    t3 = t2 * t

    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]

        m1 = tension * (p2 - p0)
        m2 = tension * (p3 - p1)

        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2

        segment = h00 * p1 + h10 * m1 + h01 * p2 + h11 * m2
        curve_points.append(segment)

    return np.vstack(curve_points).astype(np.int32)


def encode_bpsc(gray_img):
    h, w = gray_img.shape
    smoothed = cv2.bilateralFilter(gray_img, BLUR_DIAMETER, BLUR_SIGMA_COLOR, BLUR_SIGMA_SPACE)
    gray_coded = to_gray_code(smoothed)

    vertex_dict = {}
    vertex_list = []
    layers_contours = []
    total_raw_vertices = 0

    for bit in range(8):
        plane = ((gray_coded >> bit) & 1).astype(np.uint8) * 255
        contours, hierarchy = cv2.findContours(plane, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        layer_polys = []
        if contours and hierarchy is not None:
            hier = hierarchy[0]
            for idx, cnt in enumerate(contours):
                if cv2.contourArea(cnt) < MIN_CONTOUR_AREA[bit]:
                    continue

                approx = cv2.approxPolyDP(cnt, LAYER_EPSILON[bit], True)
                if len(approx) < 3:
                    continue

                is_hole = hier[idx][3] != -1
                poly_indices = []

                for pt in approx:
                    x, y = int(pt[0][0]), int(pt[0][1])
                    total_raw_vertices += 1

                    if (x, y) not in vertex_dict:
                        v_id = len(vertex_list)
                        vertex_dict[(x, y)] = v_id
                        vertex_list.append((x, y))
                    else:
                        v_id = vertex_dict[(x, y)]

                    poly_indices.append(v_id)

                layer_polys.append((is_hole, poly_indices))

        layers_contours.append(layer_polys)

    # Сборка бинарного контейнера .bpsc
    buffer = bytearray()
    buffer.extend(b"BPSC")
    buffer.extend(struct.pack("<BBHHIH", 1, 0, w, h, len(vertex_list), 0))

    for x, y in vertex_list:
        buffer.extend(struct.pack("<HH", x, y))

    for bit in range(8):
        polys = layers_contours[bit]
        buffer.extend(struct.pack("<H", len(polys)))
        for is_hole, indices in polys:
            count_and_flag = len(indices) | (0x8000 if is_hole else 0)
            buffer.extend(struct.pack("<H", count_and_flag))
            for v_id in indices:
                buffer.extend(struct.pack("<H", v_id))

    compressed_data = zlib.compress(bytes(buffer), level=9)

    stats = {
        "raw_vertices": total_raw_vertices,
        "unique_vertices": len(vertex_list),
        "shared_percent": (1 - len(vertex_list) / max(1, total_raw_vertices)) * 100,
        "compressed_bytes": len(compressed_data),
    }
    return compressed_data, stats


def decode_bpsc(compressed_data, mip_shift=0, drop_low_bits=0, upscale=1):
    raw = zlib.decompress(compressed_data)
    if raw[:4] != b"BPSC":
        raise ValueError("Invalid BPSC header")

    _, _, w, h, num_vertices, _ = struct.unpack_from("<BBHHIH", raw, 4)
    offset = 16

    vertices = []
    for _ in range(num_vertices):
        x, y = struct.unpack_from("<HH", raw, offset)
        offset += 4
        sx = (x >> mip_shift) * upscale
        sy = (y >> mip_shift) * upscale
        vertices.append((sx, sy))

    out_w = max(1, (w >> mip_shift) * upscale)
    out_h = max(1, (h >> mip_shift) * upscale)
    gray_coded_reconstructed = np.zeros((out_h, out_w), dtype=np.uint8)

    for bit in range(8):
        num_polys = struct.unpack_from("<H", raw, offset)
        offset += 2
        plane = np.zeros((out_h, out_w), dtype=np.uint8)

        for _ in range(num_polys):
            count_and_flag = struct.unpack_from("<H", raw, offset)
            offset += 2
            is_hole = bool(count_and_flag & 0x8000)
            count = count_and_flag & 0x3FFF

            pts = []
            for _ in range(count):
                v_id = struct.unpack_from("<H", raw, offset)
                offset += 2
                pt = vertices[v_id]
                if not pts or pts[-1] != pt:
                    pts.append(pt)

            if bit >= drop_low_bits and len(pts) >= 3:
                smooth_curve = evaluate_closed_catmull_rom(
                    pts, subdivisions=SPLINE_SUBDIVISIONS * upscale, tension=SPLINE_TENSION
                )
                poly_np = smooth_curve.reshape((-1, 1, 2))
                cv2.fillPoly(plane, [poly_np], 0 if is_hole else 1)

        if bit >= drop_low_bits:
            gray_coded_reconstructed |= (plane << bit)

    return from_gray_code(gray_coded_reconstructed)


def build_comparison_sheet(orig, decoded, upscaled_2x, mip_quarter):
    """Создает наглядную сравнительную картинку для витрины GitHub."""
    h, w = orig.shape
    uh, uw = upscaled_2x.shape
    cy, cx = uh // 2, uw // 2
    crop_zoom = upscaled_2x[cy - h // 2 : cy + h // 2, cx - w // 2 : cx + w // 2]

    mip_vis = cv2.resize(mip_quarter, (w, h), interpolation=cv2.INTER_NEAREST)

    def label(im, text):
        out = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(out, (8, 8), (330, 38), (0, 0, 0), -1)
        cv2.putText(out, text, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return out

    top = np.hstack([label(orig, "1. Original"), label(decoded, "2. BPSC Decoded (1:1)")])
    bottom = np.hstack([label(crop_zoom, "3. BPSC 2x Vector Zoom"), label(mip_vis, "4. Zero-Cost Mipmap (>>2)")])
    sheet = np.vstack([top, bottom])
    cv2.imwrite("comparison_sheet.png", sheet)


def main():
    if os.path.exists(IMAGE_PATH):
        img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    else:
        img = create_synthetic_texture(512, 512)

    cv2.imwrite("ref_original.png", img)
    cv2.imwrite("ref_jpeg85.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    png_size = os.path.getsize("ref_original.png")
    jpg_size = os.path.getsize("ref_jpeg85.jpg")

    compressed_blob, stats = encode_bpsc(img)
    with open("sample.bpsc", "wb") as f:
        f.write(compressed_blob)

    reconstructed = decode_bpsc(compressed_blob, mip_shift=0, drop_low_bits=0, upscale=1)
    upscaled_2x = decode_bpsc(compressed_blob, mip_shift=0, drop_low_bits=0, upscale=2)
    mip_quarter = decode_bpsc(compressed_blob, mip_shift=2, drop_low_bits=2, upscale=1)

    build_comparison_sheet(img, reconstructed, upscaled_2x, mip_quarter)

    print("\n" + "=" * 48)
    print("BPSC REFERENCE PROTOTYPE BENCHMARK")
    print("=" * 48)
    print(f"Raw Spline Control Nodes   : {stats['raw_vertices']}")
    print(f"Shared Vertex Buffer Size  : {stats['unique_vertices']}")
    print(f"Vertex Deduplication       : {stats['shared_percent']:.1f}%")
    print("-" * 48)
    print(f"PNG Lossless Size          : {png_size:,} bytes")
    print(f"JPEG (Quality 85) Size     : {jpg_size:,} bytes")
    print(f"BPSC (.bpsc) File Size     : {stats['compressed_bytes']:,} bytes")
    print("=" * 48)
    print("Generated: sample.bpsc & comparison_sheet.png")


if __name__ == "__main__":
    main()
