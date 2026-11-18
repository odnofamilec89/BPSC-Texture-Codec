#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v15_decoder_oldfilter - автономный декодер формата BPSC (ветка v15_oldfilter).

Умеет:
  - полный декод в родном разрешении контейнера (эквивалент decode() прототипа);
  - нативную растеризацию в scale x scale: сплайновые контуры считаются
    в детальной сетке (координаты вершин умножаются на scale), а не
    апскейлом готовых пикселей;
  - шумовой слой всегда живёт в родном разрешении контейнера: поле
    генерируется по (seed, sigma, mean) из файла в сетке w x h и
    накладывается nearest-апскейлом - зерно переносится as is;
  - USE_CR (Катмулл-Ром) поддержан;
  - самопроверка: среднее по блоку scale x scale составного рендера
    сравнивается с декодом в родном разрешении.

Запуск:
  python v15_decoder_oldfilter.py файл.bpsc [выход.png] [scale]

Без аргументов декодирует cat_v15.bpsc в родном разрешении.
"""
import struct
import sys
import zlib

import numpy as np

try:
    import cv2

    def imwrite(path, img):
        return cv2.imwrite(path, img)

    def resize_up(img, w, h, scale):
        return cv2.resize(img, (w * scale, h * scale),
                          interpolation=cv2.INTER_LANCZOS4)
except ImportError:
    from PIL import Image

    def imwrite(path, img):
        Image.fromarray(img).save(path)

    def resize_up(img, w, h, scale):
        return np.asarray(
            Image.fromarray(img).resize((w * scale, h * scale), Image.LANCZOS))

MAGIC = b"BPSC"
VERSION = 0x02
HDR_FMT = "<4sBBHHIHB BBBIII".replace(" ", "")
assert struct.calcsize(HDR_FMT) == 32
LONG_COUNT_BIT = 0x4000
HOLE_BIT = 0x8000
MAX_SHORT_COUNT = 0x1FFF
FLAG_USE_CR = 0x01

NOISE_GAUSS = 0
NOISE_REC_FMT = "<BHiIB"
assert struct.calcsize(NOISE_REC_FMT) == 12
NOISE_SIG_SCALE = 64
NOISE_MEAN_SCALE = 64


# ---------------- битовые плоскости ----------------
def unpack_planes(buf, off, n, nbits):
    nb = (n + 7) // 8
    v = np.zeros(n, dtype=np.uint32)
    for b in range(nbits - 1, -1, -1):
        bits = np.unpackbits(np.frombuffer(buf, np.uint8, nb, off),
                             bitorder="little")[:n]
        v |= bits.astype(np.uint32) << b
        off += nb
    return v, off


def gray_decode(g, nbits):
    q = g.copy()
    for i in range(1, nbits):
        q ^= g >> i
    return q


# ---------------- вспомогательное для колец ----------------
def _dedup_mask(pts):
    n = len(pts)
    if n < 2:
        return np.ones(n, bool)
    keep = np.ones(n, bool)
    keep[1:] = np.any(pts[1:] != pts[:-1], axis=1)
    if n > 2 and np.array_equal(pts[0], pts[-1]):
        keep[-1] = False
    return keep


# ---------------- Катмулл-Ром ----------------
def _cr_arc(p_prev, p0, p1, p_next):
    """Точки дуги CR от p0 (включительно) к p1 (не включительно)."""
    d = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    n = max(2, min(64, int(d)))
    t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
    P0, P1, P2, P3 = (np.asarray(p, np.float64) for p in (p_prev, p0, p1, p_next))
    return 0.5 * (2 * P1 + (-P0 + P2) * t
                  + (2 * P0 - 5 * P1 + 4 * P2 - P3) * t ** 2
                  + (-P0 + 3 * P1 - 3 * P2 + P3) * t ** 3)


def dense_ring(pts, fidx=None, smooth=None):
    """Кольцо в плотном виде: рёбра между гладкими вершинами - дуги CR."""
    pts_f = pts.astype(np.float64)
    if smooth is None or fidx is None:
        return pts_f
    sm = smooth[fidx]
    n = len(pts_f)
    out = []
    for k in range(n):
        a, b2 = pts_f[k], pts_f[(k + 1) % n]
        if sm[k] and sm[(k + 1) % n]:
            out.append(_cr_arc(pts_f[(k - 1) % n], a, b2, pts_f[(k + 2) % n]))
        else:
            out.append(a[None, :])
    return np.concatenate(out, 0)


# ---------------- even-odd растеризация по центрам пикселей ----------------
def rasterize_evenodd(shape, rings):
    """Пиксель закрашен, если его центр (x+0.5, y+0.5) внутри нечётного числа колец."""
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    rows = [[] for _ in range(h)]
    for pts in rings:
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            if y1 == y2:
                continue
            lo_r = max(int(np.ceil(min(y1, y2) - 0.5)), 0)
            hi_r = min(int(np.ceil(max(y1, y2) - 0.5)), h)
            for r in range(lo_r, hi_r):
                if x1 == x2:
                    rows[r].append(float(x1))
                else:
                    t = (r + 0.5 - y1) / (y2 - y1)
                    rows[r].append(x1 + t * (x2 - x1))
    for r in range(h):
        xs_ = sorted(rows[r])
        row = m[r]
        for i in range(0, len(xs_) - 1, 2):
            a, b2 = xs_[i], xs_[i + 1]
            lo = max(int(np.floor(a - 0.5)) + 1, 0)
            hi = min(int(np.ceil(b2 - 0.5)) - 1, w - 1)
            if lo <= hi:
                row[lo:hi + 1] = 1
    return m


# ---------------- шумовой слой: SplitMix64 + Box-Muller ----------------
def _sm64(z):
    with np.errstate(over="ignore"):
        z = z + np.uint64(0x9E3779B97F4A7C15)
        k = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        k = (k ^ (k >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return k ^ (k >> np.uint64(31))


def gauss_noise_field(seed, h, w, mean, sigma):
    """Детерминированное поле гауссовского шума, int16."""
    n = h * w
    if sigma <= 0.0:
        return np.zeros((h, w), np.int16)
    m2 = (n + 1) // 2 * 2
    with np.errstate(over="ignore"):
        u = (_sm64(np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
                   + np.arange(m2, dtype=np.uint64)) >> np.uint64(11)).astype(np.float64)
    u *= 1.0 / 9007199254740992.0
    u1 = np.maximum(u[0::2], 1e-300)
    u2 = u[1::2]
    r = np.sqrt(-2.0 * np.log(u1))
    th = 2.0 * np.pi * u2
    nrm = np.concatenate([r * np.cos(th), r * np.sin(th)])[:n]
    vals = np.floor(mean + sigma * nrm + 0.5)
    return vals.astype(np.int16).reshape(h, w)


# ---------------- декодер ----------------
def decode_render(data, scale=1):
    """Декод BPSC. scale=1 - родное разрешение (эквивалент decode() прототипа).
    scale>1 - нативная растеризация контуров в сетке scale-крат, шум
    генерируется в родном разрешении и накладывается nearest (зерно as is)."""
    (magic, version, flags, w, h, nv, nnl, _comp, _chroma, _chanf,
     b_planes, plen, crc, _r) = struct.unpack(HDR_FMT, data[:32])
    assert magic == MAGIC and version == VERSION, "bad magic/version"
    payload = data[32:32 + plen]
    assert len(payload) == plen, "truncated file"
    assert zlib.crc32(payload) & 0xFFFFFFFF == crc, "CRC mismatch"
    B = b_planes
    use_cr = bool(flags & FLAG_USE_CR)

    off = 0
    xs, off = unpack_planes(payload, off, nv, 16)
    ys, off = unpack_planes(payload, off, nv, 16)
    foff = off
    off += (nv + 7) // 8
    fbits = np.unpackbits(np.frombuffer(payload, np.uint8, (nv + 7) // 8, foff),
                          bitorder="little")[:nv].astype(bool)
    idx_dt = np.uint16 if nv <= 65535 else np.uint32
    ib = np.dtype(idx_dt).itemsize

    masks = []
    for b in range(B):
        ns = struct.unpack_from("<H", payload, off)[0]
        off += 2
        spls = []
        for _ in range(ns):
            fc = struct.unpack_from("<H", payload, off)[0]
            off += 2
            cnt = fc & MAX_SHORT_COUNT
            if fc & LONG_COUNT_BIT:
                cnt = struct.unpack_from("<H", payload, off)[0]
                off += 2
            spls.append(np.frombuffer(payload, idx_dt, cnt, off))
            off += cnt * ib
        masks.append(spls)

    noise = []
    for _ in range(nnl):
        noise.append(struct.unpack_from(NOISE_REC_FMT, payload, off))
        off += struct.calcsize(NOISE_REC_FMT)

    W, H = w * scale, h * scale
    X = (xs.astype(np.int64) * scale).astype(np.int32)
    Y = (ys.astype(np.int64) * scale).astype(np.int32)

    g_eff = np.zeros((H, W), np.uint16)
    for b in range(B):
        polys = []
        for idx in masks[b]:
            raw = np.stack([X[idx], Y[idx]], axis=1).astype(np.int32)
            keep = _dedup_mask(raw)
            pts = raw[keep]
            if len(pts) < 3:
                continue
            x0, y0 = pts.min(0)
            x1, y1 = pts.max(0)
            if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H:
                continue
            if use_cr:
                polys.append(dense_ring(pts, idx[keep], fbits))
            else:
                polys.append(pts.astype(np.float64))
        if polys:
            g_eff |= rasterize_evenodd((H, W), polys).astype(np.uint16) << b

    if B <= 0:
        img = np.full((H, W), 128, np.uint8)
    else:
        q = gray_decode(g_eff, B)
        top = (q >> (B - 1)) & 1
        base = ((q << (8 - B)) | (top * ((1 << (8 - B)) - 1))).astype(np.int16)
        for (gen, sig_q, mean_q, seed, _nf) in noise:
            if gen == NOISE_GAUSS:
                sig = sig_q / NOISE_SIG_SCALE
                fld = gauss_noise_field(seed, h, w,
                                        mean_q / NOISE_MEAN_SCALE, sig)
                if scale > 1:
                    fld = np.repeat(np.repeat(fld, scale, axis=0),
                                    scale, axis=1)
                base = np.clip(base + fld, 0, 255)
        img = base.astype(np.uint8)

    info = dict(w=w, h=h, nv=nv, B=B, nnl=nnl, noise=noise,
                use_cr=use_cr, flags=flags, plen=plen)
    return img, info


def psnr(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    d = a - b
    mse = float(np.mean(d * d))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main(argv):
    src = argv[1] if len(argv) > 1 else "cat_v15.bpsc"
    out = argv[2] if len(argv) > 2 else None
    scale = int(argv[3]) if len(argv) > 3 else 1
    with open(src, "rb") as f:
        data = f.read()
    img, info = decode_render(data, 1)
    print("Файл: %s (%d B)" % (src, len(data)))
    print("Заголовок: %dx%d | вершин %d | плоскостей B=%d | шумовых слоёв %d | USE_CR=%s"
          % (info["w"], info["h"], info["nv"], info["B"], info["nnl"], info["use_cr"]))
    for i, (gen, sig_q, mean_q, seed, _nf) in enumerate(info["noise"]):
        print("  шум слой %d: gen=%d sigma=%.3f mean=%.3f seed=0x%08X"
              % (i, gen, sig_q / NOISE_SIG_SCALE,
                 mean_q / NOISE_MEAN_SCALE, seed))
    print("Декод в родном разрешении: %dx%d" % (info["w"], info["h"]))
    if scale > 1:
        comp, _ = decode_render(data, scale)
        h, w = info["h"], info["w"]
        block = comp.reshape(h, scale, w, scale).mean(axis=(1, 3))
        print("Самопроверка: PSNR(блок-среднее рендера %dx%d, декод %dx%d) = %.2f дБ"
              % (w * scale, h * scale, w, h, psnr(block, img)))
        try:
            up = resize_up(img, w, h, scale)
            print("Нативная растеризация vs Ланцош-апскейл: PSNR %.2f дБ"
                  % psnr(comp, up))
        except Exception as e:
            print("Сравнение с Ланцошем пропущено:", e)
        if out is None:
            out = src.rsplit(".", 1)[0] + "_%dx%d.png" % (w * scale, h * scale)
        imwrite(out, comp)
        print("Сохранено: %s (%dx%d)" % (out, w * scale, h * scale))
        return comp
    return img


if __name__ == "__main__":
    main(sys.argv)
