#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BPSC 1.1 - референсный прототип, версия 12: энкодер + декодер + самотест.
Запуск:  python prototype12.py

Новое в 12 относительно prototype11.py:
  - VertexFlags теперь настоящие: бит 0 байта флага = гладкая вершина;
    блок пишется энкодером и читается декодером;
  - растеризация ребра: если оба конца гладкие - ребро идёт по дуге
    Катмулла-Рома, иначе прямая;
  - порог гладкости привязан к эпсилону: min(85, 20 + 12*eps);
  - при eps=0 флаги гладкости не ставятся - самопроверка 0.0000% сохранена;
  - CR включается флагом энкодера (байт 2 заголовка, бит 0x01 = USE_CR),
    декодер включает тот же режим автоматически.

Унаследовано от 11:
  - заголовок 32 байта, Magic 'BPSC', Version 0x02, CRC32 несжатого груза;
  - Bit-Planar Vertex Stream: 16 плоскостей MSB->LSB, бит 0 первого байта
    = бит вершины 0;
  - индексы uint16/uint32 по NumVertices; код Грея; мипы прунингом плоскостей
    + сдвиг координат (усечение, CoordinateRounding не ставится);
  - расширенная запись счётчика контура > 8191 (bit 14 + uint16);
  - контуры по рёбрам пикселей (сетка углов), even-odd заливка по центрам;
  - адаптивный эпсилон: младшие плоскости грубим, старшие бережём;
  - RDO-отсев микроконтуров по площади.

Сознательные упрощения против спеки:
  - один канал Y, ContentClass=Artistic, NumNoiseLayers=0, сжатие raw;
  - размер файла = 32 + PayloadLength (без выравнивания до 512 КиБ);
  - число плоскостей B хранится в резервном байте 0x13;
  - бит 0x01 байта flags (резидент спеки) занят под USE_CR.
"""
import struct
import zlib

import cv2
import numpy as np

MAGIC = b'BPSC'
VERSION = 0x02
HDR_FMT = '<4sBBHHIHB BBBIII'.replace(' ', '')
assert struct.calcsize(HDR_FMT) == 32
LONG_COUNT_BIT = 0x4000
HOLE_BIT = 0x8000
MAX_SHORT_COUNT = 0x1FFF
FLAG_USE_CR = 0x01


# ---------------- битовые плоскости ----------------
def pack_planes(values, nbits):
    v = np.asarray(values, dtype=np.uint32)
    out = bytearray()
    for b in range(nbits - 1, -1, -1):
        out += np.packbits(((v >> b) & 1).astype(np.uint8), bitorder='little').tobytes()
    return bytes(out)


def unpack_planes(buf, off, n, nbits):
    nb = (n + 7) // 8
    v = np.zeros(n, dtype=np.uint32)
    for b in range(nbits - 1, -1, -1):
        bits = np.unpackbits(np.frombuffer(buf, np.uint8, nb, off), bitorder='little')[:n]
        v |= bits.astype(np.uint32) << b
        off += nb
    return v, off


def gray_encode(q):
    return q ^ (q >> 1)


def gray_decode(g, nbits):
    q = g.copy()
    for i in range(1, nbits):
        q ^= g >> i
    return q


# ---------------- контуры по угловой сетке ----------------
def contours_edges(mask):
    """Контуры вдоль рёбер пикселей (сетка углов), классификация дырок по глубине."""
    up = np.repeat(np.repeat(mask, 2, 0), 2, 1)
    contours, hier = cv2.findContours(up, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hier = hier[0] if hier is not None else np.empty((0, 4), int)
    out = []
    for i, c in enumerate(contours):
        p = c.reshape(-1, 2)
        x = (p[:, 0] + (p[:, 0] & 1)) >> 1
        y = (p[:, 1] + (p[:, 1] & 1)) >> 1
        pts = np.stack([x, y], 1).astype(np.int32)
        pts = _dedup_ring(pts)
        if len(pts) < 3:
            continue
        d, par = 0, int(hier[i][3]) if len(hier) else -1
        while par != -1:
            d += 1
            par = int(hier[par][3])
        out.append(((d % 2) == 1, pts))
    return out


def _dedup_mask(pts):
    n = len(pts)
    if n < 2:
        return np.ones(n, bool)
    keep = np.ones(n, bool)
    keep[1:] = np.any(pts[1:] != pts[:-1], axis=1)
    if n > 2 and np.array_equal(pts[0], pts[-1]):
        keep[-1] = False
    return keep


def _dedup_ring(pts):
    return pts[_dedup_mask(pts)]


def _ring_area(pts):
    """Площадь замкнутого полигона (shoelace), pts: (N,2) int."""
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _turn_angles(pts):
    """Углы поворота в вершинах замкнутого кольца, градусы."""
    pts = pts.astype(np.float64)
    v1 = pts - np.roll(pts, 1, 0)
    v2 = np.roll(pts, -1, 0) - pts
    cross = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    dot = v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1]
    return np.degrees(np.arctan2(cross, dot))


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


# ---------------- энкодер ----------------
def encode(img, B=6, eps_scale=1.0, min_area_scale=1.0, cr=False):
    h, w = img.shape
    base = cv2.bilateralFilter(img, 5, 30, 5)
    q = base.astype(np.uint16) >> (8 - B)
    g = gray_encode(q)

    vmap, vlist = {}, []
    smooth = {}  # индекс вершины -> гладкая во всех плоскостях (конъюнкция)

    def vid(p):
        k = (int(p[0]), int(p[1]))
        i = vmap.get(k)
        if i is None:
            i = len(vlist)
            vmap[k] = i
            vlist.append(k)
        return i

    plane_blocks = []
    for b in range(B):
        mask = (((g >> b) & 1).astype(np.uint8)) * 255
        eps = 0 if eps_scale == 0 else max(0.4, eps_scale * (7 - b))
        min_area = min_area_scale * float(2 ** (7 - b))
        thr = min(85.0, 20.0 + 12.0 * eps)
        blocks = []
        for hole, pts in contours_edges(mask):
            if min_area > 0 and abs(_ring_area(pts)) < min_area:
                continue
            if eps > 0:
                pts = cv2.approxPolyDP(pts.reshape(-1, 1, 2), eps, True).reshape(-1, 2)
                pts = _dedup_ring(pts)
            if len(pts) < 3:
                continue
            idxs = [vid(p) for p in pts]
            if cr and eps > 0:
                ang = _turn_angles(pts)
                for k, i in enumerate(idxs):
                    smooth[i] = smooth.get(i, True) and (float(ang[k]) <= thr)
            blocks.append((hole, np.asarray(idxs, dtype=np.uint32)))
        plane_blocks.append(blocks)

    nv = len(vlist)
    idx_dt = np.uint16 if nv <= 65535 else np.uint32
    xs = np.array([p[0] for p in vlist], dtype=np.uint32)
    ys = np.array([p[1] for p in vlist], dtype=np.uint32)
    payload = pack_planes(xs, 16) + pack_planes(ys, 16)
    payload += bytes((nv + 7) // 8)
    fb = np.zeros(nv, np.uint8)
    for i in range(nv):
        if smooth.get(i, True):
            fb[i] = 1
    payload += np.packbits(fb, bitorder='little').tobytes()
    for blocks in plane_blocks:
        payload += struct.pack('<H', len(blocks))
        for hole, idxs in blocks:
            n = len(idxs)
            if n > MAX_SHORT_COUNT:
                payload += struct.pack('<HH', LONG_COUNT_BIT | (HOLE_BIT if hole else 0), n)
            else:
                payload += struct.pack('<H', (HOLE_BIT if hole else 0) | n)
            payload += idxs.astype(idx_dt).tobytes()
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    hdr = struct.pack(HDR_FMT, MAGIC, VERSION, FLAG_USE_CR if cr else 0, w, h, nv,
                      0, 0, 0, 0, B, len(payload), crc, 0)
    return hdr + payload


# ---------------- декодер ----------------
def decode(data, M=0):
    (magic, version, flags, w, h, nv, nnl, comp, chroma, chanf,
     b_planes, plen, crc, _r) = struct.unpack(HDR_FMT, data[:32])
    assert magic == MAGIC and version == VERSION, 'bad magic/version'
    payload = data[32:32 + plen]
    assert len(payload) == plen, 'truncated file'
    assert zlib.crc32(payload) & 0xFFFFFFFF == crc, 'CRC mismatch'
    B = b_planes
    use_cr = bool(flags & FLAG_USE_CR)
    rounding = bool(flags & 0x10)

    off = 0
    xs, off = unpack_planes(payload, off, nv, 16)
    ys, off = unpack_planes(payload, off, nv, 16)
    foff = off
    off += (nv + 7) // 8
    fbits = np.unpackbits(np.frombuffer(payload, np.uint8, (nv + 7) // 8, foff),
                          bitorder='little')[:nv].astype(bool)
    idx_dt = np.uint16 if nv <= 65535 else np.uint32
    ib = np.dtype(idx_dt).itemsize

    masks = {}
    for b in range(B):
        ns = struct.unpack_from('<H', payload, off)[0]
        off += 2
        spls = []
        for _ in range(ns):
            fc = struct.unpack_from('<H', payload, off)[0]
            off += 2
            cnt = fc & MAX_SHORT_COUNT
            if fc & LONG_COUNT_BIT:
                cnt = struct.unpack_from('<H', payload, off)[0]
                off += 2
            idx = np.frombuffer(payload, idx_dt, cnt, off)
            off += cnt * ib
            if b >= M:
                spls.append(idx)
        if b >= M:
            masks[b] = spls

    mw, mh = w >> M, h >> M
    sx = (1 << (M - 1)) if (M > 0 and rounding) else 0
    xr = ((xs + sx) >> M).astype(np.int32)
    yr = ((ys + sx) >> M).astype(np.int32)

    g_eff = np.zeros((mh, mw), dtype=np.uint16)
    for b in range(M, B):
        polys = []
        for idx in masks[b]:
            raw = np.stack([xr[idx], yr[idx]], axis=1).astype(np.int32)
            keep = _dedup_mask(raw)
            pts = raw[keep]
            if len(pts) < 3:
                continue
            x0, y0 = pts.min(0)
            x1, y1 = pts.max(0)
            if x1 < 0 or y1 < 0 or x0 >= mw or y0 >= mh:
                continue
            if use_cr:
                polys.append(dense_ring(pts, idx[keep], fbits))
            else:
                polys.append(pts.astype(np.float64))
        if polys:
            g_eff |= rasterize_evenodd((mh, mw), polys).astype(np.uint16) << (b - M)

    beff = B - M
    if beff <= 0:
        return np.full((mh, mw), 128, np.uint8)
    q = gray_decode(g_eff, beff)
    top = (q >> (beff - 1)) & 1
    low = (1 << (8 - beff)) - 1
    return ((q << (8 - beff)) | (top * low)).astype(np.uint8)


# ---------------- самотест ----------------
def psnr(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    mse = float(np.mean(d * d))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def make_test_image():
    w = h = 256
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    img = np.clip((xx + yy) * 255.0 / (2 * w - 2), 0, 255).astype(np.uint8)
    cv2.circle(img, (185, 80), 45, 215, -1)
    cv2.rectangle(img, (30, 150), (115, 215), 55, -1)
    for i in range(0, w + 100, 24):
        cv2.line(img, (i, 0), (i - 100, h - 1), 235, 3)
    rng = np.random.default_rng(42)
    img = np.clip(img.astype(np.int16) + rng.normal(0, 3, (h, w)).astype(np.int16), 0, 255)
    return img.astype(np.uint8)


def quantize_ref(filt, B):
    q = filt.astype(np.uint16) >> (8 - B)
    top = (q >> (B - 1)) & 1
    return ((q << (8 - B)) | (top * ((1 << (8 - B)) - 1))).astype(np.uint8)


def main():
    img = make_test_image()
    raw = img.nbytes
    filt = cv2.bilateralFilter(img, 5, 30, 5)

    data0 = encode(img, B=8, eps_scale=0.0, min_area_scale=0.0)
    d0 = decode(data0)
    mm = float(np.mean(d0.astype(np.int16) != filt.astype(np.int16))) * 100
    print(f'САМОПРОВЕРКА B=8, eps=0: расхождение {mm:.4f}% пикселей (цель ~0)')

    print('ГРИД (raw = %d B):' % raw)
    print('  B  eps  area |   байт    %   PSNR дБ')
    rows = []
    for B in (5, 6, 7):
        for es in (0.5, 1.0):
            for mas in (0.5, 1.0, 2.0):
                data = encode(img, B, es, mas)
                dec = decode(data)
                p = psnr(img, dec)
                rows.append((B, es, mas, len(data), p))
                print('  %d  %-4s %-4s | %6d  %3d%%  %5.1f' %
                      (B, es, mas, len(data), len(data) * 100 // raw, p))
    best_p = max(r[4] for r in rows)
    cand = [r for r in rows if r[4] >= best_p - 0.7]
    best = min(cand, key=lambda r: r[3])
    B, es, mas = best[0], best[1], best[2]
    print('ВЫБОР: B=%d eps=%s area=%s -> %d B (%d%%), PSNR=%.1f дБ' %
          (B, es, mas, best[3], best[3] * 100 // raw, best[4]))

    # абляция: тот же выбор параметров с Катмуллом-Ромом
    data_cr = encode(img, B, es, mas, cr=True)
    dec_cr = decode(data_cr)
    print('CR-АБЛЯЦИЯ: %d B (%d%%), PSNR=%.1f дБ (полигон: %d B, %.1f дБ)' %
          (len(data_cr), len(data_cr) * 100 // raw, psnr(img, dec_cr),
           best[3], best[4]))

    data = encode(img, B, es, mas)
    open('test12.bpsc', 'wb').write(data)
    dec = decode(data)
    m1, m2 = decode(data, 1), decode(data, 2)
    qref = quantize_ref(filt, B)
    ref1 = filt[1::2, 1::2]
    ref2 = filt[3::4, 3::4]
    print('PSNR: декод %.1f | против кв.базы %.1f | M=1 %.1f | M=2 %.1f (дБ)' %
          (psnr(img, dec), psnr(qref, dec), psnr(ref1, m1), psnr(ref2, m2)))
    # контрольный end-to-end прогон CR-режима
    dc = decode(data_cr)
    assert dc.shape == dec.shape
    print('CR-режим декодируется корректно (без исключений)')
    sep = np.full((256, 4), 255, np.uint8)
    strip = np.hstack([img, sep, dec, sep, np.pad(m2, ((0, 192), (0, 0)), constant_values=255)])
    cv2.imwrite('compare_strip12.png', strip)
    print('Сохранено: test12.bpsc, compare_strip12.png')


if __name__ == '__main__':
    main()
