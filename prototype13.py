#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BPSC 1.1 - референсный прототип, версия 13: энкодер + декодер + самотест.
Запуск:  python prototype13.py

Новое в 13 относительно prototype12:
  - первый шумовой слой: генератор Gaussian (gen_id = 0);
  - контейнер шумового слоя, 12 байт, формат '<BHiIB':
      B  gen_id    (0 = Gaussian)
      H  sigma_q   (std * 64)
      i  mean_q    (mean * 64)
      I  seed      (детерминированный PRNG)
      B  flags     (резерв, 0)
  - контейнеры пишутся после блоков сплайнов и входят в CRC;
    NumNoiseLayers в заголовке = число контейнеров;
  - принцип "любой заменяет любой": энкодер не подбирает форму шума
    структурно - слой описывается робастной оценкой остатка
    (медиана и MAD*1.4826); совпадение спектра/структуры не проверяется;
  - PRNG: SplitMix64 + Box-Muller на float64 - байтово одинаковый поток
    на любой платформе, зависит только от seed;
  - мипы: sigma масштабируется как sigma / 2^M (среднее по блоку 2^M x 2^M),
    mean сохраняется, seed тот же;
  - в самотест добавлены: детерминизм PRNG, детерминизм энкодера с шумом,
    абляция шумового слоя и метрика "зерна" (std высокочастотной составляющей).

Унаследовано от 12: VertexFlags (бит 0 = гладкая вершина), дуги Катмулла-Рома
на гладких рёбрах (порог min(85, 20+12*eps)), FLAG_USE_CR в байте flags,
расширенный счётчик длинных контуров (>8191), even-odd растеризация по рёбрам
пикселей (самопроверка 0.0000%), адаптивный эпсилон и RDO-отсев.

Сознательные упрощения против спеки:
  - один канал Y, ContentClass=Artistic, сжатие raw, без выравнивания 512 КиБ;
  - число плоскостей B - в резервном байте 0x13;
  - бит 0x01 байта flags занят под USE_CR (резидент спеки);
  - формат шумового контейнера - расширение прототипа;
  - шум применяется и на мипах (со масштабированным sigma);
  - при M >= B шум не применяется (фолбэк константой 128).
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

NOISE_GAUSS = 0
NOISE_REC_FMT = '<BHiIB'
assert struct.calcsize(NOISE_REC_FMT) == 12
NOISE_SIG_SCALE = 64
NOISE_MEAN_SCALE = 64


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
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _turn_angles(pts):
    pts = pts.astype(np.float64)
    v1 = pts - np.roll(pts, 1, 0)
    v2 = np.roll(pts, -1, 0) - pts
    cross = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    dot = v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1]
    return np.degrees(np.arctan2(cross, dot))


# ---------------- Катмулл-Ром ----------------
def _cr_arc(p_prev, p0, p1, p_next):
    d = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    n = max(2, min(64, int(d)))
    t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
    P0, P1, P2, P3 = (np.asarray(p, np.float64) for p in (p_prev, p0, p1, p_next))
    return 0.5 * (2 * P1 + (-P0 + P2) * t
                  + (2 * P0 - 5 * P1 + 4 * P2 - P3) * t ** 2
                  + (-P0 + 3 * P1 - 3 * P2 + P3) * t ** 3)


def dense_ring(pts, fidx=None, smooth=None):
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
    with np.errstate(over='ignore'):
        z = z + np.uint64(0x9E3779B97F4A7C15)
        k = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        k = (k ^ (k >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return k ^ (k >> np.uint64(31))


def gauss_noise_field(seed, h, w, mean, sigma):
    n = h * w
    if sigma <= 0.0:
        return np.zeros((h, w), np.int16)
    m2 = (n + 1) // 2 * 2
    with np.errstate(over='ignore'):
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


def write_noise_layers(payload, layers):
    for gen, sigma, mean, seed in layers:
        payload += struct.pack(
            NOISE_REC_FMT, gen,
            max(0, min(65535, int(round(sigma * NOISE_SIG_SCALE)))),
            max(-32768, min(32767, int(round(mean * NOISE_MEAN_SCALE)))),
            seed & 0xFFFFFFFF, 0)
    return payload


# ---------------- энкодер ----------------
def encode(img, B=6, eps_scale=1.0, min_area_scale=1.0, cr=False, noise=False):
    h, w = img.shape
    base = cv2.bilateralFilter(img, 5, 30, 5)
    q = base.astype(np.uint16) >> (8 - B)
    g = gray_encode(q)

    vmap, vlist = {}, []
    smooth = {}

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

    nnl = 0
    if noise:
        crc0 = zlib.crc32(payload) & 0xFFFFFFFF
        tmp = struct.pack(HDR_FMT, MAGIC, VERSION, FLAG_USE_CR if cr else 0,
                          w, h, nv, 0, 0, 0, 0, B, len(payload), crc0, 0) + payload
        recon = decode(tmp)
        r = img.astype(np.float64) - recon.astype(np.float64)
        mean = float(np.median(r))
        sigma = 1.4826 * float(np.median(np.abs(r - mean)))
        seed = zlib.crc32(payload) & 0xFFFFFFFF
        payload = write_noise_layers(payload, [(NOISE_GAUSS, sigma, mean, seed)])
        nnl = 1

    crc = zlib.crc32(payload) & 0xFFFFFFFF
    hdr = struct.pack(HDR_FMT, MAGIC, VERSION, FLAG_USE_CR if cr else 0, w, h, nv,
                      nnl, 0, 0, 0, B, len(payload), crc, 0)
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
    base = ((q << (8 - beff)) | (top * ((1 << (8 - beff)) - 1))).astype(np.int16)

    for _ in range(nnl):
        gen, sig_q, mean_q, seed, _nf = struct.unpack_from(NOISE_REC_FMT, payload, off)
        off += struct.calcsize(NOISE_REC_FMT)
        if gen == NOISE_GAUSS:
            sig = (sig_q / NOISE_SIG_SCALE) / float(1 << M)
            fld = gauss_noise_field(seed, mh, mw, mean_q / NOISE_MEAN_SCALE, sig)
            base = np.clip(base + fld, 0, 255)
    return base.astype(np.uint8)


# ---------------- самотест ----------------
def psnr(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    mse = float(np.mean(d * d))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def grain(x):
    return float(np.std(x.astype(np.float64) - cv2.GaussianBlur(x, (0, 0), 1.2).astype(np.float64)))


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

    f1 = gauss_noise_field(123, 64, 64, 0.0, 3.0)
    f2 = gauss_noise_field(123, 64, 64, 0.0, 3.0)
    assert np.array_equal(f1, f2), 'PRNG недетерминирован'
    assert not np.array_equal(f1, gauss_noise_field(124, 64, 64, 0.0, 3.0))
    print('PRNG: детерминизм OK, std поля %.2f (цель ~3)' % f1.std())

    data0 = encode(img, B=8, eps_scale=0.0, min_area_scale=0.0)
    d0 = decode(data0)
    mm = float(np.mean(d0.astype(np.int16) != filt.astype(np.int16))) * 100
    print('САМОПРОВЕРКА B=8, eps=0 (без шума): расхождение %.4f%% пикселей (цель ~0)' % mm)

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

    data_cr = encode(img, B, es, mas, cr=True)
    dec_cr = decode(data_cr)
    print('CR-АБЛЯЦИЯ: %d B (%d%%), PSNR=%.1f дБ (полигон: %d B, %.1f дБ)' %
          (len(data_cr), len(data_cr) * 100 // raw, psnr(img, dec_cr),
           best[3], best[4]))

    print('АБЛЯЦИЯ ШУМОВОГО СЛОЯ (B=%d, area=%s):' % (B, mas))
    for esn in (0.5, 1.0, 2.0):
        dp = encode(img, B, esn, mas)
        dn = encode(img, B, esn, mas, noise=True)
        print('  eps=%.1f | без слоя: %5d B (%2d%%) %5.1f дБ | со слоем Гаусса: %5d B (%2d%%) %5.1f дБ'
              % (esn, len(dp), len(dp) * 100 // raw, psnr(img, decode(dp)),
                 len(dn), len(dn) * 100 // raw, psnr(img, decode(dn))))

    data = encode(img, B, es, mas, noise=True)
    assert encode(img, B, es, mas, noise=True) == data, 'энкодер недетерминирован'
    print('ДЕТЕРМИНИЗМ ЭНКОДЕРА С ШУМОМ: OK (%d B)' % len(data))
    open('test13.bpsc', 'wb').write(data)
    dec = decode(data)
    dp2 = decode(data)
    assert np.array_equal(dec, dp2), 'декодер недетерминирован'
    m1, m2 = decode(data, 1), decode(data, 2)
    qref = quantize_ref(filt, B)
    ref1 = filt[1::2, 1::2]
    ref2 = filt[3::4, 3::4]
    print('ФИНАЛ (со слоем шума): %d B (%d%% от raw)' % (len(data), len(data) * 100 // raw))
    print('PSNR: декод %.1f | против кв.базы %.1f | M=1 %.1f | M=2 %.1f (дБ)' %
          (psnr(img, dec), psnr(qref, dec), psnr(ref1, m1), psnr(ref2, m2)))
    print('ЗЕРНО (std ВЧ): ориг %.2f | декод без слоя %.2f | декод со слоем %.2f'
          % (grain(img), grain(decode(encode(img, B, es, mas))), grain(dec)))
    sep = np.full((256, 4), 255, np.uint8)
    strip = np.hstack([img, sep, dec, sep, np.pad(m2, ((0, 192), (0, 0)), constant_values=255)])
    cv2.imwrite('compare_strip13.png', strip)
    print('Сохранено: test13.bpsc, compare_strip13.png')


if __name__ == '__main__':
    main()