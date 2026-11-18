#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BPSC 1.1 - референсный прототип, версия 16: энкодер + декодер + самотест.
Запуск:  python prototype16.py

Новое в 16 относительно 15/15_oldfilter (формат файла и декодер НЕ менялись,
файлы v15 остаются валидными):
  - двухступенчатый выбор фильтра: на двух якорных точках грида гоняются
    оба фильтра (surface / bilateral), по Парето-эвристике выбирается
    победитель, полный грид 18 кандидатов крутится только по нему
    (~22 кодирования вместо 36);
  - USE_CR добавлен в грид как кандидат (только при eps>0): формат и
    декодер уже умели кривые, теперь энкодер их пробует;
  - абляция расписания eps: на выбранном кандидате прогоняются
    exp/const расписания и репортятся отдельным блоком (в выбор RDO
    сознательно не включены - не смешиваем изменения);
  - mip-PSNR репортится для шорт-листа RDO (в пределах 0.7 дБ),
    в критерий выбора пока не входит - только наблюдение для v17;
  - noise_adaptive_filter принимает kind: surface (v15, сила от sigma)
    или bilateral (v14: d=5, sigma_color=30, sigma_space=5).

Оставлено без изменений относительно 15:
  - честный бюджет зерна, краевая нормировка, оценка шума до фильтрации,
    формат, PRNG, even-odd, Катмулл-Ром, пер-изображённый грид,
    sigma/2^M на мипах.

Сознательные упрощения против спеки - как в 15.
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

FILTERS = ('surface', 'bilateral')
EPS_SCHEDS = ('linear', 'exp', 'const')
RDO_PSNR_MARGIN = 0.7


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


# ---------------- шумовой слой: SplitMix64 + Box-Muller ----------------
def _sm64(z):
    with np.errstate(over='ignore'):
        z = z + np.uint64(0x9E3779B97F4A7C15)
        k = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        k = (k ^ (k >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return k ^ (k >> np.uint64(31))


def gauss_noise_field(seed, h, w, mean, sigma):
    """Детерминированное поле гауссовского шума, int16, одинаковый поток везде."""
    n = h * w
    if sigma <= 0.0:
        return np.zeros((h, w), np.int16)
    m2 = (n + 1) // 2 * 2
    with np.errstate(over='ignore'):
        u = (_sm64(np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
                   + np.arange(m2, dtype=np.uint64)) >> np.uint64(11)).astype(np.float64)
    u *= 1.0 / 9007199254740992.0  # [0, 1)
    u1 = np.maximum(u[0::2], 1e-300)
    u2 = u[1::2]
    r = np.sqrt(-2.0 * np.log(u1))
    th = 2.0 * np.pi * u2
    nrm = np.concatenate([r * np.cos(th), r * np.sin(th)])[:n]
    vals = np.floor(mean + sigma * nrm + 0.5)
    return vals.astype(np.int16).reshape(h, w)


def write_noise_layers(payload, layers):
    """layers: список (gen_id, sigma, mean, seed)."""
    for gen, sigma, mean, seed in layers:
        payload += struct.pack(
            NOISE_REC_FMT, gen,
            max(0, min(65535, int(round(sigma * NOISE_SIG_SCALE)))),
            max(-32768, min(32767, int(round(mean * NOISE_MEAN_SCALE)))),
            seed & 0xFFFFFFFF, 0)
    return payload


# ---------------- оценка шума и фильтрация (v16) ----------------
def edge_mask(img, dilate_iter=2, lo=60, hi=120):
    """Краевая маска: Canny + дилатация. True - пиксель близок к краю."""
    e = cv2.Canny(img, lo, hi)
    e = cv2.dilate(e, np.ones((3, 3), np.uint8), iterations=dilate_iter)
    return e > 0


def estimate_noise(img, sigma_scale=1.2):
    """Robust-оценка шума по оригиналу ДО фильтрации: MAD ВЧ вне краевой маски.
    Возвращает (sigma, median_hf)."""
    hf = img.astype(np.float64) - cv2.GaussianBlur(img, (0, 0), sigma_scale).astype(np.float64)
    valid = ~edge_mask(img)
    vals = hf[valid]
    if vals.size == 0:
        vals = hf.ravel()
    med = float(np.median(vals))
    sigma = 1.4826 * float(np.median(np.abs(vals - med)))
    return sigma, med


def surface_blur(img, d=5, t=25):
    """Surface blur: среднее по пикселям окна d x d, отличающимся от центра
    не более чем на t. Векторизовано через сдвиги окна."""
    h, w = img.shape
    pad = d // 2
    src = img.astype(np.float64)
    p = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE).astype(np.float64)
    acc = np.zeros_like(src)
    wgt = np.zeros_like(src)
    for dy in range(d):
        for dx in range(d):
            shift = p[dy:dy + h, dx:dx + w]
            ok = np.abs(shift - src) <= t
            acc += np.where(ok, shift, 0.0)
            wgt += ok
    out = np.where(wgt > 0, acc / np.maximum(wgt, 1.0), src)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def noise_adaptive_filter(img, sigma_est, kind='surface'):
    """Фильтр базы. surface - как в v15 (сила от снятого sigma);
    bilateral - как в v14 (d=5, sigma_color=30, sigma_space=5)."""
    if kind == 'bilateral':
        return cv2.bilateralFilter(img, 5, 30, 5)
    t = int(np.clip(4.0 * sigma_est, 10, 60))
    d = 7 if sigma_est >= 4.0 else 5
    return surface_blur(img, d, t)


def eps_for(b, eps_scale, sched):
    """Расписание eps по плоскостям (v16, абляция)."""
    if eps_scale == 0:
        return 0
    if sched == 'exp':
        return max(0.4, eps_scale * (2 ** (7 - b)) / 4.0)
    if sched == 'const':
        return eps_scale
    return max(0.4, eps_scale * (7 - b))  # linear - как в 13..15


# ---------------- энкодер ----------------
def encode(img, B=6, eps_scale=1.0, min_area_scale=1.0, cr=False, noise='auto',
           auto_sigma_thr=0.5, filter_kind='surface', eps_sched='linear'):
    h, w = img.shape
    # v15: сначала снимаем шум, потом фильтруем
    sigma_est, _ = estimate_noise(img)
    base = noise_adaptive_filter(img, sigma_est, filter_kind)
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
        eps = eps_for(b, eps_scale, eps_sched)
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
    if noise is True or noise == 'auto':
        crc0 = zlib.crc32(payload) & 0xFFFFFFFF
        tmp = struct.pack(HDR_FMT, MAGIC, VERSION, FLAG_USE_CR if cr else 0,
                          w, h, nv, 0, 0, 0, 0, B, len(payload), crc0, 0) + payload
        recon = decode(tmp)
        # v15: честный бюджет - дефицит зерна по краевой нормированной метрике
        go = grain_edge(img)
        gr = grain_edge(recon)
        sigma_eff = float(np.sqrt(max(0.0, go * go - gr * gr)))
        if noise is True:
            # форс: MAD остатка вне краевой маски
            r = img.astype(np.float64) - recon.astype(np.float64)
            valid = ~edge_mask(img)
            vals = r[valid] if valid.any() else r.ravel()
            med = float(np.median(vals))
            sigma_layer = 1.4826 * float(np.median(np.abs(vals - med)))
        else:
            enable = sigma_eff >= auto_sigma_thr
            if not enable:
                crc = zlib.crc32(payload) & 0xFFFFFFFF
                hdr = struct.pack(HDR_FMT, MAGIC, VERSION, FLAG_USE_CR if cr else 0,
                                  w, h, nv, 0, 0, 0, 0, B, len(payload), crc, 0)
                return hdr + payload
            sigma_layer = sigma_eff
            r = img.astype(np.float64) - recon.astype(np.float64)
            valid = ~edge_mask(img)
            med = float(np.median(r[valid])) if valid.any() else float(np.median(r))
        seed = zlib.crc32(payload) & 0xFFFFFFFF
        payload = write_noise_layers(payload, [(NOISE_GAUSS, sigma_layer, med, seed)])
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

    # шумовые слои: на мипе sigma масштабируется как sigma / 2^M.
    for _ in range(nnl):
        gen, sig_q, mean_q, seed, _nf = struct.unpack_from(NOISE_REC_FMT, payload, off)
        off += struct.calcsize(NOISE_REC_FMT)
        if gen == NOISE_GAUSS:
            sig = (sig_q / NOISE_SIG_SCALE) / float(1 << M)
            fld = gauss_noise_field(seed, mh, mw, mean_q / NOISE_MEAN_SCALE, sig)
            base = np.clip(base + fld, 0, 255)
    return base.astype(np.uint8)


# ---------------- RDO: двухступенчатый выбор фильтра ----------------
def choose_filter(img, filter_kind, es, mas, psnrs, sizes):
    """Мини-RDO по пробам одного фильтра: лучший PSNR, среди кандидатов
    в его пределах - минимальный размер. Возвращает (size, psnr)."""
    bp = max(psnrs)
    cand = [i for i in range(len(psnrs)) if psnrs[i] >= bp - RDO_PSNR_MARGIN]
    best = min(cand, key=lambda i: sizes[i])
    return sizes[best], psnrs[best]


def filter_verdict(pa, sa, pb, sb):
    """Парето-эвристика по пробам: победа A, победа B или ничья -> по сумме
    нормализованных преимуществ (psnr vs size). Возвращает имя фильтра."""
    # выигрыш PSNR фильтра A при большем размере B -> цена дБ за % размера
    def gain(p_hi, p_lo):
        return max(0.0, p_hi - p_lo)
    if sa <= sb and pa >= pb:
        return 'surface'
    if sb <= sa and pb >= pa:
        return 'bilateral'
    dp = pa - pb
    ds = (sa - sb) / max(sa, sb)
    # 0.3 дБ за 10% размера - точка безразличия; дальше решает нормировка
    score_a = dp / 0.3 + (-ds) / 0.10
    score_b = -dp / 0.3 + ds / 0.10
    return 'surface' if score_a >= score_b else 'bilateral'


# ---------------- самотест ----------------
def psnr(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    mse = float(np.mean(d * d))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def grain_edge(x):
    """Мера зерна (v15): std ВЧ вне краевой маски - зубцы контуров не считаются."""
    x = x.astype(np.float64)
    hf = x - cv2.GaussianBlur(x, (0, 0), 1.2)
    valid = ~edge_mask(x.astype(np.uint8))
    vals = hf[valid]
    if vals.size == 0:
        vals = hf.ravel()
    return float(np.std(vals))


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


def make_clean_image():
    """Та же сцена, что в make_test_image, но без шума."""
    w = h = 256
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    img = np.clip((xx + yy) * 255.0 / (2 * w - 2), 0, 255).astype(np.uint8)
    cv2.circle(img, (185, 80), 45, 215, -1)
    cv2.rectangle(img, (30, 150), (115, 215), 55, -1)
    for i in range(0, w + 100, 24):
        cv2.line(img, (i, 0), (i - 100, h - 1), 235, 3)
    return img


def make_grain_image():
    """Гладкая база + чистое зерно sigma=6: тест честного бюджета зерна."""
    w = h = 256
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    base = np.clip(80 + 60 * (xx + yy) / (2 * w - 2), 0, 255)
    rng = np.random.default_rng(7)
    img = base + rng.normal(0, 6, (h, w))
    return np.clip(img, 0, 255).astype(np.uint8)


def quantize_ref(filt, B):
    q = filt.astype(np.uint16) >> (8 - B)
    top = (q >> (B - 1)) & 1
    return ((q << (8 - B)) | (top * ((1 << (8 - B)) - 1))).astype(np.uint8)


# ---------------- пер-изображённый грид (v16) ----------------
def run_image(name, img, fname):
    """Ступень 1: пробы обоих фильтров на двух якорях -> выбор фильтра.
    Ступень 3: полный грид (B, eps, area, cr) на победителе, RDO,
    абляция eps-расписаний на выбранном кандидате, mip-репорт шорт-листа."""
    raw = img.nbytes
    print('=' * 62)
    print('КАРТИНКА: %s (raw = %d B)' % (name, raw))
    sig_est, _ = estimate_noise(img)
    print('ОЦЕНКА ШУМА ДО ФИЛЬТРАЦИИ: sigma=%.2f' % sig_est)

    # ступень 1: якоря (дешёвая и средняя точки), оба фильтра, без кривых
    anchors = [(5, 1.0, 2.0), (6, 1.0, 1.0)]
    probes = {}
    for kind in FILTERS:
        ps, ss = [], []
        for (B, es, mas) in anchors:
            data = encode(img, B, es, mas, noise='auto', filter_kind=kind)
            ps.append(psnr(img, decode(data)))
            ss.append(len(data))
        probes[kind] = (ps, ss)
        print('ПРОБА %-9s: %s' % (kind,
              ' | '.join('%d B %.2f дБ' % (s, p) for s, p in zip(ss, ps))))
    winner = filter_verdict(probes['surface'][0], probes['surface'][1],
                            probes['bilateral'][0], probes['bilateral'][1])
    print('ФИЛЬТР-ПОБЕДИТЕЛЬ: %s' % winner)

    # ступень 3: полный грид на победителе
    rows = []
    for B in (5, 6, 7):
        for es in (0.5, 1.0):
            for mas in (0.5, 1.0, 2.0):
                for cr in (False, True):
                    data = encode(img, B, es, mas, cr=cr, noise='auto',
                                  filter_kind=winner)
                    dec = decode(data)
                    rows.append((B, es, mas, cr, len(data), psnr(img, dec), dec, data))
    best_p = max(r[5] for r in rows)
    cand = [r for r in rows if r[5] >= best_p - RDO_PSNR_MARGIN]
    best = min(cand, key=lambda r: r[4])
    B, es, mas, cr, size, p, dec, data = best
    data_plain = encode(img, B, es, mas, cr=cr, noise=False, filter_kind=winner)
    layer_on = len(data_plain) != size
    print('  B  eps  area cr |   байт    %   PSNR дБ')
    for row in rows:
        print('  %d  %-4s %-4s %-2d | %6d  %3d%%  %5.1f' %
              (row[0], row[1], row[2], row[3], row[4], row[4] * 100 // raw, row[5]))
    print('ВЫБОР: B=%d eps=%s area=%s cr=%s -> %d B (%d%%), PSNR=%.2f дБ' %
          (B, es, mas, cr, size, size * 100 // raw, p))
    print('АВТОРЕЖИМ (честный бюджет): зерно_edge ориг %.2f / рекон %.2f -> слой %s'
          % (grain_edge(img), grain_edge(dec), 'ВКЛ' if layer_on else 'ВЫКЛ'))

    # абляция расписаний eps на выбранном кандидате (репорт, не выбор)
    print('АБЛЯЦИЯ eps-расписаний (на выбранном кандидате):')
    for sched in EPS_SCHEDS:
        if sched == 'linear':
            s2, p2 = size, p
        else:
            d2 = encode(img, B, es, mas, cr=cr, noise='auto',
                        filter_kind=winner, eps_sched=sched)
            s2, p2 = len(d2), psnr(img, decode(d2))
        print('  %-6s | %6d B (%3d%%)  %5.2f дБ' % (sched, s2, s2 * 100 // raw, p2))

    # mip-репорт шорт-листа (наблюдение для v17, в критерий не входит)
    print('MIP-РЕПОРТ шорт-листа (PSNR M=1 / M=2, дБ):')
    for row in cand:
        m1, m2 = decode(row[7], 1), decode(row[7], 2)
        print('  B=%d eps=%s area=%s cr=%d: %.2f / %.2f' %
              (row[0], row[1], row[2], row[3],
               psnr(img[1::2, 1::2], m1), psnr(img[3::4, 3::4], m2)))

    assert encode(img, B, es, mas, cr=cr, noise='auto', filter_kind=winner) == data, \
        'энкодер недетерминирован'
    assert np.array_equal(dec, decode(data)), 'декодер недетерминирован'
    open(fname, 'wb').write(data)
    m1, m2 = decode(data, 1), decode(data, 2)
    print('PSNR: декод %.2f | M=1 %.2f | M=2 %.2f (дБ)' %
          (p, psnr(img[1::2, 1::2], m1), psnr(img[3::4, 3::4], m2)))
    return winner, B, es, mas, cr, size, p


def main():
    f1 = gauss_noise_field(123, 64, 64, 0.0, 3.0)
    f2 = gauss_noise_field(123, 64, 64, 0.0, 3.0)
    assert np.array_equal(f1, f2), 'PRNG недетерминирован'
    assert not np.array_equal(f1, gauss_noise_field(124, 64, 64, 0.0, 3.0))
    print('PRNG: детерминизм OK, std поля %.2f (цель ~3)' % f1.std())

    # самопроверка B=8, eps=0 для обоих фильтров
    img0 = make_test_image()
    sig0 = estimate_noise(img0)[0]
    for kind in FILTERS:
        base0 = noise_adaptive_filter(img0, sig0, kind)
        data0 = encode(img0, B=8, eps_scale=0.0, min_area_scale=0.0, filter_kind=kind)
        d0 = decode(data0)
        mm = float(np.mean(d0.astype(np.int16) != quantize_ref(base0, 8).astype(np.int16))) * 100
        print('САМОПРОВЕРКА B=8, eps=0, %s: расхождение %.4f%% пикселей (цель ~0)'
              % (kind, mm))

    run_image('стандарт (шум sigma=3)', make_test_image(), 'test16.bpsc')
    run_image('чистая (без шума)', make_clean_image(), 'test16_clean.bpsc')
    run_image('зерно (sigma=6)', make_grain_image(), 'test16_grain.bpsc')
    print('Сохранено: test16*.bpsc')


if __name__ == '__main__':
    main()
