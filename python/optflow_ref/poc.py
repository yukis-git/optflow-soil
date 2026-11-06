"""Coarse displacement on a grid by windowed, band-limited phase-only correlation (POC)."""
from dataclasses import dataclass

import numpy as np

from .interp import gauss_blur


@dataclass
class Grid:
    xs: np.ndarray  # (nx,) pixel x of grid columns
    ys: np.ndarray  # (ny,) pixel y of grid rows
    ux: np.ndarray  # (ny, nx)
    uy: np.ndarray
    peak: np.ndarray  # normalised POC peak height, 0 = invalid


def _hann(n):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


_CACHE = {}


def _tables(shape):
    if shape not in _CACHE:
        fy = np.abs(np.fft.fftfreq(shape[0]))[:, None]
        fx = np.abs(np.fft.fftfreq(shape[1]))[None, :]
        lp = ((fy <= 0.25) & (fx <= 0.25)).astype(np.float64)
        _CACHE[shape] = (np.outer(_hann(shape[0]), _hann(shape[1])), lp, lp.mean())
    return _CACHE[shape]


def _parabola(m, c, p):
    den = m - 2.0 * c + p
    return 0.0 if den == 0 else 0.5 * (m - p) / den


def poc_shift(a, b):
    """Shift d with b(x) ~ a(x - d) for equally shaped (possibly rectangular) windows.
    Returns (dx, dy, peak) with |dx| < width/2, |dy| < height/2."""
    h, w = a.shape
    wnd, lp, lpm = _tables((h, w))
    A = np.fft.fft2((a - a.mean()) * wnd)
    B = np.fft.fft2((b - b.mean()) * wnd)
    C = B * np.conj(A)
    C /= np.abs(C) + 1e-12
    r = np.fft.fftshift(np.real(np.fft.ifft2(C * lp))) / lpm
    iy, ix = np.unravel_index(np.argmax(r), r.shape)
    iy = min(max(iy, 1), h - 2)
    ix = min(max(ix, 1), w - 2)
    sx = _parabola(r[iy, ix - 1], r[iy, ix], r[iy, ix + 1])
    sy = _parabola(r[iy - 1, ix], r[iy, ix], r[iy + 1, ix])
    return ix - w // 2 + sx, iy - h // 2 + sy, float(r[iy, ix])


def _win2(win):
    return (win, win) if np.isscalar(win) else (int(win[0]), int(win[1]))


def grid_points(h, w, step, win):
    """Grid columns/rows whose window (int or (height, width)) lies inside an h x w image."""
    wy, wx = _win2(win)
    return np.arange(wx // 2, w - wx // 2 + 1, step), np.arange(wy // 2, h - wy // 2 + 1, step)


def poc_grid(I1, I2, xs, ys, win, init=None, iters=3) -> Grid:
    """POC at each grid point; win is int or (height, width); init=(ux, uy) of shape (ny, nx) or None."""
    H, W = I1.shape
    wy, wx = _win2(win)
    hy, hx = wy // 2, wx // 2
    ny, nx = len(ys), len(xs)
    ux = np.zeros((ny, nx))
    uy = np.zeros((ny, nx))
    peak = np.zeros((ny, nx))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            dx, dy = (0.0, 0.0) if init is None else (init[0][i, j], init[1][i, j])
            a = I1[y - hy:y - hy + wy, x - hx:x - hx + wx]
            pk = 0.0
            for _ in range(iters):
                ox, oy = int(round(dx)), int(round(dy))
                y0, x0 = y - hy + oy, x - hx + ox
                if y0 < 0 or x0 < 0 or y0 + wy > H or x0 + wx > W:
                    pk = 0.0
                    break
                sx, sy, pk = poc_shift(a, I2[y0:y0 + wy, x0:x0 + wx])
                dx, dy = ox + sx, oy + sy
                if abs(sx) < 0.5 and abs(sy) < 0.5:
                    break
            ux[i, j], uy[i, j], peak[i, j] = dx, dy, pk
    return Grid(xs, ys, ux, uy, peak)


def clean(g: Grid, peak_min, thresh, max_pass=100, fallback=None) -> Grid:
    """Mark weak peaks and normalised-median outliers invalid, then fill from valid neighbours.
    When no grid point is valid there is nothing to fill from: the result is `fallback` = (ux, uy)
    of the grid's shape (the level's initial guess), or zeros without one."""
    ux, uy = g.ux.copy(), g.uy.copy()
    valid = g.peak >= peak_min
    ny, nx = ux.shape
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def neighbours(i, j, ok):
        return [(i + di, j + dj) for di, dj in offs
                if 0 <= i + di < ny and 0 <= j + dj < nx and ok[i + di, j + dj]]

    out = valid.copy()
    for i in range(ny):
        for j in range(nx):
            if not valid[i, j]:
                continue
            nb = neighbours(i, j, valid)
            if len(nb) < 3:
                continue
            for comp in (ux, uy):
                vals = np.array([comp[p] for p in nb])
                med = np.median(vals)
                res = np.median(np.abs(vals - med))
                if abs(comp[i, j] - med) / (res + 0.1) > thresh:
                    out[i, j] = False
    if not out.any():
        fx, fy = (np.zeros_like(ux), np.zeros_like(uy)) if fallback is None else (
            np.array(fallback[0], dtype=np.float64), np.array(fallback[1], dtype=np.float64))
        return Grid(g.xs, g.ys, fx, fy, g.peak * 0)
    for _ in range(max_pass):
        if out.all():
            break
        new = out.copy()
        for i in range(ny):
            for j in range(nx):
                if out[i, j]:
                    continue
                nb = neighbours(i, j, out)
                if nb:
                    ux[i, j] = np.median([ux[p] for p in nb])
                    uy[i, j] = np.median([uy[p] for p in nb])
                    new[i, j] = True
        out = new
    return Grid(g.xs, g.ys, ux, uy, np.where(valid, g.peak, 0.0))


def grid_sample(g: Grid, xq, yq):
    """Bilinear interpolation of the grid field at pixel coordinates (clamped to the grid)."""
    def frac(q, nodes):
        if len(nodes) == 1:
            return np.zeros(np.shape(q)), np.zeros(np.shape(q), dtype=np.int64)
        f = np.clip((np.asarray(q, dtype=np.float64) - nodes[0]) / (nodes[1] - nodes[0]), 0, len(nodes) - 1)
        i0 = np.minimum(np.floor(f).astype(np.int64), len(nodes) - 2)
        return f - i0, i0

    tx, x0 = frac(xq, g.xs)
    ty, y0 = frac(yq, g.ys)
    x1 = np.minimum(x0 + 1, len(g.xs) - 1)
    y1 = np.minimum(y0 + 1, len(g.ys) - 1)

    def bil(a):
        return ((1 - ty) * ((1 - tx) * a[y0, x0] + tx * a[y0, x1])
                + ty * ((1 - tx) * a[y1, x0] + tx * a[y1, x1]))

    return bil(g.ux), bil(g.uy)


def downsample(img):
    return gauss_blur(img, 1.0)[::2, ::2]


def _pow2_floor(n):
    return 1 << (int(n).bit_length() - 1) if n >= 1 else 0


def level_window(shape, win):
    """POC window (height, width) for an image of this shape. Along each axis the window may use
    at most 2/3 of the image (power of two), leaving room to shift. Square win x win when it fits;
    otherwise the short side gets what fits and the long side is widened to keep about win^2
    pixels. None if a side would be shorter than 16 px."""
    h, w = shape
    lim_h, lim_w = _pow2_floor(2 * h // 3), _pow2_floor(2 * w // 3)
    if win <= lim_h and win <= lim_w:
        return win, win
    if h <= w:
        wy = min(win, lim_h)
        wx = min(lim_w, max(win, win * win // max(wy, 1)))
    else:
        wx = min(win, lim_w)
        wy = min(lim_h, max(win, win * win // max(wx, 1)))
    return (wy, wx) if min(wy, wx) >= 16 else None


def n_levels(max_disp, win, shape):
    """Pyramid levels so that max_disp / 2**L <= win / 4, while some window still fits."""
    L = int(max(0, np.ceil(np.log2(max(max_disp, 1e-9) / (win / 4)))))
    while L > 0 and level_window((shape[0] >> L, shape[1] >> L), win) is None:
        L -= 1
    return L


def capture_range(max_disp, win, shape):
    """The displacement the coarse pyramid is designed to capture along each axis, (cy, cx) [px] as floats:
    (window / 4) * 2**L with L = n_levels(max_disp, win, shape) and the window coarse_flow uses at level L,
    which is win at L = 0 and otherwise level_window of the downsampled level-L image (ceil(n / 2) per level,
    not shape >> L). level_window shrinks that window on a small or narrow image, so the range can be
    less than max_disp, which n_levels sizes from the nominal win; the coarse stage then guarantees
    only this much."""
    L = n_levels(max_disp, win, shape)
    if L == 0:
        wy, wx = win, win
    else:
        h, w = shape
        for _ in range(L):
            h, w = (h + 1) // 2, (w + 1) // 2
        wy, wx = level_window((h, w), win)
    return wy / 4 * 2**L, wx / 4 * 2**L


def coarse_flow(I1, I2, cfg, init_dense=None) -> Grid:
    """Grid displacement at full resolution. init_dense=(ux, uy) is a dense guess (e.g. the previous
    frame); the full pyramid then searches for the correction relative to it."""
    H, W = I1.shape
    xs, ys = grid_points(H, W, cfg.grid_step, cfg.poc_win)
    L = n_levels(cfg.max_disp, cfg.poc_win, I1.shape)
    pyr = [(I1, I2)]
    for _ in range(L):
        pyr.append((downsample(pyr[-1][0]), downsample(pyr[-1][1])))
    g = None
    for lev in range(L, -1, -1):
        a, b = pyr[lev]
        wl = cfg.poc_win if lev == 0 else level_window(a.shape, cfg.poc_win)
        gx, gy = (xs, ys) if lev == 0 else grid_points(a.shape[0], a.shape[1], cfg.grid_step, wl)
        Y, X = np.meshgrid(gy, gx, indexing="ij")
        init = None
        if g is not None:  # g is on level lev+1: halve the coordinates, double the displacement
            px, py = grid_sample(g, X / 2.0, Y / 2.0)
            init = (2.0 * px, 2.0 * py)
        elif init_dense is not None:  # coarsest level starts from the guess, scaled to this level
            sc = 2**lev
            init = (init_dense[0][Y * sc, X * sc] / sc, init_dense[1][Y * sc, X * sc] / sc)
        g = clean(poc_grid(a, b, gx, gy, wl, init=init), cfg.poc_peak_min, cfg.median_thresh, fallback=init)
    return g
