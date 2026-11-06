"""Strain from a displacement field by local quadratic least-squares fits (NaN-aware Savitzky-Golay)."""
import numpy as np
from scipy.ndimage import correlate1d
from scipy.signal import fftconvolve

# quadratic basis 1, x, y, x^2, xy, y^2 as (power of x, power of y)
_BASIS = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]


def _moment(a, px, py, t):
    """sum over the window of a(p + t) * tx^px * ty^py (zero outside the image)."""
    out = correlate1d(a, t**py, axis=0, mode="constant", cval=0.0)
    return correlate1d(out, t**px, axis=1, mode="constant", cval=0.0)


# integer direction vectors of the 8 half windows (the same as phaseflow._DIRS): h = 0..7 is n_h = (cos, sin)(45 deg h)
_DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def _half_kernel(win, h):
    """Uniform window weights [row = ty, col = tx] times H(n_h . t): 1 on the side n_h points to, 1/2 on the line."""
    r = win // 2
    t = np.arange(-r, r + 1, dtype=np.float64)
    tx, ty = np.meshgrid(t, t)
    s = _DIRS[h][0] * tx + _DIRS[h][1] * ty
    return np.where(s > 0, 1.0, np.where(s == 0, 0.5, 0.0)), t


def _fit_half(ux, uy, win, min_frac, h):
    """Like the full-window fit below, but over the half window h only (min_frac of half the window area)."""
    W, t = _half_kernel(win, h)
    valid = np.isfinite(ux) & np.isfinite(uy)
    m = valid.astype(np.float64)
    u0 = np.where(valid, ux, 0.0)
    v0 = np.where(valid, uy, 0.0)

    def mom(a, px, py):
        K = W * np.outer(t**py, t**px)  # sum_t K(t) a(p + t): a correlation, zero outside the image
        return fftconvolve(a, K[::-1, ::-1], mode="same")

    M = np.empty(ux.shape + (6, 6))
    cache = {}
    for i, (ai, bi) in enumerate(_BASIS):
        for j, (aj, bj) in enumerate(_BASIS):
            key = (ai + aj, bi + bj)
            if key not in cache:
                cache[key] = mom(m, key[0], key[1])
            M[..., i, j] = cache[key]
    rhs = np.empty(ux.shape + (6, 2))
    for i, (ai, bi) in enumerate(_BASIS):
        rhs[..., i, 0] = mom(u0, ai, bi)
        rhs[..., i, 1] = mom(v0, ai, bi)
    ok = M[..., 0, 0] >= min_frac * 0.5 * win * win
    M[~ok] = np.eye(6)
    rhs[~ok] = 0.0
    coef = np.linalg.solve(M, rhs)
    nan = np.where(ok, 1.0, np.nan)
    return coef[..., 1, 0] * nan, coef[..., 2, 0] * nan, coef[..., 1, 1] * nan, coef[..., 2, 1] * nan


def displacement_gradient(ux, uy, win, min_frac=0.5, half=None):
    """Returns (dudx, dudy, dvdx, dvdy); NaN where fewer than min_frac of the window is valid.

    half: optional int array like ux, -1 = the full window (default behaviour), 0..7 = fit that pixel over the half
    window h only (spec 2.4: the side of a crack the pixel lies on); min_frac then refers to half the window area."""
    if half is not None:
        out = list(displacement_gradient(ux, uy, win, min_frac))
        for h in np.unique(half[half >= 0]):
            sel = half == h
            for o, g in zip(out, _fit_half(ux, uy, win, min_frac, int(h))):
                o[sel] = g[sel]
        return tuple(out)
    h = win // 2
    t = np.arange(-h, h + 1, dtype=np.float64)
    valid = np.isfinite(ux) & np.isfinite(uy)
    m = valid.astype(np.float64)
    u0 = np.where(valid, ux, 0.0)
    v0 = np.where(valid, uy, 0.0)
    M = np.empty(ux.shape + (6, 6))
    cache = {}
    for i, (ai, bi) in enumerate(_BASIS):
        for j, (aj, bj) in enumerate(_BASIS):
            key = (ai + aj, bi + bj)
            if key not in cache:
                cache[key] = _moment(m, key[0], key[1], t)
            M[..., i, j] = cache[key]
    rhs = np.empty(ux.shape + (6, 2))
    for i, (ai, bi) in enumerate(_BASIS):
        rhs[..., i, 0] = _moment(u0, ai, bi, t)
        rhs[..., i, 1] = _moment(v0, ai, bi, t)
    count = M[..., 0, 0]
    ok = count >= min_frac * win * win
    M[~ok] = np.eye(6)
    rhs[~ok] = 0.0
    coef = np.linalg.solve(M, rhs)  # (..., 6, 2)
    nan = np.where(ok, 1.0, np.nan)
    return coef[..., 1, 0] * nan, coef[..., 2, 0] * nan, coef[..., 1, 1] * nan, coef[..., 2, 1] * nan


def strain_fields(dudx, dudy, dvdx, dvdy):
    """Green-Lagrange E, small strain eps, max shear gamma (from E) and volumetric det F - 1."""
    exx = dudx + 0.5 * (dudx**2 + dvdx**2)
    eyy = dvdy + 0.5 * (dudy**2 + dvdy**2)
    exy = 0.5 * (dudy + dvdx) + 0.5 * (dudx * dudy + dvdx * dvdy)
    return {
        "Exx": exx,
        "Eyy": eyy,
        "Exy": exy,
        "exx": dudx,
        "eyy": dvdy,
        "exy": 0.5 * (dudy + dvdx),
        "gmax": 2.0 * np.sqrt(((exx - eyy) / 2) ** 2 + exy**2),
        "vol": (1 + dudx) * (1 + dvdy) - dudy * dvdx - 1.0,
    }
