"""Cubic B-spline interpolation and Gaussian blur.

Written with explicit formulas (no scipy interpolation) so the Rust port can match it exactly.
Images are indexed [row, col]; coordinates are (x=col, y=row).
"""
import numpy as np
from scipy.ndimage import correlate1d

_Z = np.sqrt(3.0) - 2.0
_HORIZON = int(np.ceil(np.log(1e-12) / np.log(abs(_Z))))


def _prefilter_axis0(c: np.ndarray) -> None:
    """In-place cubic B-spline prefilter along axis 0 with mirror boundary."""
    n = c.shape[0]
    if n == 1:
        return
    z = _Z
    c *= 6.0
    h = min(n, _HORIZON)
    zk = z ** np.arange(h)
    c[0] = np.tensordot(zk, c[:h], axes=(0, 0))
    for i in range(1, n):
        c[i] += z * c[i - 1]
    c[n - 1] = (z / (z * z - 1.0)) * (c[n - 1] + z * c[n - 2])
    for i in range(n - 2, -1, -1):
        c[i] = z * (c[i + 1] - c[i])


def bspline_coeffs(img: np.ndarray) -> np.ndarray:
    """Cubic B-spline coefficients of a 2-D real or complex array."""
    c = np.array(img, dtype=np.result_type(img, np.float64), copy=True)
    _prefilter_axis0(c)
    _prefilter_axis0(c.T)
    return c


def _mirror(i: np.ndarray, n: int) -> np.ndarray:
    if n == 1:
        return np.zeros_like(i)
    p = 2 * (n - 1)
    i = np.abs(i) % p
    return np.where(i >= n, p - i, i)


def _weights(t: np.ndarray):
    t2 = t * t
    t3 = t2 * t
    return ((1 - t) ** 3 / 6, (3 * t3 - 6 * t2 + 4) / 6, (-3 * t3 + 3 * t2 + 3 * t + 1) / 6, t3 / 6)


def sample(coeffs: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate the spline at (x, y). Coordinates are clamped to the array bounds."""
    h, w = coeffs.shape
    x = np.clip(x, 0.0, w - 1.0)
    y = np.clip(y, 0.0, h - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    wx = _weights(x - x0)
    wy = _weights(y - y0)
    xi = [_mirror(x0 + k - 1, w) for k in range(4)]
    out = np.zeros(np.shape(x), dtype=coeffs.dtype)
    for j in range(4):
        yi = _mirror(y0 + j - 1, h)
        acc = wx[0] * coeffs[yi, xi[0]]
        for k in range(1, 4):
            acc = acc + wx[k] * coeffs[yi, xi[k]]
        out += wy[j] * acc
    return out


def warp(img: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Return img(X + ux, Y + uy) on the pixel grid."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    return sample(bspline_coeffs(img), xx + ux, yy + uy)


def gauss_kernel(sigma: float) -> np.ndarray:
    r = int(4.0 * sigma + 0.5)
    t = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum()


def gauss_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur, radius round(4 sigma), edge values replicated."""
    k = gauss_kernel(sigma)
    if np.iscomplexobj(a):
        return gauss_blur(a.real, sigma) + 1j * gauss_blur(a.imag, sigma)
    out = correlate1d(np.asarray(a, dtype=np.float64), k, axis=0, mode="nearest")
    return correlate1d(out, k, axis=1, mode="nearest")
