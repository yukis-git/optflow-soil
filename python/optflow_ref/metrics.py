"""Error metrics against synthetic ground truth."""
import numpy as np

from .strain import strain_fields


def eval_mask(shape, field, margin):
    """Pixels >= margin from the border whose deformed position is also >= margin inside."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    x, y = (xx, yy) if field is None else field.fwd(xx, yy)
    m = (xx >= margin) & (xx <= W - 1 - margin) & (yy >= margin) & (yy <= H - 1 - margin)
    return m & (x >= margin) & (x <= W - 1 - margin) & (y >= margin) & (y <= H - 1 - margin)


def disp_errors(ux, uy, gx, gy, mask):
    ex, ey = ux - gx, uy - gy
    fin = mask & np.isfinite(ex) & np.isfinite(ey)
    if not fin.any():
        return {"rms": np.nan, "bias_x": np.nan, "bias_y": np.nan, "max": np.nan, "p99": np.nan, "valid_frac": 0.0}
    e = np.hypot(ex[fin], ey[fin])
    return {
        "rms": float(np.sqrt(np.mean(e**2))),
        "bias_x": float(ex[fin].mean()),
        "bias_y": float(ey[fin].mean()),
        "max": float(e.max()),
        "p99": float(np.percentile(e, 99)),
        "valid_frac": float(fin.sum() / mask.sum()),
    }


def true_strain(field, shape):
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    return strain_fields(*field.grad(xx, yy))


def strain_errors(est, truth, mask, keys=("Exx", "Eyy", "Exy", "gmax")):
    out = {}
    for k in keys:
        d = est[k] - truth[k]
        fin = mask & np.isfinite(d)
        out[k] = float(np.sqrt(np.mean(d[fin] ** 2))) if fin.any() else float("nan")
    return out


def crlb_rms(img, noise, sigma_w, margin=32):
    """Lower bound on the RMS displacement error of a Gaussian-windowed (sigma_w) translation
    estimate when both images carry white noise of std `noise` (reference for noisy cases)."""
    from scipy.ndimage import correlate1d

    from .interp import gauss_kernel

    k = gauss_kernel(sigma_w)
    iy, ix = np.gradient(img)

    def conv(a, kern):
        return correlate1d(correlate1d(a, kern, axis=0, mode="nearest"), kern, axis=1, mode="nearest")

    var = 0.0
    for g in (ix, iy):
        var = var + 2 * noise**2 * conv(g * g, k * k) / conv(g * g, k) ** 2
    s = np.s_[margin:-margin, margin:-margin]
    return float(np.sqrt(np.mean(var[s])))
