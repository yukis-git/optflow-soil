"""Temporal averaging front-end: the mean of the 2m + 1 frames around a sampled frame, optionally after aligning every
neighbour to the centre frame with a low-order (affine) motion estimated from the coarse POC flow."""
import numpy as np
from scipy.ndimage import map_coordinates

from .config import Config
from .poc import coarse_flow, grid_sample

MIN_POINTS = 6  # POC grid points with a valid peak needed for the affine fit


def _fit_affine(xs, ys, ux, uy, iters=4):
    """Robust least-squares affine u(x, y) = a + B (x, y) through scattered displacements; points whose residual exceeds
    max(3 x 1.4826 MAD, 0.05 px) of the previous fit are dropped. Returns the (2, 3) coefficient matrix or None."""
    A = np.stack([np.ones_like(xs), xs, ys], axis=1)
    sel = np.ones(len(xs), bool)
    coef = None
    for _ in range(iters):
        if sel.sum() < MIN_POINTS:
            return None
        cx = np.linalg.lstsq(A[sel], ux[sel], rcond=None)[0]
        cy = np.linalg.lstsq(A[sel], uy[sel], rcond=None)[0]
        coef = np.stack([cx, cy])
        r = np.hypot(A @ cx - ux, A @ cy - uy)
        mad = 1.4826 * np.median(r[sel])
        sel = r <= max(3.0 * mad, 0.05)
    return coef


def align_to(centre, other, cfg=None, mask=None):
    """`other` resampled onto the pixel grid of `centre` (cubic spline), or None when no motion could be estimated.

    The motion is the affine fit (_fit_affine) to the coarse POC flow centre -> other, c(x) = other(x + u(x)), so
    aligned(x) = other(x + u(x)); with cfg.tavg_model == "dense" u is the coarse flow grid interpolated bilinearly instead of
    the affine fit (mask is then unused). cfg supplies the coarse-stage parameters (max_disp, poc_win, grid_step, ...). mask
    (bool, image shape): only POC grid points inside it enter the fit, so that the motion of the region of interest, not
    of a static surround, is the one removed."""
    cfg = cfg or Config()
    H, W = centre.shape
    g = coarse_flow(centre, other, cfg)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    if cfg.tavg_model == "dense":
        ux, uy = grid_sample(g, xx, yy)
        return map_coordinates(other, [yy + uy, xx + ux], order=3, mode="nearest")
    Y, X = np.meshgrid(g.ys, g.xs, indexing="ij")
    ok = g.peak > 0
    if mask is not None:
        ok &= mask[Y, X]
    if ok.sum() < MIN_POINTS:
        return None
    coef = _fit_affine(X[ok].astype(np.float64), Y[ok].astype(np.float64), g.ux[ok], g.uy[ok])
    if coef is None:
        return None
    ux = coef[0, 0] + coef[0, 1] * xx + coef[0, 2] * yy
    uy = coef[1, 0] + coef[1, 1] * xx + coef[1, 2] * yy
    return map_coordinates(other, [yy + uy, xx + ux], order=3, mode="nearest")


def average_window(frames, centre, align=False, cfg=None, mask=None):
    """Mean of `frames` (a list, the window around the centre frame frames[centre]). align: every other frame is first
    aligned to the centre (align_to); a frame whose motion cannot be estimated is left out of the mean."""
    use = [frames[centre]]
    for i, f in enumerate(frames):
        if i == centre:
            continue
        if align:
            f = align_to(frames[centre], f, cfg, mask)
            if f is None:
                continue
        use.append(f)
    return np.mean(use, axis=0) if len(use) > 1 else frames[centre]
