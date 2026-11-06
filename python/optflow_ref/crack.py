"""Crack geometry, opening (COD) and slip (CSD) from the half-window bookkeeping of phaseflow (crack_split).

Inputs live on the frame-0 grid: `ci` = dict(resid_full, resid, split), the full-window residual, the adopted residual
and the adopted half window (-1 = full) of the leg that produced the displacement (see phaseflow.Flow).

* band C (spec 2.3): pixels with resid_full > split_tol in connected components of at least edge_guard_min_area pixels.
* sides: an adopted half window h points away from the crack, so the pixels on the two sides of the crack choose
  directions that differ by about 180 deg. A first normal angle phi0 is the double-angle mean of the adopted theta_h
  within crack_r; the sign of cos(theta_h - phi0) labels each well fitted adopted pixel (+ / -).
* centre line (a design choice, the spec leaves the position of the crack inside the band open): the pixels of C that
  are about equally far (|d+ - d-| <= 1.5 px) from the nearest + pixel and the nearest - pixel, both within 2 crack_r.
* normal: the principal axis of the centre-line pixels within crack_r of each centre-line pixel gives the tangent
  (exact for a straight crack, which a 45-degree quantised theta_h is not); phi0 gives its orientation and is the
  fallback where the line is short. Every pixel of C takes the normal of its nearest centre-line pixel.
* jump: for each centre-line pixel p, q+- = p +- crack_a n; u+-(p) = u(q+-) + grad u(q+-) (p - q+-); J = u+ - u-,
  COD = J . n, CSD = J . t with t = (-n_y, n_x). NaN if u or its gradient is invalid at either q. The values are then
  replaced by their median over the centre-line pixels within 2 crack_r (the opening varies slowly along a crack and
  a single pixel is noisy: scatter of ~1 px at 1 % noise). Pixels of C take the value of their nearest centre-line
  pixel, pixels outside C are NaN.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, label, map_coordinates
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

from .strain import _DIRS

_THETA = np.deg2rad(45.0 * np.arange(8))
_UNIT = np.array([(np.cos(t), np.sin(t)) for t in _THETA])
assert np.allclose(_UNIT * np.hypot(*np.array(_DIRS).T)[:, None], np.array(_DIRS))


def large_components(b, min_area):
    """Pixels of `b` belonging to 4-connected components with at least min_area pixels."""
    lab, n = label(b)
    if n == 0:
        return np.zeros_like(b)
    keep = np.bincount(lab.ravel()) >= min_area
    keep[0] = False
    return keep[lab]


def _disk(r):
    k = int(np.ceil(r))
    t = np.arange(-k, k + 1, dtype=np.float64)
    return (t[None, :] ** 2 + t[:, None] ** 2 <= r * r).astype(np.float64)


def _local_sum(a, kern):
    return fftconvolve(a, kern, mode="same")


def geometry(ci, cfg):
    """Band, centre line and normals of the crack(s) from `ci` (see module doc). Returns a dict with
    C (bool), line (bool), nx, ny (float, NaN outside C or where there is no centre line), iy, ix (index of the
    nearest centre-line pixel of every pixel; None if there is none)."""
    rf, rs, split = ci["resid_full"], ci["resid"], ci["split"]
    H, W = rf.shape
    C = large_components(rf > cfg.split_tol, cfg.edge_guard_min_area_px())
    nan = np.full((H, W), np.nan)
    empty = dict(C=C, line=np.zeros((H, W), bool), nx=nan, ny=nan, iy=None, ix=None)
    good = C & (split >= 0) & (rs < cfg.split_tol)
    if not good.any():
        return empty
    th = _THETA[np.where(good, split, 0)]
    disk = _disk(cfg.crack_r)
    c2 = _local_sum(np.where(good, np.cos(2 * th), 0.0), disk)
    s2 = _local_sum(np.where(good, np.sin(2 * th), 0.0), disk)
    phi0 = 0.5 * np.arctan2(s2, c2)
    side = np.cos(th - phi0)
    plus, minus = good & (side > 0), good & (side < 0)
    if not plus.any() or not minus.any():
        return empty
    dp, dm = distance_transform_edt(~plus), distance_transform_edt(~minus)
    R = 2.0 * cfg.crack_r
    line = C & (dp <= R) & (dm <= R) & (np.abs(dp - dm) <= 1.5)
    if not line.any():
        return empty
    # principal axis of the centre-line pixels around every pixel
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    lf = line.astype(np.float64)
    n = _local_sum(lf, disk)
    nn = np.maximum(n, 1.0)
    mx, my = _local_sum(lf * xx, disk) / nn, _local_sum(lf * yy, disk) / nn
    cxx = _local_sum(lf * xx * xx, disk) / nn - mx * mx
    cyy = _local_sum(lf * yy * yy, disk) / nn - my * my
    cxy = _local_sum(lf * xx * yy, disk) / nn - mx * my
    psi = 0.5 * np.arctan2(2 * cxy, cxx - cyy)  # tangent angle
    lam = np.sqrt(((cxx - cyy) / 2) ** 2 + cxy**2)
    straight = (n >= 6) & (lam > 0.5 * (cxx + cyy) * 0.5)
    nxp, nyp = -np.sin(psi), np.cos(psi)
    flip = nxp * np.cos(phi0) + nyp * np.sin(phi0) < 0
    nxp, nyp = np.where(flip, -nxp, nxp), np.where(flip, -nyp, nyp)
    nx_l = np.where(straight, nxp, np.cos(phi0))
    ny_l = np.where(straight, nyp, np.sin(phi0))
    _, (iy, ix) = distance_transform_edt(~line, return_indices=True)
    nx = np.where(C, nx_l[iy, ix], np.nan)
    ny = np.where(C, ny_l[iy, ix], np.nan)
    return dict(C=C, line=line, nx=nx, ny=ny, iy=iy, ix=ix)


def strain_half_map(geo, win):
    """Half window h (0..7) for the one-sided strain (spec 2.4) of every pixel closer than win/2 to the crack centre
    line, pointing from the nearest centre-line pixel to the pixel (the side away from the crack); -1 elsewhere."""
    H, W = geo["C"].shape
    half = np.full((H, W), -1, dtype=np.int8)
    if geo["iy"] is None:
        return half
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    vy, vx = yy - geo["iy"], xx - geo["ix"]
    d = np.hypot(vx, vy)
    near = (d > 0) & (d < win / 2)
    h = np.argmax(vx[..., None] * _UNIT[:, 0] + vy[..., None] * _UNIT[:, 1], axis=-1)
    half[near] = h[near]
    return half


def _at(a, x, y):
    return map_coordinates(a, [y, x], order=1, mode="constant", cval=np.nan, prefilter=False)


def _strip_extrapolation(ux, uy, geo, cfg):
    """u+ and u- at every centre-line pixel p by a local affine fit on each side (crack_sides, replacing the gradient
    extrapolation of the stage-1 jump): the valid pixels q with n . (q - p) in [crack_a - 4, crack_a + 8] (and the mirror
    image on the - side) and |t . (q - p)| <= crack_r, u(q) ~ c0 + c1 s + c2 tau; c0 is u at p. NaN if fewer than 20
    pixels or their s-range is below 4 px. Reason: the one-sided strain at q is biased by the near-crack error (about
    1e-2 at 10 px from the band), times crack_a that is 0.1-0.3 px; a fit over the whole strip is far less so."""
    ly, lx = np.nonzero(geo["line"])
    nx, ny = geo["nx"][ly, lx], geo["ny"][ly, lx]
    vy, vx = np.nonzero(np.isfinite(ux) & np.isfinite(uy))
    tree = cKDTree(np.column_stack([vx, vy]))
    reach = cfg.crack_a + 10.0 + cfg.crack_r
    out = np.full((2, 2, len(lx)), np.nan)  # [side, component, pixel]
    for i, nb in enumerate(tree.query_ball_point(np.column_stack([lx, ly]), reach)):
        if not nb:
            continue
        nb = np.asarray(nb)
        dx, dy = vx[nb] - lx[i], vy[nb] - ly[i]
        s_ = dx * nx[i] + dy * ny[i]
        tau = -dx * ny[i] + dy * nx[i]
        for si, sg in enumerate((1.0, -1.0)):
            m = (np.abs(tau) <= cfg.crack_r) & (sg * s_ >= cfg.crack_a) & (sg * s_ <= cfg.crack_a + 10.0)
            if m.sum() < 20 or np.ptp(s_[m]) < 4.0:
                continue
            A = np.column_stack([np.ones(m.sum()), s_[m], tau[m]])
            for ci, u in enumerate((ux, uy)):
                coef = np.linalg.lstsq(A, u[vy[nb][m], vx[nb][m]], rcond=None)[0]
                out[si, ci, i] = coef[0]
    return ly, lx, nx, ny, out


def jump(ux, uy, grad, geo, cfg):
    """COD and CSD maps (NaN outside C) and the line pixels' values, from the displacement (ux, uy) (NaN = invalid)
    and its gradient `grad` = (dudx, dudy, dvdx, dvdy)."""
    H, W = ux.shape
    cod, csd = np.full((H, W), np.nan), np.full((H, W), np.nan)
    if geo["iy"] is None:
        return cod, csd
    if cfg.crack_sides:
        ly, lx, nx, ny, ex = _strip_extrapolation(ux, uy, geo, cfg)
        side = [(ex[0, 0], ex[0, 1]), (ex[1, 0], ex[1, 1])]
    else:
        ly, lx = np.nonzero(geo["line"])
        nx, ny = geo["nx"][ly, lx], geo["ny"][ly, lx]
        side = []
        for sgn in (1.0, -1.0):
            qx, qy = lx + sgn * cfg.crack_a * nx, ly + sgn * cfg.crack_a * ny
            u, v = _at(ux, qx, qy), _at(uy, qx, qy)
            gxx, gxy, gyx, gyy = (_at(g, qx, qy) for g in grad)
            # p - q = -sgn a n
            dx, dy = -sgn * cfg.crack_a * nx, -sgn * cfg.crack_a * ny
            side.append((u + gxx * dx + gxy * dy, v + gyx * dx + gyy * dy))
    jx, jy = side[0][0] - side[1][0], side[0][1] - side[1][1]
    cod_l, csd_l = jx * nx + jy * ny, -jx * ny + jy * nx
    # median over the centre-line pixels within 2 crack_r (NaN ignored; NaN if none is finite)
    tree = cKDTree(np.column_stack([lx, ly]))
    sm_c, sm_s = np.full(len(lx), np.nan), np.full(len(lx), np.nan)
    for i, nb in enumerate(tree.query_ball_point(np.column_stack([lx, ly]), 2.0 * cfg.crack_r)):
        v = cod_l[nb]
        if np.isfinite(v).any():
            sm_c[i], sm_s[i] = np.nanmedian(v), np.nanmedian(csd_l[nb])
    cl, sl = np.full((H, W), np.nan), np.full((H, W), np.nan)
    cl[ly, lx], sl[ly, lx] = sm_c, sm_s
    C = geo["C"]
    cod[C], csd[C] = cl[geo["iy"][C], geo["ix"][C]], sl[geo["iy"][C], geo["ix"][C]]
    return cod, csd


def crack_dict(geo, cod, csd):
    """The outputs of crack_out (Result.crack): mask, nx, ny, cod, csd (NaN outside the band) and `line` (the
    centre line the jump was read at, diagnostics only)."""
    return dict(mask=geo["C"], nx=geo["nx"], ny=geo["ny"], cod=cod, csd=csd, line=geo["line"])


def sample_leg(ci, u0r, xx, yy):
    """The crack inputs of a leg whose reference is frame 0 displaced by u0r, on the frame-0 grid: nearest neighbour at
    (xx + u0r_x, yy + u0r_y); positions outside the leg's image count as no crack (resid 0, full window)."""
    H, W = ci["split"].shape
    X, Y = np.rint(xx + u0r[0]), np.rint(yy + u0r[1])
    ok = np.isfinite(X) & np.isfinite(Y) & (X >= 0) & (X <= W - 1) & (Y >= 0) & (Y <= H - 1)
    Xi = np.clip(np.nan_to_num(X), 0, W - 1).astype(np.int64)
    Yi = np.clip(np.nan_to_num(Y), 0, H - 1).astype(np.int64)
    return dict(resid_full=np.where(ok, ci["resid_full"][Yi, Xi], 0.0),
                resid=np.where(ok, ci["resid"][Yi, Xi], 0.0),
                split=np.where(ok, ci["split"][Yi, Xi], -1).astype(np.int8))
