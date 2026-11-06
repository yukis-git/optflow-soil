"""Dense sub-pixel refinement from the local phase of a multi-band complex Gabor filter bank.

Per outer iteration the deformed image is warped with a smoothed copy of the current field
(removing most local stretch, which would otherwise bias the phase) and filtered. The inner
iterations sample those responses at each pixel's residual position and solve, per pixel,
a Gaussian-windowed least-squares problem for a polynomial (affine or quadratic) displacement
model around p, the same idea as DIC shape functions:

    g_q . P_p(q - p) = dphi_q + g_q . u_q      for neighbours q of p,  P_p(0) = u_p

Every residual is re-expressed at the window centre's displacement (Lucas-Kanade form).
Solving for increments instead leaves high-frequency error unconstrained and it drifts.

Confidence comes from the phase residual of that local fit (a wrong, patchy field cannot be
explained by a smooth local model), times a texture factor.

Border rule (spec 2.4). The Gabor responses within about edge_guard_px() of an image border come from
reflect padding, and the local solve averages them over its Gaussian window (radius 4 sigma_w). The padded
texture is self-consistent, so neither the phase residual nor the texture factor can tell: valid-looking
pixels up to ~32 px from the border were off by up to 0.1 px at confidence 0.997. With B = 4 sigma_w the
confidence is therefore 0 where X, the pixel, lies within B of image 1's border, and where its match
X' = X + u lies within B of image 2's border or outside it (a non-finite X' counts as outside). The rule is
applied only while the edge guard is on (cfg.edge_guard_px() > 0); with the guard off B = 0 and only the
matches that leave image 2 get confidence 0. The zero-confidence strips are large low-confidence components,
so track._finish adds its usual guard band to them: the pixels reported are ~B + guard (28 px at the
defaults) from a border.

Trusted pre-warp (spec 2.3). The pre-warp s = G_sigma_s * u of every outer iteration after the first leaves out the
displacement of pixels that cannot be trusted (see prewarp). A region where the coarse stage failed holds
displacements that are ~100 px wrong and that the fine stage cannot correct; blurred into s they stretch the warped
image of the good pixels beside them (|grad s| up to 1.7 against 0.01 for the true field), and the resulting phase
bias is smooth and nearly common to all filters, so the local fit absorbs it and the pixels stay confident (conf
0.94-1.0) at errors of up to 0.4 px.

A pixel whose window carries no phase gradient energy, or only one direction of it (stripe texture:
the translation block of the normal matrix is rank deficient, the aperture problem), is declared
degenerate before the solve: it keeps its input displacement and gets resid = pi (confidence 0).
"""
from dataclasses import dataclass

import numpy as np
from scipy import fft as sfft
from scipy.ndimage import binary_dilation, correlate1d, distance_transform_edt, label

from .interp import bspline_coeffs, gauss_blur, gauss_kernel, sample, warp

PAD = 32
# A window is aperture-degenerate when det / trace^2 of its 2x2 translation block is below this
# (0.25 = isotropic gradients, 0 = one direction only; speckle and real sand stay above 0.05).
APERTURE_TOL = 1e-3


@dataclass
class Flow:
    ux: np.ndarray
    uy: np.ndarray
    conf: np.ndarray  # [0, 1]; 0 within B = 4 sigma_w of image 1's border and where the match is within B of image 2's
    resid: np.ndarray  # weighted RMS phase residual of the local model [rad]
    # crack_split bookkeeping (always filled by refine): split -1 = full window, 0..7 = half window h adopted at the
    # last inner solve; resid_full = residual of the full window there; slope = affine part of the final shape
    # function (H, W, 4) = (dux/dx, duy/dx, dux/dy, duy/dy)
    split: np.ndarray = None
    resid_full: np.ndarray = None
    slope: np.ndarray = None
    # crack_sides: split is 8 + k on the pixels of side k (k = 1..m); crack_band = the crack band C (conf 0 there),
    # all False otherwise. split_global / resid_global = the half-window bookkeeping of the ordinary stream (what
    # split / resid are without crack_sides), the input of crack.geometry.
    crack_band: np.ndarray = None
    split_global: np.ndarray = None
    resid_global: np.ndarray = None


def gabor_bank(shape, lam, scales, n_orient, bandwidth, dc_cut):
    """Frequency-domain one-sided Gaussian filters at wavelengths lam*scales, plus frequency grids."""
    h, w = shape
    fy = (np.fft.fftfreq(h) * 2 * np.pi)[:, None]
    fx = (np.fft.fftfreq(w) * 2 * np.pi)[None, :]
    # high-pass factor, zero to second order at DC: removes brightness offsets and smooth
    # illumination ramps that the lowest band would otherwise leak into the phase (see Config.dc_cut)
    s_hp = dc_cut * lam * max(scales) / (2 * np.pi)
    hp = 1.0 - np.exp(-(fx**2 + fy**2) * s_hp**2 / 2)
    bank = []
    for sc in scales:
        w0 = 2 * np.pi / (lam * sc)
        s = bandwidth * w0
        for k in range(n_orient):
            th = np.pi * k / n_orient
            bank.append(hp * np.exp(-((fx - w0 * np.cos(th)) ** 2 + (fy - w0 * np.sin(th)) ** 2) / (2 * s * s)))
    return bank, fx, fy


def _pad(img):
    return np.pad(img, PAD, mode="reflect")


def _crop(a):
    return a[PAD:-PAD, PAD:-PAD]


def reference_phase(I1, bank, fx, fy):
    """Per filter: (response, phase gradient x, phase gradient y, amplitude) of I1."""
    F = np.fft.fft2(_pad(I1))
    out = []
    for G in bank:
        P = F * G
        r = _crop(np.fft.ifft2(P))
        rx = _crop(np.fft.ifft2(P * 1j * fx))
        ry = _crop(np.fft.ifft2(P * 1j * fy))
        a2 = np.abs(r) ** 2 + 1e-12
        out.append((r, np.imag(np.conj(r) * rx) / a2, np.imag(np.conj(r) * ry) / a2, np.sqrt(a2)))
    return out


def window_moment(a, sigma, px, py):
    """sum_q W(q - p) (qx - px)^px (qy - py)^py a(q) with Gaussian W; edges replicated."""
    k = gauss_kernel(sigma)
    r = len(k) // 2
    t = np.arange(-r, r + 1, dtype=np.float64)
    out = correlate1d(a, k * t**py, axis=0, mode="nearest")
    return correlate1d(out, k * t**px, axis=1, mode="nearest")


_POWERS = {1: [(0, 0), (1, 0), (0, 1)], 2: [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]}


def shape_unknowns(order):
    """Unknowns as (displacement component 0=x/1=y, powers of the offset d); ux, uy come first."""
    pw = _POWERS[order]
    return [(0, pw[0]), (1, pw[0])] + [(c, p) for p in pw[1:] for c in (0, 1)]


def _solve(mom, momw, gcc, gw, ux, uy, order, ref_energy):
    """Core of local_solve on arrays of any shape. mom(key, pw) / momw(a) give the window moments (powers pw) of the
    products and right-hand sides named by key (a product (0,0), (0,1), (1,1), or a right-hand side 0, 1) and the
    (0,0) moment of an arbitrary array; ref_energy is the median window energy of the field (None: that of `energy` itself). Returns
    (ux, uy, energy, resid, slope, ok)."""
    unk = shape_unknowns(order)
    n = len(unk)
    M = np.empty(ux.shape + (n, n))
    for i, (ci, pi) in enumerate(unk):
        for j, (cj, pj) in enumerate(unk):
            key = (min(ci, cj), max(ci, cj))
            M[..., i, j] = mom(key, (pi[0] + pj[0], pi[1] + pj[1]))
    b = np.empty(ux.shape + (n,))
    for i, (ci, pi) in enumerate(unk):
        b[..., i] = mom(ci, pi)
    energy = M[..., 0, 0] + M[..., 1, 1]
    det2 = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] ** 2  # 2x2 translation block
    ok = energy > 1e-9 * (np.median(energy) if ref_energy is None else ref_energy) + 1e-30
    ok &= det2 > APERTURE_TOL * energy**2
    M[~ok] = np.eye(n)
    b[~ok] = 0.0
    M = M + (1e-12 * np.trace(M, axis1=-2, axis2=-1) / n)[..., None, None] * np.eye(n)
    theta = np.linalg.solve(M, b[..., None])[..., 0]
    ok &= np.isfinite(theta[..., 0]) & np.isfinite(theta[..., 1])
    # at the LS optimum the residual sum is sum W w c^2 - theta . b
    r = momw(gcc) - np.einsum("...i,...i->...", theta, b)
    resid = np.sqrt(np.maximum(r, 0.0) / (momw(gw) + 1e-30))
    resid = np.where(ok, resid, np.pi)
    slope = np.where(ok[..., None], theta[..., 2:6], 0.0)
    return np.where(ok, theta[..., 0], ux), np.where(ok, theta[..., 1], uy), energy, resid, slope, ok


def _solve_full(gxx, gxy, gyy, gxc, gyc, gcc, gw, ux, uy, sigma, order):
    prod = {(0, 0): gxx, (0, 1): gxy, (1, 1): gyy}
    rhs = {0: gxc, 1: gyc}
    cache = {}

    def mom(key, pw):
        if (key, pw) not in cache:
            cache[(key, pw)] = window_moment(prod[key] if isinstance(key, tuple) else rhs[key], sigma, pw[0], pw[1])
        return cache[(key, pw)]

    return _solve(mom, lambda a: window_moment(a, sigma, 0, 0), gcc, gw, ux, uy, order, None)


def local_solve(gxx, gxy, gyy, gxc, gyc, gcc, gw, ux, uy, sigma, order):
    """Per-pixel weighted LS for a polynomial shape function of the given order (1 = affine,
    2 = quadratic) from per-pixel sums of w*g*g^T, w*g*c, w*c^2 and w.

    Returns (ux, uy, energy, resid, slope): resid is the weighted RMS phase residual of the fit [rad]; slope
    (H, W, 4) holds the affine part of the fitted shape function at the window centre, (dux/dx, duy/dx, dux/dy,
    duy/dy) (for order 2 the first-order terms; 0 where the system is degenerate).
    Pixels whose system is degenerate (no energy, or the translation block is rank deficient by the
    APERTURE_TOL rule) keep their input displacement and get resid = pi. The remaining systems get a
    tiny ridge (1e-12 x mean diagonal) so that the solve never meets a singular matrix.
    """
    return _solve_full(gxx, gxy, gyy, gxc, gyc, gcc, gw, ux, uy, sigma, order)[:5]


# integer direction vectors of the 8 half windows; n_h = (cos, sin)(45 deg * h), integers keep H(n . d) exact
_DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


class HalfMoments:
    """Window moments sum_q G(q-p) H(n_h . (q-p)) (qx-px)^px (qy-py)^py a(q) of the 8 half windows h (spec 2.1),
    H(s) = 1 for s > 0, 1/2 for s = 0, 0 for s < 0; edges replicated like window_moment.

    The moment of h in 0..3 is a 2-D correlation done by FFT of the edge-padded input (cached kernel transforms per
    (powers, h)); h + 4 follows from H(s) + H(-s) = 1: moment(h + 4) = window_moment - moment(h). The transform of
    an input is cached per name for the lifetime of the object (build one per solve)."""

    _kernels = {}

    def __init__(self, shape, sigma):
        self.k = gauss_kernel(sigma)
        self.r = len(self.k) // 2
        self.t = np.arange(-self.r, self.r + 1, dtype=np.float64)
        H, W = shape
        self.shape = shape
        self.n = (sfft.next_fast_len(H + 2 * self.r, real=True), sfft.next_fast_len(W + 2 * self.r, real=True))
        self.sigma = sigma
        self._fa = {}
        self._full = {}

    def _kernel_fft(self, pw, h):
        key = (self.sigma, self.n, pw, h)
        if key not in HalfMoments._kernels:
            if len(HalfMoments._kernels) > 96:
                HalfMoments._kernels.clear()
            dx, dy = _DIRS[h]
            tx, ty = np.meshgrid(self.t, self.t)  # [row = ty, col = tx]
            s = dx * tx + dy * ty
            hv = np.where(s > 0, 1.0, np.where(s == 0, 0.5, 0.0))
            K = np.outer(self.k * self.t ** pw[1], self.k * self.t ** pw[0]) * hv
            P = np.zeros(self.n)
            idx = np.arange(-self.r, self.r + 1) % np.array(self.n)[:, None]
            P[np.ix_(idx[0], idx[1])] = K
            HalfMoments._kernels[key] = np.conj(sfft.rfft2(P))
        return HalfMoments._kernels[key]

    def full(self, name, a, pw):
        key = (name, pw)
        if key not in self._full:
            self._full[key] = window_moment(a, self.sigma, pw[0], pw[1])
        return self._full[key]

    def __call__(self, name, a, pw, h):
        """Moment of array `a` (identified by `name`) for powers pw and half window h, full-field."""
        if h >= 4:
            return self.full(name, a, pw) - self(name, a, pw, h - 4)
        if name not in self._fa:
            ap = np.pad(a, self.r, mode="edge")
            self._fa[name] = sfft.rfft2(ap, s=self.n)
        out = sfft.irfft2(self._fa[name] * self._kernel_fft(pw, h), s=self.n)
        return out[self.r:self.r + self.shape[0], self.r:self.r + self.shape[1]]


def split_solve(gxx, gxy, gyy, gxc, gyc, gcc, gw, ux, uy, full, sigma, order, tol, gain):
    """Spec 2.1: replace the full-window solve `full` = _solve_full(...) by the best half-window solve where the full
    residual exceeds tol. Returns (ux, uy, energy, resid, slope, split) with split = -1 (full window) or h."""
    ux_f, uy_f, en_f, rs_f, sl_f, ok_f = full
    split = np.full(ux.shape, -1, dtype=np.int8)
    cand = rs_f > tol  # includes the degenerate windows (resid = pi)
    if not cand.any():
        return ux_f, uy_f, en_f, rs_f, sl_f, split
    iy, ix = np.nonzero(cand)
    prod = {(0, 0): gxx, (0, 1): gxy, (1, 1): gyy}
    rhs = {0: gxc, 1: gyc}
    hm = HalfMoments(ux.shape, sigma)
    ref_energy = float(np.median(en_f))
    cux, cuy = ux[iy, ix], uy[iy, ix]
    best = np.full(len(iy), np.inf)
    sol = [None] * 5
    sel = np.full(len(iy), -1, dtype=np.int8)
    for h in range(8):
        def mom(key, pw, h=h):
            a = prod[key] if isinstance(key, tuple) else rhs[key]
            return hm(key, a, pw, h)[iy, ix]

        def momw(a, h=h):
            name = "cc" if a is gcc else "w"
            return hm(name, a, (0, 0), h)[iy, ix]

        ux_h, uy_h, en_h, rs_h, sl_h, ok_h = _solve(mom, momw, gcc, gw, cux, cuy, order, ref_energy)
        rs_h = np.where(ok_h, rs_h, np.inf)
        better = rs_h < best
        if sol[0] is None:
            sol = [ux_h.copy(), uy_h.copy(), en_h.copy(), rs_h.copy(), sl_h.copy()]
        else:
            for dst, src in zip(sol, (ux_h, uy_h, en_h, rs_h, sl_h)):
                dst[better] = src[better]
        best = np.where(better, rs_h, best)
        sel = np.where(better, h, sel).astype(np.int8)
    full_ok = ok_f[iy, ix]
    adopt = np.isfinite(best) & np.where(full_ok, best < gain * rs_f[iy, ix], best < tol)
    ux_n, uy_n, en_n, rs_n, sl_n = ux_f.copy(), uy_f.copy(), en_f.copy(), rs_f.copy(), sl_f.copy()
    ux_n[iy[adopt], ix[adopt]] = sol[0][adopt]
    uy_n[iy[adopt], ix[adopt]] = sol[1][adopt]
    en_n[iy[adopt], ix[adopt]] = 2.0 * sol[2][adopt]  # a half window carries about half the energy
    rs_n[iy[adopt], ix[adopt]] = sol[3][adopt]
    sl_n[iy[adopt], ix[adopt]] = sol[4][adopt]
    split[iy[adopt], ix[adopt]] = sel[adopt]
    return ux_n, uy_n, en_n, rs_n, sl_n, split


def _border_distance(x, y, W, H):
    return np.minimum.reduce([x, y, W - 1 - x, H - 1 - y])


def _reportable(xx, yy, tx, ty, B):
    """Pixels (xx, yy) of image 1 whose confidence may be non-zero: at least B from image 1's border and
    their match (tx, ty) at least B inside image 2 (NaN compares false, i.e. counts as outside)."""
    H, W = xx.shape
    return (_border_distance(xx, yy, W, H) >= B) & (_border_distance(tx, ty, W, H) >= B)


def border_width(cfg):
    """B of the border rule: 4 sigma_w while the edge guard is on, else 0."""
    return 4.0 * cfg.sigma_w if cfg.edge_guard_px() > 0 else 0.0


def reportable_flow(ux, uy, cfg):
    """Pixels of image 1 whose confidence the border rule leaves alone, for the displacement (ux, uy) of a refine."""
    H, W = ux.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    return _reportable(xx, yy, xx + ux, yy + uy, border_width(cfg))


def _confidence(energy, resid, cfg):
    """Confidence before the border rule: texture factor times the phase residual factor, not yet clipped."""
    texture = np.minimum(1.0, energy / (0.25 * np.median(energy) + 1e-30))
    return texture * np.exp(-((resid / cfg.phase_tol) ** 2))


def prewarp(ux, uy, slope, conf, cfg):
    """Smoothed pre-warp s = G_sigma_s * u (sx, sy) that the fine stage warps image 2 with, built only from the
    displacement of trusted pixels.

    Trusted: conf >= conf_min and farther than R = cfg.auto_edge_guard_px() (3 spatial sigma of the longest filter,
    12.2 px at the defaults; the automatic edge guard, not the user's edge_guard) from every pixel with
    conf < conf_min. conf is the confidence of the previous solve before the border rule. Every other pixel X takes
    the first-order extrapolation u(N) + grad u(N) (X - N) from its nearest trusted pixel N (scipy's
    distance_transform_edt indices), with grad u(N) the affine part `slope` of the shape function fitted at N.

    Why: where the coarse stage failed, u is ~100 px wrong and the fine stage cannot correct it. Blurred into s, that
    stretches the warped image of the good pixels within ~4 sigma_s plus the filter footprint of the failure
    (|grad s| up to 1.7 against 0.01 for the true field) and biases their phase smoothly and nearly equally in all
    filters, which the local fit absorbs: conf 0.94-1.0 at errors of up to 0.4 px. The guard band counts as untrusted
    (trusting it leaves 0.05 px), the fill must be first order (zeroth order leaves ~600 pixels above 3e-3 px).

    With no trusted pixel, or only trusted pixels, s is the plain G_sigma_s * u.
    """
    low = conf < cfg.conf_min
    if low.any():
        ok = distance_transform_edt(~low) > cfg.auto_edge_guard_px()
        if ok.any() and not ok.all():
            _, (iy, ix) = distance_transform_edt(~ok, return_indices=True)
            H, W = ux.shape
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
            dx, dy = xx - ix, yy - iy
            g = slope[iy, ix]
            ux = np.where(ok, ux, ux[iy, ix] + g[..., 0] * dx + g[..., 2] * dy)
            uy = np.where(ok, uy, uy[iy, ix] + g[..., 1] * dx + g[..., 3] * dy)
    return gauss_blur(ux, cfg.sigma_s), gauss_blur(uy, cfg.sigma_s)


def _moments(ref, resp, csx, csy, sx, sy, ux, uy, xx, yy, mult):
    """The seven per-pixel sums of local_solve for the responses `resp` of image 2 (pre-warped with s = (sx, sy), spline
    coefficients csx, csy) against the reference responses `ref`; the weight of every sample is multiplied by mult."""
    H, W = ux.shape
    tx, ty = xx + ux, yy + uy
    px, py = tx - sx, ty - sy
    for _ in range(2):
        px, py = tx - sample(csx, px, py), ty - sample(csy, px, py)
    gxx = np.zeros((H, W)); gxy = np.zeros((H, W)); gyy = np.zeros((H, W))
    gxc = np.zeros((H, W)); gyc = np.zeros((H, W)); gcc = np.zeros((H, W))
    wsum = np.zeros((H, W))
    for (r1, gx, gy, a1), c2 in zip(ref, resp):
        r2 = sample(c2, px + PAD, py + PAD)
        dphi = np.angle(r1 * np.conj(r2))
        wk = a1 * np.abs(r2)
        if mult is not None:
            wk = wk * mult
        c = dphi + gx * ux + gy * uy
        gxx += wk * gx * gx; gxy += wk * gx * gy; gyy += wk * gy * gy
        gxc += wk * gx * c; gyc += wk * gy * c; gcc += wk * c * c
        wsum += wk
    return gxx, gxy, gyy, gxc, gyc, gcc, wsum


def prewarp_side(ux, uy, slope, conf, cfg, fill):
    """The pre-warp of one side of a crack (spec 6.1): like prewarp, but trusted are only the pixels outside `fill`
    (the crack band and the other sides) with conf >= conf_min and farther than the automatic guard from every such
    low-confidence pixel; the displacement of `fill` and of the untrusted pixels is extrapolated to first order from the
    nearest trusted pixel."""
    keep = ~fill
    low = (conf < cfg.conf_min) & keep
    ok = keep & (distance_transform_edt(~low) > cfg.auto_edge_guard_px()) if low.any() else keep
    if ok.any() and not ok.all():
        _, (iy, ix) = distance_transform_edt(~ok, return_indices=True)
        H, W = ux.shape
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        dx, dy = xx - ix, yy - iy
        g = slope[iy, ix]
        ux = np.where(ok, ux, ux[iy, ix] + g[..., 0] * dx + g[..., 2] * dy)
        uy = np.where(ok, uy, uy[iy, ix] + g[..., 1] * dx + g[..., 3] * dy)
    return gauss_blur(ux, cfg.sigma_s), gauss_blur(uy, cfg.sigma_s)


def make_sides(I1, resid_full, resid_post, cfg):
    """Crack band C, neighbourhood N and the sides of spec 6.1 from the residuals of the last solve (C from resid_full
    for side_band "full", the literal spec; from the residual after the half-window selection for "post"). Returns None if
    there is no crack band, else dict(C, N, label (0 = none, k = 1..m), m). Components of N \\ C smaller than
    edge_guard_min_area (fragments, not sides) are added to C."""
    area = cfg.edge_guard_min_area_px()
    src = resid_full if cfg.side_band == "full" else resid_post
    lab, n = label(src > cfg.split_tol)
    if n == 0:
        return None
    big = np.bincount(lab.ravel()) >= area
    big[0] = False
    C = binary_dilation(big[lab], iterations=2)
    R = cfg.auto_edge_guard_px() + 4.0 * cfg.sigma_w
    N = distance_transform_edt(~C) <= R
    sl, m = label(N & ~C, structure=np.ones((3, 3)))
    sizes = np.bincount(sl.ravel(), minlength=m + 1)
    keep = [k for k in range(1, m + 1) if sizes[k] >= area]
    if not keep:
        return None
    out = np.zeros_like(sl)
    for i, k in enumerate(keep, start=1):
        out[sl == k] = i
    return dict(C=C | ((sl > 0) & (out == 0)), N=N, label=out, m=len(keep))


def refine(I1, I2, ux, uy, cfg) -> Flow:
    H, W = I1.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    bank, fx, fy = gabor_bank((H + 2 * PAD, W + 2 * PAD), cfg.lam, cfg.scales, cfg.n_orient, cfg.bandwidth, cfg.dc_cut)
    ref = reference_phase(I1, bank, fx, fy)
    ux = np.array(ux, dtype=np.float64)
    uy = np.array(uy, dtype=np.float64)
    resid = np.full((H, W), np.pi)
    energy = np.ones((H, W))
    slope = None
    split = np.full((H, W), -1, dtype=np.int8)
    resid_full = resid
    sides = None
    split_g, resid_g = split, resid
    for it in range(cfg.outer_iters):
        if cfg.crack_sides and it == cfg.side_from_outer:
            sides = make_sides(I1, resid_full, resid_g, cfg)
            if sides is not None:
                m1 = float(I1.mean())
                fills, refs = [], []
                for k in range(1, sides["m"] + 1):
                    F = sides["C"] | ((sides["label"] > 0) & (sides["label"] != k))
                    fills.append(F)
                    refs.append(reference_phase(np.where(F, m1, I1), bank, fx, fy))
        if it == 0:  # no solve yet to tell trusted pixels from the rest
            sx, sy = gauss_blur(ux, cfg.sigma_s), gauss_blur(uy, cfg.sigma_s)
        else:
            sx, sy = prewarp(ux, uy, slope, _confidence(energy, resid, cfg), cfg)
        F2 = np.fft.fft2(_pad(warp(I2, sx, sy)))
        resp = [bspline_coeffs(np.fft.ifft2(F2 * G)) for G in bank]
        csx, csy = bspline_coeffs(sx), bspline_coeffs(sy)
        if sides is not None:
            conf_prev = _confidence(energy, resid, cfg)
            side_state = []
            for F, rk in zip(fills, refs):
                skx, sky = prewarp_side(ux, uy, slope, conf_prev, cfg, F)
                Jk = np.where(F, m1, warp(I2, skx, sky))
                F2k = np.fft.fft2(_pad(Jk))
                side_state.append((rk, [bspline_coeffs(np.fft.ifft2(F2k * G)) for G in bank],
                                   bspline_coeffs(skx), bspline_coeffs(sky), skx, sky, ~F))
        for _ in range(cfg.inner_iters):
            # position in the pre-warped image: X' + s(X') = X + u(X)
            tx, ty = xx + ux, yy + uy
            px, py = tx - sx, ty - sy
            for _ in range(2):
                px, py = tx - sample(csx, px, py), ty - sample(csy, px, py)
            gxx = np.zeros((H, W)); gxy = np.zeros((H, W)); gyy = np.zeros((H, W))
            gxc = np.zeros((H, W)); gyc = np.zeros((H, W)); gcc = np.zeros((H, W))
            wsum = np.zeros((H, W))
            for (r1, gx, gy, a1), c2 in zip(ref, resp):
                r2 = sample(c2, px + PAD, py + PAD)
                dphi = np.angle(r1 * np.conj(r2))
                wk = a1 * np.abs(r2)
                c = dphi + gx * ux + gy * uy
                gxx += wk * gx * gx; gxy += wk * gx * gy; gyy += wk * gy * gy
                gxc += wk * gx * c; gyc += wk * gy * c; gcc += wk * c * c
                wsum += wk
            ux0, uy0 = ux, uy
            if cfg.crack_split and it >= cfg.split_from_outer:
                full = _solve_full(gxx, gxy, gyy, gxc, gyc, gcc, wsum, ux, uy, cfg.sigma_w, cfg.shape_order)
                resid_full = full[3]
                ux, uy, energy, resid, slope, split = split_solve(gxx, gxy, gyy, gxc, gyc, gcc, wsum, ux, uy, full,
                                                                  cfg.sigma_w, cfg.shape_order, cfg.split_tol,
                                                                  cfg.split_gain)
            else:
                ux, uy, energy, resid, slope = local_solve(gxx, gxy, gyy, gxc, gyc, gcc, wsum, ux, uy, cfg.sigma_w,
                                                           cfg.shape_order)
                resid_full = resid
                split = np.full((H, W), -1, dtype=np.int8)
            split_g, resid_g = split, resid
            if sides is not None:
                ux, uy = ux.copy(), uy.copy()
                energy, resid, slope, split = energy.copy(), resid.copy(), slope.copy(), split.copy()
                for k, (rk, respk, ckx, cky, skx, sky, M) in enumerate(side_state, start=1):
                    mom = _moments(rk, respk, ckx, cky, skx, sky, ux0, uy0, xx, yy, M)
                    sk = _solve_full(*mom, ux0, uy0, cfg.sigma_w, cfg.shape_order)
                    sel = sides["label"] == k
                    ux[sel], uy[sel], energy[sel], resid[sel], slope[sel] = (sk[0][sel], sk[1][sel], sk[2][sel], sk[3][sel],
                                                                             sk[4][sel])
                    split[sel] = 8 + k
                resid[sides["C"]] = np.pi
    conf = _confidence(energy, resid, cfg)
    band = sides["C"] if sides is not None else np.zeros((H, W), bool)
    return Flow(ux, uy, np.where(reportable_flow(ux, uy, cfg), np.clip(conf, 0.0, 1.0), 0.0), resid, split,
                resid_full, slope, band, split_g, resid_g)
