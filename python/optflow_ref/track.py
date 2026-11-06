"""End-to-end tracking: image pair and image sequence (cumulative, Lagrangian)."""
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import distance_transform_edt, label, uniform_filter

from . import crack as crack_mod
from . import xdic
from .phaseflow import refine, reportable_flow
from .poc import coarse_flow, grid_sample
from .strain import displacement_gradient, strain_fields


@dataclass
class Result:
    """Outcome of track_pair / one frame of track_sequence.

    ux, uy are NaN wherever the pixel is invalid (low confidence, outside the ROI, within the guard
    distance of a guard source, near the image border). conf is the confidence *before* any of that
    masking and guarding: it is not NaN-ed and not thresholded, so valid pixels are exactly the finite
    ones in ux (save_result writes them as valid.npy). In a sequence it is the frame-0 confidence of the
    whole link: the minimum over every leg.
    """
    ux: np.ndarray
    uy: np.ndarray
    conf: np.ndarray
    strain: dict
    raw_ux: np.ndarray  # unmasked field, used to initialise the next frame
    raw_uy: np.ndarray
    timing: dict = field(default_factory=dict)
    fb: np.ndarray = None  # hybrid sequences only: consecutive frames each pixel has been carried by the adjacent leg (0 = direct)
    # crack_out only: dict(mask, nx, ny, cod, csd, line) on the frame-0 grid, see crack.py (the output of the crack band,
    # its normal, opening and slip); None otherwise
    crack: dict = None
    # crack_split / crack_out only: dict(resid_full, resid, split) of the leg(s) this result came from, the input of
    # crack.geometry (a pair: the refine bookkeeping; a sequence frame: sampled on the frame-0 grid)
    crack_in: dict = None
    # crack_model "xdic" only (optflow_ref/xdic.py): dict(lines, lines_next, jumps) -- the crack polylines in the coordinates
    # of this result's reference (a pair: I1; a sequence frame: frame 0), the same polylines advanced into the target frame
    # (pairs only: the carried geometry of the next leg) and the COD / CSD samples along them; None without a crack line
    crack_lines: dict = None


def _large_components(b, min_area):
    """Pixels of `b` belonging to 4-connected components with at least min_area pixels."""
    lab, n = label(b)
    if n == 0:
        return np.zeros_like(b)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_area
    keep[0] = False
    return keep[lab]


def _finish_ex(ux, uy, conf, cfg, mask, timing, crack_in=None, xlines=None):
    """With crack_model "xdic" and crack polylines xlines (on the grid of ux): xdic.finish (returns the same 4 values; the
    COD / CSD samples along the lines go to timing["_xdic_jumps"]). Otherwise _finish plus the crack outputs: returns (mx, my, strain, crack); crack is None unless cfg.crack_out and crack_in
    (dict(resid_full, resid, split) on the grid of ux) are given. With crack_split the guard around crack-band
    sources is shortened to crack_guard_frac of it; with crack_out the strain near a crack is one-sided (spec 2.4)."""
    if cfg.crack_model == "xdic" and xlines:
        mx, my, strain, crack, jumps = xdic.finish(ux, uy, conf, cfg, mask, timing, xlines)
        timing["_xdic_jumps"] = jumps
        return mx, my, strain, crack
    t = time.perf_counter()
    low = conf < cfg.conf_min
    bad = low.copy()
    if mask is not None:
        bad |= ~mask
    geo = None
    if crack_in is not None and (cfg.crack_split or cfg.crack_out):
        geo = crack_mod.geometry(crack_in, cfg)
    guard = cfg.edge_guard_px()
    if guard > 0:
        src = _large_components(low, cfg.edge_guard_min_area_px())
        crack_src = None
        if geo is not None and cfg.crack_split and geo["C"].any():
            lab, _ = label(src)
            hit = np.unique(lab[geo["C"] & src])
            hit = hit[hit > 0]
            crack_src = np.isin(lab, hit) & src
            src = src & ~crack_src
        if mask is not None:
            src |= ~mask
        if src.any():
            bad |= distance_transform_edt(~src) <= guard
        if crack_src is not None and crack_src.any():
            bad |= distance_transform_edt(~crack_src) <= guard * cfg.crack_guard_frac
        H, W = conf.shape  # the frame just outside the image is a guard source as well
        yy, xx = np.mgrid[0:H, 0:W]
        bad |= np.minimum.reduce([xx, yy, W - 1 - xx, H - 1 - yy]) < guard
    if geo is not None and cfg.crack_split and not cfg.crack_sides and geo["line"].any():
        # the displacement within crack_a - 2 px of the crack centre line is the unreliable zone of the jump (the filters
        # straddle it, errors of several tenths of a px): invalid; the points q+- the jump is read at (crack_a from the
        # line, 2 px being the width of the centre line) lie just outside it
        bad |= distance_transform_edt(~geo["line"]) < cfg.crack_a - 2.0
    mx = np.where(bad, np.nan, ux)
    my = np.where(bad, np.nan, uy)
    crack = None
    if geo is not None and cfg.crack_out:
        grad = displacement_gradient(mx, my, cfg.strain_win, half=crack_mod.strain_half_map(geo, cfg.strain_win))
        cod, csd = crack_mod.jump(mx, my, grad, geo, cfg)
        crack = crack_mod.crack_dict(geo, cod, csd)
    else:
        grad = displacement_gradient(mx, my, cfg.strain_win)
    strain = strain_fields(*grad)
    timing["strain"] = time.perf_counter() - t
    return mx, my, strain, crack


def _finish(ux, uy, conf, cfg, mask, timing):
    return _finish_ex(ux, uy, conf, cfg, mask, timing)[:3]


def track_pair(I1, I2, cfg, mask=None, init=None, crack_hint=None) -> Result:
    """Displacement of I1's pixels into I2. init=(ux, uy) dense guess, e.g. the previous frame. crack_hint (crack_model
    "xdic" only): crack polylines in I1's coordinates carried from a previous leg; None runs the detector."""
    if I1.shape != I2.shape:
        raise ValueError(f"image sizes differ: {I1.shape} vs {I2.shape}")
    if mask is not None and mask.shape != I1.shape:
        raise ValueError(f"ROI mask shape {mask.shape} differs from the image shape {I1.shape}")
    if mask is not None and not mask.any():
        raise ValueError("ROI mask is empty")
    cfg.validate()
    if min(I1.shape) < cfg.poc_win:
        raise ValueError(f"image smaller than poc_win ({cfg.poc_win}): {I1.shape}")
    H, W = I1.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    timing = {}
    t = time.perf_counter()
    g = coarse_flow(I1, I2, cfg, init_dense=init)
    gx, gy = grid_sample(g, xx, yy)
    if init is not None:  # POC corrected the guess at grid points: add the interpolated correction
        Y, X = np.meshgrid(g.ys, g.xs, indexing="ij")
        corr = type(g)(g.xs, g.ys, g.ux - init[0][Y, X], g.uy - init[1][Y, X], g.peak)
        cx, cy = grid_sample(corr, xx, yy)
        gx, gy = init[0] + cx, init[1] + cy
    timing["coarse"] = time.perf_counter() - t
    t = time.perf_counter()
    fl = refine(I1, I2, gx, gy, cfg)
    timing["fine"] = time.perf_counter() - t
    if cfg.crack_model == "xdic":
        return _track_pair_xdic(I1, I2, fl, cfg, mask, timing, crack_hint)
    ci = dict(resid_full=fl.resid_full, resid=fl.resid_global, split=fl.split_global) if (cfg.crack_split or cfg.crack_out) else None
    ux, uy, strain, crack = _finish_ex(fl.ux, fl.uy, fl.conf, cfg, mask, timing, ci)
    return Result(ux, uy, fl.conf, strain, fl.ux, fl.uy, timing, crack=crack, crack_in=ci)


def _track_pair_xdic(I1, I2, fl, cfg, mask, timing, crack_hint):
    """The rest of track_pair for crack_model "xdic" (xdic.py): the carried lines crack_hint plus the new lines of the
    detector on the baseline result, then the per-side re-solve and xdic.finish; lines without a jump in their own solution
    are dropped, detected lines for which that solution does not explain the images better than the baseline are bands.
    No line: the baseline result."""
    t = time.perf_counter()
    hint = [np.asarray(l, np.float64) for l in crack_hint] if crack_hint else []
    mx0, my0, st0 = _finish(fl.ux, fl.uy, fl.conf, cfg, mask, {})
    found, _ = xdic.detect_lines(mx0, my0, st0["gmax"], cfg, mask)
    # carried lines first; a detected line is added unless it lies along a carried one (most of it within R_B)
    new = [l for l in found if not xdic.near_lines(l, hint, cfg.xdic_rb_px())]
    lines = hint + new
    carried = [True] * len(hint) + [False] * len(new)
    bands = []
    timing["xdic_detect"] = time.perf_counter() - t
    rejected = []
    t = time.perf_counter()
    while lines:
        flx, lines, _ = xdic.refine_xdic(I1, I2, fl, cfg, lines)
        ux, uy, strain, crack = _finish_ex(flx.ux, flx.uy, flx.conf, cfg, mask, timing, xlines=lines)
        jumps = timing.pop("_xdic_jumps")
        # a crack must show a jump in its own per-side solution: lines whose median |J| is below the detector's j_min
        # (or that have no finite jump) are dropped and the rest is solved again; so are the detected lines that are bands
        jump_ok = [xdic.jump_magnitude(j) >= xdic.DET_JMIN for j in jumps]
        gains = xdic.model_gain(I1, I2, fl, flx, lines, cfg, mask) if not all(carried) else [None] * len(lines)
        model_ok = [c or xdic.is_crack(m) for c, m in zip(carried, gains)]
        if all(jump_ok) and all(model_ok):
            break
        rejected += [l for l, k in zip(lines, jump_ok) if not k]
        bands += [l for l, j, k in zip(lines, jump_ok, model_ok) if j and not k]
        keep = [j and k for j, k in zip(jump_ok, model_ok)]
        lines = [l for l, k in zip(lines, keep) if k]
        carried = [c for c, k in zip(carried, keep) if k]
    timing["xdic"] = time.perf_counter() - t
    if not lines:
        ux, uy, strain = _finish(fl.ux, fl.uy, fl.conf, cfg, mask, timing)
        cl = dict(lines=[], lines_next=[], jumps=[], bands=bands, rejected=rejected) if (bands or rejected) else None
        return Result(ux, uy, fl.conf, strain, fl.ux, fl.uy, timing, crack_lines=cl)
    cl = dict(lines=lines, lines_next=xdic.advance_lines(lines, flx.ux, flx.uy), jumps=jumps, bands=bands, rejected=rejected)
    return Result(ux, uy, flx.conf, strain, flx.ux, flx.uy, timing, crack=crack, crack_lines=cl)


def _xdic_out(res, u0r, cx, cy, conf0, cfg, mask, timing):
    """Frame-0 outputs of a sequence frame for crack_model "xdic": the leg's lines mapped back to frame 0 (None: no line)."""
    if res.crack_lines is None or not res.crack_lines["lines"]:
        return None
    lines0 = xdic.lines_to_frame0(res.crack_lines["lines"], u0r)
    out = _finish_ex(cx, cy, conf0, cfg, mask, timing, xlines=lines0)
    return out, dict(lines=lines0, lines_next=None, jumps=timing.pop("_xdic_jumps"))


def _leg(I1, I2, cfg, init=None, hint=None, roi=None):
    """track_pair of a sequence leg; with crack_model "xdic" the carried crack geometry hint() and the ROI in the leg's
    reference roi() (None: no ROI) are passed (the detector must not see the ROI edge); otherwise the call is exactly the one
    without crack handling."""
    if cfg.crack_model == "xdic":
        return track_pair(I1, I2, cfg, mask=roi() if roi is not None else None, init=init, crack_hint=hint())
    return track_pair(I1, I2, cfg, init=init)


def _roi_in(mask, u):
    """The frame-0 ROI in the coordinates of a reference that frame 0 maps into by u (first-order inverse, nearest pixel);
    None without an ROI."""
    if mask is None:
        return None
    H, W = mask.shape
    yy, xx = np.mgrid[0:H, 0:W]
    X = np.clip(np.rint(xx - np.nan_to_num(u[0])), 0, W - 1).astype(np.int64)
    Y = np.clip(np.rint(yy - np.nan_to_num(u[1])), 0, H - 1).astype(np.int64)
    out = mask[Y, X]
    return out if out.any() else None


def _hint(res):
    """The carried crack geometry for a leg whose reference is the target frame of `res` (xdic)."""
    return None if res is None or res.crack_lines is None else res.crack_lines["lines_next"]


def _same_ref_hint(res):
    return None if res is None or res.crack_lines is None else res.crack_lines["lines"]


def sample_bilinear(a, x, y):
    """Bilinear sample with NaN propagation; outside the image -> NaN."""
    H, W = a.shape
    out_of = ~(np.isfinite(x) & np.isfinite(y)) | (x < 0) | (x > W - 1) | (y < 0) | (y > H - 1)
    xc = np.clip(np.nan_to_num(x), 0, W - 1)
    yc = np.clip(np.nan_to_num(y), 0, H - 1)
    x0 = np.minimum(np.floor(xc).astype(np.int64), W - 2)
    y0 = np.minimum(np.floor(yc).astype(np.int64), H - 2)
    tx, ty = xc - x0, yc - y0
    v = ((1 - ty) * ((1 - tx) * a[y0, x0] + tx * a[y0, x0 + 1])
         + ty * ((1 - tx) * a[y0 + 1, x0] + tx * a[y0 + 1, x0 + 1]))
    return np.where(out_of, np.nan, v)


def compose(ax, ay, bx, by):
    """u_0k(X) = u_0r(X) + u_rk(X + u_0r(X)) with a = u_0r, b = u_rk."""
    H, W = ax.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    px, py = xx + ax, yy + ay
    return ax + sample_bilinear(bx, px, py), ay + sample_bilinear(by, px, py)


def _conf_at(conf, u0r, xx, yy):
    """Confidence of a leg sampled at the positions the frame-0 pixels have in that leg's reference
    (NaN, i.e. outside the image, counts as 0)."""
    c = sample_bilinear(conf, xx + u0r[0], yy + u0r[1])
    return np.where(np.isfinite(c), c, 0.0)


def track_sequence(frames, cfg, mask=None):
    """Yield (k, Result) with displacement from frame 0 to frame k, for k = 1, 2, ...

    Each frame is matched directly against the current reference frame (no per-step drift);
    the reference moves to the previous frame when the mean confidence drops below
    cfg.ref_update_conf. The confidence of the output is that of the whole frame-0 -> k link: the
    minimum of the confidences of all legs (frame 0 -> r1 -> ... -> current reference -> k), each
    sampled where the frame-0 pixel lies in that leg's reference frame.
    """
    it = iter(frames)
    try:
        f0 = next(it)
    except StopIteration:
        raise ValueError("no frames") from None
    cfg.validate()
    H, W = f0.shape
    if mask is not None and mask.shape != f0.shape:
        raise ValueError(f"ROI mask shape {mask.shape} differs from the frame shape {f0.shape}")
    if mask is not None and not mask.any():
        raise ValueError("ROI mask is empty")
    if cfg.seq_mode == "hybrid":
        yield from _track_hybrid(f0, it, cfg, mask)
        return
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    ref, ref_idx = f0, 0
    u0r = (np.zeros((H, W)), np.zeros((H, W)))  # frame 0 -> reference
    c0r = np.ones((H, W))  # confidence of the frame 0 -> reference link, on the frame-0 grid
    prev, prev_res = f0, None
    for k, fk in enumerate(it, start=1):
        if fk.shape != f0.shape:
            raise ValueError(f"frame {k} has shape {fk.shape}, expected {f0.shape}")
        init = None if prev_res is None else (prev_res.raw_ux, prev_res.raw_uy)
        res = _leg(ref, fk, cfg, init=init, hint=lambda: _same_ref_hint(prev_res), roi=lambda: _roi_in(mask, u0r))
        cmean = _mean_conf(res, u0r, mask, xx, yy, cfg)
        if cmean < cfg.ref_update_conf and ref_idx != k - 1:
            # prev_res is the leg old reference -> k-1: its confidence extends the link of frame 0
            c0r = np.minimum(c0r, _conf_at(prev_res.conf, u0r, xx, yy))
            u0r = compose(u0r[0], u0r[1], prev_res.raw_ux, prev_res.raw_uy)
            ref, ref_idx = prev, k - 1
            res = _leg(ref, fk, cfg, hint=lambda: _hint(prev_res), roi=lambda: _roi_in(mask, u0r))
            cmean = _mean_conf(res, u0r, mask, xx, yy, cfg)
        cx, cy = compose(u0r[0], u0r[1], res.raw_ux, res.raw_uy)
        conf0 = np.minimum(c0r, _conf_at(res.conf, u0r, xx, yy))
        timing = dict(res.timing)
        ci = None if res.crack_in is None else crack_mod.sample_leg(res.crack_in, u0r, xx, yy)
        xo = _xdic_out(res, u0r, cx, cy, conf0, cfg, mask, timing) if cfg.crack_model == "xdic" else None
        if xo is not None:
            (ux, uy, strain, crack), cl = xo
        else:
            ux, uy, strain, crack = _finish_ex(cx, cy, conf0, cfg, mask, timing, ci)
            cl = None
        timing.update(reference=ref_idx, mean_conf=float(cmean))
        out = Result(ux, uy, conf0, strain, cx, cy, timing, crack=crack, crack_in=ci, crack_lines=cl)
        yield k, out
        prev, prev_res = fk, res


def _mean_conf(leg, u0r, mask, xx, yy, cfg):
    """Mean confidence of a leg over the frame-0 pixels that lie in the ROI (when given) and, in the leg's
    reference, on a reportable pixel: one the border rule of refine did not zero (spec 2.4). The border strips are a
    geometric exclusion, not a tracking-quality signal; counting them pulled the mean of frames of ~128 px under
    ref_update_conf and moved the reference at every frame. leg is the Result of the leg's track_pair."""
    X, Y = xx + u0r[0], yy + u0r[1]
    c = sample_bilinear(leg.conf, X, Y)
    rep = sample_bilinear(reportable_flow(leg.raw_ux, leg.raw_uy, cfg).astype(np.float64), X, Y)
    sel = np.isfinite(c) & (rep > 0.5)
    if mask is not None:
        sel &= mask
    return float(c[sel].mean()) if sel.any() else 0.0


def _local_affine_agreement(ux, uy, support, tx, ty, radius, min_support, tol, iters=2):
    """Spatial-consistency test of the hybrid fallback (report section "v3").

    For every pixel p an affine model u(q) = a + B (q - p) is fitted by least squares (ridge on B) to the
    displacement (ux, uy) of the `support` pixels q in the square window of half-size round(radius) around p
    (frame-0 grid). The fit is repeated `iters` times; from the second pass on, support pixels whose residual
    against their own local model from the previous pass exceeds 3 tol are dropped (robust to a displacement
    step -- a crack -- inside the window). Returns (ok, n): ok = n >= min_support and |(tx, ty) - a| <= tol,
    n = number of support pixels of the last pass in the window."""
    H, W = ux.shape
    r = max(1, int(round(radius)))
    size = 2 * r + 1
    area = float(size * size)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    ux0, uy0 = np.where(support, ux, 0.0), np.where(support, uy, 0.0)
    sel = support.copy()

    def box(a):
        return uniform_filter(a, size=size, mode="constant", cval=0.0) * area

    ridge = 1e-3 * r * r
    for it in range(iters):
        w = sel.astype(np.float64)
        S1, Sx, Sy = box(w), box(w * xx), box(w * yy)
        Sxx, Sxy, Syy = box(w * xx * xx), box(w * xx * yy), box(w * yy * yy)
        Sxc = Sx - xx * S1
        Syc = Sy - yy * S1
        Sxxc = Sxx - 2 * xx * Sx + xx * xx * S1
        Sxyc = Sxy - xx * Sy - yy * Sx + xx * yy * S1
        Syyc = Syy - 2 * yy * Sy + yy * yy * S1
        M = np.empty((H, W, 3, 3))
        M[..., 0, 0] = S1
        M[..., 0, 1] = M[..., 1, 0] = Sxc
        M[..., 0, 2] = M[..., 2, 0] = Syc
        M[..., 1, 1] = Sxxc + ridge * S1
        M[..., 1, 2] = M[..., 2, 1] = Sxyc
        M[..., 2, 2] = Syyc + ridge * S1
        ok_fit = S1 >= 3
        M[~ok_fit] = np.eye(3)
        coef = []
        for comp in (ux0, uy0):
            Su, Sxu, Syu = box(w * comp), box(w * xx * comp), box(w * yy * comp)
            rhs = np.stack([Su, Sxu - xx * Su, Syu - yy * Su], axis=-1)
            rhs[~ok_fit] = 0.0
            coef.append(np.linalg.solve(M, rhs[..., None])[..., 0])
        ax, ay = coef[0][..., 0], coef[1][..., 0]
        ax, ay = np.where(ok_fit, ax, np.nan), np.where(ok_fit, ay, np.nan)
        if it + 1 < iters:
            res = np.hypot(ux - ax, uy - ay)
            sel = support & (res <= 3 * tol)
    ok = (S1 + 0.5 >= min_support) & ok_fit & (np.hypot(tx - ax, ty - ay) <= tol)
    return ok, np.rint(S1).astype(np.int64)


def _track_hybrid(f0, it, cfg, mask):
    """seq_mode = hybrid, "direct-first" with a spatially gated fallback (report sections "v2 direct-first", "v3").

    Direct leg: exactly the chain machinery (reference frame `ref`, u0r = frame 0 -> ref, pair(ref, k) with the previous
    direct raw as init, compose, the reference moves to k-1 when the mean confidence is below ref_update_conf). Two
    state differences from chain, both per pixel:
      * link confidence = min(c0K, conf of the current leg sampled at X + u0K) -- only the CURRENT legs. c0K is not
        min-accumulated: when the reference moves to k-1 it is REPLACED by the confidence of the leg old-ref -> k-1 at
        the pixel if that pixel was direct-good at k-1, and by 0 if it was in fallback at k-1 (its u_{k-1} came from
        the adjacent leg: the pixel is then "unanchored", see recovery below).
      * u0K at a reference move is the output u_{k-1} (raw, fallback values included), not the plain composition.
    A pixel is direct-good when its u is finite and link confidence >= conf_min; its output is the chain value
    (compose(u0K, leg)) with conf = link confidence, n_fb = 0.

    Recovery across reference moves (v3). An unanchored pixel (in fallback when the reference moved) ignores c0K:
    its link confidence is the confidence of the current leg alone. It becomes direct-good again (and anchored, with
    c0K = that confidence, until the next reference move) iff
      * the reference is older than k-1 (otherwise the direct leg IS the adjacent leg and agreement proves nothing),
      * the leg confidence is >= conf_min, and
      * it agrees with the fallback value within fb_tol (|u_direct - u_fallback| <= fb_tol), or no fallback value
        exists (the adjacent confidence is < conf_min, n_fb would exceed fb_max, or u_{k-1} is not finite) and, with
        the spatial gate on (fb_min_support > 0), u_direct passes the same local-affine test as a fallback candidate
        (support = the direct-good pixels before this re-acquisition).
    Pixels that were direct-good when the reference moved are anchored and unchanged from v2.

    Other pixels are carried by the adjacent leg a = pair(k-1 -> k) (init: the previous adjacent raw if that leg was
    computed at k-1, else none; the leg is computed only if some ROI pixel needs it, and it is the direct leg itself
    when ref == k-1): u_k = u_{k-1} + a(X + u_{k-1}); n_fb += 1. Candidate iff n_fb <= fb_max and the adjacent confidence
    at X + u_{k-1} is >= conf_min. Spatial gate (v3, fb_min_support > 0): the candidate is valid only if
    |u_k - model| <= fb_tol, model = robust affine fit (see _local_affine_agreement) of the direct-good ROI pixels in the
    square window of half-size fb_radius around it, which must hold >= fb_min_support of them; without support the
    pixel stays invalid. Output conf = adjacent conf * max(0, 1 - n_fb / (fb_max + 1)); a valid fallback pixel
    may therefore report conf < conf_min (validity is decided by the rules above, via the conf passed to _finish).
    timing: reference, mean_conf, fallback_frac (share of ROI pixels valid by fallback), n_fb_max, fb_gate_rejected
    (share of ROI pixels that were candidates but failed the gate), reacquired (share of ROI pixels re-acquired)."""
    H, W = f0.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    roi = np.ones((H, W), bool) if mask is None else mask
    ref, ref_idx = f0, 0
    u0r = (np.zeros((H, W)), np.zeros((H, W)))  # frame 0 -> reference
    c0r = np.ones((H, W))  # c0K
    anch = np.ones((H, W), bool)  # c0K is trusted (the pixel was direct-good when the reference moved)
    u_prev = (np.zeros((H, W)), np.zeros((H, W)))  # raw output of frame k-1
    nfb = np.zeros((H, W), np.int64)
    good_prev = np.ones((H, W), bool)  # direct-good at k-1 (frame 0 counts as good)
    prev, prev_res, prev_adj = f0, None, None
    cap = cfg.fb_max + 1
    for k, fk in enumerate(it, start=1):
        if fk.shape != f0.shape:
            raise ValueError(f"frame {k} has shape {fk.shape}, expected {f0.shape}")
        init = None if prev_res is None else (prev_res.raw_ux, prev_res.raw_uy)
        res = _leg(ref, fk, cfg, init=init, hint=lambda: _same_ref_hint(prev_res), roi=lambda: _roi_in(mask, u0r))
        cmean = _mean_conf(res, u0r, mask, xx, yy, cfg)
        if cmean < cfg.ref_update_conf and ref_idx != k - 1:
            c0r = np.where(good_prev, _conf_at(prev_res.conf, u0r, xx, yy), 0.0)  # replaced, not min-accumulated
            anch = good_prev.copy()
            u0r = (u_prev[0].copy(), u_prev[1].copy())
            ref, ref_idx = prev, k - 1
            res = _leg(ref, fk, cfg, hint=lambda: _hint(prev_res), roi=lambda: _roi_in(mask, u0r))
            cmean = _mean_conf(res, u0r, mask, xx, yy, cfg)
        cx, cy = compose(u0r[0], u0r[1], res.raw_ux, res.raw_uy)
        leg_c = _conf_at(res.conf, u0r, xx, yy)
        link = np.where(anch, np.minimum(c0r, leg_c), leg_c)
        direct_ok = np.isfinite(cx) & np.isfinite(cy) & (link >= cfg.conf_min)
        good = direct_ok & anch
        cand_re = direct_ok & ~anch & roi & (ref_idx < k - 1)  # re-acquisition candidates
        need = ~good & roi & np.isfinite(u_prev[0]) & np.isfinite(u_prev[1]) & (nfb + 1 <= cfg.fb_max)
        timing = dict(res.timing)
        adj = None
        if need.any():
            if ref_idx == k - 1:
                adj = res
            else:
                adj = _leg(prev, fk, cfg, init=prev_adj, hint=lambda: _hint(prev_res), roi=lambda: _roi_in(mask, u_prev))
                timing["coarse"] += adj.timing["coarse"]
                timing["fine"] += adj.timing["fine"]
            ax, ay = compose(u_prev[0], u_prev[1], adj.raw_ux, adj.raw_uy)
            ca = _conf_at(adj.conf, u_prev, xx, yy)
            ok_a = np.isfinite(ax) & np.isfinite(ay)
            ax, ay = np.where(ok_a, ax, u_prev[0]), np.where(ok_a, ay, u_prev[1])
        else:
            ax, ay, ca = u_prev[0], u_prev[1], np.zeros((H, W))
        has_fb = need & (ca >= cfg.conf_min)  # a fallback value exists (before the spatial gate)
        with np.errstate(invalid="ignore"):
            agree = np.hypot(cx - ax, cy - ay) <= cfg.fb_tol
        re_nofb = cand_re & ~has_fb
        if cfg.fb_min_support > 0 and (re_nofb & roi).any():
            # no fallback value to compare with: the direct value must agree with the local model of the direct-good pixels
            gate0, _ = _local_affine_agreement(np.where(good, cx, np.nan), np.where(good, cy, np.nan), good & roi, cx, cy,
                                               cfg.fb_radius_px(), cfg.fb_min_support, cfg.fb_tol)
            re_nofb = re_nofb & gate0
        re = (cand_re & has_fb & agree) | re_nofb
        c0r = np.where(re, leg_c, c0r)
        anch = anch | re
        good = good | re
        nfb = np.where(good, 0, np.minimum(nfb + 1, cap))
        fb_cand = ~good & need & (nfb <= cfg.fb_max) & (ca >= cfg.conf_min)
        if cfg.fb_min_support > 0 and (fb_cand & roi).any():
            gate, _ = _local_affine_agreement(np.where(good, cx, np.nan), np.where(good, cy, np.nan), good & roi, ax, ay,
                                              cfg.fb_radius_px(), cfg.fb_min_support, cfg.fb_tol)
            fb_ok = fb_cand & gate
        else:
            fb_ok = fb_cand
        ux, uy = np.where(good, cx, ax), np.where(good, cy, ay)
        conf = np.where(good, link, ca * np.maximum(0.0, 1.0 - nfb / cap))
        conf_in = np.where(good, link, np.where(fb_ok, np.maximum(conf, cfg.conf_min), -1.0))
        ci = None if res.crack_in is None else crack_mod.sample_leg(res.crack_in, u0r, xx, yy)
        xo = _xdic_out(res, u0r, ux, uy, conf_in, cfg, mask, timing) if cfg.crack_model == "xdic" else None
        if xo is not None:
            (mx, my, strain, crack), cl = xo
        else:
            mx, my, strain, crack = _finish_ex(ux, uy, conf_in, cfg, mask, timing, ci)
            cl = None
        timing.update(reference=ref_idx, mean_conf=float(cmean), n_fb_max=int(nfb.max()),
                      fallback_frac=float((fb_ok & roi).sum() / roi.sum()),
                      fb_gate_rejected=float((fb_cand & ~fb_ok & roi).sum() / roi.sum()),
                      reacquired=float((re & roi).sum() / roi.sum()))
        yield k, Result(mx, my, conf, strain, ux, uy, timing, nfb, crack=crack, crack_in=ci, crack_lines=cl)
        u_prev, good_prev = (ux, uy), good
        prev, prev_res = fk, res
        prev_adj = None if adj is None else (adj.raw_ux, adj.raw_uy)
