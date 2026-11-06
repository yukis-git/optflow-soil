"""Crack model "xdic" (experimental): a displacement field
with a jump across explicit crack lines, solved per side of each line by local-phase matching (the X-DIC idea).

Geometry. A crack is a polyline in the coordinates of the leg's reference image I1 (x = column, y = row). phi(X) is the
signed distance to the nearest polyline (+ on the left normal n = (-t_y, t_x) of the polyline direction t); a pixel is
"interior" when its nearest point on the polyline is not beyond an end of it. The side of an interior pixel is sign phi
(phi = 0 counts as +). The strip S = interior & |phi| <= R_B (cfg.xdic_rb_px(), 28 px at the defaults) is solved per side;
everything else is solved as before. The line has zero width: in reference coordinates the two sides tile the plane, the
gap of an opening crack only exists in I2.

1. Initialisation (spec 3.1): `detect_lines`, a minimal field-jump detector on the baseline result. Candidates are the
   pixels within DET_GMAX px of a large invalid hole (inside the ROI) and the valid pixels with gmax > DET_GMAX_STRAIN, on a
   DET_STEP px lattice, at least B + guard from the image border. At each candidate p and each of DET_NTHETA normals n_theta
   the disk of radius DET_R is split into the half disks n . (q - p) > DET_LINE_GUARD and < -DET_LINE_GUARD; an affine model
   is fitted by least squares to the valid pixels of each half (both components), J = u+(p) - u-(p) is the difference of
   the two fits at p, and sigma_J follows from the fit residuals (var a0 = kappa s^2 (M^-1)_00 per half and component,
   kappa = 4 pi sigma_w^2 for the spatially correlated errors of the windowed solve; sigma_J^2 = mean over the two
   components of var+ + var-). z = |J| / sigma_J; the theta of the largest z is kept. A candidate is a crack point when
   z >= DET_ZMIN and |J| >= DET_JMIN; halves with fewer than DET_MINPTS valid pixels, or whose mean tangential offset from p
   exceeds DET_R / 4, are not tested. Crack points are clustered (8-connected on the lattice after two dilations), each
   cluster gets its principal axis (z-weighted); per bin of DET_BIN px along the axis the vertex is the median normal offset
   of the baseline's invalid pixels there (the hole straddles the crack; the jump test is flat across it), else the
   z-weighted offset of the points with z >= 0.7 of the bin's maximum; median 3. Collinear pieces facing each other across
   <= 3 DET_R are joined, polylines shorter than 2 DET_R dropped, an end within 2 DET_R of the border extended to it.
   A continuous shear band passes this jump test as well; `is_crack` (after the solve, step 2) tells them apart.
2. Per-side solve (spec 3.2, `refine_xdic`), warm-started from the baseline refine. For side s the paint region is
   F_s = S & (other side | |phi| < MIX_PX): the mixed-pixel layer of MIX_PX = 2 px is painted for both sides (a pixel there
   can hold both materials, and the cubic-spline warp of I2 leaks the gap into it: one such column left 0.46 px at the line,
   0.19 px 6 px from it; painted 0.04-0.05). Its pixels are still solved, from their side's neighbours. I1_s = I1 with F_s
   painted with the mean of I1; per outer iteration the pre-warp s_k of side s is the blurred first-order extrapolation of
   side s's trusted displacement over the other side (2 R_B deep), J2_s = warp(I2, s_k) painted the same way, and one
   Gauss-Newton step is taken at the grid positions with the Jacobian of `masked_reference` (the paint edge does not move
   with the residual; with the plain phase gradient the estimate stays anchored to the pre-warp) and the LS weights times
   M_s = 1 - F_s. It replaces u / resid / energy / slope on the pixels of side s in S. Before the first iteration the
   low-confidence pixels of S take the unblurred extrapolation of their side.
3. Geometry update (spec 3.3, `_update_geometry`) after outer iterations 2, 4, ... (xdic_iters times): for the interior
   pixels within UPD_BAND px of the line, the displacement of either side is extrapolated to first order from that side's
   trusted pixels outside the band, and the squared intensity difference between I1 at X and gain I2 + offset at
   X + u_s(X) is taken (blurred by sigma = 1 px; gain / offset fitted on the confident pixels off the strip). Deviation from
   the spec's phase residual: the filter footprint (~12 px) straddles the line, so the phase residual switches sides over
   124-133 px for a line at 128 (and 3 px off, towards the gap), the intensity residual within 1 px. A pixel goes to the
   side with the smaller residual; per bin of UPD_BIN px along the line the offset delta in [-UPD_BAND, UPD_BAND] that
   minimises the |e+ - e-|-weighted misassignment is found and the polyline moves by delta along its normal (median 3 over
   the bins). Only pixels whose two residuals differ by more than UPD_INFO times the noise vote: inside a gap of I1 both
   motions fit, so such a line does not move. Pixels that change side restart from their new side's extrapolation (conf 0
   for the next pre-warp).
4. Outputs (spec 3.4, `finish`): validity by the usual confidence; large low-confidence components inside S are not guard
   sources (there is no special guard near the line), those outside S are guarded as before. Strain is one-sided within
   strain_win / 2 of the line (the field of the other side masked, min_frac 0.25). COD / CSD at points every 2 px along the
   line: per side a robust affine fit (in the line's normal / tangent offsets) to the valid pixels of that side with
   JUMP_SKIP <= |normal offset| and distance <= JUMP_R, evaluated at the point (the per-side field extrapolated to the
   line), J = u+ - u-, COD = J . n, CSD = J . t, t = (-n_y, n_x), then the median over +-JUMP_SMOOTH px along the line.
   crack = dict(mask, nx, ny, cod, csd, line) like crack_out (mask = line = interior & |phi| <= 1).

Sequences (track.track_sequence): every leg (pair reference -> frame) runs its own xdic refine in its own reference
coordinates. The geometry is carried as a polyline: a leg whose reference is the one of the previous direct leg starts from
that leg's final (updated) polyline; a leg whose reference is frame k-1 (a re-reference, or the adjacent leg of hybrid)
starts from the previous direct leg's polyline advanced into frame k-1, every vertex P moved by the mean of the two sides'
displacement at P (raw field sampled 1.5 px to either side). Every leg also runs the detector; a detected crack is added
unless most of it lies within R_B of a carried line. After the solve a line whose median |J| is below DET_JMIN is dropped
and a detected (not carried) line for which the per-side solve does not explain the images better than the baseline
(`model_gain`, `is_crack`) is reported as a band (track._track_pair_xdic). The frame-0 outputs use the polyline mapped back to frame 0 (`lines_to_frame0`: the cell where the
forward map of frame 0 jumps over the vertex).
"""
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label, map_coordinates, median_filter
from scipy.spatial import cKDTree

from . import phaseflow as pf
from .crack import large_components
from .interp import bspline_coeffs, gauss_blur, sample, warp
from .strain import displacement_gradient, strain_fields

DET_R = 36.0  # radius of the half-disk fits [px] (spec A: 24; the baseline hole of a 3 px opening is +-20 px wide)
DET_ZMIN = 6.0  # significance |J| / sigma_J
DET_JMIN = 0.3  # minimum jump [px]
DET_GMAX = 12.0  # candidates: pixels within this distance of a large invalid hole [px]
DET_GMAX_STRAIN = 0.05  # ... and valid pixels with gmax above this
DET_NTHETA = 16
DET_MINPTS = 30
DET_STEP = 4
DET_LINE_GUARD = 8.0  # half-disk pixels closer than this to the dividing line are left out (the baseline smears a jump over ~2 sigma_w)
DET_BIN = 12.0
UPD_BAND = 8.0
UPD_BIN = 16.0
UPD_INFO = 2.0  # a pixel votes in the geometry update when |e+ - e-| exceeds this times the noise level of e
JUMP_R = 16.0
JUMP_MIN = 12
JUMP_SMOOTH = 32.0
JUMP_SKIP = 4.0
LINE_STEP = 2.0
MIX_PX = 2.0


# ----------------------------------------------------------------------------------------------------------------- geometry
def polyline_fields(lines, shape):
    """phi (signed distance to the nearest polyline, + on the left normal), interior (nearest point not beyond an end), nx,
    ny (the unit normal of the nearest segment, pointing to +), lid (index of the nearest polyline, -1 without lines) and
    tau (arc length of the nearest point along its polyline)."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    best = np.full((H, W), np.inf)
    out = dict(phi=np.zeros((H, W)), interior=np.zeros((H, W), bool), nx=np.zeros((H, W)), ny=np.zeros((H, W)),
               lid=np.full((H, W), -1, np.int64), tau=np.zeros((H, W)))
    for li, P in enumerate(lines):
        P = np.asarray(P, np.float64)
        d = np.diff(P, axis=0)
        L = np.hypot(d[:, 0], d[:, 1])
        keep = L > 1e-9
        P = np.vstack([P[:1], P[1:][keep]])
        d, L = d[keep], L[keep]
        m = len(L)
        if m == 0:
            continue
        nrm = np.column_stack([-d[:, 1], d[:, 0]]) / L[:, None]
        cum = np.r_[0.0, np.cumsum(L)]
        for i in range(m):
            ax, ay = P[i]
            rx, ry = xx - ax, yy - ay
            t_raw = (rx * d[i, 0] + ry * d[i, 1]) / (L[i] ** 2)
            t = np.clip(t_raw, 0.0, 1.0)
            ex, ey = rx - t * d[i, 0], ry - t * d[i, 1]
            dist = np.hypot(ex, ey)
            s = rx * nrm[i, 0] + ry * nrm[i, 1]
            # at an inner vertex the sign comes from the pseudo-normal of the two segments that meet there
            if i > 0:
                pn = nrm[i - 1] + nrm[i]
                s = np.where(t <= 0.0, rx * pn[0] + ry * pn[1], s)
            if i < m - 1:
                pn = nrm[i] + nrm[i + 1]
                bx, by = xx - P[i + 1, 0], yy - P[i + 1, 1]
                s = np.where(t >= 1.0, bx * pn[0] + by * pn[1], s)
            inner = np.ones((H, W), bool)
            if i == 0:
                inner &= t_raw >= 0.0
            if i == m - 1:
                inner &= t_raw <= 1.0
            closer = dist < best
            best = np.where(closer, dist, best)
            out["phi"] = np.where(closer, np.where(s >= 0, dist, -dist), out["phi"])
            out["interior"] = np.where(closer, inner, out["interior"])
            out["nx"] = np.where(closer, nrm[i, 0], out["nx"])
            out["ny"] = np.where(closer, nrm[i, 1], out["ny"])
            out["lid"] = np.where(closer, li, out["lid"])
            out["tau"] = np.where(closer, cum[i] + t * L[i], out["tau"])
    return out


def _resample(P, step):
    """Points every `step` px of arc length along polyline P (both ends included), their unit normals and arc lengths."""
    P = np.asarray(P, np.float64)
    d = np.diff(P, axis=0)
    L = np.hypot(d[:, 0], d[:, 1])
    keep = L > 1e-9
    P = np.vstack([P[:1], P[1:][keep]])
    d, L = d[keep], L[keep]
    cum = np.r_[0.0, np.cumsum(L)]
    n = max(int(np.ceil(cum[-1] / step)), 1)
    s = np.linspace(0.0, cum[-1], n + 1)
    i = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(L) - 1)
    t = (s - cum[i]) / L[i]
    pts = P[i] + t[:, None] * d[i]
    nrm = np.column_stack([-d[i, 1], d[i, 0]]) / L[i][:, None]
    return pts, nrm, s


def _extend_to_border(P, shape, reach):
    """Extend an end of P to the image border along its end tangent when the border is within `reach` px."""
    H, W = shape
    P = [np.asarray(p, np.float64) for p in P]

    def exit_len(p, t):
        ls = []
        for c, dc, lo, hi in ((p[0], t[0], 0.0, W - 1.0), (p[1], t[1], 0.0, H - 1.0)):
            if dc > 1e-12:
                ls.append((hi - c) / dc)
            elif dc < -1e-12:
                ls.append((lo - c) / dc)
        return min(ls) if ls else np.inf

    k = min(4, len(P))
    for end in (0, 1):
        Q = np.array(P[:k] if end == 0 else P[-k:])
        b = P[0] if end == 0 else P[-1]
        _, _, vt = np.linalg.svd(Q - Q.mean(0))  # tangent of the last k vertices, pointing out of the line
        t = vt[0] * np.sign(np.dot(vt[0], (Q[0] - Q[-1]) if end == 0 else (Q[-1] - Q[0])) + 1e-300)
        ln = exit_len(b, t)
        if 0 < ln <= reach:
            q = b + (ln + 2.0) * t
            P = [q] + P if end == 0 else P + [q]
    return np.array(P)


# ----------------------------------------------------------------------------------------------------------------- detector
def _disk_offsets(R):
    k = int(np.ceil(R))
    t = np.arange(-k, k + 1, dtype=np.float64)
    oy, ox = np.meshgrid(t, t, indexing="ij")
    m = ox**2 + oy**2 <= R * R
    return ox[m], oy[m]


def jump_test(ux, uy, valid, px, py, R=DET_R, ntheta=DET_NTHETA, guard=DET_LINE_GUARD, minpts=DET_MINPTS, kappa=1.0):
    """Half-disk jump test at the integer points (px, py): returns J (n, 2) = u+ - u-, z (n,) and theta (n,) of the normal
    with the largest z (NaN / nan where no direction could be tested). kappa: variance inflation for spatially correlated
    errors (the fit residuals of neighbouring pixels are not independent). A direction is tested only if both halves hold
    minpts valid pixels whose mean tangential offset from p is within R / 4 (p lies between them, not off their end:
    otherwise J is an extrapolation along the line, e.g. next to the border strip)."""
    H, W = ux.shape
    ox, oy = _disk_offsets(R)
    th = np.pi * np.arange(ntheta) / ntheta
    s = ox[:, None] * np.cos(th)[None] + oy[:, None] * np.sin(th)[None]
    halves = np.concatenate([s > guard, s < -guard], axis=1).astype(np.float64)  # (no, 2 nt): + halves, then - halves
    nh = halves.shape[1]
    one = np.ones_like(ox)
    prods = [one, ox, oy, ox * ox, ox * oy, oy * oy]
    Bm = np.concatenate([halves * p[:, None] for p in prods], axis=1)
    Bb = np.concatenate([halves * p[:, None] for p in (one, ox, oy)], axis=1)
    n = len(px)
    J = np.full((n, 2), np.nan)
    z = np.full(n, np.nan)
    theta = np.full(n, np.nan)
    u0 = np.where(valid, ux, 0.0)
    v0 = np.where(valid, uy, 0.0)
    idx = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 1): 3, (1, 2): 4, (2, 2): 5}
    for c0 in range(0, n, 512):
        cx, cy = px[c0:c0 + 512], py[c0:c0 + 512]
        qx = cx[:, None] + ox[None].astype(np.int64)
        qy = cy[:, None] + oy[None].astype(np.int64)
        ins = (qx >= 0) & (qx < W) & (qy >= 0) & (qy < H)
        qxc, qyc = np.clip(qx, 0, W - 1), np.clip(qy, 0, H - 1)
        V = ins & valid[qyc, qxc]
        Vf = V.astype(np.float64)
        S = (Vf @ Bm).reshape(len(cx), 6, nh)
        M = np.empty((len(cx), nh, 3, 3))
        for (a, b), k in idx.items():
            M[:, :, a, b] = S[:, k]
            M[:, :, b, a] = S[:, k]
        cnt = S[:, 0]
        ok = cnt >= minpts
        M[~ok] = np.eye(3)
        M = M + 1e-9 * np.eye(3)
        Minv = np.linalg.inv(M)
        a0 = np.empty((len(cx), nh, 2))
        var = np.empty((len(cx), nh, 2))
        for ci, comp in enumerate((u0, v0)):
            U = np.where(V, comp[qyc, qxc], 0.0)
            b = (U @ Bb).reshape(len(cx), 3, nh).transpose(0, 2, 1)
            uu = (U * U) @ halves
            coef = np.einsum("chij,chj->chi", Minv, b)
            ss = np.maximum(uu - np.einsum("chi,chi->ch", coef, b), 0.0)
            s2 = ss / np.maximum(cnt - 3.0, 1.0)
            a0[..., ci] = coef[..., 0]
            var[..., ci] = s2 * Minv[..., 0, 0]
        nt = ntheta
        th2 = np.r_[th, th]
        mt = (-np.sin(th2)[None] * S[:, 1] + np.cos(th2)[None] * S[:, 2]) / np.maximum(cnt, 1.0)
        ok &= np.abs(mt) <= R / 4.0
        jj = a0[:, :nt] - a0[:, nt:]
        vj = kappa * 0.5 * (var[:, :nt] + var[:, nt:]).sum(axis=-1)
        okt = ok[:, :nt] & ok[:, nt:]
        zz = np.where(okt, np.hypot(jj[..., 0], jj[..., 1]) / np.sqrt(vj + 1e-12), -1.0)
        k = np.argmax(zz, axis=1)
        r = np.arange(len(cx))
        good = zz[r, k] >= 0
        J[c0:c0 + 512][good] = jj[r, k][good]
        z[c0:c0 + 512][good] = zz[r, k][good]
        theta[c0:c0 + 512][good] = th[k][good]
    return J, z, theta


def _lines_from_points(px, py, z, shape, hole=None):
    """Polylines through clustered crack points (see the module doc). hole: invalid pixels of the baseline; where a bin
    has invalid pixels within DET_R of the cluster axis, the vertex is their median normal offset (the hole of the baseline
    straddles the crack symmetrically, while the jump test is flat across it)."""
    H, W = shape
    hy, hx = np.nonzero(hole) if hole is not None else (np.zeros(0, int), np.zeros(0, int))
    if len(px) == 0:
        return []
    gi, gj = py // DET_STEP, px // DET_STEP
    grid = np.zeros((H // DET_STEP + 1, W // DET_STEP + 1), bool)
    grid[gi, gj] = True
    lab, nlab = label(binary_dilation(grid, structure=np.ones((3, 3)), iterations=2), structure=np.ones((3, 3)))
    cl = lab[gi, gj]
    lines = []
    for c in range(1, nlab + 1):
        sel = cl == c
        if sel.sum() < 4:
            continue
        x, y, w = px[sel].astype(np.float64), py[sel].astype(np.float64), np.minimum(z[sel], 50.0)
        mx, my = np.average(x, weights=w), np.average(y, weights=w)
        C = np.cov(np.vstack([x - mx, y - my]), aweights=w)
        ev, evec = np.linalg.eigh(C)
        t = evec[:, 1]
        nrm = np.array([-t[1], t[0]])
        tau = (x - mx) * t[0] + (y - my) * t[1]
        nu = (x - mx) * nrm[0] + (y - my) * nrm[1]
        htau = (hx - mx) * t[0] + (hy - my) * t[1]
        hnu = (hx - mx) * nrm[0] + (hy - my) * nrm[1]
        edges = np.arange(tau.min(), tau.max() + DET_BIN, DET_BIN)
        verts = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (tau >= a) & (tau < b)
            if not m.any():
                continue
            zm = w[m]
            top = zm >= 0.7 * zm.max()
            nu0 = np.average(nu[m][top], weights=zm[top])
            hm = (htau >= a) & (htau < b) & (np.abs(hnu - nu0) <= DET_R)
            if hm.sum() >= 0.25 * DET_BIN * DET_BIN:
                nu0 = float(np.median(hnu[hm]))
            verts.append((0.5 * (a + b), nu0))
        if len(verts) < 2:
            continue
        verts = np.array(verts)
        if len(verts) >= 3:
            verts[:, 1] = median_filter(verts[:, 1], size=3, mode="nearest")
        P = np.column_stack([mx + verts[:, 0] * t[0] + verts[:, 1] * nrm[0], my + verts[:, 0] * t[1] + verts[:, 1] * nrm[1]])
        lines.append(P)
    lines = _merge_collinear(lines)
    return [_extend_to_border(P, shape, 2 * DET_R) for P in lines if _length(P) >= 2 * DET_R]


def _length(P):
    return float(np.hypot(*np.diff(P, axis=0).T).sum())


def _merge_collinear(segs, gap=3 * DET_R, cos_min=0.95, off_max=8.0):
    """Join polylines whose ends face each other across a gap of at most `gap` px along a common direction (a crack whose
    jump test failed over a short stretch)."""
    segs = [np.asarray(P, np.float64) for P in segs]

    def unit(v):
        return v / max(np.hypot(*v), 1e-12)

    merged = True
    while merged and len(segs) > 1:
        merged = False
        for i in range(len(segs)):
            for j in range(len(segs)):
                if i == j:
                    continue
                for B in (segs[j], segs[j][::-1]):
                    A = segs[i]
                    tA, tB = unit(A[-1] - A[0]), unit(B[-1] - B[0])
                    v = B[0] - A[-1]
                    if (np.hypot(*v) <= gap and tA @ tB >= cos_min and v @ tA > 0
                            and abs(v[0] * tA[1] - v[1] * tA[0]) <= off_max):
                        segs = [P for k, P in enumerate(segs) if k not in (i, j)] + [np.vstack([A, B])]
                        merged = True
                        break
                if merged:
                    break
            if merged:
                break
    return segs


def detect_lines(mx, my, gmax, cfg, mask=None):
    """Crack polylines of the baseline result (mx, my: NaN = invalid; gmax its max shear strain). Returns (lines, info)."""
    H, W = mx.shape
    valid = np.isfinite(mx) & np.isfinite(my)
    roi = np.ones((H, W), bool) if mask is None else mask
    holes = large_components(~valid & roi, cfg.edge_guard_min_area_px())
    cand = distance_transform_edt(~holes) <= DET_GMAX if holes.any() else np.zeros((H, W), bool)
    if gmax is not None:
        cand |= valid & (np.nan_to_num(gmax, nan=0.0) > DET_GMAX_STRAIN)
    cand &= roi
    yy, xx = np.mgrid[0:H, 0:W]
    inner = np.minimum.reduce([xx, yy, W - 1 - xx, H - 1 - yy]) >= pf.border_width(cfg) + cfg.edge_guard_px()
    if mask is not None and not mask.all():  # the guard band along the ROI edge is a hole for the same reason
        inner &= distance_transform_edt(mask) > pf.border_width(cfg) + cfg.edge_guard_px()
    cand &= inner
    lat = np.zeros((H, W), bool)
    lat[::DET_STEP, ::DET_STEP] = True
    py, px = np.nonzero(cand & lat)
    J, z, th = jump_test(mx, my, valid, px, py, kappa=4.0 * np.pi * cfg.sigma_w**2)
    jm = np.hypot(J[:, 0], J[:, 1])
    acc = np.isfinite(z) & (z >= DET_ZMIN) & (jm >= DET_JMIN)
    lines = _lines_from_points(px[acc], py[acc], z[acc], (H, W), holes & inner)
    return lines, dict(n_cand=int(len(px)), n_acc=int(acc.sum()), px=px[acc], py=py[acc], z=z[acc], J=J[acc])


# ----------------------------------------------------------------------------------------------------------------- solve
def side_extrapolate(ux, uy, slope, conf, cfg, fill):
    """phaseflow.prewarp_side without the blur: the trusted displacement of one side (outside `fill`, conf >= conf_min and
    farther than the automatic guard from the low-confidence pixels outside `fill`), every other pixel the first-order
    extrapolation u(N) + grad u(N) (X - N) from its nearest trusted pixel N. Returns (ux, uy, ok) (ok = trusted)."""
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
    return ux, uy, ok


class _Geo:
    def __init__(self, lines, shape, cfg):
        self.lines = [np.asarray(P, np.float64) for P in lines]
        f = polyline_fields(self.lines, shape)
        self.__dict__.update(f)
        RB = cfg.xdic_rb_px()
        self.side = np.where(self.phi >= 0, 1, -1).astype(np.int8)
        self.strip = self.interior & (np.abs(self.phi) <= RB)
        # the mixed-pixel layer |phi| < MIX_PX is painted for both sides (its intensity may hold both materials, or the gap
        # in I2); its pixels are still solved, from their side's neighbours
        mix = np.abs(self.phi) < MIX_PX
        self.fill = {s: self.strip & ((self.side == -s) | mix) for s in (1, -1)}
        self.fill_pw = {s: self.interior & (self.side == -s) & (np.abs(self.phi) <= 2 * RB) for s in (1, -1)}
        self.sel = {s: self.strip & (self.side == s) for s in (1, -1)}


def _phase_resid_at(ref, c2raw, iy, ix, ux, uy):
    """Amplitude-weighted mean squared phase difference over the bank between I1 at the pixels (ix, iy) and I2 at
    (ix + ux, iy + uy)."""
    num = np.zeros(ix.shape)
    den = np.zeros(ix.shape)
    for (r1f, _, _, a1f), c2 in zip(ref, c2raw):
        r1, a1 = r1f[iy, ix], a1f[iy, ix]
        r2 = sample(c2, ix + ux + pf.PAD, iy + uy + pf.PAD)
        w = a1 * np.abs(r2)
        num += w * np.angle(r1 * np.conj(r2)) ** 2
        den += w
    return num / (den + 1e-30)


def _intensity_resid_at(I1, c2, gain, iy, ix, ux, uy):
    """Squared intensity difference between I1 at the pixels (ix, iy) and gain[0] I2 + gain[1] at (ix + ux, iy + uy)
    (c2: spline coefficients of I2)."""
    return (I1[iy, ix] - (gain[0] * sample(c2, ix + ux, iy + uy) + gain[1])) ** 2


def _update_geometry(geo, ux, uy, slope, conf, cfg, I1, c2, gain, shape, noise=0.0):
    """Spec 3.3: reassign the pixels near the line by the residual of either side's motion and move the polylines. A pixel
    votes only when the two residuals differ by more than UPD_INFO times the noise level `noise` (mean squared intensity
    difference of well-matched pixels): both motions explain a pixel of a gap in I1 (dark to dark), and with a gap wider
    than the band every vote was noise and moved the line by up to +-UPD_BAND per update. Without votes the line stays; with
    the votes of one face only it moves to the middle of the free range, i.e. towards the middle of the gap.
    Returns (new lines, ext) with ext[s] = (ux, uy) the extrapolation of side s (for the pixels that change side)."""
    H, W = shape
    band = geo.interior & (np.abs(geo.phi) <= UPD_BAND)
    if not band.any():
        return geo.lines, None
    iy, ix = np.nonzero(band)
    ext = {}
    e = {}
    for s in (1, -1):
        trusted_fill = ~(geo.sel[s] & (np.abs(geo.phi) > UPD_BAND))  # side s pixels of S beyond the band are the source
        ex, ey, ok = side_extrapolate(ux, uy, slope, conf, cfg, trusted_fill)
        if not ok.any():
            return geo.lines, None
        ext[s] = (ex, ey)
        res = np.zeros((H, W))
        res[iy, ix] = _intensity_resid_at(I1, c2, gain, iy, ix, ex[iy, ix], ey[iy, ix])
        e[s] = gauss_blur(np.where(band, res, 0.0), 1.0) / np.maximum(gauss_blur(band.astype(np.float64), 1.0), 1e-9)
    lab_plus = e[1] < e[-1]
    wgt = np.abs(e[1] - e[-1])
    wgt = np.where(wgt > UPD_INFO * noise, wgt, 0.0)
    new = []
    deltas = np.arange(-UPD_BAND, UPD_BAND + 1e-9, 0.25)
    for li, P in enumerate(geo.lines):
        m = band & (geo.lid == li)
        pts, nrm, sarc = _resample(P, UPD_BIN)
        if not m.any() or len(sarc) < 2:
            new.append(P)
            continue
        tau, phi, lp, w = geo.tau[m], geo.phi[m], lab_plus[m], wgt[m]
        nb = len(sarc) - 1
        dl = np.full(nb, np.nan)
        for b in range(nb):
            mb = (tau >= sarc[b]) & (tau < sarc[b + 1] if b < nb - 1 else tau <= sarc[b + 1])
            if mb.sum() < 20:
                continue
            # cost(delta) = weight of + labels with phi < delta plus weight of - labels with phi > delta
            pp, ww, ll = phi[mb], w[mb], lp[mb]
            cost = ((ll[None] & (pp[None] < deltas[:, None])) * ww[None]).sum(1) \
                + ((~ll[None] & (pp[None] > deltas[:, None])) * ww[None]).sum(1)
            best = np.flatnonzero(cost <= cost.min() + 1e-12)
            dl[b] = float(np.median(deltas[best]))
        if not np.isfinite(dl).any():
            new.append(P)
            continue
        fin = np.isfinite(dl)
        dl = np.interp(np.arange(nb), np.flatnonzero(fin), dl[fin])
        if nb >= 3:
            dl = median_filter(dl, size=3, mode="nearest")
        dv = np.r_[dl[0], 0.5 * (dl[:-1] + dl[1:]), dl[-1]]  # at the stations (bin edges)
        new.append(pts + dv[:, None] * nrm)
    return new, ext


def masked_reference(I1, fill, m1, bank, fx, fy):
    """Reference responses of one side: per filter (response, phase gradient x, y, amplitude) of I1 with `fill` painted m1,
    where the phase gradient is that of a shift of the side's texture inside the FIXED mask, Im(conj r G*(M dI1)) / |r|^2
    with M = 1 - fill and dI1 the (spectral) derivative of the unpainted I1 -- not the gradient of r itself, which also
    holds the edge of the paint. Reason: the paint boundary is at the same reference position in I1_s and in the
    pre-warped I2_s, so it does not move with the residual displacement; with the plain phase gradient the LS reads its
    response as "no motion" and the estimate stays anchored to the pre-warp (a 0.3 px pre-warp error left 0.27 px at the
    line, 0.14 px 4 px from it). The side's J2 is then compared at the same grid position (no resampling, which would
    move the paint edge as well): dphi = g' . (u - s)."""
    F = np.fft.fft2(pf._pad(np.where(fill, m1, I1)))
    Fd = np.fft.fft2(pf._pad(I1))
    keep = ~fill
    dx = np.where(keep, pf._crop(np.real(np.fft.ifft2(Fd * 1j * fx))), 0.0)
    dy = np.where(keep, pf._crop(np.real(np.fft.ifft2(Fd * 1j * fy))), 0.0)
    Fdx, Fdy = np.fft.fft2(pf._pad(dx)), np.fft.fft2(pf._pad(dy))
    out = []
    for G in bank:
        r = pf._crop(np.fft.ifft2(F * G))
        rx = pf._crop(np.fft.ifft2(Fdx * G))
        ry = pf._crop(np.fft.ifft2(Fdy * G))
        a2 = np.abs(r) ** 2 + 1e-12
        out.append((r, np.imag(np.conj(r) * rx) / a2, np.imag(np.conj(r) * ry) / a2, np.sqrt(a2)))
    return out


def _moments_grid(ref, resp, sx, sy, mult):
    """The seven per-pixel sums of local_solve for responses `resp` of the side's pre-warped I2 (pre-warp s = (sx, sy))
    taken at the grid position itself: c = dphi + g' . s, weights times mult."""
    H, W = sx.shape
    gxx = np.zeros((H, W)); gxy = np.zeros((H, W)); gyy = np.zeros((H, W))
    gxc = np.zeros((H, W)); gyc = np.zeros((H, W)); gcc = np.zeros((H, W))
    wsum = np.zeros((H, W))
    for (r1, gx, gy, a1), r2 in zip(ref, resp):
        dphi = np.angle(r1 * np.conj(r2))
        wk = a1 * np.abs(r2) * mult
        c = dphi + gx * sx + gy * sy
        gxx += wk * gx * gx; gxy += wk * gx * gy; gyy += wk * gy * gy
        gxc += wk * gx * c; gyc += wk * gy * c; gcc += wk * c * c
        wsum += wk
    return gxx, gxy, gyy, gxc, gyc, gcc, wsum


def refine_xdic(I1, I2, fl0, cfg, lines):
    """Spec 3.2 / 3.3: the per-side re-solve of the strip around `lines`, warm-started from the baseline refine fl0.
    Returns (Flow, lines after the geometry updates, _Geo of those lines)."""
    H, W = I1.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    bank, fx, fy = pf.gabor_bank((H + 2 * pf.PAD, W + 2 * pf.PAD), cfg.lam, cfg.scales, cfg.n_orient, cfg.bandwidth,
                                 cfg.dc_cut)
    ref = pf.reference_phase(I1, bank, fx, fy)
    m1 = float(I1.mean())
    ux, uy = np.array(fl0.ux, np.float64), np.array(fl0.uy, np.float64)
    slope = np.array(fl0.slope, np.float64)
    conf = np.array(fl0.conf, np.float64)
    resid = np.array(fl0.resid, np.float64)
    energy = None
    geo = _Geo(lines, (H, W), cfg)
    n_outer = max(cfg.outer_iters, 2 * cfg.xdic_iters + 2)
    sched = {2 + 2 * i for i in range(cfg.xdic_iters)}
    c2raw = None

    def init_sides(geo, ux, uy):
        for s in (1, -1):
            ex, ey, _ = side_extrapolate(ux, uy, slope, conf, cfg, geo.fill_pw[s])
            ux = np.where(geo.sel[s] & (conf < cfg.conf_min), ex, ux)
            uy = np.where(geo.sel[s] & (conf < cfg.conf_min), ey, uy)
        return ux, uy

    ux, uy = init_sides(geo, ux, uy)
    refs = None
    for it in range(n_outer):
        if it in sched:
            if c2raw is None:
                c2raw = bspline_coeffs(I2)
                # brightness of I2 against I1 (gain, offset) from the confident pixels off the strip
                ok = (conf >= 0.5) & ~geo.strip
                if ok.sum() > 100:
                    w2 = sample(c2raw, (xx + ux)[ok], (yy + uy)[ok])
                    A = np.column_stack([w2, np.ones(len(w2))])
                    gain = np.linalg.lstsq(A, I1[ok], rcond=None)[0]
                    noise = float(np.mean((I1[ok] - A @ gain) ** 2))
                else:
                    gain = np.array([1.0, 0.0])
                    noise = 0.0
            new, ext = _update_geometry(geo, ux, uy, slope, conf, cfg, I1, c2raw, gain, (H, W), noise)
            if ext is not None:
                g2 = _Geo(new, (H, W), cfg)
                for s in (1, -1):
                    ch = g2.sel[s] & ~geo.sel[s]
                    ux = np.where(ch, ext[s][0], ux)
                    uy = np.where(ch, ext[s][1], uy)
                    conf = np.where(ch, 0.0, conf)
                geo = g2
                refs = None
        if refs is None:
            refs = {s: (masked_reference(I1, geo.fill[s], m1, bank, fx, fy), 1.0 - geo.fill[s])
                    for s in (1, -1) if geo.sel[s].any()}
        sx, sy = pf.prewarp(ux, uy, slope, conf, cfg)
        F2 = np.fft.fft2(pf._pad(warp(I2, sx, sy)))
        resp = [bspline_coeffs(np.fft.ifft2(F2 * G)) for G in bank]
        csx, csy = bspline_coeffs(sx), bspline_coeffs(sy)
        side_state = {}
        for s, (rk, Mk) in refs.items():
            ex, ey, _ = side_extrapolate(ux, uy, slope, conf, cfg, geo.fill_pw[s])
            skx, sky = gauss_blur(ex, cfg.sigma_s), gauss_blur(ey, cfg.sigma_s)
            Jk = np.where(geo.fill[s], m1, warp(I2, skx, sky))
            F2k = np.fft.fft2(pf._pad(Jk))
            side_state[s] = (rk, [pf._crop(np.fft.ifft2(F2k * G)) for G in bank], skx, sky, Mk)
        ux_in, uy_in = ux, uy
        for _ in range(cfg.inner_iters):
            mom = pf._moments(ref, resp, csx, csy, sx, sy, ux, uy, xx, yy, None)
            ux, uy, energy, resid, slope = pf.local_solve(*mom, ux, uy, cfg.sigma_w, cfg.shape_order)
        ux, uy = ux.copy(), uy.copy()
        for s, (rk, respk, skx, sky, Mk) in side_state.items():
            # one Gauss-Newton step of side s about its pre-warp (see masked_reference)
            momk = _moments_grid(rk, respk, skx, sky, Mk)
            sk = pf._solve_full(*momk, ux_in, uy_in, cfg.sigma_w, cfg.shape_order)
            sel = geo.sel[s]
            ux[sel], uy[sel], energy[sel], resid[sel], slope[sel] = (sk[0][sel], sk[1][sel], sk[2][sel],
                                                                     sk[3][sel], sk[4][sel])
        conf = pf._confidence(energy, resid, cfg)
    conf_out = np.where(pf.reportable_flow(ux, uy, cfg), np.clip(conf, 0.0, 1.0), 0.0)
    split = np.where(geo.strip, np.where(geo.side > 0, 9, 10), -1).astype(np.int8)
    fl = pf.Flow(ux, uy, conf_out, resid, split, resid, slope, np.zeros((H, W), bool), split, resid)
    return fl, geo.lines, geo


CRACK_GAIN = 3.0  # a crack: the per-side solve removes at least 2/3 of the baseline's excess residual at the line ...
CRACK_E0_MIN = 1.5  # ... and that excess stands above the noise (median residual at the line / off the strip);
CRACK_MEAN = 2.0  # or: the mean residual within CRACK_MEAN_PX of the line falls by this factor (gross mismatches of the baseline)
CRACK_MEAN_PX = 8.0


def near_lines(P, others, dist):
    """True when most (> 50 %) of polyline P (sampled every 2 px) lies within `dist` px of one of the polylines `others`."""
    if not others:
        return False
    q, _, _ = _resample(P, 2.0)
    best = np.full(len(q), np.inf)
    for O in others:
        O = np.asarray(O, np.float64)
        for p0, p1 in zip(O[:-1], O[1:]):
            d = p1 - p0
            tt = np.clip(((q - p0) @ d) / max(float(d @ d), 1e-12), 0.0, 1.0)
            best = np.minimum(best, np.hypot(*(q - (p0 + tt[:, None] * d)).T))
    return bool(np.mean(best <= dist) > 0.5)


def jump_magnitude(j):
    """Median |J| = hypot(COD, CSD) of one line's jump samples (0 when none is finite)."""
    m = np.hypot(j["cod"], j["csd"])
    return float(np.nanmedian(m)) if np.isfinite(m).any() else 0.0


def model_gain(I1, I2, fl0, flx, lines, cfg, mask=None, reach=4.0):
    """Per polyline: does the jump explain the images better than the continuous baseline? Median squared intensity
    difference between I1 at X and gain I2 + offset at X + u, within `reach` px of the line (interior pixels, inside 4 sigma_w
    of the border and the ROI), under the baseline field fl0 (e0) and the per-side field flx (e1), against the same median
    on the confident baseline pixels off the strip (bg, the noise). Returns a list of dict(gain, e0, e1, mean, n): e0 and e1
    in units of bg, gain = (e0 - 1) / max(e1 - 1, 0.1), mean = ratio of the mean residuals within CRACK_MEAN_PX px of the
    line. `is_crack` decides from it. Phase residual and confidence of the
    baseline do not separate: the dark, textureless gap of a real video has a residual below the typical one and
    confidence 0.8 at a 2.5 px opening (0.78 x typical), a continuous 6 px band of width 3 px 2.9 x typical."""
    H, W = I1.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    f = polyline_fields(lines, (H, W))
    inner = np.minimum.reduce([xx, yy, W - 1 - xx, H - 1 - yy]) >= 4.0 * cfg.sigma_w
    if mask is not None:
        inner &= mask
    off = (fl0.conf >= 0.5) & inner & ~(f["interior"] & (np.abs(f["phi"]) <= cfg.xdic_rb_px()))
    c2 = bspline_coeffs(I2)
    if off.sum() <= 100:
        return [dict(gain=0.0, e0=0.0, e1=0.0, mean=0.0, n=0) for _ in lines]
    w2 = sample(c2, (xx + fl0.ux)[off], (yy + fl0.uy)[off])
    A = np.column_stack([w2, np.ones(len(w2))])
    g = np.linalg.lstsq(A, I1[off], rcond=None)[0]
    bg = max(float(np.median((I1[off] - A @ g) ** 2)), 1e-12)
    out = []
    for li in range(len(lines)):
        iy, ix = np.nonzero(f["interior"] & (f["lid"] == li) & inner & (np.abs(f["phi"]) <= reach))
        if len(iy) < 20:
            out.append(dict(gain=0.0, e0=0.0, e1=0.0, mean=0.0, n=int(len(iy))))
            continue
        e0 = float(np.median(_intensity_resid_at(I1, c2, g, iy, ix, fl0.ux[iy, ix], fl0.uy[iy, ix]))) / bg
        e1 = float(np.median(_intensity_resid_at(I1, c2, g, iy, ix, flx.ux[iy, ix], flx.uy[iy, ix]))) / bg
        iy, ix = np.nonzero(f["interior"] & (f["lid"] == li) & inner & (np.abs(f["phi"]) <= CRACK_MEAN_PX))
        m0 = float(np.mean(_intensity_resid_at(I1, c2, g, iy, ix, fl0.ux[iy, ix], fl0.uy[iy, ix])))
        m1 = float(np.mean(_intensity_resid_at(I1, c2, g, iy, ix, flx.ux[iy, ix], flx.uy[iy, ix])))
        out.append(dict(gain=(e0 - 1.0) / max(e1 - 1.0, 0.1), e0=e0, e1=e1, mean=m0 / max(m1, 1e-12), n=int(len(iy))))
    return out


def is_crack(m):
    """A line found by the detector is a crack when the per-side solve explains the images better (see model_gain, m its
    dict): gain >= CRACK_GAIN and e0 >= CRACK_E0_MIN (a fresh opening of about 1 px: gain 6, mean 1.2), or mean >= CRACK_MEAN
    (the widening dark gap of a real video: mean >= 2.5, gain down to 1.6). Otherwise a band: the continuous baseline fits it
    as well and the field there stays the baseline one."""
    return (m["gain"] >= CRACK_GAIN and m["e0"] >= CRACK_E0_MIN) or m["mean"] >= CRACK_MEAN


# ----------------------------------------------------------------------------------------------------------------- outputs
def _side_value(a, x, y):
    return map_coordinates(a, [y, x], order=1, mode="nearest")


def advance_lines(lines, ux, uy):
    """The polylines of a leg (reference coordinates) moved into the leg's target frame: every vertex by the mean of the
    raw displacement 1.5 px to either side of it."""
    out = []
    for P in lines:
        P = np.asarray(P, np.float64)
        nrm = _vertex_normals(P)
        up = [_side_value(a, P[:, 0] + 1.5 * nrm[:, 0], P[:, 1] + 1.5 * nrm[:, 1]) for a in (ux, uy)]
        um = [_side_value(a, P[:, 0] - 1.5 * nrm[:, 0], P[:, 1] - 1.5 * nrm[:, 1]) for a in (ux, uy)]
        mvx, mvy = 0.5 * (up[0] + um[0]), 0.5 * (up[1] + um[1])
        out.append(np.column_stack([P[:, 0] + np.nan_to_num(mvx), P[:, 1] + np.nan_to_num(mvy)]))
    return out


def lines_to_frame0(lines, u0r, w=6.0):
    """Polylines of a leg's reference mapped back to the frame-0 grid (u0r = frame 0 -> reference, raw). A vertex P lies in
    the crack of the reference; the forward map jumps over P at the crack of frame 0, so the crack is found directly: along
    the normal through P, the 1 px cell where g(s) = (X(s) + u0r(X(s)) - P) . n goes from < 0 to >= 0, of the largest rise when
    noise gives several; `cross` is the zero of g in the cell. u = mean of u0r at +- w n about the zero. X0 = P - u when that
    lies within 1/2 px of the zero (the midpoint motion gives the sub-pixel position when P is the middle of the gap), the
    zero itself beyond 1 px (a line off the middle of the gap maps to the same crack), linear in between: X0 is a continuous
    function of P and u0r, a choice by "inside the cell" jumped by up to 1 px. Replaces the fixed point X0 = P - mean u0r(X0 +- w n), which is a 2-cycle between the two sides once the jump of
    u0r exceeds 2 w (it ended half a gap off: 8 px at a 16 px gap with P exactly in the middle). Tangentially the crack
    is searched again after moving the search line by the tangential motion. A vertex without a cell (the ends beyond the
    image, a hole of u0r) has no measurement of its own and moves with the nearest vertex of the line that has one: reading
    u0r about an arbitrary search point mixed the zeros of the NaN pixels into u, and which pixels depended on the length of
    the search line, ceil(max |u0r|), so u0r near an integer moved the vertex (0.35 px for a 2e-3 px change of the maximum).
    A line without any cell: X0 = P - mean u0r at +- w n about the search point of the smallest |g|, X0 = P without a finite
    sample."""
    if u0r is None or not (np.any(np.nan_to_num(u0r[0])) or np.any(np.nan_to_num(u0r[1]))):
        return [np.asarray(P, np.float64) for P in lines]
    ax, ay = np.nan_to_num(u0r[0]), np.nan_to_num(u0r[1])
    fin = (np.isfinite(u0r[0]) & np.isfinite(u0r[1])).astype(np.float64)
    reach = int(np.ceil(np.hypot(ax, ay).max())) + 2
    ss = np.arange(-reach, reach + 1, dtype=np.float64)
    out = []
    for P in lines:
        P = np.asarray(P, np.float64)
        nrm = _vertex_normals(P)
        tan = np.column_stack([-nrm[:, 1], nrm[:, 0]])
        A = P.copy()
        for _ in range(2):
            qx = A[:, 0][:, None] + ss[None] * nrm[:, 0][:, None]
            qy = A[:, 1][:, None] + ss[None] * nrm[:, 1][:, None]
            vx, vy, ok = (_side_value(a, qx, qy) for a in (ax, ay, fin))
            g = (qx + vx - P[:, 0][:, None]) * nrm[:, 0][:, None] + (qy + vy - P[:, 1][:, None]) * nrm[:, 1][:, None]
            good = ok > 1.0 - 1e-9
            rise = np.where((g[:, :-1] < 0) & (g[:, 1:] >= 0) & good[:, :-1] & good[:, 1:], g[:, 1:] - g[:, :-1], -1.0)
            i = np.argmax(rise, axis=1)
            r = np.arange(len(P))
            found = rise[r, i] > 0
            near = np.argmin(np.where(good, np.abs(g), np.inf), axis=1)
            g0 = g[r, i]
            cross = np.where(found, ss[i] - g0 / np.where(found, rise[r, i], 1.0), ss[near])  # the zero of g in the cell
            c = A + cross[:, None] * nrm
            um = np.column_stack([0.5 * (_side_value(a, c[:, 0] + w * nrm[:, 0], c[:, 1] + w * nrm[:, 1])
                                         + _side_value(a, c[:, 0] - w * nrm[:, 0], c[:, 1] - w * nrm[:, 1])) for a in (ax, ay)])
            cells = np.flatnonzero(found)
            if len(cells):
                um[~found] = um[cells[np.argmin(np.abs(np.flatnonzero(~found)[:, None] - cells), axis=1)]]
            else:
                um[~good.any(axis=1)] = 0.0
            A = P - np.sum(um * tan, axis=1)[:, None] * tan
        X = P - um
        sx = np.sum((X - A) * nrm, axis=1)
        d = sx - cross
        sc = np.where(found, cross + np.clip(2.0 * (1.0 - np.abs(d)), 0.0, 1.0) * d, sx)
        out.append(A + sc[:, None] * nrm)
    return out


def _vertex_normals(P):
    t = np.gradient(P, axis=0) if len(P) > 1 else np.array([[1.0, 0.0]])
    t = t / np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-12)[:, None]
    return np.column_stack([-t[:, 1], t[:, 0]])


def jump_along(mx, my, lines, phi, shape, r=None, min_pts=JUMP_MIN, smooth=None):
    """COD / CSD at points every LINE_STEP px along each polyline (inside the image): list of dicts per line with pts, n,
    s (arc length), cod, csd (smoothed: median over +-smooth px), cod_raw, csd_raw, u_plus, u_minus (2, n)."""
    H, W = shape
    r = JUMP_R if r is None else r
    smooth = JUMP_SMOOTH if smooth is None else smooth
    vy, vx = np.nonzero(np.isfinite(mx) & np.isfinite(my))
    out = []
    if len(vx) == 0:
        return out
    tree = cKDTree(np.column_stack([vx, vy]))
    sgn = phi[vy, vx] >= 0
    for P in lines:
        pts, nrm, sarc = _resample(P, LINE_STEP)
        inside = (pts[:, 0] >= 0) & (pts[:, 0] <= W - 1) & (pts[:, 1] >= 0) & (pts[:, 1] <= H - 1)
        pts, nrm, sarc = pts[inside], nrm[inside], sarc[inside]
        n = len(pts)
        up = np.full((2, n), np.nan)
        um = np.full((2, n), np.nan)
        for i, nb in enumerate(tree.query_ball_point(pts, r)):
            if not nb:
                continue
            nb = np.asarray(nb)
            dx, dy = vx[nb] - pts[i, 0], vy[nb] - pts[i, 1]
            dn = dx * nrm[i, 0] + dy * nrm[i, 1]
            dt = -dx * nrm[i, 1] + dy * nrm[i, 0]
            for side, dst in ((True, up), (False, um)):
                m = (sgn[nb] == side) & (np.abs(dn) >= JUMP_SKIP)
                if m.sum() < min_pts:
                    continue
                A = np.column_stack([np.ones(m.sum()), dn[m], dt[m]])
                for c, a in enumerate((mx, my)):
                    val = a[vy[nb][m], vx[nb][m]]
                    coef = np.linalg.lstsq(A, val, rcond=None)[0]
                    res = val - A @ coef
                    mad = np.median(np.abs(res - np.median(res)))
                    keep = np.abs(res) <= 3.0 * 1.4826 * mad + 0.05
                    if keep.sum() >= min_pts and not keep.all():
                        coef = np.linalg.lstsq(A[keep], val[keep], rcond=None)[0]
                    dst[c, i] = coef[0]
        jx, jy = up[0] - um[0], up[1] - um[1]
        cod_r = jx * nrm[:, 0] + jy * nrm[:, 1]
        csd_r = -jx * nrm[:, 1] + jy * nrm[:, 0]
        cod, csd = np.full(n, np.nan), np.full(n, np.nan)
        for i in range(n):
            m = np.abs(sarc - sarc[i]) <= smooth
            if np.isfinite(cod_r[m]).any():
                cod[i], csd[i] = np.nanmedian(cod_r[m]), np.nanmedian(csd_r[m])
        out.append(dict(pts=pts, n=nrm, s=sarc, cod=cod, csd=csd, cod_raw=cod_r, csd_raw=csd_r, u_plus=up, u_minus=um))
    return out


def finish(ux, uy, conf, cfg, mask, timing, lines):
    """track._finish for crack_model "xdic" (module doc, item 4): returns (mx, my, strain, crack, jumps)."""
    import time
    t = time.perf_counter()
    H, W = conf.shape
    geo = polyline_fields(lines, (H, W))
    RB = cfg.xdic_rb_px()
    strip = geo["interior"] & (np.abs(geo["phi"]) <= RB)
    low = conf < cfg.conf_min
    bad = low.copy()
    if mask is not None:
        bad |= ~mask
    guard = cfg.edge_guard_px()
    if guard > 0:
        src = large_components(low & ~strip, cfg.edge_guard_min_area_px())
        if mask is not None:
            src |= ~mask
        if src.any():
            bad |= distance_transform_edt(~src) <= guard
        yy, xx = np.mgrid[0:H, 0:W]
        bad |= np.minimum.reduce([xx, yy, W - 1 - xx, H - 1 - yy]) < guard
    mx = np.where(bad, np.nan, ux)
    my = np.where(bad, np.nan, uy)
    phi, inter = geo["phi"], geo["interior"]
    full = displacement_gradient(mx, my, cfg.strain_win)
    near = inter & (np.abs(phi) < cfg.strain_win / 2.0 + 1.0)
    grad = [g.copy() for g in full]
    for s in (1, -1):
        on = (phi >= 0) if s > 0 else (phi < 0)
        keep = on | ~inter
        gs = displacement_gradient(np.where(keep, mx, np.nan), np.where(keep, my, np.nan), cfg.strain_win, min_frac=0.25)
        sel = near & on
        for g, h in zip(grad, gs):
            g[sel] = h[sel]
    strain = strain_fields(*grad)
    jumps = jump_along(mx, my, lines, phi, (H, W))
    line_px = inter & (np.abs(phi) <= 1.0)
    cod_m, csd_m = np.full((H, W), np.nan), np.full((H, W), np.nan)
    pts = [j["pts"] for j in jumps if len(j["pts"])]
    if pts:
        allp = np.vstack(pts)
        allc = np.concatenate([j["cod"] for j in jumps if len(j["pts"])])
        alls = np.concatenate([j["csd"] for j in jumps if len(j["pts"])])
        ly, lx = np.nonzero(line_px)
        if len(ly):
            _, k = cKDTree(allp).query(np.column_stack([lx, ly]))
            cod_m[ly, lx], csd_m[ly, lx] = allc[k], alls[k]
    nan = np.full((H, W), np.nan)
    crack = dict(mask=line_px, nx=np.where(line_px, geo["nx"], nan), ny=np.where(line_px, geo["ny"], nan), cod=cod_m,
                 csd=csd_m, line=line_px)
    timing["strain"] = time.perf_counter() - t
    return mx, my, strain, crack, jumps
