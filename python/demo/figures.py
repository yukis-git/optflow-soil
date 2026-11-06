"""Figures and the animation of the demo: overview maps, settlement curve, and the strain-overlay video."""
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from optflow_ref import viz  # noqa: E402
from optflow_ref.track import sample_bilinear  # noqa: E402

from .field import analytic_budget, surface_budget, surface_profile  # noqa: E402
from .footing import Scene, VideoWriter  # noqa: E402


def view_window(scene: Scene):
    """Part of the frame-0 image shown in the maps: the active zone under and beside the footing (the whole mechanism up
    to its exit at the surface), as (x0, x1, y0, y1) in frame-0 pixels."""
    p = scene.params
    half = min(p.xe + 2.5 * p.w, p.half_box)
    return (int(p.xc - half), int(p.xc + half), int(p.y0), int(p.y0 + 0.8 * p.depth))


def selected_frames(n_frames):
    """Frames of the overview: about 1/3, 2/3 and the last."""
    last = n_frames - 1
    return sorted({max(1, int(round(last / 3))), max(1, int(round(2 * last / 3))), last})


# overview rows: (title, field source, colour map name key, scale kind)
ROWS = [("u_x [px]\nestimate", "ux", "est"), ("u_y [px]\nestimate", "uy", "est"),
        ("displacement error\n|est - true| [px]", "disp_err", "err_pos"),
        ("max shear strain\ngmax, estimate", "gmax", "est"), ("gmax\ntruth", "gmax", "true"),
        ("gmax error\n(est - true)", "gmax", "err"),
        ("volumetric strain\ndet F - 1, estimate", "vol", "est"), ("det F - 1\ntruth", "vol", "true"),
        ("det F - 1 error\n(est - true)", "vol", "err")]


def _masked(a, valid):
    return np.where(valid, a, np.nan)


def make_overview(ev, ref_img, path, title=""):
    """Estimate, truth and error maps of the selected frames on the frame-0 image, one column per frame, shared
    colour scales (rows of the same quantity share one: estimate and truth; each error row has its own,
    symmetric, shared by all frames)."""
    scene = ev.scene
    bx0, by0, bx1, by1 = ev.truth.bbox
    vx0, vx1, vy0, vy1 = view_window(scene)
    vx0, vx1, vy0, vy1 = max(vx0, bx0), min(vx1, bx1), max(vy0, by0), min(vy1, by1)  # inside the ROI's bounding box
    frames = sorted(ev.kept)
    sl = np.s_[vy0 - by0:vy1 - by0, vx0 - bx0:vx1 - bx0]
    img = ref_img[vy0:vy1, vx0:vx1]
    data = {}
    for k in frames:
        d = ev.kept[k]
        v = d["valid"] & np.isfinite(d["gmax"])
        e_ux, e_uy = d["ux"] - d["t_ux"], d["uy"] - d["t_uy"]
        data[k] = {
            "est": {q: _masked(d[q], v)[sl] for q in ("ux", "uy", "gmax", "vol")},
            "true": {q: _masked(d["t_" + q], v)[sl] for q in ("ux", "uy", "gmax", "vol")},
            "disp_err": _masked(np.hypot(e_ux, e_uy), v)[sl],
            "gmax_err": _masked(d["gmax"] - d["t_gmax"], v)[sl],
            "vol_err": _masked(d["vol"] - d["t_vol"], v)[sl],
        }
    # colour scales: estimate and truth of a quantity share one (the viz rules: symmetric for signed fields)
    fsets = []
    for k in frames:
        for kind in ("est", "true"):
            f = dict(data[k][kind])
            f["conf"], f["Exy"] = np.zeros(1), np.zeros(1)
            fsets.append(f)
    lim = viz.shared_limits(fsets, pct=99.9)

    def err_lim(name, signed):
        vals = np.concatenate([np.abs(data[k][name][np.isfinite(data[k][name])]) for k in frames])
        v = viz.nice_ceil(np.percentile(vals, 99.5)) if vals.size else 1.0
        return (-v, v) if signed else (0.0, v)

    lim["disp_err"], lim["gmax_err"], lim["vol_err"] = err_lim("disp_err", False), err_lim("gmax_err", True), err_lim("vol_err", True)

    # figure geometry in inches
    ph, pw = img.shape
    panel_w = 4.3
    panel_h = panel_w * ph / pw
    lm, rm, gap_x, gap_y, top = 1.7, 1.0, 0.18, 0.1, 0.75
    nrow, ncol = len(ROWS), len(frames)
    fw = lm + ncol * panel_w + (ncol - 1) * gap_x + rm
    fh = top + nrow * panel_h + (nrow - 1) * gap_y + 0.15
    fig = plt.figure(figsize=(fw, fh))
    if title:
        fig.suptitle(title, y=1 - 0.1 / fh, va="top", fontsize=12)
    cmaps = {"ux": "RdBu_r", "uy": "RdBu_r", "gmax": "Reds", "vol": "RdBu_r", "disp_err": "viridis",
             "gmax_err": "RdBu_r", "vol_err": "RdBu_r"}
    for r, (label, q, kind) in enumerate(ROWS):
        y1 = fh - top - r * (panel_h + gap_y)
        key = {"est": q, "true": q, "err_pos": "disp_err", "err": q + "_err"}[kind]
        lo, hi = lim[key]
        im = None
        for c, k in enumerate(frames):
            ax = fig.add_axes([(lm + c * (panel_w + gap_x)) / fw, (y1 - panel_h) / fh, panel_w / fw, panel_h / fh])
            ax.set_axis_off()
            field = data[k][kind][q] if kind in ("est", "true") else data[k][key]
            ax.imshow(img, cmap="gray", aspect="auto", vmin=0, vmax=1)
            im = ax.imshow(np.ma.masked_invalid(field), cmap=cmaps[key], vmin=lo, vmax=hi, alpha=0.88, aspect="auto")
            if r == 0:
                ax.set_title(f"frame {k}: settlement {scene.settlement(k):.1f} px", fontsize=10)
        fig.text(0.12 / fw, (y1 - panel_h / 2) / fh, label, ha="left", va="center", fontsize=9)
        cax = fig.add_axes([(lm + ncol * panel_w + (ncol - 1) * gap_x + 0.12) / fw, (y1 - panel_h) / fh, 0.14 / fw, panel_h / fh])
        fig.colorbar(im, cax=cax)
        cax.tick_params(labelsize=8)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return lim


def make_curves(ev, path, title=""):
    """Settlement true vs estimated, per-frame errors, valid fraction, peak gmax, the true volume budget of the field
    (heave area vs intrusion area vs settlement), the split of the settlement into the diffuse and the mechanism part, and
    the true ground surface (height above the initial level) at the selected frames."""
    rows = ev.rows
    k = np.array([r["k"] for r in rows])
    sc = ev.scene
    p = sc.params
    k_all = np.arange(sc.n_frames)
    s_all = sc.settlements()
    fig = plt.figure(figsize=(11, 14))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.85])
    axs = np.array([[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(3)])
    ax = axs[0, 0]
    ax.plot(k_all, s_all, "k-", lw=1.5, label="footing settlement s")
    ax.plot(k, [r["settlement_ref"] for r in rows], "--", color="C2", lw=1.5, label="true: median u_y of the soil in the wedge")
    ax.plot(k, [r["settlement_est"] for r in rows], "o", ms=3.5, color="C3", label="estimated (median u_y in the wedge)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("frame"), ax.set_ylabel("settlement [px]"), ax.legend(fontsize=7), ax.grid(alpha=0.3)
    ax.set_title("settlement curve")
    ax2 = ax.twinx()
    ax2.plot(k, [r["settlement_err"] for r in rows], "-", color="C0", lw=1, alpha=0.8)
    ax2.set_ylabel("error est - true soil [px]", color="C0")
    ax = axs[0, 1]
    ax.plot(k, [r["disp_rms"] for r in rows], label="RMS")
    ax.plot(k, [r["disp_p99"] for r in rows], label="99th percentile")
    ax.plot(k, [r["disp_rms_band"] for r in rows], "--", label="RMS inside shear bands")
    ax.plot(k, [r["disp_rms_out"] for r in rows], "--", label="RMS outside shear bands")
    ax.set_xlabel("frame"), ax.set_ylabel("displacement error [px]"), ax.legend(fontsize=8), ax.grid(alpha=0.3)
    ax.set_title("displacement error (valid pixels, frame-0 grid)")
    ax = axs[1, 0]
    ax.plot(k, [r["valid_frac"] for r in rows], label="valid / ROI")
    ax.plot(k, [r["valid_frac_eff"] for r in rows], label="valid / (ROI beyond the guard distance)")
    ax.plot(k, [r["mean_conf"] for r in rows], label="mean confidence of the current leg")
    for r in rows:
        if r["ref_update"]:
            ax.axvline(r["k"], color="gray", ls=":", lw=1)
    ax.set_ylim(0, 1.05), ax.set_xlabel("frame"), ax.legend(fontsize=8, loc="lower left"), ax.grid(alpha=0.3)
    ax.set_title("valid fraction (dotted: reference updates)")
    ax = axs[1, 1]
    ax.plot(k, [r["gmax_peak_true"] for r in rows], "k-", label="true, 99.9th percentile (valid pixels)")
    ax.plot(k, [r["gmax_peak_true_all"] for r in rows], "k:", label="true, 99.9th percentile (all pixels beyond the guard)")
    ax.plot(k, [r["gmax_peak_est"] for r in rows], "-", color="C3", label="estimated, 99.9th percentile (valid pixels)")
    ax.plot(k, [r["gmax_rms_band"] for r in rows], "--", color="C1", label="gmax RMS error inside bands")
    ax.plot(k, [r["gmax_rms_out"] for r in rows], "--", color="C2", label="gmax RMS error outside bands")
    ax.set_xlabel("frame"), ax.set_ylabel("gmax"), ax.legend(fontsize=7), ax.grid(alpha=0.3)
    ax.set_title("maximum shear strain")
    # the true volume budget of the field (areas of the 2-D scene): what the footing pushes in and where it goes
    sb = [surface_budget(p, float(s), step=0.25) for s in s_all]
    first = [analytic_budget(p, float(s))["total"] for s in s_all]
    ax = axs[2, 0]
    ax.plot(k_all, [b["intrusion_area"] for b in sb], "k-", lw=1.5, label="intrusion B s")
    ax.plot(k_all, [b["heave_area"] for b in sb], "-", color="C3", label="heave area beside the footing (part above the level)")
    ax.plot(k_all, [-b["trough_area"] for b in sb], "-", color="C0", label="settlement trough beside the footing (minus)")
    ax.plot(k_all, [b["net_dv"] for b in sb], "-", color="C1", label="net change of the soil's area (exact)")
    ax.plot(k_all, first, ":", color="C1", label="... first order: pre-peak compression + 2 D s_m")
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xlabel("frame"), ax.set_ylabel("area [px^2]"), ax.legend(fontsize=7), ax.grid(alpha=0.3)
    ax.set_title("true volume budget of the field")
    ax = axs[2, 1]
    ax.plot(k_all, s_all, "k-", label="s = s_e + s_m")
    ax.plot(k_all, p.s_e(s_all), "-", color="C0", label="s_e: diffuse (pre-peak) part")
    ax.plot(k_all, p.s_m(s_all), "-", color="C3", label="s_m: failure mechanism")
    ax.set_xlabel("frame"), ax.set_ylabel("settlement [px]"), ax.legend(fontsize=8), ax.grid(alpha=0.3)
    ax.set_title("settlement: diffuse part and mechanism")
    for ax in axs.ravel():
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # the true ground surface: height above the initial level (positive up) against the deformed position, at the frames of
    # the overview; the footing's base is at -s; the dotted lines mark the footing's edges and the mechanism's exit
    ax = fig.add_subplot(gs[3, :])
    for kk, col in zip(sorted(ev.kept), ("C0", "C1", "C3")):
        s = float(sc.settlement(kk))
        _, xd, h = surface_profile(p, s, step=0.25)
        for side, lab in ((xd < -p.b, f"frame {kk}: s = {s:.0f} px"), (xd > p.b, None)):  # beside the footing, left and right
            ax.plot(xd[side], h[side], "-", color=col, label=lab)
        ax.plot([-p.b, p.b], [-s, -s], "-", color=col, lw=3, alpha=0.5)  # the footing's base
    ax.axhline(0.0, color="gray", lw=0.8)
    for x in (-p.b, p.b):
        ax.axvline(x, color="k", ls=":", lw=1)
    for x in (-p.xe, p.xe):
        ax.axvline(x, color="gray", ls=":", lw=1)
    ax.set_xlim(-p.half_box, p.half_box)
    ax.set_xlabel("x' from the centre line [px] (dotted: footing edges, mechanism exit)"), ax.set_ylabel("height above the initial level [px]")
    ax.legend(fontsize=8, loc="lower right"), ax.grid(alpha=0.3)
    ax.set_title("true ground surface (thick line: footing base): the soil at the edge is dragged down, then a ridge "
                 "and the passive wedge's plateau", fontsize=9)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------
# animation: the strain at the material's current positions
# ---------------------------------------------------------------------------------------------------------
def invert_displacement(ux, uy, x, y, tol=0.02, maxit=40):
    """Material coordinates X with X + u(X) = x for the pixel positions (x, y) (arrays of one shape): fixed-point
    iteration X <- x - u(X) with bilinear samples of the estimated displacement; NaN where X reaches an invalid
    pixel or the image's edge.  Only the pixels that are still moving are iterated."""
    X, Y = x.astype(np.float64).copy(), y.astype(np.float64).copy()
    active = np.arange(X.size)
    Xf, Yf, xf, yf = X.ravel(), Y.ravel(), x.ravel().astype(np.float64), y.ravel().astype(np.float64)
    for _ in range(maxit):
        dx = sample_bilinear(ux, Xf[active], Yf[active])
        dy = sample_bilinear(uy, Xf[active], Yf[active])
        nx, ny = xf[active] - dx, yf[active] - dy
        change = np.maximum(np.abs(nx - Xf[active]), np.abs(ny - Yf[active]))
        Xf[active], Yf[active] = nx, ny
        keep = np.isfinite(change) & (change > tol)
        active = active[keep]
        if active.size == 0:
            break
    return Xf.reshape(X.shape), Yf.reshape(Y.shape)


def _outline_points(polygon, step=2.0):
    pts = []
    n = len(polygon)
    for i in range(n):
        a, b = np.array(polygon[i], float), np.array(polygon[(i + 1) % n], float)
        m = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
        t = np.linspace(0.0, 1.0, m, endpoint=False)[:, None]
        pts.append(a + t * (b - a))
    return np.vstack(pts)


def crop_window(scene: Scene):
    """Part of the full frame shown in the video: the box with a margin and part of the rod above (even size)."""
    p = scene.params
    x0 = max(int(scene.box_l - 1.1 * scene.wall - 4) // 2 * 2, 0)
    x1 = min(int(scene.box_r + 1.1 * scene.wall + 4) // 2 * 2 + 2, scene.width)
    y0 = max(int(p.y0 - 1.4 * scene.foot_h - 10) // 2 * 2, 0)
    y1 = min(int(scene.y_bottom + 1.2 * scene.wall + 6) // 2 * 2 + 2, scene.height)
    return x0, x1, y0, y1


def _put(img, text, org, scale, color=(255, 255, 255), thick=1):
    import cv2

    org = (int(round(org[0])), int(round(org[1])))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def make_animation(scene: Scene, video_path, track_dir, out_path, gmax_hi, roi_polygon, progress=None, alpha=0.78):
    """Side by side: the input frame, and the same frame with the estimated gmax drawn at the current positions of
    the soil particles (transparent where invalid or where the strain is near zero), the ROI outline moved with the
    material.

    gmax_hi: top of the colour scale.  The overlay of frame k needs the estimated displacement of frame k (frame-0
    grid): its inverse is found by fixed-point iteration, then gmax of the frame-0 grid is sampled there."""
    import cv2

    cmap = colormaps["inferno"]
    x0, x1, y0, y1 = crop_window(scene)
    w, h = x1 - x0, y1 - y0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"cannot open {video_path}")
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    outline = _outline_points(roi_polygon)
    poly = np.asarray(roi_polygon, dtype=float)
    rx0, ry0 = int(np.floor(poly[:, 0].min())), int(np.floor(poly[:, 1].min()))
    rx1, ry1 = int(np.ceil(poly[:, 0].max())) + 1, int(np.ceil(poly[:, 1].max())) + 1
    s_k = scene.settlements()
    bar_h = int(0.55 * h)
    label_h = 34
    fs = float(np.clip(w / 1700.0, 0.32, 0.6))  # font scale
    out = np.zeros((h + label_h, 2 * w, 3), np.uint8)
    with VideoWriter(out_path, 2 * w, h + label_h, scene.fps, crf=14) as vw:
        for k in range(scene.n_frames):
            ok, bgr = cap.read()
            if not ok:
                raise OSError(f"{video_path}: frame {k} is missing")
            frame = bgr[y0:y1, x0:x1].copy()
            right = frame.copy()
            if k > 0:
                d = os.path.join(track_dir, f"frame_{k:05d}")
                # the inversion uses the unmasked estimate (valid everywhere in the crop); gmax decides what is drawn
                rux = np.load(os.path.join(d, "raw_ux.npy")).astype(np.float64)
                ruy = np.load(os.path.join(d, "raw_uy.npy")).astype(np.float64)
                gm = np.load(os.path.join(d, "gmax.npy")).astype(np.float64)
                roi_sel = np.s_[ry0:ry1, rx0:rx1]
                # only the part of the frame the ROI's soil can have moved into (the rest is air, wall, footing)
                pad = int(np.ceil(np.nanmax(np.abs(np.concatenate([rux[roi_sel], ruy[roi_sel]]))))) + 6
                sx0, sx1 = max(rx0 - pad, x0) - x0, min(rx1 + pad, x1) - x0
                sy0, sy1 = max(ry0 - pad, y0) - y0, min(ry1 + pad, y1) - y0
                X, Y = invert_displacement(rux, ruy, xx[sy0:sy1, sx0:sx1], yy[sy0:sy1, sx0:sx1], tol=0.03, maxit=25)
                g = np.full((h, w), np.nan)
                g[sy0:sy1, sx0:sx1] = sample_bilinear(gm, X, Y)
                m = np.isfinite(g)
                rgb = (cmap(np.clip(g / gmax_hi, 0, 1))[..., :3] * 255.0)[..., ::-1]  # BGR
                # transparent where invalid, and fading out toward zero strain so the soil stays visible
                a = np.where(m, alpha * np.clip(np.nan_to_num(g) / (0.12 * gmax_hi), 0.0, 1.0), 0.0)[..., None]
                right = ((1 - a) * right + a * rgb).astype(np.uint8)
                px = outline[:, 0] + sample_bilinear(rux, outline[:, 0], outline[:, 1])
                py = outline[:, 1] + sample_bilinear(ruy, outline[:, 0], outline[:, 1])
            else:
                px, py = outline[:, 0], outline[:, 1]
            good = np.isfinite(px) & np.isfinite(py)
            if good.sum() > 2:
                pts = np.stack([px[good] - x0, py[good] - y0], axis=1)
                cv2.polylines(right, [np.round(pts * 16).astype(np.int32)], True, (255, 255, 255), 1, cv2.LINE_AA, shift=4)
            out[:] = 0
            out[label_h:, :w] = frame
            out[label_h:, w:] = right
            _put(out, "input video", (8, 23), fs)
            _put(out, "estimated gmax at the soil particles' current positions", (w + 8, 23), fs)
            txt = f"frame {k}  settlement {s_k[k]:.1f} px"
            tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0]
            _put(out, txt, (w - tw - 8, 23), fs)
            # colour bar (the colour at full opacity), ticks 0 and the top of the scale at its left, a title above it
            bar_w = max(int(round(14 * w / 1000)), 8)
            bx, by = 2 * w - 3 * bar_w - 20, label_h + 12 + int(36 * fs / 0.6)
            for i in range(bar_h):
                out[by + i, bx:bx + bar_w] = (np.array(cmap(1 - i / (bar_h - 1))[:3]) * 255)[::-1]
            tf = 0.9 * fs
            for text, y in ((f"{gmax_hi:g}", by + 8), ("0", by + bar_h)):
                tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, tf, 1)[0][0]
                _put(out, text, (bx - tw - 5, y), tf)
            _put(out, "gmax", (bx - 6, by - 14), tf)
            vw.write(out[..., ::-1])
            if progress:
                progress(k, scene.n_frames)
    cap.release()
