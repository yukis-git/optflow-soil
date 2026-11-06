"""Overlay plots of displacement, confidence and strain on the reference image.

Colour scales: every signed field (u_x, u_y, det F - 1, E_xy) uses a diverging map centred on zero, so
zero is always white; gamma_max (>= 0) starts from white at zero; confidence is fixed to [0, 1]. A set of
maps (the pairs 1->2, 2->3, 3->4, or the cumulative frames of a sequence) shares one scale per field,
so one colour means one value in every figure of the set."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# key, label, colormap, scale mode ("sym": [-v, v]; "pos": [0, v]; tuple: fixed)
PANELS = [("ux", "u_x [px]", "RdBu_r", "sym"), ("uy", "u_y [px]", "RdBu_r", "sym"),
          ("conf", "confidence", "magma", (0.0, 1.0)), ("gmax", "max shear strain gamma_max", "Reds", "pos"),
          ("vol", "volumetric strain det F - 1", "RdBu_r", "sym"), ("Exy", "shear strain E_xy", "RdBu_r", "sym")]


def nice_ceil(v):
    """Smallest number of the form {1, 1.5, 2, 2.5, 3, 4, 5, 6, 8} x 10^k that is >= v (v > 0; float32
    rounding of v is tolerated), for round colourbar ends."""
    v = float(v)
    if not np.isfinite(v) or v <= 0:
        return 1.0
    e = 10.0 ** np.floor(np.log10(v))
    for m in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if m * e >= v * (1 - 1e-6):
            return float(round(m * e, 12))
    return float(10.0 * e)


def shared_limits(field_sets, pct=99.0):
    """One (lo, hi) per panel key over all field dicts: the pct-th percentile of |value| over every finite
    value of the set, rounded up by nice_ceil."""
    lims = {}
    for key, _, _, mode in PANELS:
        if isinstance(mode, tuple):
            lims[key] = mode
            continue
        vals = [np.abs(f[key][np.isfinite(f[key])]) for f in field_sets]
        vals = np.concatenate(vals) if vals else np.zeros(0)
        v = nice_ceil(np.percentile(vals, pct)) if vals.size else 1.0
        lims[key] = (-v, v) if mode == "sym" else (0.0, v)
    return lims


def fields_of(res):
    return {"ux": res.ux, "uy": res.uy, "conf": res.conf, **res.strain}


def _overlay(ax, ref_img, field, cmap, lo, hi):
    ax.imshow(ref_img, cmap="gray", aspect="auto")
    return ax.imshow(np.ma.masked_invalid(field), cmap=cmap, vmin=lo, vmax=hi, alpha=0.85, aspect="auto")


def _blocks_figure(columns, n, shape, title=""):
    """Fixed-geometry figure (sizes in inches, so every strip and colourbar lines up): `columns` is a list
    of columns, each a list of block labels; a block is a title, n image strips with the image's aspect
    ratio and a colourbar axis to the right of them. Returns (fig, {(column, block): (strip_axes, cax)})."""
    h, w = shape
    strip_w = 7.2 if w >= 1.6 * h else 4.5
    sh = strip_w * h / w
    title_h, vgap, bgap, top = 0.3, 0.06, 0.22, (0.5 if title else 0.1)
    cb_gap, cb_w, lab_w, cgap, lm = 0.08, 0.16, 0.8, 0.3, 0.1
    colw = strip_w + cb_gap + cb_w + lab_w
    nb = max(len(c) for c in columns)
    block_h = title_h + n * sh + (n - 1) * vgap
    fw = lm + len(columns) * colw + (len(columns) - 1) * cgap
    fh = top + nb * block_h + (nb - 1) * bgap + 0.1
    fig = plt.figure(figsize=(fw, fh))
    if title:
        fig.suptitle(title, y=1 - 0.12 / fh, va="top")
    out = {}
    for c, labels in enumerate(columns):
        x0 = lm + c * (colw + cgap)
        for b, label in enumerate(labels):
            yt = fh - top - b * (block_h + bgap)
            fig.text((x0 + strip_w / 2) / fw, (yt - 0.04) / fh, label, ha="center", va="top", fontsize=10)
            axes = []
            for i in range(n):
                y1 = yt - title_h - i * (sh + vgap)
                ax = fig.add_axes([x0 / fw, (y1 - sh) / fh, strip_w / fw, sh / fh])
                ax.set_axis_off()
                axes.append(ax)
            ybot = yt - title_h - n * sh - (n - 1) * vgap
            cax = fig.add_axes([(x0 + strip_w + cb_gap) / fw, ybot / fh, cb_w / fw, (yt - title_h - ybot) / fh])
            out[(c, b)] = (axes, cax)
    return fig, out


def save_maps(ref_img, res, path, title="", limits=None):
    """One figure: the reference image and the six panels. limits: {key: (lo, hi)} (default: this
    result alone, by the same rule as shared_limits)."""
    fields = res if isinstance(res, dict) else fields_of(res)
    limits = limits or shared_limits([fields])
    h, w = ref_img.shape
    blocks = [("reference " + title, None, None)] + [(label, key, cmap) for key, label, cmap, _ in PANELS]
    ncol = 2 if w < 1.6 * h else 1
    per = int(np.ceil(len(blocks) / ncol))
    columns = [blocks[i:i + per] for i in range(0, len(blocks), per)]
    fig, axs = _blocks_figure([[b[0] for b in col] for col in columns], 1, ref_img.shape)
    for c, col in enumerate(columns):
        for b, (_, key, cmap) in enumerate(col):
            (ax,), cax = axs[(c, b)]
            if key is None:
                ax.imshow(ref_img, cmap="gray", aspect="auto")
                cax.remove()
                continue
            fig.colorbar(_overlay(ax, ref_img, fields[key], cmap, *limits[key]), cax=cax)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_overview(items, path, title="", limits=None):
    """All maps of a set in one figure: a block per field (displacement / confidence on the left, strain
    on the right), one strip per item, one colourbar per block. items: [(ref_img, fields, label)]."""
    limits = limits or shared_limits([f for _, f, _ in items])
    columns = [PANELS[:3], PANELS[3:]]
    fig, axs = _blocks_figure([[p[1] for p in col] for col in columns], len(items), items[0][0].shape, title)
    for c, col in enumerate(columns):
        for b, (key, _, cmap, _) in enumerate(col):
            axes, cax = axs[(c, b)]
            for ax, (ref_img, fields, name) in zip(axes, items):
                im = _overlay(ax, ref_img, fields[key], cmap, *limits[key])
                ax.text(0.006, 0.96, name, transform=ax.transAxes, va="top", fontsize=8,
                        bbox=dict(fc="white", alpha=0.75, lw=0, pad=1.5))
            fig.colorbar(im, cax=cax)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_map_set(items, overview_path=None, overview_title=""):
    """Per-item figures and an optional overview with one shared scale per field.
    items: [(ref_img, res_or_fields, path, title, short_label)]."""
    fsets = [r if isinstance(r, dict) else fields_of(r) for _, r, _, _, _ in items]
    limits = shared_limits(fsets)
    for (ref_img, _, path, title, _), f in zip(items, fsets):
        save_maps(ref_img, f, path, title=title + " [shared scale]", limits=limits)
    if overview_path:
        save_overview([(it[0], f, it[4]) for it, f in zip(items, fsets)], overview_path, overview_title,
                      limits=limits)
    return limits
