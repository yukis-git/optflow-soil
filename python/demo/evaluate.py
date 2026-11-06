"""Comparison of the tracker's output (one directory per frame, frame 0 grid) with the truth of the demo scene."""
import json
import os

import numpy as np
from scipy import ndimage as ndi

from optflow_ref.config import Config
from optflow_ref.io import polygon_mask

from .footing import Scene, Truth

STRAIN_KEYS = ("Exx", "Eyy", "Exy", "gmax", "vol")
BAND_FRACTION = 0.4  # shear bands: where the true gmax of the last frame exceeds this share of its peak


class Evaluation:
    """Per-frame records and the fields of a few kept frames.

    rows[i]: dict of the frame k = i + 1 (frame 0 is the reference and has no result).  kept[k]: dict of the
    estimated fields ('ux', 'uy', 'gmax', 'vol', ... , NaN = invalid), the true ones ('t_ux', ...) and the
    ROI mask, cropped to the ROI's bounding box."""

    def __init__(self, scene: Scene, truth: Truth, roi, interior, band, rigid):
        self.scene, self.truth = scene, truth
        self.roi, self.interior, self.band, self.rigid = roi, interior, band, rigid
        self.rows = []
        self.kept = {}
        self.extra = {}  # facts about the last frame (where the errors are)


def roi_masks(scene: Scene, truth: Truth, cfg: Config):
    """ROI mask on the bounding box, its interior (at least the tracker's guard distance from the ROI's edge:
    the pixels that can be valid at all), the shear-band mask from the true gmax of the last frame, and the
    rigid-wedge pixels used for the estimated settlement."""
    x0, y0, x1, y1 = truth.bbox
    roi = polygon_mask((scene.height, scene.width), scene.roi_polygon())[y0:y1, x0:x1]
    # distance to the ROI's exterior; pad so the bounding box's own border counts as exterior too
    dist = ndi.distance_transform_edt(np.pad(roi, 1))[1:-1, 1:-1]
    guard = cfg.edge_guard_px()
    interior = dist > guard
    gm = truth.frame(float(scene.settlement(scene.n_frames - 1)))["gmax"]
    peak = float(np.percentile(gm[interior], 99.9))
    band = interior & (gm >= BAND_FRACTION * peak)
    rigid = truth.rigid_mask() & (dist > guard + 3.0)
    return roi, interior, band, rigid


def _load(d, name, bbox):
    x0, y0, x1, y1 = bbox
    return np.load(os.path.join(d, f"{name}.npy"), mmap_mode="r")[y0:y1, x0:x1].astype(np.float64)


def _rms(a):
    return float(np.sqrt(np.mean(a ** 2))) if a.size else float("nan")


def evaluate(scene: Scene, truth: Truth, track_dir, keep=(), progress=None) -> Evaluation:
    """Walk over frame_00001 ... of a tracker output directory (the CLI's layout)."""
    with open(os.path.join(track_dir, "config.json")) as f:
        cfg = Config.from_json(f.read())
    roi, interior, band, rigid = roi_masks(scene, truth, cfg)
    ev = Evaluation(scene, truth, roi, interior, band, rigid)
    bbox = truth.bbox
    n_roi, n_int = int(roi.sum()), int(interior.sum())
    prev_ref = 0
    for k in range(1, scene.n_frames):
        d = os.path.join(track_dir, f"frame_{k:05d}")
        s_true = float(scene.settlement(k))
        t = truth.frame(s_true)
        est = {name: _load(d, name, bbox) for name in ("ux", "uy") + STRAIN_KEYS}
        with open(os.path.join(d, "timing.json")) as f:
            timing = json.load(f)
        valid = roi & np.isfinite(est["ux"]) & np.isfinite(est["uy"])
        ex, ey = (est["ux"] - t["ux"])[valid], (est["uy"] - t["uy"])[valid]
        e = np.hypot(ex, ey)
        rv = rigid & valid
        # settlement_true: the footing's settlement s.  The soil in the wedge settles slightly less (the pre-peak flow of
        # the field decays with depth), so the tracker's estimate (median u_y over the wedge pixels) is compared with the
        # true median over the same pixels, settlement_ref
        row = {"k": k, "settlement_true": s_true,
               "settlement_est": float(np.median(est["uy"][rv])) if rv.any() else float("nan"),
               "settlement_ref": float(np.median(t["uy"][rv])) if rv.any() else float("nan"),
               "disp_rms": _rms(e), "disp_p99": float(np.percentile(e, 99)) if e.size else float("nan"),
               "disp_max": float(e.max()) if e.size else float("nan"),
               "valid_frac": float(valid.sum()) / n_roi, "valid_frac_eff": float((valid & interior).sum()) / n_int,
               "reference": int(timing["reference"]), "mean_conf": float(timing["mean_conf"]),
               "t_coarse": float(timing["coarse"]), "t_fine": float(timing["fine"]), "t_strain": float(timing["strain"])}
        row["ref_update"] = row["reference"] != prev_ref
        prev_ref = row["reference"]
        row["settlement_err"] = row["settlement_est"] - row["settlement_ref"]
        for key in STRAIN_KEYS:
            ok = valid & np.isfinite(est[key])
            diff = est[key] - t[key]
            row[f"{key}_rms_band"] = _rms(diff[ok & band])
            row[f"{key}_rms_out"] = _rms(diff[ok & ~band & interior])
        ok = valid & np.isfinite(est["gmax"]) & interior
        row["gmax_peak_est"] = float(np.percentile(est["gmax"][ok], 99.9)) if ok.any() else float("nan")
        row["gmax_peak_true"] = float(np.percentile(t["gmax"][ok], 99.9)) if ok.any() else float("nan")
        row["gmax_peak_true_all"] = float(np.percentile(t["gmax"][interior], 99.9))  # wherever the tracker is valid or not
        row["rigid_valid_frac"] = float(rv.sum()) / max(int(rigid.sum()), 1)  # share of the wedge pixels with a valid estimate
        # displacement error inside / outside the bands
        vb, vo = valid & band, valid & ~band & interior
        row["disp_rms_band"] = _rms(np.hypot(est["ux"] - t["ux"], est["uy"] - t["uy"])[vb])
        row["disp_rms_out"] = _rms(np.hypot(est["ux"] - t["ux"], est["uy"] - t["uy"])[vo])
        ev.rows.append(row)
        if k == scene.n_frames - 1:
            ev.extra = _last_frame_facts(scene, truth, est, t, valid, band, interior)
        if k in keep:
            ev.kept[k] = {**est, **{f"t_{n}": a for n, a in t.items()}, "valid": valid}
        if progress:
            progress(k, scene.n_frames - 1)
    return ev


def _last_frame_facts(scene, truth, est, t, valid, band, interior):
    """Where the last frame's errors are: shares of the squared displacement error and positions of the largest
    errors, in the footing frame (x' from the centre line, y' down from the initial surface)."""
    x0, y0, _, _ = truth.bbox
    p = scene.params
    e2 = np.where(valid, (est["ux"] - t["ux"]) ** 2 + (est["uy"] - t["uy"]) ** 2, 0.0)
    ok = valid & interior
    ge = np.where(valid & np.isfinite(est["gmax"]), np.abs(est["gmax"] - t["gmax"]), 0.0)

    def loc(a):
        i, j = np.unravel_index(int(np.argmax(a)), a.shape)
        return {"x": float(x0 + j - p.xc), "y": float(y0 + i - p.y0), "value": float(a[i, j])}

    gx, gy = np.meshgrid(np.arange(x0, x0 + e2.shape[1]) - p.xc, np.arange(y0, y0 + e2.shape[0]) - p.y0)
    near = np.hypot(gx, gy) < 2.5 * p.b
    return {"band_pixel_share": float((band & ok).sum() / max(ok.sum(), 1)),
            "band_sq_error_share": float(e2[band & ok].sum() / max(e2[ok].sum(), 1e-300)),
            "near_footing_pixel_share": float((near & ok).sum() / max(ok.sum(), 1)),
            "near_footing_sq_error_share": float(e2[near & ok].sum() / max(e2[ok].sum(), 1e-300)),
            "disp_error_max_at": {**loc(np.sqrt(e2)), "value": float(np.sqrt(e2.max()))},
            "gmax_error_max_at": loc(ge)}


def summarize(ev: Evaluation):
    """Headline numbers of an Evaluation.  The settlement estimate is NaN at a frame where no pixel of the wedge is valid
    (strong bands swallow a narrow wedge); settlement_*_last_valid belong to the last frame that has an estimate."""
    rows = ev.rows
    last = rows[-1]
    worst = max(rows, key=lambda r: r["disp_rms"])
    with_est = [r for r in rows if np.isfinite(r["settlement_est"])]
    lv = with_est[-1] if with_est else None
    errs = [abs(r["settlement_err"]) for r in with_est]
    return {
        "frames": len(rows) + 1,
        "disp_rms_final": last["disp_rms"], "disp_rms_worst": worst["disp_rms"], "disp_rms_worst_frame": worst["k"],
        "disp_p99_final": last["disp_p99"], "disp_max_final": last["disp_max"],
        "settlement_true_final": last["settlement_true"], "settlement_ref_final": last["settlement_ref"],
        "settlement_est_final": last["settlement_est"],
        "settlement_last_valid_frame": lv["k"] if lv else None,
        "settlement_true_last_valid": lv["settlement_true"] if lv else float("nan"),
        "settlement_ref_last_valid": lv["settlement_ref"] if lv else float("nan"),
        "settlement_est_last_valid": lv["settlement_est"] if lv else float("nan"),
        "settlement_err_max_abs": max(errs) if errs else float("nan"),
        "rigid_valid_frac_final": last["rigid_valid_frac"],
        "valid_frac_min": min(r["valid_frac"] for r in rows),
        "valid_frac_eff_min": min(r["valid_frac_eff"] for r in rows),
        "reference_updates": int(sum(r["ref_update"] for r in rows)),
        "gmax_rms_band_final": last["gmax_rms_band"], "gmax_rms_out_final": last["gmax_rms_out"],
        "gmax_peak_est_final": last["gmax_peak_est"], "gmax_peak_true_final": last["gmax_peak_true"],
        "gmax_peak_true_all_final": last["gmax_peak_true_all"],
        "band_pixels": int(ev.band.sum()), "interior_pixels": int(ev.interior.sum()), "roi_pixels": int(ev.roi.sum()),
    }
