"""Temporal averaging front-end on a synthetic high-frame-rate video with truth (report section "temporal averaging").

The small footing scene is rendered at `--fps-ratio` frames per sampled frame (30 fps video, step 15 -> ratio 15) with
per-frame Gaussian noise (--noise, fraction of full range) and gain flicker, written as 8-bit PNG files and read back
through optflow_ref.io.iter_frames(tavg, align) -- the same front-end `python -m optflow_ref seq` uses. Each variant is
tracked in chain and hybrid mode; per sampled frame: valid fraction, trimmed RMS error of the valid pixels, and the error
on the pixels valid here but not in the tavg = 0 run of the same mode.

  python bench/tavg_eval.py --out ../results --tag _small
"""
import argparse
import dataclasses
import datetime
import json
import os
import sys
import tempfile
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.hybrid_eval import trimmed_rms  # noqa: E402
from demo import footing  # noqa: E402
from optflow_ref.config import Config  # noqa: E402
from optflow_ref.io import iter_frames, polygon_mask  # noqa: E402
from optflow_ref.track import track_sequence  # noqa: E402


def render_video(d, n_sampled, ratio, s_final, noise, flicker, seed=1):
    """Write the frames of a ratio * (n_sampled - 1) + 1 frame video into directory d; returns (scene, truth, roi)."""
    sc = footing.make_scene("small", n_frames=ratio * (n_sampled - 1) + 1, s_final=s_final, seed=seed)
    sc = dataclasses.replace(sc, noise=noise, flicker=flicker)
    r = footing.Renderer(sc)
    for k, s in enumerate(sc.settlements()):
        Image.fromarray(r.frame(k, float(s), np.random.default_rng([sc.seed, 1000 + k]))).save(os.path.join(d, f"f{k:04d}.png"))
    return sc, footing.Truth(sc), polygon_mask((sc.height, sc.width), sc.roi_polygon())


def evaluate(res, truth, sc, roi, k_hr, other=None):
    x0, y0, x1, y1 = truth.bbox
    fr = truth.frame(float(sc.settlement(k_hr)))
    H, W = sc.height, sc.width
    gx, gy = (np.full((H, W), np.nan) for _ in range(2))
    gx[y0:y1, x0:x1], gy[y0:y1, x0:x1] = fr["ux"], fr["uy"]
    m = roi & np.isfinite(gx)
    ex, ey = res.ux - gx, res.uy - gy
    v = m & np.isfinite(ex) & np.isfinite(ey)
    e = np.hypot(ex, ey)
    out = {"valid_frac": float(v.sum() / m.sum()), "rms_trim": trimmed_rms(e[v]) if v.any() else None,
           "rms": float(np.sqrt(np.mean(e[v] ** 2))) if v.any() else None}
    if other is not None:
        ov = m & np.isfinite(other.ux)
        new, both = v & ~ov, v & ov
        out |= {"n_new": int(new.sum()), "rms_new": trimmed_rms(e[new]) if new.any() else None,
                "n_lost": int((ov & ~v).sum()), "n_both": int(both.sum()),
                "rms_both": trimmed_rms(e[both]) if both.any() else None}
    return out


VARIANTS = [("tavg0", 0, False, "affine"), ("plain3", 3, False, "affine"), ("plain7", 7, False, "affine"),
            ("aligned3", 3, True, "affine"), ("aligned7", 7, True, "affine"), ("dense3", 3, True, "dense"),
            ("dense7", 7, True, "dense")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sampled", type=int, default=6)
    ap.add_argument("--fps-ratio", type=int, default=15, help="video frames per sampled frame (step)")
    ap.add_argument("--s-final", type=float, default=6.0)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--flicker", type=float, default=0.006)
    ap.add_argument("--modes", default="chain,hybrid")
    ap.add_argument("--only", default="", help="comma-separated variant names to run (tavg0 is always needed for 'new')")
    ap.add_argument("--new-vs", default="", help="mode whose tavg0 run is the baseline of 'new' / 'both' for every mode "
                    "(default: each mode's own tavg0), e.g. 'chain' = the shipped default")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "results"))
    a = ap.parse_args()
    rec = {"date": datetime.date.today().isoformat(), "n_sampled": a.n_sampled, "step": a.fps_ratio, "noise": a.noise,
           "flicker": a.flicker, "s_final": a.s_final}
    only = set(a.only.split(",")) if a.only else None
    with tempfile.TemporaryDirectory() as d:
        sc, truth, roi = render_video(d, a.n_sampled, a.fps_ratio, a.s_final, a.noise, a.flicker)
        cfg0 = Config(max_disp=32)
        base = {}
        for mode in a.modes.split(","):
            for name, m, al, model in VARIANTS:
                if only and name not in only:
                    continue
                cfg = dataclasses.replace(cfg0, seq_mode=mode, tavg=m, tavg_align=al)
                t0 = time.perf_counter()
                frames = list(iter_frames(d, 0, None, a.fps_ratio, 1, m, al, Config(max_disp=8.0, tavg_model=model), lambda shape: roi))
                t_avg = time.perf_counter() - t0
                per, kept = [], {}
                for k, res in track_sequence(frames, cfg, mask=roi):
                    ref_mode = a.new_vs or mode
                    e = evaluate(res, truth, sc, roi, k * a.fps_ratio,
                                 base.get((ref_mode, k)) if (name != "tavg0" or ref_mode != mode) else None)
                    e["frame"] = k
                    per.append(e)
                    kept[k] = res
                if name == "tavg0":
                    base.update({(mode, k): r for k, r in kept.items()})
                rec.setdefault(mode, {})[name] = {"tavg": m, "align": al, "model": model, "seconds_front_end": t_avg, "per_frame": per}
                print(mode, name, [(p["frame"], round(p["valid_frac"], 3), None if p["rms_trim"] is None else round(p["rms_trim"], 4))
                                   for p in per], flush=True)
    path = os.path.join(a.out, f"{rec['date']}_tavg_eval{a.tag}.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
