"""Chain vs hybrid sequence tracking on synthetic sequences with ground truth (spec 2026-11-02-hybrid-sequence-design.md
section 5, revised: v2 direct-first). One JSON per case in <out>/: per frame trimmed RMS error, valid fraction, gmax
error; for hybrid also the fallback share and the comparison with chain (error on the pixels valid in both modes, on
the pixels valid only in hybrid, chain-valid pixels lost).

  python bench/hybrid_eval.py --cases sequence_20,crack_seq --mode both
  python bench/hybrid_eval.py --cases long_weak --frames 200 --size 384x216 --mode hybrid

Cases: sequence_20, video_20 (MJPG round trip), crack_seq, static_scratches_seq, illumination_seq (the pair cases of
bench/cases.py with their field scaled over 6 frames), footing_small (demo/footing.py small scene) and long_weak (new:
weak texture, rigid settlement + a vertical shear band, a static additive pattern and a slowly changing low-frequency
"reflection" layer).
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
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.cases import all_cases  # noqa: E402
from bench.run_synth import SEED_DEF, SEED_REF, frame as synth_frame, write_video  # noqa: E402
from optflow_ref import synth  # noqa: E402
from optflow_ref.config import Config  # noqa: E402
from optflow_ref.io import iter_frames, polygon_mask  # noqa: E402
from optflow_ref.metrics import eval_mask, true_strain  # noqa: E402
from optflow_ref.track import track_sequence  # noqa: E402

TRIM = 0.01  # the worst 1 % of the valid pixels are left out of the trimmed RMS


class SettleBand:
    """Rigid settlement S (+ drift tx) plus a vertical shear band: x = X + tx,  y = Y + S + 0.5 delta tanh(-(X - xc) / w).
    The motion is a function of X alone in y, so the inverse is exact. Same interface as synth.Affine."""

    def __init__(self, S, tx, delta, xc, w):
        self.S, self.tx, self.delta, self.xc, self.w = S, tx, delta, xc, w

    def _uy(self, X):
        return self.S + 0.5 * self.delta * np.tanh(-(X - self.xc) / self.w)

    def fwd(self, X, Y):
        return X + self.tx, Y + self._uy(X)

    def inv(self, x, y):
        X = x - self.tx
        return X, y - self._uy(X)

    def grad(self, X, Y):
        z = np.zeros(np.shape(X))
        return z, z, -0.5 * self.delta / self.w / np.cosh((X - self.xc) / self.w) ** 2, z


def scaled_affine(f, s):
    """The affine field f applied s times as much (s = 1 is f): x = c + (I + s (A - I)) (X - c) + s t."""
    return synth.Affine(np.eye(2) + s * (f.A - np.eye(2)), s * f.t, f.c)


def smooth_noise(h, w, sigma, seed):
    a = gaussian_filter(np.random.default_rng(seed).normal(size=(h, w)), sigma, mode="wrap")
    return a / a.std()


class Seq:
    """A sequence with truth: frames (list of HxW float arrays), fields (true deformation of frame k relative to frame 0,
    k = 1..), margin of the evaluation mask, roi (mask for track_sequence) and the config to track with."""

    def __init__(self, name, frames, fields, margin, cfg, roi=None, shape_note=""):
        self.name, self.frames, self.fields, self.margin, self.cfg, self.roi = name, frames, fields, margin, cfg, roi
        self.note = shape_note

    def truth(self, k):
        f = self.fields[k - 1]
        H, W = self.frames[0].shape
        gx, gy = synth.displacement(f, H, W)
        return gx, gy, true_strain(f, (H, W))["gmax"], eval_mask((H, W), f, self.margin)


def case_from_bench(name, n, seed, nframes=None, video=False, scale_to=None):
    """One of bench/cases.py. scale_to: a pair case becomes a sequence of that many frames, frame k deforming by
    k / scale_to of the case's (affine) field."""
    base = next(c for c in all_cases(n) if c.name == name)
    fields = base.fields
    if scale_to:
        fields = [scaled_affine(base.fields[0], k / scale_to) for k in range(1, scale_to + 1)]
    case = dataclasses.replace(base, fields=fields)
    pattern = synth.make_pattern(n, n, seed=seed)
    frames = [synth_frame(pattern, n, None, case, SEED_REF)]
    frames += [synth_frame(pattern, n, f, case, SEED_DEF + k) for k, f in enumerate(fields)]
    if video:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "seq.avi")
            write_video(frames, path)
            frames = list(iter_frames(path))
    return Seq(name, frames, fields, base.margin, base.cfg)


def case_footing(frames_n=8, s_final=4.0):
    from demo import footing
    sc = footing.make_scene("small", n_frames=frames_n, s_final=s_final)
    r = footing.Renderer(sc)
    frames = []
    for k, s in enumerate(sc.settlements()):
        rgb = r.frame(k, float(s), np.random.default_rng([sc.seed, 1000 + k]))
        frames.append((rgb.astype(np.float64) / 255.0) @ footing.LUMA)
    truth = footing.Truth(sc)
    x0, y0, x1, y1 = truth.bbox
    roi = polygon_mask(frames[0].shape, sc.roi_polygon())

    class FootingSeq(Seq):
        def truth(self, k):
            fr = truth.frame(float(sc.settlement(k)))
            H, W = self.frames[0].shape
            gx, gy, gm = (np.full((H, W), np.nan) for _ in range(3))
            gx[y0:y1, x0:x1], gy[y0:y1, x0:x1], gm[y0:y1, x0:x1] = fr["ux"], fr["uy"], fr["gmax"]
            return gx, gy, gm, roi & np.isfinite(gx)

    return FootingSeq("footing_small", frames, [None] * (frames_n - 1), 0, Config(max_disp=32), roi=roi)


def case_long_weak(n_frames=200, size=(384, 216), seed=5, contrast=0.5, refl_amp=0.02, noise=0.004, scr_amp=(0.0013, 0.004)):
    """Video-1-like: weak texture, rigid settlement + shear band, static additive pattern (scratches), reflection layer
    changing slowly (two fixed low-frequency fields cross-faded over ~100 frames; they drift too)."""
    W, H = size
    pat = synth.make_pattern(H, W, seed=seed)
    static = synth.scratches(H, W, 18, seed + 1, amp=scr_amp)
    R = [smooth_noise(H + 40, W + 40, 6.0, seed + 10 + i) for i in range(3)]
    S_tot, tx_tot, delta_tot = 0.12 * H, 0.04 * W, 3.0  # px over the whole sequence; the band peak shear is 0.5 delta / w = 0.1
    xc = 0.62 * W
    fields, frames = [], []
    for k in range(n_frames):
        a = k / (n_frames - 1)
        f = SettleBand(S_tot * a, tx_tot * a, delta_tot * a, xc, 15.0)
        img = 0.5 + contrast * (synth.render(pat, H, W, None if k == 0 else f) - 0.5) + static
        ph = 2 * np.pi * k / 100.0
        wts = [np.cos(ph), np.sin(ph), 0.5 * np.cos(0.37 * ph)]
        dx, dy = int(round(0.05 * k)), int(round(0.03 * k))
        refl = sum(w * r[20 + dy:20 + dy + H, 20 + dx:20 + dx + W] for w, r in zip(wts, R))
        img = img + refl_amp * refl / np.sqrt(sum(w * w for w in wts))
        img = synth.degrade(img, seed=1000 + k, noise=noise, bits=8)
        frames.append(img)
        if k:
            fields.append(f)
    cfg = Config(max_disp=32)
    return Seq("long_weak", frames, fields, 40, cfg)


def trimmed_rms(e, frac=TRIM):
    e = np.sort(e)
    keep = e[:max(1, int(np.ceil(len(e) * (1 - frac))))]
    return float(np.sqrt(np.mean(keep ** 2)))


def _rms(e2):
    return (trimmed_rms(np.sqrt(e2)) if e2.size else None)


def metrics(res, truth, other=None):
    """Errors of one mode's result at one frame. other = the chain result of the same frame (for the hybrid): adds the
    comparison on the pixels valid in both modes (rms_both, rms_other_both), on the pixels valid only here (rms_new,
    n_new) and the count of chain-valid pixels lost (n_lost)."""
    gx, gy, gm, m = truth
    ex, ey = res.ux - gx, res.uy - gy
    valid = m & np.isfinite(ex) & np.isfinite(ey)
    out = {"valid_frac": float(valid.sum() / m.sum()), "n_valid": int(valid.sum())}
    e2full = ex ** 2 + ey ** 2
    out["rms_trim"] = _rms(e2full[valid])
    out["rms"] = float(np.sqrt(e2full[valid].mean())) if valid.any() else None
    g = valid & np.isfinite(res.strain["gmax"]) & np.isfinite(gm)
    out["gmax_rms"] = float(np.sqrt(np.mean((res.strain["gmax"][g] - gm[g]) ** 2))) if g.any() else None
    if res.fb is not None:
        out["fallback_frac"] = res.timing["fallback_frac"]
        out["fb_valid_frac"] = float((valid & (res.fb > 0)).sum() / m.sum())
    if other is not None:
        ox, oy = other.ux - gx, other.uy - gy
        ov = m & np.isfinite(ox) & np.isfinite(oy)
        both, new = valid & ov, valid & ~ov
        oe2 = ox ** 2 + oy ** 2
        out |= {"n_both": int(both.sum()), "rms_both": _rms(e2full[both]), "rms_chain_both": _rms(oe2[both]),
                "n_new": int(new.sum()), "rms_new": _rms(e2full[new]), "n_lost": int((ov & ~valid).sum()),
                "rms_new_rms_nontrim": float(np.sqrt(e2full[new].mean())) if new.any() else None}
    return out


def run(seq, mode, cfg_over, every=1, chain_by_k=None):
    cfg = dataclasses.replace(seq.cfg, seq_mode=mode, **cfg_over)
    per, t0, keep = [], time.perf_counter(), {}
    for k, res in track_sequence(seq.frames, cfg, mask=seq.roi):
        if k % every and k != len(seq.frames) - 1:
            continue
        e = metrics(res, seq.truth(k), None if chain_by_k is None else chain_by_k.get(k))
        e |= {"frame": k, "reference": res.timing["reference"]}
        keep[k] = res
        per.append(e)
        print(f"{seq.name:22s} {mode:6s} k={k:3d} rms_trim={e['rms_trim']} valid={e['valid_frac']:.3f} "
              f"fb={e.get('fallback_frac', '-')} t={time.perf_counter() - t0:.0f}s", flush=True)
    return {"config": dataclasses.asdict(cfg), "seconds": time.perf_counter() - t0, "per_frame": per}, keep


def build(name, a):
    n = a.n
    if name in ("sequence_20", "video_20"):
        return case_from_bench(name, n, 0, video=name == "video_20")
    if name in ("crack_seq", "static_scratches_seq", "illumination_seq"):
        return case_from_bench(name[:-4], n, 0, scale_to=a.pair_frames)
    if name == "footing_small":
        return case_footing()
    if name == "long_weak":
        w, h = (int(x) for x in a.size.split("x"))
        return case_long_weak(a.frames, (w, h))
    if name == "long_weak_hard":  # weaker texture, stronger reflection and noise: the confidence falls below conf_min
        w, h = (int(x) for x in a.size.split("x"))
        s = case_long_weak(a.frames, (w, h), contrast=0.25, refl_amp=0.04, noise=0.008, scr_amp=(0.0013, 0.004))
        s.name = name
        return s
    raise SystemExit(f"unknown case {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="sequence_20")
    ap.add_argument("--mode", choices=("chain", "hybrid", "both"), default="both")
    ap.add_argument("--n", type=int, default=256, help="image size of the bench/cases.py cases")
    ap.add_argument("--pair-frames", type=int, default=6, help="frames of the pair cases turned into sequences")
    ap.add_argument("--frames", type=int, default=200, help="long_weak: number of frames")
    ap.add_argument("--size", default="384x216", help="long_weak: WxH")
    ap.add_argument("--every", type=int, default=1, help="evaluate every n-th frame (and the last)")
    ap.add_argument("--set", default="", help="hybrid config overrides, e.g. fb_tol=0.4,fb_max=6")
    ap.add_argument("--variants", default="", help="extra hybrid variants run against the same chain result, "
                    "'name:k=v,k=v;name2:k=v' (JSON key = name, e.g. 'nogate:fb_min_support=0')")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "results"))
    a = ap.parse_args()
    over = {k: (int(v) if v.lstrip("-").isdigit() else float(v)) for k, v in (kv.split("=") for kv in a.set.split(",") if kv)}
    os.makedirs(a.out, exist_ok=True)
    for name in a.cases.split(","):
        seq = build(name, a)
        rec = {"case": name, "date": datetime.date.today().isoformat(), "frames": len(seq.frames),
               "size": list(seq.frames[0].shape), "trim": TRIM, "hybrid_overrides": over}
        chain_res = None
        for mode in (("chain", "hybrid") if a.mode == "both" else (a.mode,)):
            rec[mode], kept = run(seq, mode, over if mode == "hybrid" else {}, a.every, chain_res if mode == "hybrid" else None)
            chain_res = kept if mode == "chain" else chain_res
        for v in filter(None, a.variants.split(";")):
            vname, _, vset = v.partition(":")
            vover = over | {k: (int(x) if x.lstrip("-").isdigit() else float(x)) for k, x in (kv.split("=") for kv in vset.split(",") if kv)}
            rec[vname], _ = run(seq, "hybrid", vover, a.every, chain_res)
        path = os.path.join(a.out, f"{rec['date']}_hybrid_eval_{name}{a.tag}.json")
        with open(path, "w") as f:
            json.dump(rec, f, indent=1)
        print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
