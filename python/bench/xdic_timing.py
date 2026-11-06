"""Timing and accuracy of crack_model "xdic": Python reference against the Rust port, same images.

    python/.venv/bin/python python/bench/xdic_timing.py prepare OUT [--size 384] [--alpha 0] [--opening 3] [--slip 1]
    python/.venv/bin/python python/bench/xdic_timing.py python OUT [--runs 1]        # Python base / xdic, with the stage breakdown
    python/.venv/bin/python python/bench/xdic_timing.py metrics OUT RUST_DIR         # Python vs Rust metrics from an `optflow pair` run

`prepare` writes OUT/a.npy, OUT/b.npy (float32; the images of demo/crack_ab_eval.py's Case) and the 16-bit PNGs
OUT/a.png, OUT/b.png the Rust command line reads. The reference is run on the float32-rounded images of the npy files
and the Rust tool on the PNGs, which were made from the same float32 values rounded to 16 bits: for a
bit-exact input use the `bench_pair --files OUT/a.npy,OUT/b.npy` example of optflow-core (timing) and
`optflow pair` on OUT/a.png, OUT/b.png (accuracy; the 16-bit rounding is far below the 0.01 noise).

Thread count: run with OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 for a single-threaded
Python (numpy's FFT is single-threaded anyway) and RAYON_NUM_THREADS=1 for Rust.

The Python stage breakdown re-runs `xdic.refine_xdic` with timers around its parts (a copy of its loop: its output
is checked to be identical to the original's) and times the other stages of `track._track_pair_xdic` with wrappers.
Stages, named as in the Rust `prof` stages: detect (the baseline's validity and strain, the detector), responses
(`masked_reference` and the per-side pre-warped I2: warp, FFT, filters), side solve (extrapolation, moments, per-side
solve), geometry (`_update_geometry`, `_Geo`), finish (`xdic.finish`, `model_gain`, `advance_lines`); the baseline
iterations of `refine_xdic` (pre-warp, warp, responses of I2, inner solves) are reported as "baseline iterations"
and the baseline fine stage before it as "fine".
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from bench.rustcli import write_png16  # noqa: E402
from demo.crack_ab_eval import Case, field_metrics, geometry, line_metrics  # noqa: E402
from optflow_ref import phaseflow as pf  # noqa: E402
from optflow_ref import synth, track, xdic  # noqa: E402
from optflow_ref.config import Config  # noqa: E402
from optflow_ref.interp import bspline_coeffs, gauss_blur, sample, warp  # noqa: E402


def prepare(a):
    os.makedirs(a.out, exist_ok=True)
    case = Case(a.size, a.alpha, a.opening, a.slip)
    ai, bi = case.images()
    ai, bi = ai.astype(np.float32).astype(np.float64), bi.astype(np.float32).astype(np.float64)
    np.save(os.path.join(a.out, "a.npy"), ai.astype(np.float32))
    np.save(os.path.join(a.out, "b.npy"), bi.astype(np.float32))
    write_png16(os.path.join(a.out, "a.png"), ai)
    write_png16(os.path.join(a.out, "b.png"), bi)
    json.dump(dict(size=a.size, alpha=a.alpha, opening=a.opening, slip=a.slip), open(os.path.join(a.out, "case.json"), "w"))
    print("wrote", a.out)


def load(out):
    return (np.load(os.path.join(out, "a.npy")).astype(np.float64), np.load(os.path.join(out, "b.npy")).astype(np.float64),
            json.load(open(os.path.join(out, "case.json"))))


class Timers(dict):
    def add(self, k, dt):
        self[k] = self.get(k, 0.0) + dt

    def wrap(self, mod, name, key):
        f = getattr(mod, name)
        timers = self

        def g(*args, **kw):
            t = time.perf_counter()
            try:
                return f(*args, **kw)
            finally:
                timers.add(key, time.perf_counter() - t)

        setattr(mod, name, g)
        return f


def refine_xdic_timed(I1, I2, fl0, cfg, lines, T):
    """xdic.refine_xdic with timers (the loop of the reference, copied); same arguments and result."""
    H, W = I1.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    bank, fx, fy = pf.gabor_bank((H + 2 * pf.PAD, W + 2 * pf.PAD), cfg.lam, cfg.scales, cfg.n_orient, cfg.bandwidth, cfg.dc_cut)
    ref = pf.reference_phase(I1, bank, fx, fy)
    m1 = float(I1.mean())
    ux, uy = np.array(fl0.ux, np.float64), np.array(fl0.uy, np.float64)
    slope, conf, resid = np.array(fl0.slope, np.float64), np.array(fl0.conf, np.float64), np.array(fl0.resid, np.float64)
    energy = None
    t = time.perf_counter()
    geo = xdic._Geo(lines, (H, W), cfg)
    T.add("geometry", time.perf_counter() - t)
    n_outer = max(cfg.outer_iters, 2 * cfg.xdic_iters + 2)
    sched = {2 + 2 * i for i in range(cfg.xdic_iters)}
    c2raw = None
    t = time.perf_counter()
    for s in (1, -1):
        ex, ey, _ = xdic.side_extrapolate(ux, uy, slope, conf, cfg, geo.fill_pw[s])
        ux = np.where(geo.sel[s] & (conf < cfg.conf_min), ex, ux)
        uy = np.where(geo.sel[s] & (conf < cfg.conf_min), ey, uy)
    T.add("side solve", time.perf_counter() - t)
    refs = None
    for it in range(n_outer):
        if it in sched:
            t = time.perf_counter()
            if c2raw is None:
                c2raw = bspline_coeffs(I2)
                ok = (conf >= 0.5) & ~geo.strip
                if ok.sum() > 100:
                    w2 = sample(c2raw, (xx + ux)[ok], (yy + uy)[ok])
                    A = np.column_stack([w2, np.ones(len(w2))])
                    gain = np.linalg.lstsq(A, I1[ok], rcond=None)[0]
                    noise = float(np.mean((I1[ok] - A @ gain) ** 2))
                else:
                    gain, noise = np.array([1.0, 0.0]), 0.0
            new, ext = xdic._update_geometry(geo, ux, uy, slope, conf, cfg, I1, c2raw, gain, (H, W), noise)
            if ext is not None:
                g2 = xdic._Geo(new, (H, W), cfg)
                for s in (1, -1):
                    ch = g2.sel[s] & ~geo.sel[s]
                    ux, uy = np.where(ch, ext[s][0], ux), np.where(ch, ext[s][1], uy)
                    conf = np.where(ch, 0.0, conf)
                geo = g2
                refs = None
            T.add("geometry", time.perf_counter() - t)
        if refs is None:
            t = time.perf_counter()
            refs = {s: (xdic.masked_reference(I1, geo.fill[s], m1, bank, fx, fy), 1.0 - geo.fill[s])
                    for s in (1, -1) if geo.sel[s].any()}
            T.add("responses", time.perf_counter() - t)
        t = time.perf_counter()
        sx, sy = pf.prewarp(ux, uy, slope, conf, cfg)
        F2 = np.fft.fft2(pf._pad(warp(I2, sx, sy)))
        resp = [bspline_coeffs(np.fft.ifft2(F2 * G)) for G in bank]
        csx, csy = bspline_coeffs(sx), bspline_coeffs(sy)
        T.add("baseline iterations", time.perf_counter() - t)
        side_state = {}
        for s, (rk, Mk) in refs.items():
            t = time.perf_counter()
            ex, ey, _ = xdic.side_extrapolate(ux, uy, slope, conf, cfg, geo.fill_pw[s])
            skx, sky = gauss_blur(ex, cfg.sigma_s), gauss_blur(ey, cfg.sigma_s)
            T.add("side solve", time.perf_counter() - t)
            t = time.perf_counter()
            Jk = np.where(geo.fill[s], m1, warp(I2, skx, sky))
            F2k = np.fft.fft2(pf._pad(Jk))
            side_state[s] = (rk, [pf._crop(np.fft.ifft2(F2k * G)) for G in bank], skx, sky, Mk)
            T.add("responses", time.perf_counter() - t)
        t = time.perf_counter()
        ux_in, uy_in = ux, uy
        for _ in range(cfg.inner_iters):
            mom = pf._moments(ref, resp, csx, csy, sx, sy, ux, uy, xx, yy, None)
            ux, uy, energy, resid, slope = pf.local_solve(*mom, ux, uy, cfg.sigma_w, cfg.shape_order)
        ux, uy = ux.copy(), uy.copy()
        T.add("baseline iterations", time.perf_counter() - t)
        t = time.perf_counter()
        for s, (rk, respk, skx, sky, Mk) in side_state.items():
            momk = xdic._moments_grid(rk, respk, skx, sky, Mk)
            sk = pf._solve_full(*momk, ux_in, uy_in, cfg.sigma_w, cfg.shape_order)
            sel = geo.sel[s]
            ux[sel], uy[sel], energy[sel], resid[sel], slope[sel] = sk[0][sel], sk[1][sel], sk[2][sel], sk[3][sel], sk[4][sel]
        conf = pf._confidence(energy, resid, cfg)
        T.add("side solve", time.perf_counter() - t)
    conf_out = np.where(pf.reportable_flow(ux, uy, cfg), np.clip(conf, 0.0, 1.0), 0.0)
    return ux, uy, conf_out, geo.lines


def run_python(a):
    I1, I2, case = load(a.out)
    res = {}
    for name, cfg in (("base", Config()), ("xdic", Config(crack_model="xdic"))):
        best = None
        for _ in range(a.runs):
            t = time.perf_counter()
            r = track.track_pair(I1, I2, cfg)
            dt = time.perf_counter() - t
            if best is None or dt < best[0]:
                best = (dt, r)
        dt, r = best
        res[name] = dict(total=dt, timing={k: float(v) for k, v in r.timing.items() if isinstance(v, (int, float))})
    # the stage breakdown of the xdic pair
    T = Timers()
    orig = {"refine_xdic": xdic.refine_xdic}
    for mod, nm, key in ((xdic, "finish", "finish"), (xdic, "model_gain", "finish"), (xdic, "advance_lines", "finish"),
                         ):
        orig[nm] = T.wrap(mod, nm, key)
    # refine_xdic is replaced by the timed copy; check it against the original on the first call
    checked = []

    def timed(I1_, I2_, fl0, cfg_, lines):
        t0 = time.perf_counter()
        out = refine_xdic_timed(I1_, I2_, fl0, cfg_, lines, T)
        T.add("refine_timed", time.perf_counter() - t0)
        if not checked:
            fl, lns, _ = orig["refine_xdic"](I1_, I2_, fl0, cfg_, lines)
            checked.append(float(np.nanmax(np.abs(fl.ux - out[0]))) + float(np.nanmax(np.abs(fl.uy - out[1]))))
        return _Flow(out), out[3], None

    class _Flow:
        def __init__(self, out):
            self.ux, self.uy, self.conf = out[0], out[1], out[2]

    xdic.refine_xdic = timed
    T.clear()
    cfg = Config(crack_model="xdic")
    t = time.perf_counter()
    r = track.track_pair(I1, I2, cfg)
    total = time.perf_counter() - t
    xdic.refine_xdic = orig["refine_xdic"]
    stages = dict(T)
    stages.pop("refine_total", None)
    stages.pop("refine_timed", None)
    res["xdic_stages"] = dict(total=total, detect=r.timing["xdic_detect"], fine=r.timing["fine"], coarse=r.timing["coarse"],
                              **{k: v for k, v in stages.items()}, check_max_abs_diff_vs_reference=checked[0] if checked else None)
    res["case"] = case
    json.dump(res, open(os.path.join(a.out, "python_timing.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


def cmd_metrics(a):
    """Tracking rate and error of the Python reference and of an `optflow pair` run, on the same case."""
    I1, I2, case = load(a.out)
    N = I1.shape[0]
    c = Case(N, case["alpha"], case["opening"], case["slip"])
    sd, tau, n, t, cen = geometry(c.field, N)
    gx, gy = synth.displacement(c.field, N, N)
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
    gexx = c.field.grad(xx, yy)[0]
    out = {}

    class R:  # the fields of a track.Result that field_metrics reads
        pass

    def from_dir(d):
        r = R()
        r.ux, r.uy = np.load(os.path.join(d, "ux.npy")).astype(np.float64), np.load(os.path.join(d, "uy.npy")).astype(np.float64)
        r.strain = {"exx": np.load(os.path.join(d, "Exx.npy")).astype(np.float64) * 0 + np.nan}
        r.crack_lines = None
        lj = os.path.join(d, "crack_lines.json")
        if os.path.exists(lj):
            j = json.load(open(lj))
            r.crack_lines = dict(lines=[np.array(p) for p in j["lines"]], bands=j["bands"], rejected=j["rejected"],
                                 jumps=[dict(pts=np.array(x["pts"]), cod=np.array([np.nan if v is None else v for v in x["cod"]]),
                                             csd=np.array([np.nan if v is None else v for v in x["csd"]])) for x in j["jumps"]])
        return r

    py = track.track_pair(I1, I2, Config(crack_model="xdic"))
    for name, r in (("python", py), ("rust", from_dir(a.rust))):
        m = field_metrics(r, gx, gy, gexx, sd, tau, N)
        out[name] = dict(valid=m["valid"], rms=m["rms"], far_valid=m["far_valid"], far_rms=m["far_rms"],
                         line=line_metrics(r, c.field, n, t, cen, N))
    json.dump(out, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("out")
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--opening", type=float, default=3.0)
    p.add_argument("--slip", type=float, default=1.0)
    p = sub.add_parser("python")
    p.add_argument("out")
    p.add_argument("--runs", type=int, default=1)
    p = sub.add_parser("metrics")
    p.add_argument("out")
    p.add_argument("rust")
    a = ap.parse_args()
    {"prepare": prepare, "python": run_python, "metrics": cmd_metrics}[a.cmd](a)


if __name__ == "__main__":
    main()
