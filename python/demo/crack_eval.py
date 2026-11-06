"""Synthetic evaluation of the crack-aware tracking.

    python -m demo.crack_eval [--quick] [--size 384] [--seq-size 256] [--frames 20] [--no-seq] [--no-pairs]

Pairs (stage 2, spec 6.3: columns base / split / sides): a straight crack through the image centre (normal angle alpha, opening o and slip s in px, optional uniform
strain exx on the + side), 1 % noise. Each case is tracked twice with the Python reference: the current model
(baseline) and crack_split + crack_out. Sequences: the opening grows 0 -> 6 px over --frames frames, tracked in chain
and hybrid mode, baseline vs crack_split + crack_out.

Metrics (distance d from the crack line in the frame-0 grid; the central 240 px along the crack only, so that the image
ends do not count): band = 3 <= d <= 16 px; far = d > 32 px and 40 px inside the image.
  valid   share of the band's pixels that are valid (finite displacement): the tracking rate
  rms     RMS displacement error [px] of the valid band pixels (and of the valid far pixels)
  COD/CSD median |estimate - truth| over the crack band C at the pixels where it is finite (finite share in brackets),
          for crack_a = 6 (the spec's default) and 12
  exx     RMS error of the small strain exx over the valid band pixels (the baseline's false band against the one-sided fit)
Tracking rate and accuracy have to be read together: a model that makes more pixels valid is also asked to be right there.
"""
import argparse
import time
from dataclasses import replace

import numpy as np

from optflow_ref import crack as crack_mod
from optflow_ref import synth
from optflow_ref.config import Config
from optflow_ref.strain import displacement_gradient
from optflow_ref.track import _finish_ex, track_pair, track_sequence

NOISE = 0.01
BAND = (3.0, 16.0)
ALONG = 120.0  # half length of the evaluated stretch of the crack [px]


class Case:
    def __init__(self, N, alpha, o, s, exx=0.0, seed=21):
        self.N, self.alpha, self.o, self.s, self.exx = N, alpha, o, s, exx
        c = (N / 2.0, N / 2.0)
        plus = synth.uniform_strain(exx, 0.0, c) if exx else None
        self.field = synth.Crack(c, alpha, o, s, plus=plus)
        self.pat = synth.make_pattern(N, N, seed=seed)
        yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
        f = self.field
        self.d = np.abs(f.n[0] * (xx - c[0]) + f.n[1] * (yy - c[1]))
        self.tau = f.t[0] * (xx - c[0]) + f.t[1] * (yy - c[1])
        self.gx, self.gy = synth.displacement(f, N, N)
        self.gexx = f.grad(xx, yy)[0]
        self.band = (self.d >= BAND[0]) & (self.d <= BAND[1]) & (np.abs(self.tau) < ALONG)
        m = 40
        self.far = (self.d > 32) & (xx >= m) & (xx <= N - 1 - m) & (yy >= m) & (yy <= N - 1 - m)
        # the true jump at the crack point with along-coordinate tau: J plus the strain of the + side there
        P = np.array(c)[:, None, None] + f.t[:, None, None] * self.tau[None]
        self.cod_true = f.J[0] * f.n[0] + f.J[1] * f.n[1] + exx * (P[0] - c[0]) * f.n[0]
        self.csd_true = f.J[0] * f.t[0] + f.J[1] * f.t[1] + exx * (P[0] - c[0]) * f.t[0]

    def images(self, seed=1):
        a = synth.degrade(synth.render(self.pat, self.N, self.N), seed=seed, noise=NOISE)
        b = synth.degrade(synth.render(self.pat, self.N, self.N, self.field), seed=seed + 1, noise=NOISE)
        return a, b

    def label(self):
        return f"a={self.alpha:g} o={self.o:g} s={self.s:g} exx={self.exx * 100:g}%"


def _rms(e):
    return float(np.sqrt(np.mean(e**2))) if e.size else float("nan")


def metrics(case, r):
    """valid / rms of the band, rms of the far field, exx rms of the band for one Result."""
    fin = np.isfinite(r.ux) & np.isfinite(r.uy)
    e = np.hypot(r.ux - case.gx, r.uy - case.gy)
    ex = r.strain["exx"] - case.gexx
    mb = fin & case.band
    mf = fin & case.far
    mx = mb & np.isfinite(ex)
    return dict(valid=float(mb.sum() / case.band.sum()), rms=_rms(e[mb]), far_valid=float(mf.sum() / case.far.sum()),
                far_rms=_rms(e[mf]), exx=_rms(ex[mx]))


def jump_metrics(case, r, cfg, a):
    """Median |COD - truth|, median |CSD - truth| over the crack band C (where finite) and the finite share, with
    crack_a = a. The geometry and the gradient are recomputed from the Result (u is already masked)."""
    cfg = replace(cfg, crack_a=a)
    geo = crack_mod.geometry(r.crack_in, cfg)
    grad = displacement_gradient(r.ux, r.uy, cfg.strain_win, half=crack_mod.strain_half_map(geo, cfg.strain_win))
    cod, csd = crack_mod.jump(r.ux, r.uy, grad, geo, cfg)
    sel = geo["C"] & (np.abs(case.tau) < ALONG)
    if not sel.any():
        return dict(cod=float("nan"), csd=float("nan"), fin=0.0, cod_med=float("nan"))
    f = np.isfinite(cod[sel])
    ok = sel & np.isfinite(cod)
    return dict(cod=float(np.median(np.abs(cod[ok] - case.cod_true[ok]))) if f.any() else float("nan"),
                csd=float(np.median(np.abs(csd[ok] - case.csd_true[ok]))) if f.any() else float("nan"),
                fin=float(f.mean()), cod_med=float(np.median(cod[ok])) if f.any() else float("nan"))


def with_frac(r, cfg, frac):
    """The Result r re-finished with crack_guard_frac = frac (it only changes the masks of the output, not the tracking)."""
    c2 = replace(cfg, crack_guard_frac=frac)
    ux, uy, strain, crack = _finish_ex(r.raw_ux, r.raw_uy, r.conf, c2, None, {}, r.crack_in)
    return replace(r, ux=ux, uy=uy, strain=strain, crack=crack)


CONFIGS = {
    "split": Config(crack_split=True, crack_out=True, crack_a=6.0),  # stage 1 (spec 2)
    "sides": Config(crack_split=True, crack_sides=True, crack_out=True),  # stage 2 (spec 6): crack_a 12, frac 0.5
}
PROFILE_D = (4, 6, 8, 10, 12, 16)


def profile(case, r):
    """Median displacement error of the valid pixels at distance 4/6/8/10/12/16 px (+-1 px) from the crack, central
    240 px, on the - side and the + side: {'-': [...], '+': [...]} (NaN: no valid pixel)."""
    e = np.hypot(r.ux - case.gx, r.uy - case.gy)
    fin = np.isfinite(e) & (np.abs(case.tau) < ALONG)
    f = case.field
    yy, xx = np.mgrid[0:case.N, 0:case.N].astype(np.float64)
    sgn = f.n[0] * (xx - case.N / 2) + f.n[1] * (yy - case.N / 2)
    out = {}
    for name, side in (("-", sgn < 0), ("+", sgn > 0)):
        row = []
        for d in PROFILE_D:
            m = fin & side & (np.abs(case.d - d) <= 1.0)
            row.append(float(np.median(e[m])) if m.any() else float("nan"))
        out[name] = row
    return out


def run_pair(case):
    a, b = case.images()
    out = {"time": []}
    runs = {"base": (Config(), track_pair(a, b, Config()))}
    for name, cfg in CONFIGS.items():
        t = time.time()
        runs[name] = (cfg, track_pair(a, b, cfg))
        out["time"].append(time.time() - t)
    cfg_s, r_s = runs["sides"]
    runs["sides0"] = (replace(cfg_s, crack_guard_frac=0.0), with_frac(r_s, cfg_s, 0.0))
    for name, (cfg, r) in runs.items():
        out[name] = metrics(case, r)
        out[name]["profile"] = profile(case, r)
        if name != "base":
            out[name]["jump"] = jump_metrics(case, r, cfg, cfg.crack_a)
    return out


def pair_cases(N, quick=False):
    cases = []
    for alpha in (0.0, 30.0):
        for o, s in ((1.0, 0.0), (3.0, 1.0), (6.0, 2.0)):
            for exx in (0.0, 0.005):
                cases.append(Case(N, alpha, o, s, exx))
    if quick:
        cases = [c for c in cases if (c.alpha, c.o, c.exx) in ((0.0, 3.0, 0.0), (30.0, 3.0, 0.005))]
    return cases


def run_sequence(N, nframes, mode, cfg, o_end=6.0):
    """Opening 0 -> o_end over nframes frames (alpha 0, no slip) tracked with cfg (seq_mode set here): per frame the band
    valid share and RMS, the median COD (read at cfg.crack_a) and the true opening."""
    pat = synth.make_pattern(N, N, seed=21)
    c = (N / 2.0, N / 2.0)
    fields = [synth.Crack(c, 0.0, o_end * k / nframes, 0.0) for k in range(nframes + 1)]
    frames = [synth.degrade(synth.render(pat, N, N, f), seed=100 + k, noise=NOISE) for k, f in enumerate(fields)]
    cfg = replace(cfg, seq_mode=mode)
    rows = []
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
    d = np.abs(xx - c[0])
    band = (d >= BAND[0]) & (d <= BAND[1]) & (np.abs(yy - c[1]) < ALONG)
    for k, r in track_sequence(frames, cfg):
        f = fields[k]
        gx, gy = synth.displacement(f, N, N)
        fin = np.isfinite(r.ux)
        e = np.hypot(r.ux - gx, r.uy - gy)
        mb = fin & band
        row = dict(k=k, valid=float(mb.sum() / band.sum()), rms=_rms(e[mb]), o=f.J[0], ref=r.timing["reference"])
        if cfg.crack_split and r.crack_in is not None:
            row.update(jump_metrics_simple(r, cfg, cfg.crack_a, f.J[0], yy - c[1]))
        rows.append(row)
    return rows


def jump_metrics_simple(r, cfg, a, o_true, tau):
    cfg = replace(cfg, crack_a=a)
    geo = crack_mod.geometry(r.crack_in, cfg)
    grad = displacement_gradient(r.ux, r.uy, cfg.strain_win, half=crack_mod.strain_half_map(geo, cfg.strain_win))
    cod, _ = crack_mod.jump(r.ux, r.uy, grad, geo, cfg)
    sel = geo["C"] & (np.abs(tau) < ALONG) & np.isfinite(cod)
    return dict(cod=float(np.median(cod[sel])) if sel.any() else float("nan"),
                cod_err=float(np.median(np.abs(cod[sel] - o_true))) if sel.any() else float("nan"))


def _f(x, w=5, p=2):
    return f"{x:{w}.{p}f}" if np.isfinite(x) else " " * (w - 3) + "n/a"


COLS = ("base", "split", "sides", "sides0")
HEAD = ("PAIRS (384x384 synthetic crack, 1% noise; band = 3..16 px from the crack, central 240 px). Columns: base = current model;\n"
        "split = crack_split+crack_out (stage 1, crack_a 6); sides = crack_sides (stage 2, crack_a 12, crack_guard_frac 0.5);\n"
        "sides0 = the same with crack_guard_frac 0.\n"
        "case                       | band valid                | band rms [px]                  | far rms [px]  | COD err (fin)         | exx rms 1e-3\n"
        "                           | base split sides sides0   | base   split  sides  sides0    | base   sides  | split  sides  sides0  | base  split  sides  sides0")


def print_pair_table(results):
    print(HEAD)
    for case, r in results:
        v = " ".join(_f(r[c]["valid"]) for c in COLS)
        m = " ".join(_f(r[c]["rms"], 6, 3) for c in COLS)
        far = " ".join(_f(r[c]["far_rms"], 6, 4) for c in ("base", "sides"))
        j = " ".join(f"{_f(r[c]['jump']['cod'], 5, 2)}({r[c]['jump']['fin']:.2f})" for c in ("split", "sides", "sides0"))
        x = " ".join(_f(r[c]["exx"] * 1e3, 6, 1) for c in COLS)
        print(f"{case.label():26s} | {v} | {m} | {far} | {j} | {x}")
    print("far valid share (min over cases): " + " ".join(f"{c} {min(r[c]['far_valid'] for _, r in results):.3f}" for c in COLS))
    print("PER-DISTANCE median error [px] of the valid pixels, distance 4/6/8/10/12/16 px (- side | + side); n/a = no valid pixel")
    for case, r in results:
        for c in ("base", "split", "sides", "sides0"):
            if c != "sides" and not (case.alpha == 0 and case.o == 3 and case.exx == 0):
                continue
            p = r[c]["profile"]
            print(f"  {case.label():26s} {c:7s} -: " + " ".join(_f(x, 5, 2) for x in p["-"]) + "  | +: " + " ".join(_f(x, 5, 2) for x in p["+"]))


def print_seq_table(name, runs):
    n = len(runs["base"])
    pick = [0, n // 2 - 1, n - 1]
    print(f"SEQUENCE {name}: opening 0 -> {runs['base'][-1]['o']:.0f} px over {n} frames (256x256); valid = band share, rms [px]; "
          f"cod = median COD, err = median |COD - true|")
    print("   k  true o | valid: " + " ".join(f"{c:>6s}" for c in runs) + " | rms: " + " ".join(f"{c:>6s}" for c in runs)
          + " | cod/err sides | cod/err sides0")
    for i in range(n):
        v = " ".join(_f(runs[c][i]["valid"], 6) for c in runs)
        m = " ".join(_f(runs[c][i]["rms"], 6, 3) for c in runs)
        s, s0 = runs["sides"][i], runs["sides0"][i]
        mark = " <" if i in pick else ""
        print(f"  {runs['base'][i]['k']:2d}  {runs['base'][i]['o']:5.2f} |        {v} |      {m} | "
              f"{_f(s.get('cod', np.nan), 5, 2)}/{_f(s.get('cod_err', np.nan), 4, 2)} | {_f(s0.get('cod', np.nan), 5, 2)}/{_f(s0.get('cod_err', np.nan), 4, 2)}{mark}")
    print("  start/mid/end (k = %s) valid: " % "/".join(str(runs['base'][i]['k']) for i in pick)
          + "  ".join(c + " " + "/".join(f"{runs[c][i]['valid']:.2f}" for i in pick) for c in runs))
    print("                          rms:   " + "  ".join(c + " " + "/".join(f"{runs[c][i]['rms']:.3f}" for i in pick) for c in runs))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="two pair cases only")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--seq-size", type=int, default=256)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--no-seq", action="store_true")
    ap.add_argument("--no-pairs", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    if not a.no_pairs:
        res = []
        for case in pair_cases(a.size, a.quick):
            res.append((case, run_pair(case)))
        print_pair_table(res)
        print("time per case [s] split / sides (mean): %.1f / %.1f" % tuple(np.mean([r["time"] for _, r in res], axis=0)))
    if not a.no_seq:
        for mode in ("chain", "hybrid"):
            runs = {"base": run_sequence(a.seq_size, a.frames, mode, Config()),
                    "split": run_sequence(a.seq_size, a.frames, mode, CONFIGS["split"]),
                    "sides": run_sequence(a.seq_size, a.frames, mode, CONFIGS["sides"]),
                    "sides0": run_sequence(a.seq_size, a.frames, mode, replace(CONFIGS["sides"], crack_guard_frac=0.0))}
            print_seq_table(mode, runs)
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
