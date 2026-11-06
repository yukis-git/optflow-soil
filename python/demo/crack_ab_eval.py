"""Synthetic evaluation of the experimental crack model, crack_model "xdic".

    python -m demo.crack_ab_eval pairs [--size 384] [--jobs 4]     # 12 synthetic crack pairs, base vs xdic
    python -m demo.crack_ab_eval neg [--size 384]                  # negative controls: weak texture, shear band
    python -m demo.crack_ab_eval seq MODE [--size 256] [--frames 20]   # opening 0 -> 6 px, MODE chain | hybrid
    python -m demo.crack_ab_eval report                            # tables (stdout), figures and summary.json

Every step writes its raw results as JSON to --scratch; `report` writes figures and summary JSON to --out.
Defaults: scratch crack_eval_work/, out crack_eval_out/ (in the current directory).

Metrics (synthetic; distance d from the true crack line, central 240 px along it; base = the current model, Config()):
  profile    per distance 1/2/4/6/8/10/12/16 px (|d - D| <= 0.5) and side: valid share and median |error| [px]
  band       3 <= d <= 16: valid share and RMS error; far: d > 32 px and 40 px inside the image, RMS (the noise level)
  detection  mean distance of the detected polyline points (|tau| < 120) to the true line; share of the true line (|tau| < 120)
             within 2 px of a detected line; number of lines
  COD / CSD  at the line samples with |tau| < 120: median |estimate - truth| and the bias (signed mean)
  exx        RMS of the small-strain exx error over the valid band pixels; time per pair
Negative controls: (1) uniform translation (1.3, -0.7) px with a 60 x 20 px patch at 0.05 x contrast, (2) ShearBand delta 3 px,
width 6 px: the detected crack lines (count, length, pixels within 1 px of them = false detection area) and the bands.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from optflow_ref import synth  # noqa: E402
from optflow_ref.config import Config  # noqa: E402
from optflow_ref.track import track_pair, track_sequence  # noqa: E402
from optflow_ref import xdic  # noqa: E402

NOISE = 0.01
BAND = (3.0, 16.0)
ALONG = 120.0
PROFILE_D = (1, 2, 4, 6, 8, 10, 12, 16)
SCRATCH = "crack_eval_work"
OUT = "crack_eval_out"
XCFG = Config(crack_model="xdic")


def _rms(e):
    return float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")


def _med(e):
    return float(np.median(e)) if e.size else float("nan")


class Case:
    """A straight crack through the image centre (normal angle alpha, opening o, slip s; strain exx on the + side)."""

    def __init__(self, N, alpha, o, s, exx=0.0, seed=21):
        self.N, self.alpha, self.o, self.s, self.exx = N, alpha, o, s, exx
        c = (N / 2.0, N / 2.0)
        plus = synth.uniform_strain(exx, 0.0, c) if exx else None
        self.field = synth.Crack(c, alpha, o, s, plus=plus)
        self.pat = synth.make_pattern(N, N, seed=seed)
        self.c = np.array(c)

    def images(self, seed=1):
        a = synth.degrade(synth.render(self.pat, self.N, self.N), seed=seed, noise=NOISE)
        b = synth.degrade(synth.render(self.pat, self.N, self.N, self.field), seed=seed + 1, noise=NOISE)
        return a, b

    def label(self):
        return f"a={self.alpha:g} o={self.o:g} s={self.s:g} exx={self.exx * 100:g}%"

    def key(self):
        return f"a{self.alpha:g}_o{self.o:g}_s{self.s:g}_e{self.exx * 100:g}"


def geometry(field, N):
    """Signed distance and along-line coordinate of every pixel for a synth.Crack / ShearBand-like straight field."""
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
    n = field.n if hasattr(field, "n") else np.array([field.nx, field.ny])
    t = np.array([-n[1], n[0]])
    c = np.asarray(field.c, np.float64)
    sd = n[0] * (xx - c[0]) + n[1] * (yy - c[1])
    tau = t[0] * (xx - c[0]) + t[1] * (yy - c[1])
    return sd, tau, n, t, c


def field_metrics(r, gx, gy, gexx, sd, tau, N):
    d = np.abs(sd)
    fin = np.isfinite(r.ux) & np.isfinite(r.uy)
    e = np.hypot(r.ux - gx, r.uy - gy)
    yy, xx = np.mgrid[0:N, 0:N]
    along = np.abs(tau) < min(ALONG, N / 2.0 - 8)
    band = (d >= BAND[0]) & (d <= BAND[1]) & along
    far = (d > 32) & (xx >= 40) & (xx <= N - 41) & (yy >= 40) & (yy <= N - 41)
    ex = r.strain["exx"] - gexx
    mb = fin & band
    out = dict(valid=float(mb.sum() / max(band.sum(), 1)), rms=_rms(e[mb]), far_valid=float((fin & far).sum() / max(far.sum(), 1)),
               far_rms=_rms(e[fin & far]), exx=_rms(ex[mb & np.isfinite(ex)]), profile={})
    for name, side in (("-", sd < 0), ("+", sd > 0)):
        rows = []
        for D in PROFILE_D:
            ring = along & side & (np.abs(d - D) <= 0.5)
            m = ring & fin
            rows.append([float(m.sum() / max(ring.sum(), 1)), _med(e[m])])
        out["profile"][name] = rows
    return out


def line_metrics(r, field, n, t, c, N, exx=0.0):
    """Detection and COD / CSD of an xdic Result against a straight true line."""
    cl = r.crack_lines
    out = dict(n_lines=0, n_bands=0, dist=float("nan"), covered=0.0, cod_err=float("nan"), cod_bias=float("nan"),
               csd_err=float("nan"), csd_bias=float("nan"), cod_med=float("nan"), csd_med=float("nan"), length=0.0)
    if cl is None:
        return out
    out["n_bands"] = len(cl.get("bands", []))
    lines = cl["lines"]
    out["n_lines"] = len(lines)
    if not lines:
        return out
    pts = []
    for P in lines:
        q, _, s = xdic._resample(P, 1.0)
        inside = (q[:, 0] >= 0) & (q[:, 0] <= N - 1) & (q[:, 1] >= 0) & (q[:, 1] <= N - 1)
        pts.append(q[inside])
    pts = np.vstack(pts)
    out["length"] = float(len(pts))
    sd = (pts - c) @ n
    ta = (pts - c) @ t
    cen = np.abs(ta) < min(ALONG, N / 2.0 - 8)
    out["dist"] = float(np.mean(np.abs(sd[cen]))) if cen.any() else float("nan")
    # coverage: true-line points (1 px apart) within 2 px of a detected point
    from scipy.spatial import cKDTree
    tl = np.arange(-min(ALONG, N / 2.0 - 8), min(ALONG, N / 2.0 - 8) + 1e-9, 1.0)
    true_pts = c[None] + tl[:, None] * t[None]
    dd, _ = cKDTree(pts).query(true_pts)
    out["covered"] = float(np.mean(dd <= 2.0))
    cods, csds, codt, csdt = [], [], [], []
    J = field.J if hasattr(field, "J") else np.zeros(2)
    for j in cl["jumps"]:
        if not len(j["pts"]):
            continue
        ta_j = (j["pts"] - c) @ t
        m = np.abs(ta_j) < min(ALONG, N / 2.0 - 8)
        cods.append(j["cod"][m]); csds.append(j["csd"][m])
        x_off = ta_j[m] * t[0]  # x - c_x of the true-line point with this tau (the strain of the + side)
        codt.append(J @ n + exx * x_off * n[0]); csdt.append(J @ t + exx * x_off * t[0])
    if cods:
        cod, csd, ct, st = (np.concatenate(v) for v in (cods, csds, codt, csdt))
        ok = np.isfinite(cod) & np.isfinite(csd)
        if ok.any():
            out.update(cod_err=_med(np.abs(cod[ok] - ct[ok])), cod_bias=float(np.mean(cod[ok] - ct[ok])),
                       csd_err=_med(np.abs(csd[ok] - st[ok])), csd_bias=float(np.mean(csd[ok] - st[ok])),
                       cod_med=_med(cod[ok]), csd_med=_med(csd[ok]), cod_fin=float(ok.mean()))
    return out


def run_pair(args):
    N, alpha, o, s, exx = args
    case = Case(N, alpha, o, s, exx)
    a, b = case.images()
    sd, tau, n, t, c = geometry(case.field, N)
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
    gx, gy = synth.displacement(case.field, N, N)
    gexx = case.field.grad(xx, yy)[0]
    out = dict(label=case.label(), key=case.key())
    for name, cfg in (("base", Config()), ("xdic", XCFG)):
        t0 = time.time()
        r = track_pair(a, b, cfg)
        out[name] = field_metrics(r, gx, gy, gexx, sd, tau, N)
        out[name]["time"] = time.time() - t0
        out[name]["timing"] = {k: float(v) for k, v in r.timing.items() if isinstance(v, (int, float))}
        if name == "xdic":
            out[name]["line"] = line_metrics(r, case.field, n, t, c, N, exx)
    print(case.label(), f"base {out['base']['time']:.0f}s xdic {out['xdic']['time']:.0f}s", flush=True)
    return out


def pair_args(N):
    return [(N, alpha, o, s, exx) for alpha in (0.0, 30.0) for o, s in ((1.0, 0.0), (3.0, 1.0), (6.0, 2.0)) for exx in (0.0, 0.005)]


def cmd_pairs(a):
    args = pair_args(a.size)
    if a.jobs > 1:
        with get_context("spawn").Pool(a.jobs) as pool:
            res = pool.map(run_pair, args, chunksize=1)
    else:
        res = [run_pair(x) for x in args]
    _dump(a, "pairs.json", res)


def neg_images(kind, N):
    pat = synth.make_pattern(N, N, seed=21)
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
    if kind == "weak":
        tr = (1.3, -0.7)
        f = synth.translation(*tr)
        a, b = synth.render(pat, N, N), synth.render(pat, N, N, f)
        x0, y0 = N / 2 - 30, N / 2 - 10
        m1 = (xx >= x0) & (xx < x0 + 60) & (yy >= y0) & (yy < y0 + 20)
        m2 = (xx - tr[0] >= x0) & (xx - tr[0] < x0 + 60) & (yy - tr[1] >= y0) & (yy - tr[1] < y0 + 20)
        a = 0.5 + (a - 0.5) * np.where(m1, 0.05, 1.0)
        b = 0.5 + (b - 0.5) * np.where(m2, 0.05, 1.0)
        region = m1
    else:
        f = synth.ShearBand(3.0, 6.0, (N / 2, N / 2), 90.0)  # band along y (normal along x)
        a, b = synth.render(pat, N, N), synth.render(pat, N, N, f)
        region = np.abs(xx - N / 2) <= 16
    a = synth.degrade(a, seed=1, noise=NOISE)
    b = synth.degrade(b, seed=2, noise=NOISE)
    gx, gy = synth.displacement(f, N, N)
    return a, b, gx, gy, region


def cmd_neg(a):
    N = a.size
    res = {}
    for kind in ("weak", "band"):
        I1, I2, gx, gy, region = neg_images(kind, N)
        rec = {}
        for name, cfg in (("base", Config()), ("xdic", XCFG)):
            t0 = time.time()
            r = track_pair(I1, I2, cfg)
            fin = np.isfinite(r.ux)
            e = np.hypot(r.ux - gx, r.uy - gy)
            m = dict(time=time.time() - t0, valid=float(fin.mean()), rms=_rms(e[fin]), region_valid=float(fin[region].mean()),
                     region_rms=_rms(e[fin & region]))
            cl = r.crack_lines
            lines = [] if cl is None else cl["lines"]
            m["n_lines"] = len(lines)
            m["n_bands"] = 0 if cl is None else len(cl.get("bands", []))
            m["band_lines"] = [] if cl is None else [np.asarray(l).round(1).tolist() for l in cl.get("bands", [])]
            m["false_area_px"] = int(r.crack["mask"].sum()) if r.crack is not None else 0
            m["false_length_px"] = float(sum(len(xdic._resample(l, 1.0)[0]) for l in lines))
            if cl is not None and cl["jumps"]:
                m["cod_med"] = float(np.nanmedian(np.concatenate([j["cod"] for j in cl["jumps"]])))
                m["csd_med"] = float(np.nanmedian(np.concatenate([j["csd"] for j in cl["jumps"]])))
            rec[name] = m
            print(kind, name, m, flush=True)
        res[kind] = rec
    _dump(a, "neg.json", res)


def cmd_seq(a):
    N, nf, mode = a.size, a.frames, a.mode
    pat = synth.make_pattern(N, N, seed=21)
    c = (N / 2.0, N / 2.0)
    fields = [synth.Crack(c, 0.0, 6.0 * k / nf, 0.0) for k in range(nf + 1)]
    frames = [synth.degrade(synth.render(pat, N, N, f), seed=100 + k, noise=NOISE) for k, f in enumerate(fields)]
    sd, tau, n, t, cc = geometry(fields[-1], N)
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
    res = {}
    for name, cfg in (("base", Config(seq_mode=mode)), ("xdic", Config(seq_mode=mode, crack_model="xdic"))):
        rows = []
        t0 = time.time()
        for k, r in track_sequence(frames, cfg):
            f = fields[k]
            gx, gy = synth.displacement(f, N, N)
            fm = field_metrics(r, gx, gy, np.zeros((N, N)), sd, tau, N)
            row = dict(k=k, o=float(f.J[0]), valid=fm["valid"], rms=fm["rms"], far_rms=fm["far_rms"], profile=fm["profile"],
                       time=time.time() - t0, ref=int(r.timing["reference"]))
            if name == "xdic":
                row["line"] = line_metrics(r, f, n, t, cc, N)
            rows.append(row)
            print(mode, name, k, f"valid {fm['valid']:.2f} rms {fm['rms']:.3f}",
                  "" if name == "base" else f"cod {row['line']['cod_med']:.2f} dist {row['line']['dist']:.2f}", flush=True)
        res[name] = rows
    _dump(a, f"seq_{mode}.json", res)


def _dump(a, name, obj):
    os.makedirs(a.scratch, exist_ok=True)
    with open(os.path.join(a.scratch, name), "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: None if o != o else (o.tolist() if hasattr(o, "tolist") else str(o)))


def _load(a, name):
    p = os.path.join(a.scratch, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


# ------------------------------------------------------------------------------------------------------------- report
def _f(x, w=5, p=2):
    return f"{x:{w}.{p}f}" if x is not None and np.isfinite(x) else " " * (w - 3) + "n/a"


def cmd_report(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(a.out, exist_ok=True)
    summary = {}
    pairs = _load(a, "pairs.json")
    if pairs:
        print("PAIRS: band (3-16 px) valid / RMS [px], far RMS, detection, COD/CSD err (bias), exx RMS 1e-3, time [s]")
        print("case                     | valid base xdic | rms base  xdic  | far base xdic   | dist cover | COD err(bias)  CSD err(bias) | exx base xdic | t base xdic")
        for p in pairs:
            b, x, ln = p["base"], p["xdic"], p["xdic"]["line"]
            print(f"{p['label']:24s} | {_f(b['valid'])} {_f(x['valid'])}   | {_f(b['rms'], 5, 3)} {_f(x['rms'], 5, 3)} | "
                  f"{_f(b['far_rms'], 6, 4)} {_f(x['far_rms'], 6, 4)} | {_f(ln['dist'])} {_f(ln['covered'])} | "
                  f"{_f(ln['cod_err'])}({_f(ln['cod_bias'], 5, 2)}) {_f(ln['csd_err'])}({_f(ln['csd_bias'], 5, 2)}) | "
                  f"{_f(b['exx'] * 1e3, 5, 1)} {_f(x['exx'] * 1e3, 5, 1)} | {b['time']:4.0f} {x['time']:4.0f}")
        print("PER-DISTANCE valid share / median error [px], D = " + " ".join(map(str, PROFILE_D)))
        for p in pairs:
            for name in ("base", "xdic"):
                for side in ("-", "+"):
                    row = " ".join(f"{v:.2f}/{_f(e, 4, 2)}" for v, e in p[name]["profile"][side])
                    print(f"  {p['label']:24s} {name:5s} {side}: {row}")
        summary["pairs"] = pairs
        # figure: per-distance profile of all cases
        fig, ax = plt.subplots(2, 2, figsize=(13, 8))
        for p in pairs:
            for j, side in enumerate(("-", "+")):
                pr = np.array(p["xdic"]["profile"][side], float)
                ax[0, j].plot(PROFILE_D, pr[:, 1], "o-", lw=0.8, label=p["label"])
                ax[1, j].plot(PROFILE_D, pr[:, 0], "o-", lw=0.8, label=p["label"])
                prb = np.array(p["base"]["profile"][side], float)
                ax[1, j].plot(PROFILE_D, prb[:, 0], "k:", lw=0.5)
        for j, side in enumerate(("-", "+")):
            ax[0, j].set_title(f"xdic: median |error| [px] vs distance, side {side}"); ax[0, j].set_ylim(0, 0.4)
            ax[1, j].set_title(f"valid share vs distance, side {side} (dotted: base)")
            for k in (0, 1):
                ax[k, j].grid(alpha=.3); ax[k, j].set_xlabel("distance from the crack [px]")
        ax[0, 0].legend(fontsize=6)
        fig.tight_layout(); fig.savefig(f"{a.out}/pairs_profile.png", dpi=70); plt.close(fig)
    neg = _load(a, "neg.json")
    if neg:
        print("NEGATIVE CONTROLS")
        for kind, rec in neg.items():
            for name, m in rec.items():
                print(f"  {kind:5s} {name:5s} valid {m['valid']:.3f} rms {m['rms']:.3f} region valid {m['region_valid']:.2f} rms "
                      f"{_f(m['region_rms'], 5, 3)} | cracks {m['n_lines']} (length {m['false_length_px']:.0f} px, area "
                      f"{m['false_area_px']} px) bands {m['n_bands']} | {m['time']:.0f} s")
        summary["neg"] = neg
    for mode in ("chain", "hybrid"):
        seq = _load(a, f"seq_{mode}.json")
        if not seq:
            continue
        print(f"SEQUENCE {mode}: k o | valid base xdic | rms base xdic | xdic COD / err / line dist")
        for rb, rx in zip(seq["base"], seq["xdic"]):
            ln = rx["line"]
            print(f"  {rb['k']:2d} {rb['o']:4.2f} | {_f(rb['valid'])} {_f(rx['valid'])} | {_f(rb['rms'], 5, 3)} {_f(rx['rms'], 5, 3)} | "
                  f"{_f(ln['cod_med'])} {_f(ln['cod_err'])} {_f(ln['dist'])}")
        print(f"  time base {seq['base'][-1]['time']:.0f} s, xdic {seq['xdic'][-1]['time']:.0f} s")
        summary[f"seq_{mode}"] = seq
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        k = [r["k"] for r in seq["base"]]
        for name in ("base", "xdic"):
            ax[0].plot(k, [r["valid"] for r in seq[name]], "o-", label=name)
            ax[1].plot(k, [r["rms"] for r in seq[name]], "o-", label=name)
        ax[2].plot(k, [r["o"] for r in seq["base"]], "k-", label="true opening")
        ax[2].plot(k, [r["line"]["cod_med"] for r in seq["xdic"]], "o-", label="xdic COD")
        for axx, tt in zip(ax, ("band valid share", "band RMS [px]", "COD [px]")):
            axx.set_title(f"{mode}: {tt}"); axx.legend(); axx.grid(alpha=.3); axx.set_xlabel("frame")
        fig.tight_layout(); fig.savefig(f"{a.out}/seq_{mode}.png", dpi=70); plt.close(fig)
    with open(f"{a.out}/summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=lambda o: None if o != o else o)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("pairs", "neg", "seq", "report"))
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--scratch", default=SCRATCH)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.cmd == "pairs":
        a.size = a.size or 384
        cmd_pairs(a)
    elif a.cmd == "neg":
        a.size = a.size or 384
        cmd_neg(a)
    elif a.cmd == "seq":
        a.size = a.size or 256
        a.mode = a.arg or "chain"
        cmd_seq(a)
    else:
        cmd_report(a)
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
