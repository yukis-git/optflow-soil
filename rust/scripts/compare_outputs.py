#!/usr/bin/env python3
"""Compare two optflow output directories (A = reference CPU, B = candidate GPU).

usage: compare_outputs.py A B [--frame N] [--tier plain|robust]
  A, B   output dirs of `optflow pair` (ux.npy, uy.npy, conf.npy, Exx.npy, ...) or `optflow seq`
         (frame_00001/ ... below it; choose one with --frame)
Run with numpy only, e.g. python/.venv/bin/python rust/scripts/compare_outputs.py A B --tier robust

Prints the plain GPU-spec section 5 metrics and the robust ones; pass/fail is judged for --tier
(default plain). Exit status 1 on fail, 0 on pass.

plain  (synthetic and 512^2): ux/uy RMS <= 1e-5 over pixels finite in both; conf RMS <= 1e-5 over all
       pixels; strain RMS <= 1e-6 over finite-both; valid agreement >= 0.999.
robust (real full frames and sequences): ux/uy RMS of the 99.9 % smallest |d| <= 1e-5; ux/uy p99 |d|
       <= 1e-5; fraction of finite-both pixels with |du| > 1e-4 <= 0.5 %; conf RMS over finite-both
       pixels <= 2e-5; strain RMS of the 99.9 % smallest <= 2e-6; valid agreement >= 0.999.
"""
import argparse
import os
import sys

import numpy as np

STRAIN = ("Exx", "Eyy", "Exy", "gmax", "vol")


def load(d, n):
    p = os.path.join(d, n + ".npy")
    return np.load(p).astype(np.float64) if os.path.exists(p) else None


def rms(x):
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")


def tail(d):
    """sorted |d|: (p99, rms of smallest 99.9 %, fraction > 1e-4)"""
    s = np.sort(np.abs(d))
    if not s.size:
        return float("nan"), float("nan"), float("nan")
    return float(s[int(len(s) * 0.99)]), rms(s[: int(len(s) * 0.999)]), float(np.mean(s > 1e-4))


def metrics(a, b):
    m = {}
    ux1, ux2 = load(a, "ux"), load(b, "ux")
    both = np.isfinite(ux1) & np.isfinite(ux2)
    m["both_finite"] = int(both.sum())
    m["valid_agree"] = float(np.mean(np.isfinite(ux1) == np.isfinite(ux2)))
    m["one_sided"] = int((np.isfinite(ux1) ^ np.isfinite(ux2)).sum())
    for n in ("ux", "uy"):
        d = (load(a, n) - load(b, n))[both]
        p99, trim, frac = tail(d)
        m[n] = dict(rms=rms(d), p99=p99, trim=trim, frac=frac, max=float(np.abs(d).max()) if d.size else float("nan"))
    c = load(a, "conf") - load(b, "conf")
    cf = c[both]
    # conf over pixels finite in both is taken from ux finiteness of both runs
    fin = np.isfinite(c)
    m["conf"] = dict(rms_all=rms(c[fin]), rms_both=rms(cf[np.isfinite(cf)]), trim_all=tail(c[fin])[1])
    for n in STRAIN:
        x, y = load(a, n), load(b, n)
        if x is None or y is None:
            continue
        f = np.isfinite(x) & np.isfinite(y)
        d = (x - y)[f]
        m[n] = dict(rms=rms(d), trim=tail(d)[1], max=float(np.abs(d).max()) if d.size else float("nan"))
    return m


def judge(m, tier):
    """list of (name, value, limit, ok)"""
    r = []

    def add(name, v, lim):
        r.append((name, v, lim, bool(v <= lim)))

    robust = tier == "robust"
    for n in ("ux", "uy"):
        add(f"{n} " + ("trim99.9 rms" if robust else "rms"), m[n]["trim" if robust else "rms"], 1e-5)
        if robust:
            add(f"{n} p99 |d|", m[n]["p99"], 1e-5)
    if robust:
        add("ux frac |d|>1e-4", m["ux"]["frac"], 5e-3)
        add("uy frac |d|>1e-4", m["uy"]["frac"], 5e-3)
        add("conf rms (finite-both)", m["conf"]["rms_both"], 2e-5)
    else:
        add("conf rms (all)", m["conf"]["rms_all"], 1e-5)
    for n in STRAIN:
        if n in m:
            add(f"{n} " + ("trim99.9 rms" if robust else "rms"), m[n]["trim" if robust else "rms"], 2e-6 if robust else 1e-6)
    r.append(("valid agreement", m["valid_agree"], 0.999, m["valid_agree"] >= 0.999))
    return r


def resolve(d, frame):
    if os.path.exists(os.path.join(d, "ux.npy")):
        return d
    if frame is None:
        sys.exit(f"{d}: no ux.npy (a seq output); pass --frame N")
    return os.path.join(d, f"frame_{frame:05d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--frame", type=int)
    ap.add_argument("--tier", choices=("plain", "robust"), default="plain")
    o = ap.parse_args()
    a, b = resolve(o.a, o.frame), resolve(o.b, o.frame)
    m = metrics(a, b)
    print(f"A={a}\nB={b}")
    print(f"both-finite px {m['both_finite']}, one-sided px {m['one_sided']}, valid agreement {m['valid_agree']:.7f}")
    for n in ("ux", "uy"):
        v = m[n]
        print(f"{n}: rms {v['rms']:.3g} trim99.9 {v['trim']:.3g} p99 {v['p99']:.3g} frac>1e-4 {v['frac']:.3g} max {v['max']:.3g}")
    c = m["conf"]
    print(f"conf: rms(all) {c['rms_all']:.3g} rms(finite-both) {c['rms_both']:.3g} trim99.9(all) {c['trim_all']:.3g}")
    for n in STRAIN:
        if n in m:
            v = m[n]
            print(f"{n}: rms {v['rms']:.3g} trim99.9 {v['trim']:.3g} max {v['max']:.3g}")
    for tier in ("plain", "robust"):
        res = judge(m, tier)
        ok = all(x[3] for x in res)
        mark = "  <== judged" if tier == o.tier else ""
        print(f"[{tier}] {'PASS' if ok else 'FAIL'}{mark}")
        for name, v, lim, k in res:
            print(f"  {'ok  ' if k else 'FAIL'} {name}: {v:.3g} (limit {lim:g})")
        if tier == o.tier:
            final = ok
    print(f"RESULT tier={o.tier} {'pass' if final else 'fail'}")
    sys.exit(0 if final else 1)


if __name__ == "__main__":
    main()
