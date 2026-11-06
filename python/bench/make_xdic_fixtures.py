"""Python-reference inputs and expected outputs for the Rust parity tests of crack_model "xdic"
(rust/crates/optflow-core/tests/parity_xdic.rs).

    python/.venv/bin/python python/bench/make_xdic_fixtures.py [--only NAME,...] [--jobs 2]

Output: rust/crates/optflow-core/tests/fixtures/xdic/xd_*.npy and xdic_meta.json (a directory of its own: the
size and dtype rules of the other fixtures, f32 and under 8 MB, do not apply to these u16 images). The images are 16-bit
quantised (u16, so that the fixtures are small): the reference runs on exactly the values the Rust
test reads, (u16 / 65535) rounded to float32. The cases (N = 160): crack pairs at normal angle 0 and
30 degrees with opening 1 and 6 px, the negative controls (ShearBand, weak texture), a step field for
the detector, a polyline for the distance field, a chain and a hybrid sequence with a widening
opening (0 -> 24 px, the reference moves every frame: the frame-0 line mapping past 12 px), and the
frame-0 mapping of a line in a gap. Re-run whenever python/optflow_ref/xdic.py or track.py changes.
"""
import argparse
import json
import os
import sys
from multiprocessing import get_context

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.dirname(HERE)
sys.path.insert(0, PY)
from demo.crack_ab_eval import Case, neg_images  # noqa: E402
from optflow_ref import synth, xdic  # noqa: E402
from optflow_ref.config import Config  # noqa: E402
from optflow_ref.track import track_pair, track_sequence  # noqa: E402

ROOT = os.path.dirname(PY)
OUT = os.path.join(ROOT, "rust", "crates", "optflow-core", "tests", "fixtures", "xdic")
N = 192  # pairs and negative controls (the opening-1 crack is not detected at 160)
NSEQ = 160
PAIRS = {  # name: (alpha, opening, slip)
    "a0_o1": (0.0, 1.0, 0.0),
    "a0_o6": (0.0, 6.0, 2.0),
    "a30_o1": (30.0, 1.0, 0.0),
    "a30_o6": (30.0, 6.0, 2.0),
}
NEG = ("weak", "band")


def q16(img):
    return np.clip(np.rint(np.clip(img, 0.0, 1.0) * 65535.0), 0, 65535).astype(np.uint16)


def deq(u):
    """The images as the Rust tests read them: (u16 / 65535) rounded to float32, widened to float64."""
    return (u.astype(np.float64) / 65535.0).astype(np.float32).astype(np.float64)


def save(name, a):
    np.save(os.path.join(OUT, name + ".npy"), np.ascontiguousarray(a))


def f32(a):
    return np.asarray(a, dtype=np.float32)


def lines_json(lines):
    return [np.asarray(p, float).tolist() for p in lines]


def jumps_json(jumps):
    return [dict(pts=j["pts"].tolist(), cod=j["cod"].tolist(), csd=j["csd"].tolist()) for j in jumps]


def result_meta(r):
    cl = r.crack_lines
    meta = dict(n_lines=0, bands=[], rejected=[], lines=[], jumps=[], crack_px=0)
    if cl is not None:
        meta.update(n_lines=len(cl["lines"]), bands=lines_json(cl.get("bands", [])), rejected=lines_json(cl.get("rejected", [])),
                    lines=lines_json(cl["lines"]), jumps=jumps_json(cl["jumps"]))
    if r.crack is not None:
        meta["crack_px"] = int(r.crack["mask"].sum())
    return meta


def run_pair(args):
    key, a16, b16 = args
    a, b = deq(a16), deq(b16)
    r = track_pair(a, b, Config(crack_model="xdic"))
    if key in PAIRS:  # the negative controls are compared with the Rust baseline, bit for bit
        save(f"xd_{key}_ux", f32(r.ux))
        save(f"xd_{key}_uy", f32(r.uy))
        save(f"xd_{key}_conf", f32(r.conf))
    if key == "a30_o6" and r.crack is not None:
        save(f"xd_{key}_Exx", f32(r.strain["Exx"]))
        save(f"xd_{key}_cod", f32(r.crack["cod"]))
    print("pair", key, "lines", result_meta(r)["n_lines"], flush=True)
    return key, result_meta(r)


def run_seq(args):
    mode, frames16 = args
    frames = [deq(f) for f in frames16]
    cfg = Config(crack_model="xdic", seq_mode=mode, ref_update_conf=0.99)
    per = []
    last = None
    for k, r in track_sequence(frames, cfg):
        m = result_meta(r)
        m["k"] = k
        m["reference"] = int(r.timing["reference"])
        per.append(m)
        last = r
        print("seq", mode, k, m["n_lines"], flush=True)
    save(f"xd_seq_{mode}_ux", f32(last.ux))
    save(f"xd_seq_{mode}_uy", f32(last.uy))
    save(f"xd_seq_{mode}_conf", f32(last.conf))
    return mode, per


def step_field(n=160, j=(2.0, 0.6), hole=12, noise=0.03, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    s = xx - 80.3
    ux = 0.5 + 0.002 * yy + np.where(s >= 0, j[0], 0.0) + noise * rng.standard_normal((n, n))
    uy = -0.3 + np.where(s >= 0, j[1], 0.0) + noise * rng.standard_normal((n, n))
    bad = (np.abs(s) < hole) | (np.minimum.reduce([xx, yy, n - 1 - xx, n - 1 - yy]) < 28)
    return np.where(bad, np.nan, ux), np.where(bad, np.nan, uy)


def units():
    """Detector, distance field and geometry helpers on small inputs."""
    meta = {}
    cfg = Config(crack_model="xdic")
    ux, uy = step_field()
    ux, uy = f32(ux).astype(np.float64), f32(uy).astype(np.float64)
    save("xd_step_ux", f32(ux))
    save("xd_step_uy", f32(uy))
    lines, info = xdic.detect_lines(ux, uy, None, cfg)
    pts = np.array([[80, 80], [60, 60], [100, 100], [40, 100], [80, 40]])
    J, z, th = xdic.jump_test(ux, uy, np.isfinite(ux), pts[:, 0], pts[:, 1], kappa=4 * np.pi * 16)
    meta["step"] = dict(lines=lines_json(lines), n_cand=info["n_cand"], n_acc=info["n_acc"], pts=pts.tolist(),
                        J=np.nan_to_num(J, nan=-999).tolist(), z=np.nan_to_num(z, nan=-999).tolist(),
                        theta=np.nan_to_num(th, nan=-999).tolist())
    # distance field of a bent polyline pair on 64 x 64
    P = [np.array([[10.0, 5.0], [30.0, 25.0], [50.0, 28.0], [60.0, 55.0]]), np.array([[5.0, 50.0], [20.0, 58.0]])]
    f = xdic.polyline_fields(P, (64, 64))
    for k in ("phi", "nx", "ny", "tau", "lid"):
        save(f"xd_poly_{k}", f32(f[k]))
    save("xd_poly_interior", f["interior"])
    meta["poly"] = dict(lines=lines_json(P))
    # resample, vertex normals, advance and the frame-0 mapping
    pts_, nrm_, s_ = xdic._resample(P[0], 7.0)
    meta["resample"] = dict(pts=pts_.tolist(), n=nrm_.tolist(), s=s_.tolist())
    H = W = 64
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    ax = 0.5 + 0.03 * xx - 0.01 * yy + np.where(xx > 30, 1.5, 0.0)
    ay = -0.4 + 0.02 * yy + 0.005 * xx
    meta["advance"] = dict(lines=lines_json(xdic.advance_lines(P, ax, ay)))
    near = xdic.near_lines(P[0], [P[0] + 2.0], 5.0), xdic.near_lines(P[0], [P[0] + 12.0], 5.0)
    meta["near"] = [bool(v) for v in near]
    wide = []
    Hh = Ww = 256
    xg = np.mgrid[0:Hh, 0:Ww][1].astype(float)
    for o in (16.0, 30.0):
        for off in (0.0, 0.25, -0.25):
            u = np.where(xg >= 128, o, 0.0)
            v = np.full((Hh, Ww), -3.0)
            Pw = np.array([[128 + o / 2 + off * o, 20.0], [128 + o / 2 + off * o, 230.0]])
            X = xdic.lines_to_frame0([Pw], (u, v))[0]
            wide.append(dict(o=o, off=off, P=Pw.tolist(), X=X.tolist()))
    meta["wide"] = wide
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    meta_path = os.path.join(OUT, "xdic_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    meta.update(N=N, NSEQ=NSEQ)
    want = lambda k: not only or k in only  # noqa: E731
    jobs = []
    for key, (alpha, o, s) in PAIRS.items():
        if want(key):
            case = Case(N, alpha, o, s)
            ai, bi = case.images()
            a16, b16 = q16(ai), q16(bi)
            save(f"xd_{key}_a", a16)
            save(f"xd_{key}_b", b16)
            jobs.append((key, a16, b16))
    for kind in NEG:
        if want(kind):
            ai, bi, _, _, _ = neg_images(kind, N)
            a16, b16 = q16(ai), q16(bi)
            save(f"xd_{kind}_a", a16)
            save(f"xd_{kind}_b", b16)
            jobs.append((kind, a16, b16))
    seq_jobs = []
    if want("seq"):
        pat = synth.make_pattern(NSEQ, NSEQ, seed=21)
        c = (NSEQ / 2.0, NSEQ / 2.0)
        fields = [synth.Crack(c, 0.0, 4.0 * k, 0.0) for k in range(7)]
        frames = [q16(synth.degrade(synth.render(pat, NSEQ, NSEQ, f), seed=100 + k, noise=0.01)) for k, f in enumerate(fields)]
        save("xd_seq_frames", np.stack(frames))
        seq_jobs = [(m, frames) for m in ("chain", "hybrid")]
    if want("units"):
        meta["units"] = units()
    with get_context("spawn").Pool(a.jobs) as pool:
        res_pairs = pool.map(run_pair, jobs, chunksize=1)
        res_seq = pool.map(run_seq, seq_jobs, chunksize=1)
    for key, m in res_pairs:
        meta[f"pair_{key}"] = m
    for mode, per in res_seq:
        meta[f"seq_{mode}"] = per
    with open(meta_path, "w") as fh:
        json.dump(meta, fh)
    print("wrote", meta_path)


if __name__ == "__main__":
    main()
