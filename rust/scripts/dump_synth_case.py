"""Writes one benchmark pair as .npy files for `engine_compare` (and the GPU's `bench_gpu --compare`).

    python/.venv/bin/python rust/scripts/dump_synth_case.py --case band_D6_w5_o1 --n 512 --out DIR
    python/.venv/bin/python rust/scripts/dump_synth_case.py --real REF.JPG \
        DEF.JPG --scale 4 --roi data/example_roi.json --out DIR

    python/.venv/bin/python rust/scripts/dump_synth_case.py --seq 20 --w 1280 --h 720 --out DIR

--seq N writes N+1 16-bit PNG frames DIR/f_000.png ... for `optflow seq --frames DIR`: the motion of
the synthetic sequence case (algorithm spec 4.2, case 9): frame k is the pattern deformed by a
translation of (1.0, 0.5) px and a strain of (0.1 %, -0.03 %) per frame, times k. No noise.

Run with PYTHONPATH set to the `python/` directory of the checkout (so that `bench` and
`optflow_ref` import). Output: DIR/a.npy, DIR/b.npy (float32, the grey values of the frames: the
synthetic ones as the bench renders them, the real ones box-averaged by --scale and cropped to the
ROI's bounding box plus a margin) and DIR/set.txt (`key=value` lines: the case's config where it
differs from the default). The real-image config is (max_disp 400/scale,
sigma_w 6, sigma_s 8).
"""
import argparse
import json
import os
from dataclasses import asdict

import numpy as np

from bench.rustcli import write_png16
from bench.cases import all_cases
from bench.run_synth import SEED_DEF, SEED_REF, frame
from optflow_ref import synth
from optflow_ref.config import Config
from optflow_ref.io import downscale, load_roi, polygon_mask, read_image


def write_set(path, cfg):
    base = asdict(Config())
    with open(path, "w") as fh:
        for k, v in asdict(cfg).items():
            if v != base[k]:
                fh.write(f"{k}={json.dumps(v)}\n")


def synthetic(a):
    case = next((c for c in all_cases(a.n) if c.name == a.case), None)
    if case is None:
        raise SystemExit(f"no case {a.case!r} in all_cases({a.n})")
    if len(case.fields) != 1:
        raise SystemExit(f"{a.case}: a sequence case; only pair cases are dumped")
    pattern = synth.make_pattern(a.n, a.n, seed=a.seed)
    ref = frame(pattern, a.n, None, case, seed=SEED_REF)
    dfm = frame(pattern, a.n, case.fields[0], case, seed=SEED_DEF)
    return ref, dfm, case.cfg


def sequence(a):
    pattern = synth.make_pattern(a.h, a.w, seed=a.seed)
    c = (a.w / 2, a.h / 2)
    fields = [None] + [synth.uniform_strain(0.001 * k, -0.0003 * k, c, t=(1.0 * k, 0.5 * k)) for k in range(1, a.seq + 1)]
    os.makedirs(a.out, exist_ok=True)
    for k, f in enumerate(fields):
        write_png16(os.path.join(a.out, f"f_{k:03d}.png"), synth.render(pattern, a.h, a.w, f))
    print(f"{a.out}: {len(fields)} frames {a.w}x{a.h}")


def real(a):
    s = a.scale
    cfg = Config(max_disp=400 / s, sigma_w=6.0, sigma_s=8.0)
    full = [downscale(read_image(p), s) for p in a.real]
    roi = polygon_mask(full[0].shape, load_roi(a.roi), s)
    ys, xs = np.where(roi)
    mx, my = a.margin_x // s, a.margin_y // s
    y0, y1 = max(ys.min() - my, 0), min(ys.max() + my + 1, roi.shape[0])
    x0, x1 = max(xs.min() - mx, 0), min(xs.max() + mx + 1, roi.shape[1])
    return [np.ascontiguousarray(f[y0:y1, x0:x1]) for f in full] + [cfg]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real", nargs=2, metavar=("REF", "DEF"))
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--roi")
    ap.add_argument("--margin-x", type=int, default=640, help="full-resolution px left and right of the ROI")
    ap.add_argument("--margin-y", type=int, default=128, help="full-resolution px above and below the ROI")
    ap.add_argument("--seq", type=int, help="write a synthetic sequence of this many steps (needs --w, --h)")
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.seq:
        return sequence(a)
    if bool(a.case) == bool(a.real):
        ap.error("give exactly one of --case and --real")
    if a.real and not a.roi:
        ap.error("--real needs --roi")
    ref, dfm, cfg = synthetic(a) if a.case else real(a)
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "a.npy"), np.ascontiguousarray(ref, dtype=np.float32))
    np.save(os.path.join(a.out, "b.npy"), np.ascontiguousarray(dfm, dtype=np.float32))
    write_set(os.path.join(a.out, "set.txt"), cfg)
    print(f"{a.out}: {ref.shape[1]}x{ref.shape[0]}")


if __name__ == "__main__":
    main()
