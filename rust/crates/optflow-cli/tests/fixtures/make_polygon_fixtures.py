"""Writes polygon_cases.json: Pillow's ImageDraw.polygon(fill=1) masks, the reference for polygon.rs.

Run from the repository root:  python/.venv/bin/python rust/crates/optflow-cli/tests/fixtures/make_polygon_fixtures.py
(Pillow 13.0.0, arm64). Contents:
  random: simple polygons with integer vertices on a 20x28 image, some partly outside it;
          each mask is np.packbits of the row-major boolean mask, as hex.
  scaled: float vertices divided by a scale (the vertices are truncated toward zero by Pillow).
  roi:    data/example_roi.json at scales 1, 2 and 4: per row the list x_first, x_last, ... of its runs.
"""
import json
import os
import random

import numpy as np
import PIL
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))


def pil_mask(shape, pts):
    im = Image.new("1", (shape[1], shape[0]), 0)
    ImageDraw.Draw(im).polygon(pts, fill=1)
    return np.asarray(im, dtype=bool)


def orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_seg(a, b, c):
    return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])


def cross(p1, p2, p3, p4):
    d1, d2, d3, d4 = orient(p3, p4, p1), orient(p3, p4, p2), orient(p1, p2, p3), orient(p1, p2, p4)
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True
    return ((d1 == 0 and on_seg(p3, p4, p1)) or (d2 == 0 and on_seg(p3, p4, p2))
            or (d3 == 0 and on_seg(p1, p2, p3)) or (d4 == 0 and on_seg(p1, p2, p4)))


def simple(pts):
    n = len(pts)
    if len(set(pts)) != n:
        return False
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if cross(pts[i], pts[(i + 1) % n], pts[j], pts[(j + 1) % n]):
                return False
    return True


def packed(mask):
    return np.packbits(mask.ravel()).tobytes().hex()


def runs(mask):
    """Per row, the flat list [first, last, first, last, ...] of the runs of set pixels."""
    out = []
    for row in mask:
        xs = np.flatnonzero(row)
        flat = []
        if xs.size:
            breaks = np.flatnonzero(np.diff(xs) > 1)
            starts = np.r_[xs[0], xs[breaks + 1]]
            ends = np.r_[xs[breaks], xs[-1]]
            for a, b in zip(starts, ends):
                flat += [int(a), int(b)]
        out.append(flat)
    return out


def main():
    rng = random.Random(20261022)
    h, w = 20, 28
    cases = []
    while len(cases) < 160:
        n = rng.choice([3, 3, 4, 4, 5, 6, 7])
        pts = [(rng.randint(-3, w + 2), rng.randint(-3, h + 2)) for _ in range(n)]
        if simple(pts):
            cases.append({"shape": [h, w], "pts": pts, "mask": packed(pil_mask((h, w), pts))})
    scaled = []
    while len(scaled) < 30:
        sc = rng.choice([2, 3, 4])
        n = rng.choice([4, 5, 6])
        pts = [(rng.randint(-2 * sc, (w + 2) * sc) + rng.choice([0, 0.5, 0.25]),
                rng.randint(-2 * sc, (h + 2) * sc) + rng.choice([0, 0.5, 0.75])) for _ in range(n)]
        if simple([(int(x / sc), int(y / sc)) for x, y in pts]):
            scaled.append({"shape": [h, w], "scale": sc, "pts": pts,
                           "mask": packed(pil_mask((h, w), [(x / sc, y / sc) for x, y in pts]))})
    with open(os.path.join(ROOT, "data", "example_roi.json")) as f:
        roi = json.load(f)["polygon"]
    rois = []
    for sc in (1, 2, 4):
        shape = (4160 // sc, 6240 // sc)
        m = pil_mask(shape, [(x / sc, y / sc) for x, y in roi])
        rois.append({"scale": sc, "shape": list(shape), "area": int(m.sum()), "runs": runs(m)})
    out = {"pillow": PIL.__version__, "random": cases,
           "scaled": scaled, "roi": {"polygon": roi, "cases": rois}}
    path = os.path.join(HERE, "polygon_cases.json")
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(path, os.path.getsize(path), "bytes")


if __name__ == "__main__":
    main()
