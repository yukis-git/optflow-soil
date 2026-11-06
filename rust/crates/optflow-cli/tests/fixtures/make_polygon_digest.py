"""Prints the digest that tests/polygon_digest.rs expects: Pillow's ImageDraw.polygon(fill=1) masks of
tens of thousands of pseudo-random simple polygons, folded into one FNV-1a 64-bit hash.

Run from the repository root:  python/.venv/bin/python rust/crates/optflow-cli/tests/fixtures/make_polygon_digest.py
(Pillow 13.0.0, arm64). The polygons come from the small linear congruential generator below, which
the Rust test repeats exactly, so no fixture file is needed: random simple polygons with integer
vertices (some partly outside the image), on images of three sizes. Per accepted polygon the hash
takes the image size (h, w as two bytes each), the number of vertices and the
`np.packbits` bytes of the row-major boolean mask.
"""
import numpy as np
from PIL import Image, ImageDraw

MASK64 = (1 << 64) - 1
POLYGONS = 40000
SIZES = [(12, 16), (20, 28), (33, 25)]
VERTEX_COUNTS = [3, 3, 4, 4, 5, 6, 7, 8, 9]


class Lcg:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next(self):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & MASK64
        return self.state >> 33

    def below(self, n):
        return self.next() % n

    def range(self, lo, hi):
        return lo + self.below(hi - lo + 1)


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


def fnv(h, data):
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & MASK64
    return h


def main():
    rng = Lcg(0x5EED0F901)
    h64 = 0xCBF29CE484222325
    accepted = 0
    while accepted < POLYGONS:
        h, w = SIZES[rng.below(len(SIZES))]
        n = VERTEX_COUNTS[rng.below(len(VERTEX_COUNTS))]
        pts = [(rng.range(-3, w + 2), rng.range(-3, h + 2)) for _ in range(n)]
        if not simple(pts):
            continue
        im = Image.new("1", (w, h), 0)
        ImageDraw.Draw(im).polygon(pts, fill=1)
        mask = np.asarray(im, dtype=bool)
        h64 = fnv(h64, bytes([h >> 8, h & 255, w >> 8, w & 255, n]))
        h64 = fnv(h64, np.packbits(mask.ravel()).tobytes())
        accepted += 1
    print(f"0x{h64:016X}")


if __name__ == "__main__":
    main()
