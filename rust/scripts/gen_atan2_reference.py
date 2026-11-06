#!/usr/bin/env python3
"""Writes rust/crates/optflow-core/tests/atan2_mpmath.txt: inputs of atan2 with the
correctly rounded f64 of the exact angle, the reference of the atan2 tests in optflow-core (kept
outside tests/fixtures, whose size budget is for the parity fixtures of the Python reference).

    python3 rust/scripts/gen_atan2_reference.py > rust/crates/optflow-core/tests/atan2_mpmath.txt

Needs mpmath (any Python 3.9+; the table in the repository was written with Python 3.9.6 and
mpmath 1.4.1). The angle is computed at 200 bits and rounded to the nearest f64 by comparing the
neighbouring doubles in that precision, so the result does not depend on mpmath's own rounding to
float. The inputs are deterministic (random.Random(20261023)): the whole plane, small angles about
the positive x axis (phase differences of a converged fit), coordinates of very different sizes,
finite doubles over the whole exponent range (subnormal to 1.7e308), the interval ends 7/16 and
11/16 of the reduction, the ratios near 1/4 and 1/8 where the f64 implementation errs most, and
the diagonals at extreme sizes; no zero coordinate (its signed-zero rule is not modelled by
mpmath and is tested on its own). Each line: y, x and the reference as the hexadecimal bits of
the f64.
"""
import math
import random
import struct
import sys

import mpmath

mpmath.mp.prec = 200


def bits(v):
    return struct.unpack("<Q", struct.pack("<d", v))[0]


def from_bits(b):
    return struct.unpack("<d", struct.pack("<Q", b))[0]


def nearest(exact):
    """The f64 nearest to the mpf `exact` (ties to even), by comparing the neighbours of float()."""
    f = float(exact)
    cands = {f, math.nextafter(f, math.inf), math.nextafter(f, -math.inf)}
    best = min(cands, key=lambda c: (abs(mpmath.mpf(c) - exact), bits(c) & 1))
    return best


rnd = random.Random(20261023)
pts = []
for _ in range(500):
    pts.append((rnd.uniform(-1, 1), rnd.uniform(-1, 1)))
for _ in range(400):
    m, a = 0.1 + rnd.random(), (rnd.random() - 0.5) * 0.6
    pts.append((m * math.sin(a), m * math.cos(a)))
for _ in range(300):
    m, a = 0.1 + rnd.random(), (rnd.random() - 0.5) * 0.02
    pts.append((m * math.sin(a), m * math.cos(a)))
for _ in range(400):
    e1, e2 = 2.0 ** rnd.uniform(-40, 40), 2.0 ** rnd.uniform(-40, 40)
    pts.append(((rnd.random() - 0.5) * e1, (rnd.random() - 0.5) * e2))
while len(pts) < 1900:
    # finite doubles over the whole range: random sign, exponent field 0..2046, mantissa
    def anyfinite():
        return from_bits((rnd.getrandbits(1) << 63) | (rnd.randrange(0, 2047) << 52) | rnd.getrandbits(52))
    pts.append((anyfinite(), anyfinite()))
for c in (7 / 16, 11 / 16, 0.25, 0.125, 0.0625, 1.0):
    for _ in range(80):
        x = rnd.uniform(0.5, 2.0) * 2.0 ** rnd.randrange(-20, 20)
        y = x * c * (1 + rnd.uniform(-1e-3, 1e-3))
        sx, sy = rnd.choice((1, -1)), rnd.choice((1, -1))
        pts.append((sy * y, sx * x) if rnd.random() < 0.5 else (sx * x, sy * y))
for v in (1.0, 1e-300, 1e300, 5e-324, 1.7976931348623157e308):
    for y, x in ((v, v), (-v, v), (v, -v), (-v, -v), (v, 1.0), (1.0, -v)):
        pts.append((y, x))
# a zero coordinate has a signed-zero rule (C99) that mpmath does not model: tested elsewhere
pts = [(y, x) for y, x in pts if y != 0.0 and x != 0.0]

out = sys.stdout
out.write("# atan2 reference: y x correctly-rounded-atan2(y, x), f64 bits in hex; written by\n")
out.write("# rust/scripts/gen_atan2_reference.py (mpmath %s, 200 bits); do not edit.\n" % mpmath.__version__)
for y, x in pts:
    exact = mpmath.atan2(mpmath.mpf(y), mpmath.mpf(x))
    ref = nearest(exact)
    out.write("%016x %016x %016x\n" % (bits(y), bits(x), bits(ref)))
