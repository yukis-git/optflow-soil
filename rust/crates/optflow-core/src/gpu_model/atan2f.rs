//! `atan2` in f32 from `+ - * /`, `mul_add` and comparisons only, so that MSL can mirror it line
//! by line.
//!
//! With `t = min(|y|, |x|) / max(|y|, |x|)` in `[0, 1]`: for `t > 2 - sqrt(3)` the argument is
//! reduced by `atan(t) = pi/6 + atan((sqrt(3) t - 1) / (sqrt(3) + t))`, so `|t'| <= tan(pi/12)`;
//! `atan(t') = t' (1 + t'^2 P(t'^2))` with `P` of degree 4. The quadrant follows from the swap of
//! min and max and the sign of `x`; the sign of `y` is copied last.
//!
//! Coefficients: minimax of `z (P(z) - g(z))`, `g(z) = (atan(sqrt z) / sqrt z - 1) / z`, on
//! `[0, (2 - sqrt 3)^2]`, found by a Remez exchange at 40 digits with mpmath (degree 4: maximum
//! error 4.6e-12 relative to `atan`, far below the f32 rounding), then rounded to f32.
//! Measured against `f64::atan2` rounded to f32 on 10^6 inputs over eight decades of magnitude
//! (`tests/gpu_model_parts.rs`, which prints it and asserts at most 2): 1.65 ulp at most, of which
//! about 0.5 is the rounding of the reference itself.

/// `sqrt(3)`, `2 - sqrt(3)`, `pi/6`, `pi/2` and `pi` as `hi + lo` (`hi` the f32 rounding).
use super::df64::two_sum;

const SQRT3_HI: f32 = 1.732_050_8;
const SQRT3_LO: f32 = 3.108_725e-8;
const T_REDUCE: f32 = 0.267_949_2;
const PIO6: (f32, f32) = (std::f32::consts::FRAC_PI_6, -1.457_046_3e-8);
const PIO2: (f32, f32) = (std::f32::consts::FRAC_PI_2, -4.371_139e-8);
const PI: (f32, f32) = (std::f32::consts::PI, -8.742_278e-8);

const P: [f32; 5] = [
    -0.333_333_33,
    0.199_999_4,
    -0.142_818_65,
    0.110_011_3,
    -0.076_371_97,
];

/// atan2 in f32 with the same quadrant rules as `f64::atan2` (signed zeros, infinities, NaN),
/// within 2 ulp of the correctly rounded result. Range reduction to |t| <= tan(pi/12) and a
/// minimax polynomial; only +, -, *, /, mul_add, comparisons, so MSL can mirror it line by line.
pub fn atan2f(y: f32, x: f32) -> f32 {
    if y.is_nan() || x.is_nan() {
        return f32::NAN;
    }
    let (mut ax, mut ay) = (x.abs(), y.abs());
    if ax == f32::INFINITY || ay == f32::INFINITY {
        // an infinite coordinate as its limit
        ax = if ax == f32::INFINITY { 1.0 } else { 0.0 };
        ay = if ay == f32::INFINITY { 1.0 } else { 0.0 };
    }
    let swap = ay > ax;
    let n = if swap { ax } else { ay };
    let d = if swap { ay } else { ax };
    let t = if d == 0.0 { 0.0 } else { n / d };
    // the reduction is only a decision; its argument is formed from n and d (not from the rounded
    // t), (sqrt(3) n - d) / (sqrt(3) d + n), with sqrt(3) as two parts
    let reduce = t > T_REDUCE;
    let u = if reduce {
        let num = SQRT3_HI.mul_add(n, -d) + SQRT3_LO * n;
        let den = SQRT3_HI.mul_add(d, n) + SQRT3_LO * d;
        num / den
    } else {
        t
    };
    let z = u * u;
    let p = P[4]
        .mul_add(z, P[3])
        .mul_add(z, P[2])
        .mul_add(z, P[1])
        .mul_add(z, P[0]);
    // atan(t) = [pi/6 +] u + u z p, kept as s + corr with the rounding of pi/6 + u recovered
    let (s, c0) = if reduce {
        let (s, e) = two_sum(PIO6.0, u);
        (s, e + PIO6.1)
    } else {
        (u, 0.0)
    };
    let corr = c0 + (u * z) * p;
    // angle = base + sign * atan(t): (pi/2 - a) for |y| > |x|, pi - a for x < 0, both: pi/2 + a
    let left = x.is_sign_negative();
    let (bh, bl) = if swap {
        PIO2
    } else if left {
        PI
    } else {
        (0.0, 0.0)
    };
    let (ss, sc) = if swap == left { (s, corr) } else { (-s, -corr) };
    let (h, e) = two_sum(bh, ss);
    let res = h + ((e + bl) + sc);
    res.copysign(y)
}
