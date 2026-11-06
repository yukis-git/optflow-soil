//! `atan2` in `f64` from plain arithmetic, the same on every platform.
//!
//! `accumulate` of [`crate::phaseflow`] takes the phase difference of every filter at every
//! pixel and inner iteration, 50 million `atan2` per 512 x 512 pair; through the platform's libm
//! that was about a quarter of the stage. [`atan2_all`] evaluates a whole filter bank's worth at
//! once without branches, so that the compiler vectorises it, and its results do not depend on
//! the platform's libm.
//!
//! Accuracy, measured against `mpmath.atan2` at 200 bits on 200 000 inputs (the whole plane,
//! small angles about the positive x axis, coordinates from 2^-40 to 2^40): at most 1.46 ulp of
//! the exact value, 99.9 % within 1 ulp; macOS libm: at most 0.73 ulp. The two agree on 83 % of
//! the inputs. The errors above 1 ulp come from the rounding of `n / d` where `atan` crosses a
//! power of two just below `t` (`t` near 1/4, 1/8). The tests check the function against a table
//! of correctly rounded values (`tests/atan2_mpmath.txt`, written by
//! `rust/scripts/gen_atan2_reference.py` with mpmath), so they do not depend on the platform's
//! libm, which they compare against only loosely.
//!
//! The method is that of fdlibm's `atan` (`s_atan.c`, also in musl and FreeBSD msun), with its
//! coefficients: with `n = min(|y|, |x|)`, `d = max(|y|, |x|)`, the ratio `t = n / d` in `[0, 1]`
//! is reduced to `|t'| <= 7/16` by `atan(t) = atan(c) + atan((t - c) / (1 + c t))` with
//! `c = 0` on `[0, 7/16]`, `c = 1/2` on `(7/16, 11/16)` and `c = 1` on `[11/16, 1]` (the reduced
//! argument formed from `n` and `d` with one division), `atan(t')` is an odd polynomial of
//! degree 23, and `atan(c)` enters as a two-part constant. The quadrant follows from the swap of
//! `n` and `d` and the sign of `x`, with `pi / 2` and `pi` as two-part constants as well; the sign
//! of `y` is copied last. Plain `f64` operations only (no fused multiply-add), so the results are
//! the same bit for bit wherever IEEE 754 double arithmetic is.

/// `aT[0..11]` of fdlibm `s_atan.c`: `atan(t) = t - t (aT0 z + aT1 z^2 + ...)`, `z = t^2`.
const AT: [f64; 11] = [
    f64::from_bits(0x3FD5_5555_5555_550D), // 3.33333333333329318027e-01
    f64::from_bits(0xBFC9_9999_9998_EBC4), // -1.99999999998764832476e-01
    f64::from_bits(0x3FC2_4924_9200_83FF), // 1.42857142725034663711e-01
    f64::from_bits(0xBFBC_71C6_FE23_1671), // -1.11111104054623557880e-01
    f64::from_bits(0x3FB7_45CD_C54C_206E), // 9.09088713343650656196e-02
    f64::from_bits(0xBFB3_B0F2_AF74_9A6D), // -7.69187620504482999495e-02
    f64::from_bits(0x3FB1_0D66_A0D0_3D51), // 6.66107313738753120669e-02
    f64::from_bits(0xBFAD_DE2D_52DE_FD9A), // -5.83357013379057348645e-02
    f64::from_bits(0x3FA9_7B4B_2476_0DEB), // 4.97687799461593236017e-02
    f64::from_bits(0xBFA2_B444_2C6A_6C2F), // -3.65315727442169155270e-02
    f64::from_bits(0x3F90_AD3A_E322_DA11), // 1.62858201153657823623e-02
];

/// `atan(1/2)` and `atan(1) = pi / 4` as `hi + lo` (fdlibm `atanhi`, `atanlo`).
const ATAN_HALF: (f64, f64) = (
    f64::from_bits(0x3FDD_AC67_0561_BB4F),
    f64::from_bits(0x3C7A_2B7F_222F_65E2),
);
const PIO4: (f64, f64) = (
    f64::from_bits(0x3FE9_21FB_5444_2D18),
    f64::from_bits(0x3C81_A626_3314_5C07),
);
/// `pi / 2` and `pi` as `hi + lo`.
const PIO2: (f64, f64) = (
    f64::from_bits(0x3FF9_21FB_5444_2D18),
    f64::from_bits(0x3C91_A626_3314_5C07),
);
const PI: (f64, f64) = (
    f64::from_bits(0x4009_21FB_5444_2D18),
    f64::from_bits(0x3CA1_A626_3314_5C07),
);

/// The largest `|y|`, `|x|` that [`atan2_finite`] takes, 2^1020: up to it the reduced argument's
/// `d + d + n` (at most `3 max(|y|, |x|)`) stays finite.
const BIG: f64 = f64::from_bits(0x7FB0_0000_0000_0000);

/// The factor 2^-4 that brings finite coordinates above [`BIG`] into range: a power of two, so
/// the ratio of the two is unchanged (the smaller loses bits only below 2^-1018, where its ratio
/// to the larger underflows to 0 anyway).
const SHRINK: f64 = f64::from_bits(0x3FB0_0000_0000_0000);

/// `atan2(y, x)` for `|y|, |x| <= BIG` (any other input gives a meaningless value). No
/// branches: the selections are between computed values, so a loop of these vectorises.
#[inline(always)]
fn atan2_finite(y: f64, x: f64) -> f64 {
    let (ax, ay) = (x.abs(), y.abs());
    let swap = ay > ax;
    let n = if swap { ax } else { ay };
    let d = if swap { ay } else { ax };
    // t = n / d: c = 0 on [0, 7/16] (also n = d = 0), 1/2 on (7/16, 11/16), 1 on [11/16, 1]
    let c0 = n <= 0.4375 * d;
    let mid = !c0 & (n < 0.6875 * d);
    // t' = (t - c) / (1 + c t) as (p n - q d) / (q n + p d), (p, q) = (1, 0), (2, 1), (1, 1)
    let num = (n + if mid { n } else { 0.0 }) - if c0 { 0.0 } else { d };
    let den = (d + if mid { d } else { 0.0 }) + if c0 { 0.0 } else { n };
    let den = if den == 0.0 { 1.0 } else { den }; // n = d = 0: t' = 0
    let (hi, lo) = if c0 {
        (0.0, 0.0)
    } else if mid {
        ATAN_HALF
    } else {
        PIO4
    };
    let t = num / den;
    let z = t * t;
    let w = z * z;
    let w2 = w * w;
    // the even and odd parts of fdlibm's polynomial in w, in pairs (Estrin): a shorter chain
    let s1 = z * ((AT[0] + w * AT[2]) + w2 * ((AT[4] + w * AT[6]) + w2 * (AT[8] + w * AT[10])));
    let s2 = w * ((AT[1] + w * AT[3]) + w2 * ((AT[5] + w * AT[7]) + w2 * AT[9]));
    // a = atan(n / d) = atan(c) + atan(t')
    let a = hi - ((t * (s1 + s2) - lo) - t);
    // the angle of (|x|, |y|) is a, or pi/2 - a if swapped; for x < 0 it is pi minus that:
    // base - (s a - base_lo) with (base, s) = (0, -1), (pi/2, 1), (pi, 1), (pi/2, -1)
    let left = x.is_sign_negative();
    let (bh, bl) = if swap {
        PIO2
    } else if left {
        PI
    } else {
        (0.0, 0.0)
    };
    let sa = if swap != left { a } else { -a };
    (bh - (sa - bl)).copysign(y)
}

/// The angle of the point `(x, y)` in `[-pi, pi]`, as `f64::atan2` for every input (C99 Annex F
/// for the zeros, infinities and NaN), within 2 ulp of the correctly rounded value (see the
/// module doc), and the same on every platform.
pub(crate) fn atan2(y: f64, x: f64) -> f64 {
    // false for NaN and the infinities
    if y.abs() <= BIG && x.abs() <= BIG {
        return atan2_finite(y, x);
    }
    if y.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if y.is_finite() && x.is_finite() {
        // above 2^1020 the sums of the reduced argument could overflow: both scaled by 2^-4
        return atan2_finite(y * SHRINK, x * SHRINK);
    }
    // an infinite coordinate as its limit: (inf, finite) -> (1, 0), (inf, inf) -> (1, 1)
    let unit = |v: f64| if v.is_infinite() { 1.0f64 } else { 0.0 };
    atan2_finite(unit(y).copysign(y), unit(x).copysign(x))
}

/// `out[k] = atan2(y[k], x[k])` for every `k < out.len()` ([`atan2`], bit for bit). When every
/// value is at most [`BIG`] in magnitude (the usual case: finite and not near `f64::MAX`) the loop
/// has no branches and vectorises.
#[inline(always)]
pub(crate) fn atan2_all(y: &[f64], x: &[f64], out: &mut [f64]) {
    let n = out.len();
    let (y, x) = (&y[..n], &x[..n]);
    let in_range = y
        .iter()
        .zip(x)
        .fold(true, |ok, (a, b)| ok & (a.abs() <= BIG) & (b.abs() <= BIG));
    if in_range {
        for ((o, &a), &b) in out.iter_mut().zip(y).zip(x) {
            *o = atan2_finite(a, b);
        }
    } else {
        for ((o, &a), &b) in out.iter_mut().zip(y).zip(x) {
            *o = atan2(a, b);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distance in units in the last place (0 for two NaNs).
    /// The largest distance accepted from the platform's libm, which is not the reference: this
    /// function is within 1.5 ulp of the correctly rounded value (the mpmath table), a libm within
    /// 2.5 ulp of it may then be 4 ulp away.
    const LIBM_ULPS: u64 = 4;

    fn ulps(a: f64, b: f64) -> u64 {
        if a.is_nan() && b.is_nan() {
            return 0;
        }
        let key = |v: f64| {
            let bits = v.to_bits() as i64;
            if bits < 0 {
                i64::MIN - bits
            } else {
                bits
            }
        };
        (key(a) - key(b)).unsigned_abs()
    }

    fn lcg(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// The constants are fdlibm's (the decimal values of `s_atan.c`; the bit patterns above
    /// were transcribed from its hexadecimal comments), the high parts are the angles rounded
    /// to `f64`, and each low part is below half an ulp of its high part.
    #[test]
    fn atan2_constants_are_those_of_fdlibm() {
        let dec = |s: &str| s.parse::<f64>().expect("decimal");
        let at = [
            "3.33333333333329318027e-01",
            "-1.99999999998764832476e-01",
            "1.42857142725034663711e-01",
            "-1.11111104054623557880e-01",
            "9.09088713343650656196e-02",
            "-7.69187620504482999495e-02",
            "6.66107313738753120669e-02",
            "-5.83357013379057348645e-02",
            "4.97687799461593236017e-02",
            "-3.65315727442169155270e-02",
            "1.62858201153657823623e-02",
        ];
        for (c, d) in AT.iter().zip(at) {
            assert_eq!(*c, dec(d), "{d}");
        }
        // atan(1/2) to 50 digits, rounded to f64, is the high part
        assert_eq!(
            ATAN_HALF.0,
            dec("0.46364760900080611621425623146121440202853705428612")
        );
        assert_eq!(ATAN_HALF.1, dec("2.26987774529616870924e-17"));
        assert_eq!(PIO4.1, dec("3.06161699786838301793e-17"));
        assert_eq!(PIO4.0, std::f64::consts::FRAC_PI_4);
        assert_eq!(PIO2.0, std::f64::consts::FRAC_PI_2);
        assert_eq!(PI.0, std::f64::consts::PI);
        assert_eq!(PIO2.1, 2.0 * PIO4.1);
        assert_eq!(PI.1, 4.0 * PIO4.1);
        for (hi, lo) in [ATAN_HALF, PIO4, PIO2, PI] {
            assert!(lo > 0.0 && lo < 0.5 * (f64::from_bits(hi.to_bits() + 1) - hi));
        }
    }

    /// Against correctly rounded values (mpmath at 200 bits, `tests/atan2_mpmath.txt`:
    /// the plane, small angles, coordinates from 2^-1074 to 1.7e308, the interval ends 7/16 and
    /// 11/16, and the ratios near 1/4 and 1/8 where the error is largest): within 2 ulp
    /// everywhere and within 1 ulp almost everywhere. Platform independent: the function and the
    /// table are.
    #[test]
    fn atan2_is_within_2_ulp_of_the_correctly_rounded_value() {
        let table = include_str!("../tests/atan2_mpmath.txt");
        let hex = |s: &str| f64::from_bits(u64::from_str_radix(s, 16).expect("hex"));
        let mut hist = [0usize; 3];
        for line in table
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
        {
            let v: Vec<&str> = line.split_whitespace().collect();
            let (y, x, want) = (hex(v[0]), hex(v[1]), hex(v[2]));
            let got = atan2(y, x);
            let u = ulps(got, want);
            assert!(
                u <= 2,
                "atan2({y:e}, {x:e}) = {got:e}, correctly rounded {want:e}"
            );
            hist[u as usize] += 1;
        }
        let n: usize = hist.iter().sum();
        eprintln!("atan2 against the correctly rounded table (0, 1, 2 ulp): {hist:?}");
        assert!(n >= 2000, "{n} table entries");
        assert!(hist[2] * 100 <= n, "{hist:?}");
    }

    /// Finite coordinates: small, up to 1.7e308, and of any exponent (2^-1074 .. 2^1023).
    fn coord() -> impl proptest::strategy::Strategy<Value = f64> {
        use proptest::prelude::*;
        prop_oneof![
            -2.0f64..2.0,
            -1.7e308f64..1.7e308,
            (-1.0f64..1.0, -1074i32..1024).prop_map(|(m, e)| m * 2f64.powi(e.min(1023)))
        ]
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(if cfg!(miri) { 16 } else { 4096 }))]

        /// Any finite input gives a finite angle in `[-pi, pi]` within 4 ulp of the platform's
        /// libm (a loose reference: this function is within 1.5 ulp of the exact value, and 4 ulp
        /// leaves room for a libm within 2.5 ulp; the exact reference is the mpmath table), up to 1.7e308 in magnitude; and scaling both coordinates by a power of two
        /// changes nothing while nothing under- or overflows.
        #[test]
        fn atan2_finite_inputs_give_finite_angles_also_near_f64_max(
            y in coord(),
            x in coord(),
            k in -900i32..900,
        ) {
            let a = atan2(y, x);
            proptest::prop_assert!(a.is_finite() && a.abs() <= std::f64::consts::PI, "{}", a);
            proptest::prop_assert!(ulps(a, y.atan2(x)) <= LIBM_ULPS, "atan2({:e}, {:e}) = {:e}", y, x, a);
            let s = 2f64.powi(k);
            let (ys, xs) = (y * s, x * s);
            let inside = |v: f64, w: f64| v == 0.0 || (w.abs() >= 1e-290 && w.abs() <= 1e300);
            if inside(y, ys) && inside(x, xs) && (y != 0.0 || x != 0.0) {
                proptest::prop_assert_eq!(atan2(ys, xs).to_bits(), a.to_bits(), "scaled by 2^{}", k);
            }
        }
    }

    /// The case of the review and its relatives: both coordinates near `f64::MAX`, in every
    /// quadrant (the sums of the reduced argument overflowed there before the scaling by 2^-4).
    #[test]
    fn atan2_near_f64_max_is_finite_and_close_to_libm() {
        for (y, x) in [
            (1e308, 1.5e308),
            (-1.7e308, 1e-300),
            (f64::MAX, -f64::MAX),
            (-f64::MAX, -f64::MAX),
            (3.0, -1e308),
            (1.2e308, 1.1e308),
        ] {
            let a = atan2(y, x);
            assert!(
                a.is_finite() && ulps(a, y.atan2(x)) <= LIBM_ULPS,
                "atan2({y:e}, {x:e}) = {a:e}"
            );
        }
    }

    /// The platform's libm against this function on 300 000 inputs, as a report (printed with
    /// `--nocapture`; macOS: 83 to 90 % equal, the rest 1 ulp, one in 10^7 2 ulp). Asserted only
    /// to [`LIBM_ULPS`]: libms differ.
    #[test]
    fn atan2_agreement_with_libm_report() {
        let mut rnd = lcg(12345);
        let (mut hist, mut worst) = ([0usize; 4], 0u64);
        let n = if cfg!(miri) { 300 } else { 300_000 };
        for i in 0..n {
            let (y, x) = match i % 3 {
                0 => (rnd() * 2.0 - 1.0, rnd() * 2.0 - 1.0),
                1 => {
                    let (m, a) = (0.1 + rnd(), (rnd() - 0.5) * 0.6);
                    (m * a.sin(), m * a.cos())
                }
                _ => {
                    let (e1, e2) = ((rnd() * 80.0 - 40.0).exp2(), (rnd() * 80.0 - 40.0).exp2());
                    ((rnd() - 0.5) * e1, (rnd() - 0.5) * e2)
                }
            };
            let u = ulps(atan2(y, x), y.atan2(x));
            worst = worst.max(u);
            hist[(u as usize).min(3)] += 1;
        }
        eprintln!("atan2 against libm (0, 1, 2, more ulp): {hist:?}, at most {worst} ulp");
        assert!(worst <= LIBM_ULPS, "{hist:?}, at most {worst} ulp");
    }

    #[test]
    fn atan2_special_values_follow_c99() {
        use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI as P};
        let (inf, nan) = (f64::INFINITY, f64::NAN);
        let cases = [
            (0.0, 0.0, 0.0),
            (-0.0, 0.0, -0.0),
            (0.0, -0.0, P),
            (-0.0, -0.0, -P),
            (0.0, 3.0, 0.0),
            (-0.0, 3.0, -0.0),
            (0.0, -3.0, P),
            (-0.0, -3.0, -P),
            (2.0, 0.0, FRAC_PI_2),
            (2.0, -0.0, FRAC_PI_2),
            (-2.0, 0.0, -FRAC_PI_2),
            (-2.0, -0.0, -FRAC_PI_2),
            (inf, 5.0, FRAC_PI_2),
            (-inf, -5.0, -FRAC_PI_2),
            (5.0, inf, 0.0),
            (-5.0, inf, -0.0),
            (5.0, -inf, P),
            (-5.0, -inf, -P),
            (inf, inf, FRAC_PI_4),
            (-inf, inf, -FRAC_PI_4),
            (1.0, 1.0, FRAC_PI_4),
            (1.0, -1.0, 3.0 * FRAC_PI_4),
            (f64::MAX, f64::MAX, FRAC_PI_4),
            (f64::MIN_POSITIVE, 1.0, f64::MIN_POSITIVE),
        ];
        for (y, x, want) in cases {
            let got = atan2(y, x);
            assert!(
                got.to_bits() == want.to_bits() || (want != 0.0 && ulps(got, want) <= 1),
                "atan2({y}, {x}) = {got} (want {want})"
            );
        }
        // 3 pi / 4 from both infinities, within an ulp
        assert!(ulps(atan2(inf, -inf), 3.0 * FRAC_PI_4) <= 1);
        assert!(ulps(atan2(-inf, -inf), -3.0 * FRAC_PI_4) <= 1);
        for (y, x) in [(nan, 1.0), (1.0, nan), (nan, inf), (inf, nan), (nan, nan)] {
            assert!(atan2(y, x).is_nan());
        }
    }

    /// Odd in `y`, and `atan2(y, x)` and `atan2(x, y)` add up to `pi / 2` in the first quadrant
    /// (to rounding).
    #[test]
    fn atan2_symmetries_hold() {
        let mut rnd = lcg(7);
        for _ in 0..10_000 {
            let (y, x) = (rnd() * 4.0 - 2.0, rnd() * 4.0 - 2.0);
            assert_eq!(atan2(-y, x).to_bits(), (-atan2(y, x)).to_bits());
            let (a, b) = (atan2(y.abs(), x.abs()), atan2(x.abs(), y.abs()));
            assert!((a + b - std::f64::consts::FRAC_PI_2).abs() < 4e-16);
        }
    }

    /// The vector path equals the scalar function bit for bit, with and without a non-finite
    /// value in the batch.
    #[test]
    fn atan2_all_equals_atan2_bit_for_bit() {
        let mut rnd = lcg(99);
        let mut y: Vec<f64> = (0..37).map(|_| rnd() * 2.0 - 1.0).collect();
        let x: Vec<f64> = (0..37).map(|_| rnd() * 2.0 - 1.0).collect();
        y[3] = 0.0;
        let mut x = x;
        x[3] = -0.0;
        for batch in [false, true] {
            if batch {
                y[20] = f64::INFINITY;
            }
            let mut out = vec![0.0; 37];
            atan2_all(&y, &x, &mut out);
            for k in 0..37 {
                assert_eq!(out[k].to_bits(), atan2(y[k], x[k]).to_bits(), "k {k}");
            }
            let mut fixed = [0.0; 12];
            atan2_all(&y[..12], &x[..12], &mut fixed);
            assert_eq!(
                fixed.map(f64::to_bits),
                std::array::from_fn(|k| out[k].to_bits())
            );
        }
    }
}
