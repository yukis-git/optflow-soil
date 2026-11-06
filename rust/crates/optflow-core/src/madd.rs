//! The multiply-add `c + a b` of the hot loops (window moments, spline sampling, the per-filter
//! sums, the lockstep Cholesky, the B-spline prefilters).
//!
//! By default the product and the sum are rounded separately, as everywhere else in optflow-core,
//! so the results are those of the plain expression on every platform. With the cargo feature
//! `fma` (off by default) it is a fused multiply-add, rounded once: faster on aarch64, where it
//! is one instruction, but the results differ in the last bits, and on an x86-64 build without
//! the FMA target feature (the default for Windows) `f64::mul_add` is a library call and slow.
//! Whether to enable it is left to the user (Task P2).

/// `c + a * b`: two roundings by default, one with the feature `fma`.
#[inline(always)]
pub(crate) fn madd(a: f64, b: f64, c: f64) -> f64 {
    #[cfg(feature = "fma")]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(feature = "fma"))]
    {
        c + a * b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Without the feature, the plain expression bit for bit (also for the signed zeros of
    /// `c + a b` and `c - a b` written as `madd(-a, b, c)`); with it, one rounding.
    #[test]
    fn madd_rounds_as_configured() {
        let (a, b, c) = (1.0 + f64::EPSILON, 1.0 - f64::EPSILON, -1.0);
        let fused = a.mul_add(b, c);
        let plain = c + a * b;
        assert_ne!(
            fused.to_bits(),
            plain.to_bits(),
            "the example needs a difference"
        );
        let want = if cfg!(feature = "fma") { fused } else { plain };
        assert_eq!(madd(a, b, c).to_bits(), want.to_bits());
        if !cfg!(feature = "fma") {
            for (a, b, c) in [
                (0.0, 1.0, -0.0),
                (-0.0, 1.0, -0.0),
                (2.0, 3.0, 0.5),
                (1e300, 1e10, 1.0),
            ] {
                assert_eq!(madd(a, b, c).to_bits(), (c + a * b).to_bits());
                assert_eq!(madd(-a, b, c).to_bits(), (c - a * b).to_bits());
            }
        }
    }
}
