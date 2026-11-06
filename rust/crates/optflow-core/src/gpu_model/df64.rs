//! Double-float ("df64") arithmetic: a value is `hi + lo` exactly with `|lo| <= ulp(hi) / 2`
//! after normalisation. Only `+`, `-`, `*` and `f32::mul_add` (the one fused operation) on `f32`
//! are used, in an explicit order, so that the Metal kernels can mirror it line by line.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Df64 {
    pub hi: f32,
    pub lo: f32,
}

/// `(s, e)` with `s = fl(a + b)` and `s + e = a + b` exactly (Knuth).
#[inline]
pub fn two_sum(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// `(p, e)` with `p = fl(a * b)` and `p + e = a * b` exactly (barring over- and underflow).
#[inline]
pub fn two_prod(a: f32, b: f32) -> (f32, f32) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

/// Like [`two_sum`] for `|a| >= |b|` (Dekker).
#[inline]
fn quick_two_sum(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    (s, b - (s - a))
}

// `add`/`sub` are inherent methods by the interface: the Metal port has the same names
#[allow(clippy::should_implement_trait)]
impl Df64 {
    pub fn from_f64(v: f64) -> Self {
        let hi = v as f32;
        Df64 {
            hi,
            lo: (v - f64::from(hi)) as f32,
        }
    }

    pub fn to_f64(self) -> f64 {
        f64::from(self.hi) + f64::from(self.lo)
    }

    pub fn from_f32(v: f32) -> Self {
        Df64 { hi: v, lo: 0.0 }
    }

    pub fn add(self, o: Df64) -> Df64 {
        let (s, e) = two_sum(self.hi, o.hi);
        let (t, f) = two_sum(self.lo, o.lo);
        let (s, e) = quick_two_sum(s, e + t);
        let (hi, lo) = quick_two_sum(s, e + f);
        Df64 { hi, lo }
    }

    pub fn sub(self, o: Df64) -> Df64 {
        self.add(Df64 {
            hi: -o.hi,
            lo: -o.lo,
        })
    }

    pub fn add_f32(self, v: f32) -> Df64 {
        self.add(Df64::from_f32(v))
    }

    pub fn mul(self, o: Df64) -> Df64 {
        let (p, e) = two_prod(self.hi, o.hi);
        let (hi, lo) = quick_two_sum(p, e + (self.hi * o.lo + self.lo * o.hi));
        Df64 { hi, lo }
    }

    pub fn mul_f32(self, v: f32) -> Df64 {
        let (p, e) = two_prod(self.hi, v);
        let (hi, lo) = quick_two_sum(p, self.lo.mul_add(v, e));
        Df64 { hi, lo }
    }

    /// `(self - o)` rounded to f32: for differences of nearby values.
    pub fn diff_f32(self, o: Df64) -> f32 {
        let d = self.sub(o);
        d.hi + d.lo
    }

    /// Integer part (floor) and fraction in `[0, 1)` as f32; requires `|value| < 2^23`.
    pub fn floor_frac(self) -> (i32, f32) {
        let mut fl = self.hi.floor();
        // `hi - fl` is exact; only `lo` can make the sum negative (hi an integer, lo < 0)
        let mut f = (self.hi - fl) + self.lo;
        if f < 0.0 {
            fl -= 1.0;
            f += 1.0;
        }
        // a fraction that rounded up to 1.0 stays below it, the integer part is kept
        if f >= 1.0 {
            f = f32::from_bits(1.0f32.to_bits() - 1);
        }
        (fl as i32, f)
    }
}
