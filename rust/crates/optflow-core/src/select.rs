//! Order statistics of `f64` slices: the median as `np.median` computes it, and the radix select
//! it uses for large inputs. Shared by the coarse stage ([`crate::poc`]) and the fine stage
//! ([`crate::phaseflow`]).

use rayon::prelude::*;

/// Below this many values the middle is found by `select_nth_unstable_by` on a copy; from there on
/// by the parallel radix select [`ranks`] without a copy (from this size on the copy and
/// the serial select take longer than the parallel passes, e.g. for the energy of a 512 x 512
/// image, whose threshold every local solve needs).
const RADIX_FROM: usize = 1 << 16;

/// `np.median` of all values: the middle value, or the mean of the two middle ones for an even
/// count; NaN if any value is NaN or there are none. Below `RADIX_FROM` values the middle is
/// found by `select_nth_unstable_by` on a copy; from there on by a parallel radix select
/// ([`ranks`]) without a copy (the energy of an image or a tile, the whole image's for the
/// texture scale). Both give the same values, bit for bit: the order of `f64::total_cmp` puts
/// equal values (including `-0.0` before `0.0`) in one place.
pub(crate) fn median(v: &[f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n < RADIX_FROM {
        if v.iter().any(|x| x.is_nan()) {
            return f64::NAN;
        }
        let mut a = v.to_vec();
        let (below, &mut upper, _) = a.select_nth_unstable_by(n / 2, f64::total_cmp);
        if !n.is_multiple_of(2) {
            return upper;
        }
        let lower = below.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        return (lower + upper) / 2.0;
    }
    match ranks(v, n / 2, n.is_multiple_of(2)) {
        None => f64::NAN,
        Some((upper, None)) => upper,
        Some((upper, Some(lower))) => (lower + upper) / 2.0,
    }
}

/// `v` mapped to `u64` keys whose unsigned order is the order of `f64::total_cmp`.
fn order_key(x: f64) -> u64 {
    let b = x.to_bits();
    if b >> 63 == 1 {
        !b
    } else {
        b | (1 << 63)
    }
}

/// The value with key `k` (inverse of [`order_key`]).
fn from_order_key(k: u64) -> f64 {
    f64::from_bits(if k >> 63 == 1 { k & !(1 << 63) } else { !k })
}

/// Values per parallel piece of the passes of [`ranks`] (any size gives the same counts; this
/// one splits a 512 x 512 image into 16 pieces).
const CHUNK: usize = 1 << 14;

/// Bits of the key that one counting pass of [`ranks`] resolves.
const DIGIT: u32 = 11;

/// Once the bucket of the wanted rank holds at most this many values, they are collected and
/// the rank is selected among them.
const SMALL: usize = 1 << 15;

/// The value of rank `k` of `v` (from 0, `k < v.len()`) in the order of `f64::total_cmp`, i.e.
/// the value `select_nth_unstable_by(k, f64::total_cmp)` puts at index `k`, and with `lower`
/// also the value of rank `k - 1` (`k >= 1`); `None` if `v` holds a NaN.
///
/// Radix select on [`order_key`]: each pass counts, in parallel, the keys that share the prefix
/// found so far by their next `DIGIT` bits (the first pass also counts the NaNs) and keeps the
/// digit whose bucket holds rank `k`. Once that bucket holds at most `SMALL` values they are
/// collected and both ranks are selected among them; rank `k - 1` lies below the bucket only if
/// rank `k` is the bucket's smallest, and then it is the largest value below the bucket (one
/// more pass).
fn ranks(v: &[f64], k: usize, lower: bool) -> Option<(f64, Option<f64>)> {
    let (mut prefix, mut bits, mut kb) = (0u64, 0u32, k);
    let matches = move |key: u64, prefix: u64, bits: u32| bits == 0 || key >> (64 - bits) == prefix;
    loop {
        let width = DIGIT.min(64 - bits);
        let shift = 64 - bits - width;
        let mask = (1u64 << width) - 1;
        let first = bits == 0;
        let zero = || (vec![0usize; 1 << width], 0usize);
        let (hist, nan) = v
            .par_chunks(CHUNK)
            .fold(zero, |(mut h, mut nan), chunk| {
                for &x in chunk {
                    let key = order_key(x);
                    if matches(key, prefix, bits) {
                        h[((key >> shift) & mask) as usize] += 1;
                    }
                    if first {
                        nan += usize::from(x.is_nan());
                    }
                }
                (h, nan)
            })
            .reduce(zero, |(mut a, na), (b, nb)| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
                (a, na + nb)
            });
        if nan > 0 {
            return None;
        }
        let mut below = 0;
        let mut digit = 0;
        for (d, &c) in hist.iter().enumerate() {
            if below + c > kb {
                digit = d;
                break;
            }
            below += c;
        }
        kb -= below;
        prefix = (prefix << width) | digit as u64;
        bits += width;
        if bits == 64 {
            // every value of the bucket has this key: rank k - 1 is in it unless rank k is first
            let x = from_order_key(prefix);
            let low = (lower && kb >= 1).then_some(x);
            return Some((x, low.or_else(|| lower.then(|| below_max(v, prefix)))));
        }
        if hist[digit] <= SMALL {
            let mut rest: Vec<f64> = v
                .par_chunks(CHUNK)
                .map(|chunk| {
                    chunk
                        .iter()
                        .copied()
                        .filter(|&x| matches(order_key(x), prefix, bits))
                        .collect::<Vec<f64>>()
                })
                .collect::<Vec<Vec<f64>>>()
                .concat();
            let (left, &mut x, _) = rest.select_nth_unstable_by(kb, f64::total_cmp);
            let low = if !lower {
                None
            } else if kb >= 1 {
                left.iter().copied().max_by(f64::total_cmp)
            } else {
                Some(below_max(v, prefix << (64 - bits)))
            };
            return Some((x, low));
        }
    }
}

/// The largest value of `v` whose key is below `key` (one parallel pass); there must be one.
fn below_max(v: &[f64], key: u64) -> f64 {
    let max = v
        .par_chunks(CHUNK)
        .map(|chunk| {
            chunk
                .iter()
                .map(|&x| order_key(x))
                .filter(|&k| k < key)
                .max()
        })
        .reduce(|| None, Option::max);
    from_order_key(max.expect("below_max: a value below the key"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_median_radix_select_equals_sorting() {
        let mut s = 99u64;
        let mut rnd = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        // spread values, heavy duplicates (few distinct keys: the radix passes run to the
        // last digit), signed zeros and negatives; odd and even counts around the small-set
        // cut-off
        let sets: Vec<Vec<f64>> = vec![
            (0..200_001).map(|_| rnd() * 3.0 - 1.0).collect(),
            (0..200_000).map(|_| (rnd() * 7.0).floor() - 2.0).collect(),
            (0..70_000)
                .map(|i| match i % 4 {
                    0 => 0.0,
                    1 => -0.0,
                    2 => 1e-300 * rnd(),
                    _ => -rnd(),
                })
                .collect(),
            (0..65_537).map(|_| 1.0 + rnd() * 1e-12).collect(),
            // the smallest count of the radix path, and just below it (the copy and the select)
            (0..RADIX_FROM).map(|_| rnd() * 5.0).collect(),
            (0..RADIX_FROM - 2)
                .map(|i| match i % 3 {
                    0 => 0.0,
                    1 => -0.0,
                    _ => rnd() - 0.5,
                })
                .collect(),
            // even count, the two middle values distinct and equal ones elsewhere
            (0..262_144).map(|i| (i % 1000) as f64 + 0.5).collect(),
            // the upper middle is the smallest of its bucket (the lower one lies below it): two
            // well separated halves, spread and all equal (the passes run to the last digit)
            (0..80_000)
                .map(|i| if i % 2 == 0 { rnd() } else { 1e3 + rnd() })
                .collect(),
            (0..80_000)
                .map(|i| if i % 2 == 0 { -0.0 } else { 2.0 })
                .collect(),
            vec![7.0; 70_000],
            vec![2.5],
        ];
        for v in sets {
            let mut sorted = v.clone();
            sorted.sort_by(f64::total_cmp);
            let n = v.len();
            for k in [0, 1, n / 2, n - 1].into_iter().filter(|&k| k < n) {
                let (x, low) = ranks(&v, k, k >= 1).expect("no NaN");
                assert_eq!(x.to_bits(), sorted[k].to_bits(), "n {n} k {k}");
                assert_eq!(
                    low.map(f64::to_bits),
                    (k >= 1).then(|| sorted[k - 1].to_bits()),
                    "n {n} k {k}: rank k - 1"
                );
            }
            let want = if n % 2 == 1 {
                sorted[n / 2]
            } else {
                (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
            };
            assert_eq!(median(&v).to_bits(), want.to_bits(), "n {n}");
        }
    }

    #[test]
    fn select_median_follows_numpy() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median(&[5.0]), 5.0);
        assert_eq!(median(&[2.0, 2.0, 2.0, 9.0]), 2.0);
        assert!(median(&[1.0, f64::NAN, 3.0]).is_nan());
        assert!(median(&[]).is_nan());
    }

    /// Below `RADIX_FROM` values a scan finds a NaN, from there on the first counting pass of
    /// the radix select; a single NaN, at the start, the middle or the end, makes the median NaN
    /// on either side of that size.
    #[test]
    fn select_median_sees_a_nan_at_any_position_and_size() {
        for n in [7, RADIX_FROM - 1, RADIX_FROM, 3 * RADIX_FROM + 1] {
            let v: Vec<f64> = (0..n).map(|i| ((i * 7919) % 1009) as f64).collect();
            assert!(!median(&v).is_nan(), "n {n}");
            for at in [0, n / 2, n - 1] {
                let mut w = v.clone();
                w[at] = f64::NAN;
                assert!(median(&w).is_nan(), "n {n} NaN at {at}");
            }
        }
    }

    /// The `-0.0` / `0.0` pair keeps its sign in the median of two middle values, as in a sort by
    /// `f64::total_cmp`: the mean of `-0.0` and `-0.0` is `-0.0`, anything else with a `0.0` is `0.0`.
    #[test]
    fn select_median_keeps_the_sign_of_zero_like_a_sort() {
        let bits = |v: &[f64]| median(v).to_bits();
        assert_eq!(bits(&[-0.0, -0.0]), (-0.0f64).to_bits());
        assert_eq!(bits(&[-0.0, 0.0]), 0.0f64.to_bits());
        assert_eq!(bits(&[0.0, -0.0, 0.0, 0.0]), 0.0f64.to_bits());
        assert_eq!(bits(&[-1.0, -0.0, -0.0, 5.0]), (-0.0f64).to_bits());
    }
}

/// Tests added after mutation testing (cargo-mutants, Task 11m).
#[cfg(test)]
mod mutation_gap_tests {
    use super::*;

    /// The NaN counts of the pieces of the first pass of the radix select add up: one NaN in each
    /// of two pieces is two NaNs, not a count that cancels (`ranks` itself, on the smallest input
    /// of two pieces).
    #[test]
    fn select_ranks_counts_nans_of_two_pieces_together() {
        let n = 2 * CHUNK;
        let base: Vec<f64> = (0..n).map(|i| ((i * 7919) % 1009) as f64).collect();
        assert!(ranks(&base, n / 2, true).is_some());
        let mut v = base.clone();
        v[5] = f64::NAN;
        v[CHUNK + 5] = f64::NAN;
        assert!(ranks(&v, n / 2, true).is_none());
        // a single NaN, in either piece
        for at in [5, CHUNK + 5] {
            let mut v = base.clone();
            v[at] = f64::NAN;
            assert!(ranks(&v, n / 2, false).is_none(), "NaN at {at}");
        }
    }

    /// The same through `median`, for a NaN in every `step`-th piece of 16 (one NaN in each of the
    /// pieces of an even number of them is the case that cancels).
    #[test]
    #[cfg_attr(miri, ignore)] // 262144 values, ten medians: slow under Miri
    fn select_radix_counts_nans_of_all_pieces_together() {
        let n = 16 * CHUNK;
        assert!(n >= RADIX_FROM);
        let base: Vec<f64> = (0..n).map(|i| ((i * 7919) % 1009) as f64).collect();
        assert!(!median(&base).is_nan());
        // NaN in pieces 0 .. every `step`-th piece
        for step in [1, 2, 4, 8, 15] {
            let mut v = base.clone();
            for piece in (0..16).step_by(step) {
                v[piece * CHUNK + 5] = f64::NAN;
            }
            assert!(median(&v).is_nan(), "a NaN in every {step}-th piece");
            assert!(ranks(&v, n / 2, true).is_none(), "step {step}");
        }
    }
}
