//! FFT helpers built on rustfft.

use std::sync::Arc;

use num_complex::Complex32;
use rayon::prelude::*;
use rustfft::{Fft, FftPlanner};

/// Lines (rows or columns) per parallel task of a 2-D transform. Even, so that every task starts at
/// an even line index (see [`Fft2`]).
pub(crate) const LINES: usize = 8;
const _: () = assert!(LINES.is_multiple_of(2));

/// Side of the square blocks a transpose copies at a time.
const BLOCK: usize = 16;

/// Two-dimensional FFT of a fixed `h` x `w` size with numpy's `fft2` / `ifft2` conventions:
/// `forward` is unnormalised, `inverse` divides by `h * w`. Data is row-major (`data[y * w + x]`).
/// Any size works, not only powers of two. The plans are built once and shared across calls.
///
/// Lines are handed to the plans in tasks of [`LINES`] lines, in parallel. A line is not always
/// transformed on its own: rustfft's SIMD kernels for some short lengths transform two
/// neighbouring lines at a time, which can round a line differently from transforming it alone.
/// The pairing depends only on the position of a line inside its task; every task starts at an
/// even line index (`LINES` is even), so the pairs are the same for any number of threads and
/// any order of the tasks, and so are the values.
pub struct Fft2 {
    h: usize,
    w: usize,
    row_fwd: Arc<dyn Fft<f32>>,
    row_inv: Arc<dyn Fft<f32>>,
    col_fwd: Arc<dyn Fft<f32>>,
    col_inv: Arc<dyn Fft<f32>>,
}

impl Fft2 {
    /// Plans transforms for `h` rows and `w` columns. Panics if either is zero.
    pub fn new(h: usize, w: usize) -> Self {
        assert!(h >= 1 && w >= 1, "Fft2::new: empty size {h}x{w}");
        let mut planner = FftPlanner::<f32>::new();
        Self {
            h,
            w,
            row_fwd: planner.plan_fft_forward(w),
            row_inv: planner.plan_fft_inverse(w),
            col_fwd: planner.plan_fft_forward(h),
            col_inv: planner.plan_fft_inverse(h),
        }
    }

    pub fn height(&self) -> usize {
        self.h
    }

    pub fn width(&self) -> usize {
        self.w
    }

    fn check(&self, len: usize) {
        assert_eq!(
            len,
            self.h * self.w,
            "Fft2: {} values for a {}x{} transform",
            len,
            self.h,
            self.w
        );
    }

    /// In-place 2-D transform: every row, then every column. `data.len()` must be `h * w`.
    fn run(&self, data: &mut [Complex32], row: &dyn Fft<f32>, col: &dyn Fft<f32>) {
        self.check(data.len());
        lines(data, row);
        // columns: transpose to (w x h), transform the contiguous rows, transpose back
        let mut t = vec![Complex32::new(0.0, 0.0); data.len()];
        transpose(data, &mut t, self.h, self.w);
        lines(&mut t, col);
        transpose(&t, data, self.w, self.h);
    }

    /// The factor `1 / (h * w)` of the inverse transform.
    fn inverse_scale(&self) -> f32 {
        1.0 / (self.h * self.w) as f32
    }

    /// The steps of [`Fft2::inverse`] for callers that run them on their own blocks of lines: the
    /// plan along a row (`w` values), the plan along a column (`h` values) and the factor
    /// `1 / (h * w)`. `inverse` is every row, then every column, then `v *= scale`, with the lines
    /// handed to the plans in blocks of [`LINES`]. rustfft's SIMD kernels for some short lengths
    /// transform two lines at a time, which can round a line differently from transforming it
    /// alone; blocks that start at an even line index (as blocks of `LINES` lines do) pair the
    /// lines exactly as `inverse` does, and then give the same values bit for bit.
    pub(crate) fn inverse_parts(&self) -> (&dyn Fft<f32>, &dyn Fft<f32>, f32) {
        (
            self.row_inv.as_ref(),
            self.col_inv.as_ref(),
            self.inverse_scale(),
        )
    }

    /// Forward transform, unnormalised (numpy `fft2`).
    pub fn forward(&self, data: &mut [Complex32]) {
        self.run(data, self.row_fwd.as_ref(), self.col_fwd.as_ref());
    }

    /// Inverse transform divided by `h * w` (numpy `ifft2`).
    pub fn inverse(&self, data: &mut [Complex32]) {
        self.run(data, self.row_inv.as_ref(), self.col_inv.as_ref());
        let scale = self.inverse_scale();
        data.par_chunks_mut(self.w * LINES).for_each(|c| {
            for v in c {
                *v *= scale;
            }
        });
    }
}

/// Transforms every `plan.len()`-long line of `data` in place, `LINES` lines per parallel task.
fn lines(data: &mut [Complex32], plan: &dyn Fft<f32>) {
    let n = plan.len();
    let scratch = plan.get_inplace_scratch_len();
    data.par_chunks_mut(n * LINES).for_each_init(
        || vec![Complex32::new(0.0, 0.0); scratch],
        |s, chunk| plan.process_with_scratch(chunk, s),
    );
}

/// `dst` (`cols` rows of `rows` values) = transpose of `src` (`rows` rows of `cols` values).
///
/// A parallel task fills `BLOCK` rows of `dst` (the columns `x0..x0 + BLOCK` of `src`) and walks
/// down `src` in steps of `BLOCK` rows (`y0`). One step is a `BLOCK` x `BLOCK` tile: it reads a run
/// of `BLOCK` consecutive values from each of `BLOCK` rows of `src` and writes them, as `BLOCK`
/// consecutive values each, into the `BLOCK` rows of `dst`, so the tile stays in cache while it
/// is copied (the strided writes of an untiled transpose touch a new cache line per value).
fn transpose(src: &[Complex32], dst: &mut [Complex32], rows: usize, cols: usize) {
    debug_assert_eq!(src.len(), rows * cols);
    debug_assert_eq!(dst.len(), rows * cols);
    dst.par_chunks_mut(rows * BLOCK)
        .enumerate()
        .for_each(|(b, chunk)| {
            let x0 = b * BLOCK;
            let nx = chunk.len() / rows;
            for y0 in (0..rows).step_by(BLOCK) {
                for y in y0..(y0 + BLOCK).min(rows) {
                    let line = &src[y * cols + x0..y * cols + x0 + nx];
                    for (j, &v) in line.iter().enumerate() {
                        chunk[j * rows + y] = v;
                    }
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Naive f64 2-D DFT with sign `-1` (forward) or `+1` (inverse, unnormalised).
    fn naive_dft(data: &[Complex32], h: usize, w: usize, sign: f64) -> Vec<(f64, f64)> {
        let mut out = vec![(0.0, 0.0); h * w];
        for v in 0..h {
            for u in 0..w {
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for y in 0..h {
                    for x in 0..w {
                        let a = sign
                            * 2.0
                            * std::f64::consts::PI
                            * ((u * x) as f64 / w as f64 + (v * y) as f64 / h as f64);
                        let s = data[y * w + x];
                        let (c, sn) = (a.cos(), a.sin());
                        re += f64::from(s.re) * c - f64::from(s.im) * sn;
                        im += f64::from(s.re) * sn + f64::from(s.im) * c;
                    }
                }
                out[v * w + u] = (re, im);
            }
        }
        out
    }

    fn ramp(h: usize, w: usize) -> Vec<Complex32> {
        (0..h * w)
            .map(|i| Complex32::new((i as f32 * 0.37).sin(), (i as f32 * 0.11).cos()))
            .collect()
    }

    #[test]
    fn fft_forward_matches_naive_dft_on_non_square_odd_sizes() {
        for (h, w) in [(1, 1), (1, 7), (5, 6), (6, 5), (9, 4), (7, 11)] {
            let data = ramp(h, w);
            let want = naive_dft(&data, h, w, -1.0);
            let mut got = data.clone();
            Fft2::new(h, w).forward(&mut got);
            for (g, (wr, wi)) in got.iter().zip(&want) {
                assert!(
                    (f64::from(g.re) - wr).abs() < 1e-4 && (f64::from(g.im) - wi).abs() < 1e-4,
                    "{h}x{w}: got {g} want {wr}+{wi}i"
                );
            }
        }
    }

    #[test]
    fn fft_inverse_is_normalised_by_h_times_w() {
        let (h, w) = (4, 6);
        let data = ramp(h, w);
        let want = naive_dft(&data, h, w, 1.0);
        let mut got = data;
        Fft2::new(h, w).inverse(&mut got);
        for (g, (wr, wi)) in got.iter().zip(&want) {
            let n = (h * w) as f64;
            assert!((f64::from(g.re) - wr / n).abs() < 1e-5);
            assert!((f64::from(g.im) - wi / n).abs() < 1e-5);
        }
    }

    #[test]
    fn fft_forward_of_constant_is_hw_at_dc() {
        let (h, w) = (5, 8);
        let mut d = vec![Complex32::new(1.0, 0.0); h * w];
        Fft2::new(h, w).forward(&mut d);
        assert!((d[0].re - (h * w) as f32).abs() < 1e-3 && d[0].im.abs() < 1e-3);
        assert!(d[1..].iter().all(|v| v.norm() < 1e-3));
    }

    #[test]
    fn fft_inverse_parts_on_blocks_of_lines_give_inverse_bit_for_bit() {
        for (h, w) in [(1, 1), (5, 6), (37, 20), (64, 40), (13, 11)] {
            let data = ramp(h, w);
            let fft = Fft2::new(h, w);
            let mut want = data.clone();
            fft.inverse(&mut want);
            let (row, col, scale) = fft.inverse_parts();
            let mut work = data;
            for block in work.chunks_mut(w * LINES) {
                row.process(block);
            }
            for x0 in (0..w).step_by(LINES) {
                let n = LINES.min(w - x0);
                let mut lines: Vec<Complex32> =
                    (0..n * h).map(|i| work[(i % h) * w + x0 + i / h]).collect();
                col.process(&mut lines);
                for (i, v) in lines.into_iter().enumerate() {
                    work[(i % h) * w + x0 + i / h] = v * scale;
                }
            }
            for y in 0..h {
                for x in 0..w {
                    let (a, b) = (work[y * w + x], want[y * w + x]);
                    assert_eq!(
                        (a.re.to_bits(), a.im.to_bits()),
                        (b.re.to_bits(), b.im.to_bits())
                    );
                }
            }
        }
    }

    #[test]
    fn fft_transpose_handles_sizes_off_the_block_grid() {
        for (rows, cols) in [(1, 1), (3, 17), (17, 3), (33, 40)] {
            let src = ramp(rows, cols);
            let mut dst = vec![Complex32::new(0.0, 0.0); rows * cols];
            transpose(&src, &mut dst, rows, cols);
            for y in 0..rows {
                for x in 0..cols {
                    assert_eq!(dst[x * rows + y], src[y * cols + x]);
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "Fft2")]
    fn fft_rejects_wrong_length() {
        Fft2::new(4, 4).forward(&mut [Complex32::new(0.0, 0.0); 15]);
    }

    proptest! {
        #[test]
        fn fft_round_trip_is_identity(
            h in 1usize..=40,
            w in 1usize..=40,
            seed in proptest::collection::vec(-1.0f32..1.0, 2 * 40 * 40),
        ) {
            let data: Vec<Complex32> = (0..h * w)
                .map(|i| Complex32::new(seed[2 * i], seed[2 * i + 1]))
                .collect();
            let fft = Fft2::new(h, w);
            let mut x = data.clone();
            fft.forward(&mut x);
            fft.inverse(&mut x);
            for (a, b) in x.iter().zip(&data) {
                prop_assert!((a - b).norm() < 1e-5, "{h}x{w}: {a} vs {b}");
            }
        }
    }
}
