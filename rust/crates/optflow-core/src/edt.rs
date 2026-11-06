//! Euclidean distance transform and 4-connected component filtering.
use rayon::prelude::*;

use crate::image::Image;

/// Scratch buffers for [`dt_1d`] on lines of length `n`.
struct Line {
    f: Vec<f64>,
    v: Vec<usize>,
    z: Vec<f64>,
}

impl Line {
    fn new(n: usize) -> Self {
        Self {
            f: vec![0.0; n],
            v: vec![0; n],
            z: vec![0.0; n + 1],
        }
    }
}

/// Lower envelope of the parabolas `(q - p)^2 + f[p]` of the finite entries of `line.f`: the
/// rightmost parabola of the envelope ends at `v[k]`, and `z[k]..z[k + 1]` is the stretch of
/// `q` where the parabola of `v[k]` is the lowest. Entries of `f` that are infinite (no source on
/// that line yet) are skipped, so no `inf - inf` is ever formed. Returns false if there is no
/// finite entry (the envelope is empty).
fn envelope(line: &mut Line) -> bool {
    let Line { f, v, z } = line;
    let n = f.len();
    let mut k = 0usize; // index of the rightmost parabola of the lower envelope
    let mut any = false;
    for q in 0..n {
        if f[q].is_infinite() {
            continue;
        }
        if !any {
            any = true;
            v[0] = q;
            z[0] = f64::NEG_INFINITY;
            z[1] = f64::INFINITY;
            continue;
        }
        let qf = q as f64;
        loop {
            let p = v[k];
            let pf = p as f64;
            let s = ((f[q] + qf * qf) - (f[p] + pf * pf)) / (2.0 * (qf - pf));
            // z[0] = -inf, so the envelope never empties
            if s <= z[k] {
                k -= 1;
            } else {
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = f64::INFINITY;
                break;
            }
        }
    }
    any
}

/// Calls `visit(q, p, d2)` for every `q` of the line with the parabola `p` that is lowest there
/// and its value `d2 = (q - p)^2 + f[p]`; where two parabolas are equally low (`q` on the
/// boundary `z[k + 1]`) the one with the smaller `p` is taken. Needs a non-empty [`envelope`].
fn read_envelope(line: &Line, mut visit: impl FnMut(usize, usize, f64)) {
    let Line { f, v, z } = line;
    let mut k = 0usize;
    for q in 0..f.len() {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let d = q as f64 - v[k] as f64;
        visit(q, v[k], d * d + f[v[k]]);
    }
}

/// Felzenszwalb-Huttenlocher 1-D squared distance transform:
/// `out[q] = min_p ((q - p)^2 + line.f[p])`. A line without a finite entry comes out all
/// infinite.
fn dt_1d(line: &mut Line, out: &mut [f64]) {
    if !envelope(line) {
        out.fill(f64::INFINITY);
        return;
    }
    read_envelope(line, |q, _, d2| out[q] = d2);
}

/// [`dt_1d`] that also gives the position `p` of the minimum, ties to the smaller `p`: `src[q]`
/// is the `p` with `out[q] = (q - p)^2 + line.f[p]`. A line without a finite entry comes out
/// all infinite (`src` is left alone).
fn nearest_1d(line: &mut Line, out: &mut [f64], src: &mut [u32]) {
    if !envelope(line) {
        out.fill(f64::INFINITY);
        return;
    }
    read_envelope(line, |q, p, d2| {
        out[q] = d2;
        src[q] = p as u32;
    });
}

/// Exact Euclidean distance from every pixel to the nearest invalid (`false`) pixel
/// (`scipy.ndimage.distance_transform_edt(valid)`): 0 at invalid pixels. Squared distances are
/// computed exactly in f64 (separable Felzenszwalb-Huttenlocher), then square-rooted and stored
/// as f32. The pixel `(x, y)` is at row `y`, column `x`; distances are in pixels.
///
/// With no invalid pixel at all there is no meaningful answer. scipy 1.18.1 (the version that
/// wrote the fixtures; `python/requirements.txt` allows any scipy from 1.13, and this case was
/// not checked on the others) returns `sqrt((y + 1)^2 + x^2)` there, as if a single background
/// pixel sat at row -1, column 0, and this function reproduces that so the two agree on every
/// input. The tracking pipeline never relies on it (it only calls the transform when the guard
/// source has a pixel set).
pub fn distance_to_invalid(valid: &Image<bool>) -> Image<f32> {
    let (w, h) = (valid.w, valid.h);
    let mut out = Image::new(w, h);
    if w == 0 || h == 0 {
        return out;
    }
    let any_invalid = valid.data.iter().any(|&b| !b);

    // pass 1 along x: squared distance to the nearest invalid pixel of the same row
    let mut rows = vec![0.0f64; w * h];
    rows.par_chunks_mut(w)
        .zip(valid.data.par_chunks(w))
        .for_each_init(
            || Line::new(w),
            |line, (dst, src)| {
                for (f, &ok) in line.f.iter_mut().zip(src) {
                    *f = if ok { f64::INFINITY } else { 0.0 };
                }
                dt_1d(line, dst);
            },
        );

    // pass 2 along y, on the transpose so that every column is a contiguous chunk
    let mut cols = vec![0.0f64; w * h];
    cols.par_chunks_mut(h).enumerate().for_each_init(
        || Line::new(h),
        |line, (x, dst)| {
            for (y, f) in line.f.iter_mut().enumerate() {
                *f = rows[y * w + x];
            }
            dt_1d(line, dst);
        },
    );

    out.data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let d2 = if any_invalid {
                cols[x * h + y]
            } else {
                ((y + 1) * (y + 1) + x * x) as f64
            };
            *o = d2.sqrt() as f32;
        }
    });
    out
}

/// Whether a distance from [`distance_to_invalid`] is at most `limit`. scipy's float64 distance is
/// the square root of an integer; the transform here stores that root as `f32`, so the integer
/// is restored first (exact for distances below about 2000 px) and the root taken in `f64`, which
/// decides every pixel like the reference does.
pub(crate) fn within(d: f32, limit: f64) -> bool {
    let d = f64::from(d);
    (d * d).round().sqrt() <= limit
}

/// For every pixel, the row-major index `y * w + x` of the nearest source (`true`) pixel
/// (`scipy.ndimage.distance_transform_edt(~src, return_indices=True)`, the feature transform), or
/// `None` if there is no source. Distances are exact (squared, in integers held in f64). Of
/// several equally near sources the one in the smallest column is taken, and in that column the
/// smallest row; this is the rule of scipy 1.18.1 (the only version checked; the rule is not
/// documented, so another scipy, `python/requirements.txt` allows 1.13 and later, could differ).
/// Two separate checks against scipy 1.18.1 found no difference:
/// the closed form "(squared distance, column, row) is lexicographically smallest" on 7052 random
/// masks, and this algorithm (Felzenszwalb-Huttenlocher, the pass along y first) on 5403 random
/// masks with 214 291 tied pixels; the `edt_ft_*` fixtures pin it in the tests. It comes out of
/// the algorithm without a comparison of its own: the pass along y runs first (scipy's axis 0)
/// and the lower envelope (`read_envelope`) takes the smaller position on a tie, in both passes.
pub fn nearest_source(src: &Image<bool>) -> Option<Vec<u32>> {
    let (w, h) = (src.w, src.h);
    if !src.data.iter().any(|&s| s) {
        return None;
    }
    assert!(
        w * h <= u32::MAX as usize,
        "nearest_source: {w}x{h} pixels do not fit u32 indices"
    );

    // pass 1 along y, one column x at a time (stored transposed, `[x * h + y]`): the squared
    // distance to the nearest source of the column and that source's row
    let mut col_d2 = vec![f64::INFINITY; w * h];
    let mut col_row = vec![0u32; w * h];
    col_d2
        .par_chunks_mut(h)
        .zip(col_row.par_chunks_mut(h))
        .enumerate()
        .for_each_init(
            || Line::new(h),
            |line, (x, (d2, row))| {
                for (y, f) in line.f.iter_mut().enumerate() {
                    *f = if src.data[y * w + x] {
                        0.0
                    } else {
                        f64::INFINITY
                    };
                }
                nearest_1d(line, d2, row);
            },
        );

    // pass 2 along x, one row at a time, over the columns' candidates
    let mut out = vec![0u32; w * h];
    out.par_chunks_mut(w).enumerate().for_each_init(
        || (Line::new(w), vec![0.0f64; w], vec![0u32; w]),
        |(line, d2, col), (y, dst)| {
            for (x, f) in line.f.iter_mut().enumerate() {
                *f = col_d2[x * h + y];
            }
            nearest_1d(line, d2, col);
            for (o, &c) in dst.iter_mut().zip(col.iter()) {
                *o = col_row[c as usize * h + y] * w as u32 + c;
            }
        },
    );
    Some(out)
}

/// Pixels of `b` that belong to a 4-connected component of at least `min_area` pixels
/// (`track._large_components`; `scipy.ndimage.label` default structure: edge neighbours only).
pub fn large_components(b: &Image<bool>, min_area: f64) -> Image<bool> {
    let (w, h) = (b.w, b.h);
    let mut out = Image::new(w, h);
    let mut seen = vec![false; w * h];
    let mut comp: Vec<usize> = Vec::new(); // breadth-first queue = members of the component
    for start in 0..w * h {
        if !b.data[start] || seen[start] {
            continue;
        }
        seen[start] = true;
        comp.clear();
        comp.push(start);
        let mut i = 0;
        while i < comp.len() {
            let p = comp[i];
            i += 1;
            let (x, y) = (p % w, p / w);
            let mut visit = |q: usize| {
                if b.data[q] && !seen[q] {
                    seen[q] = true;
                    comp.push(q);
                }
            };
            if x > 0 {
                visit(p - 1);
            }
            if x + 1 < w {
                visit(p + 1);
            }
            if y > 0 {
                visit(p - w);
            }
            if y + 1 < h {
                visit(p + w);
            }
        }
        if comp.len() as f64 >= min_area {
            for &p in &comp {
                out.data[p] = true;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn mask(rows: &[&str]) -> Image<bool> {
        let w = rows[0].len();
        let data = rows
            .iter()
            .flat_map(|r| r.bytes().map(|c| c == b'#'))
            .collect();
        Image::from_vec(w, rows.len(), data)
    }

    /// Brute force: the minimum over all invalid pixels, scipy convention without any.
    fn brute_force(valid: &Image<bool>) -> Vec<f32> {
        let (w, h) = (valid.w as i64, valid.h as i64);
        let invalid: Vec<(i64, i64)> = (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .filter(|&(x, y)| !valid.data[(y * w + x) as usize])
            .collect();
        let mut out = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let d2 = if invalid.is_empty() {
                    (y + 1) * (y + 1) + x * x
                } else {
                    invalid
                        .iter()
                        .map(|&(ix, iy)| (x - ix) * (x - ix) + (y - iy) * (y - iy))
                        .min()
                        .unwrap()
                };
                out.push((d2 as f64).sqrt() as f32);
            }
        }
        out
    }

    #[test]
    fn edt_hand_case() {
        let valid = mask(&[
            "#####", //
            "##.##", //
            "#####",
        ]);
        let d = distance_to_invalid(&valid);
        let (s5, s2) = (5.0f64.sqrt() as f32, 2.0f64.sqrt() as f32);
        #[rustfmt::skip]
        let want = [
            s5,  s2,  1.0, s2,  s5,
            2.0, 1.0, 0.0, 1.0, 2.0,
            s5,  s2,  1.0, s2,  s5,
        ];
        assert_eq!(d.data, want.to_vec());
    }

    #[test]
    fn edt_invalid_pixels_are_zero_and_far_pixels_are_exact() {
        let mut valid = Image::from_vec(40, 3, vec![true; 120]);
        valid.set(0, 1, false);
        let d = distance_to_invalid(&valid);
        assert_eq!(d.get(0, 1), 0.0);
        assert_eq!(d.get(39, 1), 39.0);
        assert_eq!(d.get(39, 0), (39.0f64 * 39.0 + 1.0).sqrt() as f32);
    }

    #[test]
    fn edt_without_invalid_pixel_follows_scipy() {
        // scipy 1.18.1: distance_transform_edt(np.ones((4, 3), bool))**2
        let d = distance_to_invalid(&Image::from_vec(3, 4, vec![true; 12]));
        let sq: Vec<f32> = d.data.iter().map(|v| (v * v).round()).collect();
        assert_eq!(
            sq,
            [1., 2., 5., 4., 5., 8., 9., 10., 13., 16., 17., 20.].to_vec()
        );
    }

    #[test]
    fn edt_all_invalid_is_zero_and_empty_image_is_fine() {
        let d = distance_to_invalid(&Image::<bool>::new(5, 4));
        assert!(d.data.iter().all(|&v| v == 0.0));
        let e = distance_to_invalid(&Image::<bool>::new(0, 7));
        assert_eq!((e.w, e.h, e.data.len()), (0, 7, 0));
    }

    /// Brute force: of the sources nearest to each pixel (squared distance in integers), the one in
    /// the smallest column, and in that column the smallest row: scipy's rule (see [`nearest_source`]).
    fn brute_force_nearest(src: &Image<bool>) -> Vec<u32> {
        let (w, h) = (src.w, src.h);
        let sources: Vec<(usize, usize)> = (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .filter(|&(x, y)| src.data[y * w + x])
            .collect();
        let mut out = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let d2 = |&(sx, sy): &(usize, usize)| {
                    (x as i64 - sx as i64).pow(2) + (y as i64 - sy as i64).pow(2)
                };
                // (d2, column, row) in lexicographic order
                let (sx, sy) = *sources
                    .iter()
                    .min_by_key(|s| (d2(s), s.0, s.1))
                    .expect("at least one source");
                out.push((sy * w + sx) as u32);
            }
        }
        out
    }

    #[test]
    fn nearest_source_hand_cases_and_ties() {
        // two sources equally far from (2, 0) and (2, 2): the one in the smaller column wins
        let src = mask(&[
            ".....", //
            "#...#", //
            ".....",
        ]);
        let idx = nearest_source(&src).unwrap();
        let at = |x: usize, y: usize| idx[y * 5 + x] as usize;
        let (left, right) = (5, 5 + 4); // row-major indices of the two sources
        assert_eq!(
            (at(0, 1), at(4, 1)),
            (left, right),
            "a source is its own nearest"
        );
        assert_eq!(
            at(2, 0),
            left,
            "x = 2 is equally far from both: smaller column"
        );
        assert_eq!(at(2, 2), left);
        assert_eq!((at(1, 0), at(3, 2)), (left, right));
        // two sources in one column, equally far from the pixel between them: smaller row
        let src = mask(&[
            ".#.", //
            "...", //
            ".#.", //
            "...", //
            "...",
        ]);
        let idx = nearest_source(&src).unwrap();
        let at = |x: usize, y: usize| idx[y * 3 + x] as usize;
        let (top, bottom) = (1usize, 2 * 3 + 1usize); // row-major indices of the two sources
        assert_eq!(at(1, 1), top, "(1, 1): rows 0 and 2 are equally far");
        assert_eq!((at(1, 3), at(0, 4)), (bottom, bottom));
        assert_eq!(
            (at(1, 0), at(1, 2)),
            (top, bottom),
            "a source is its own nearest"
        );
        // no source: no answer
        assert_eq!(nearest_source(&Image::<bool>::new(4, 3)), None);
        assert_eq!(nearest_source(&Image::<bool>::new(0, 3)), None);
    }

    fn sources_strategy() -> impl Strategy<Value = Image<bool>> {
        (1usize..=24, 1usize..=24, 0.0f64..=1.0).prop_flat_map(|(w, h, p)| {
            (
                proptest::collection::vec(proptest::bool::weighted(p), w * h),
                0..w * h,
            )
                .prop_map(move |(mut d, forced)| {
                    d[forced] = true; // at least one source
                    Image::from_vec(w, h, d)
                })
        })
    }

    /// Images up to 64 x 64 that are valid except for at most 6 pixels (none: the scipy convention
    /// without an invalid pixel). The dense masks below reach 20 x 20 only; here most pixels are far
    /// from every invalid one, the lines of the transform hold long runs of infinite entries and
    /// the lower envelopes have few, widely spaced parabolas.
    fn sparse_invalid_strategy() -> impl Strategy<Value = Image<bool>> {
        (2usize..=64, 2usize..=64).prop_flat_map(|(w, h)| {
            proptest::collection::vec(0..w * h, 0..=6).prop_map(move |holes| {
                let mut valid = vec![true; w * h];
                for p in holes {
                    valid[p] = false;
                }
                Image::from_vec(w, h, valid)
            })
        })
    }

    /// Images up to 64 x 64 with 1 to 6 source pixels.
    fn sparse_sources_strategy() -> impl Strategy<Value = Image<bool>> {
        (1usize..=64, 1usize..=64).prop_flat_map(|(w, h)| {
            proptest::collection::vec(0..w * h, 1..=6).prop_map(move |srcs| {
                let mut src = vec![false; w * h];
                for p in srcs {
                    src[p] = true;
                }
                Image::from_vec(w, h, src)
            })
        })
    }

    fn mask_strategy() -> impl Strategy<Value = Image<bool>> {
        (2usize..=20, 2usize..=20, 0.0f64..=1.0).prop_flat_map(|(w, h, p)| {
            proptest::collection::vec(proptest::bool::weighted(p), w * h)
                .prop_map(move |d| Image::from_vec(w, h, d))
        })
    }

    proptest! {
        #[test]
        fn edt_matches_brute_force(valid in mask_strategy()) {
            let got = distance_to_invalid(&valid);
            prop_assert_eq!(got.data, brute_force(&valid));
        }

        #[test]
        fn nearest_source_matches_brute_force_with_scipys_tie_rule(src in sources_strategy()) {
            prop_assert_eq!(nearest_source(&src).unwrap(), brute_force_nearest(&src));
        }

        #[test]
        fn edt_matches_brute_force_on_large_sparse_masks(valid in sparse_invalid_strategy()) {
            let got = distance_to_invalid(&valid);
            prop_assert_eq!(got.data, brute_force(&valid));
        }

        #[test]
        fn nearest_source_matches_brute_force_on_large_sparse_source_sets(
            src in sparse_sources_strategy(),
        ) {
            prop_assert_eq!(nearest_source(&src).unwrap(), brute_force_nearest(&src));
        }

        #[test]
        fn nearest_source_agrees_with_the_distance_transform(src in sources_strategy()) {
            let idx = nearest_source(&src).unwrap();
            let valid = Image::from_vec(src.w, src.h, src.data.iter().map(|&s| !s).collect());
            let dist = distance_to_invalid(&valid);
            for (p, &s) in idx.iter().enumerate() {
                let (x, y, sx, sy) = ((p % src.w) as i64, (p / src.w) as i64, (s as usize % src.w) as i64, (s as usize / src.w) as i64);
                prop_assert!(src.data[s as usize]);
                let d = (((x - sx).pow(2) + (y - sy).pow(2)) as f64).sqrt() as f32;
                prop_assert_eq!(d, dist.data[p]);
            }
        }

        #[test]
        fn edt_is_zero_exactly_on_invalid_pixels(valid in mask_strategy()) {
            let got = distance_to_invalid(&valid);
            for (&d, &ok) in got.data.iter().zip(&valid.data) {
                prop_assert_eq!(d == 0.0, !ok);
            }
        }

        #[test]
        fn large_components_matches_label_propagation(
            b in mask_strategy(),
            min_area in 0.0f64..30.0,
        ) {
            // independent reference: propagate the minimum label to 4-neighbours until stable
            let (w, h) = (b.w, b.h);
            let mut lab: Vec<usize> = (0..w * h).collect();
            loop {
                let mut changed = false;
                for y in 0..h {
                    for x in 0..w {
                        let p = y * w + x;
                        if !b.data[p] {
                            continue;
                        }
                        let nb = [
                            (x > 0).then(|| p - 1),
                            (x + 1 < w).then(|| p + 1),
                            (y > 0).then(|| p - w),
                            (y + 1 < h).then(|| p + w),
                        ];
                        for q in nb.into_iter().flatten() {
                            if b.data[q] && lab[q] < lab[p] {
                                lab[p] = lab[q];
                                changed = true;
                            }
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
            let want: Vec<bool> = (0..w * h)
                .map(|p| {
                    b.data[p] && {
                        let size = (0..w * h).filter(|&q| b.data[q] && lab[q] == lab[p]).count();
                        size as f64 >= min_area
                    }
                })
                .collect();
            prop_assert_eq!(large_components(&b, min_area).data, want);
        }
    }
}
