//! Strain from a displacement field by local quadratic least-squares fits (NaN-aware
//! Savitzky-Golay), port of `python/optflow_ref/strain.py`.
use rayon::prelude::*;

use crate::image::Image;
use crate::interp::{correlate_sep, Boundary};

/// Quadratic basis 1, x, y, x^2, xy, y^2 as (power of x, power of y).
const BASIS: [(usize, usize); 6] = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)];
/// Highest total power of a window moment: the product of two basis functions.
const MAX_POW: usize = 4;
/// A window is fitted only if at least this fraction of it (`num / den`) is valid.
const MIN_FRAC: (u64, u64) = (1, 2);

/// Displacement gradient `du/dx, du/dy, dv/dx, dv/dy` (NaN where the fit is not available).
#[derive(Clone, Debug, PartialEq)]
pub struct Gradient {
    pub dudx: Image<f32>,
    pub dudy: Image<f32>,
    pub dvdx: Image<f32>,
    pub dvdy: Image<f32>,
}

/// Green-Lagrange strain `E`, small strain `eps`, maximum shear `gamma` (from `E`) and the
/// volumetric strain `det F - 1`.
#[derive(Clone, Debug, PartialEq)]
pub struct Strain {
    pub exx_gl: Image<f32>,
    pub eyy_gl: Image<f32>,
    pub exy_gl: Image<f32>,
    pub exx: Image<f32>,
    pub eyy: Image<f32>,
    pub exy: Image<f32>,
    pub gmax: Image<f32>,
    pub vol: Image<f32>,
}

/// Whether a window holding `count` valid pixels is dense enough, i.e. whether
/// `count >= (num / den) * win^2`, decided in exact integer arithmetic (the Python code compares
/// the float `count >= min_frac * win * win`). Exact for all `u64` arguments.
pub fn window_ok(count: u64, win: u64, min_frac_num: u64, min_frac_den: u64) -> bool {
    let lhs = u128::from(count) * u128::from(min_frac_den);
    let rhs = u128::from(min_frac_num).saturating_mul(u128::from(win) * u128::from(win));
    lhs >= rhs
}

/// Index of the moment `sum a(p + t) tx^px ty^py` in the moment tables.
fn mom(px: usize, py: usize) -> usize {
    px * (MAX_POW + 1) + py
}

/// `sum over the window of a(p + t) * tx^px * ty^py`, zero outside the image
/// (`strain._moment`: `t^py` along rows, then `t^px` along columns).
fn moment(a: &Image<f64>, px: usize, py: usize, t: &[f64]) -> Vec<f64> {
    let ky: Vec<f64> = t.iter().map(|&v| v.powi(py as i32)).collect();
    let kx: Vec<f64> = t.iter().map(|&v| v.powi(px as i32)).collect();
    correlate_sep(a, &ky, &kx, Boundary::Constant0).data
}

/// Solves `M c = rhs` (two right-hand sides) by Cholesky and returns coefficients 1 and 2
/// (the x and y slopes) of both columns as `[[c1_u, c1_v], [c2_u, c2_v]]`.
/// `None` when `M` is not numerically positive definite (the window cannot determine the fit).
#[allow(clippy::needless_range_loop)] // textbook index form of the 6x6 factorisation
fn solve6(m: &[[f64; 6]; 6], rhs: &[[f64; 2]; 6]) -> Option<[[f64; 2]; 2]> {
    let mut l = [[0.0f64; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let mut s = m[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                // relative pivot test: a rank-deficient window leaves only rounding noise here
                if s.is_nan() || s <= 1e-12 * m[i][i] {
                    return None;
                }
                l[i][i] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    let mut c = [[0.0f64; 2]; 6];
    for col in 0..2 {
        let mut y = [0.0f64; 6];
        for i in 0..6 {
            let mut s = rhs[i][col];
            for k in 0..i {
                s -= l[i][k] * y[k];
            }
            y[i] = s / l[i][i];
        }
        for i in (0..6).rev() {
            let mut s = y[i];
            for k in i + 1..6 {
                s -= l[k][i] * c[k][col];
            }
            c[i][col] = s / l[i][i];
        }
    }
    Some([c[1], c[2]])
}

/// Displacement gradient by local quadratic least-squares fits over `win` x `win` windows
/// (`strain.displacement_gradient`, `min_frac = 1/2`). Pixels where `ux` or `uy` is not finite
/// are left out of every fit. The output is NaN where fewer than half the window pixels are
/// valid (the window is cut off at the image border) and, where the valid pixels cannot
/// determine the six coefficients, also NaN (Python raises `LinAlgError` there).
///
/// `win` is the odd window size; as in Python an even `win` gives the `win + 1` wide window with
/// the `win^2` pixel threshold. Moments are accumulated in f64 and each pixel's 6x6 normal
/// equations are assembled from the moment maps and solved independently, so the result does
/// not depend on the number of threads.
pub fn displacement_gradient(ux: &Image<f32>, uy: &Image<f32>, win: usize) -> Gradient {
    displacement_gradient_frac(ux, uy, win, MIN_FRAC)
}

/// [`displacement_gradient`] with the share of the window that must be valid, `min_frac =
/// (num, den)` of Python's `displacement_gradient(..., min_frac=num / den)` (`crack_model =
/// "xdic"` fits one side of a crack with `(1, 4)`).
pub fn displacement_gradient_frac(
    ux: &Image<f32>,
    uy: &Image<f32>,
    win: usize,
    min_frac: (u64, u64),
) -> Gradient {
    assert_eq!(
        (ux.w, ux.h),
        (uy.w, uy.h),
        "displacement_gradient: ux and uy differ in size"
    );
    let (w, h) = (ux.w, ux.h);
    let mut out = Gradient {
        dudx: Image::new(w, h),
        dudy: Image::new(w, h),
        dvdx: Image::new(w, h),
        dvdy: Image::new(w, h),
    };
    if w == 0 || h == 0 {
        return out;
    }
    let half = win / 2;
    let t: Vec<f64> = (0..=2 * half).map(|i| i as f64 - half as f64).collect();

    let valid: Vec<bool> = ux
        .data
        .iter()
        .zip(&uy.data)
        .map(|(a, b)| a.is_finite() && b.is_finite())
        .collect();
    let to_f64 = |src: &Image<f32>| {
        Image::from_vec(
            w,
            h,
            src.data
                .iter()
                .zip(&valid)
                .map(|(&v, &ok)| if ok { f64::from(v) } else { 0.0 })
                .collect(),
        )
    };
    let mask = Image::from_vec(
        w,
        h,
        valid.iter().map(|&ok| f64::from(u8::from(ok))).collect(),
    );
    let (u0, v0) = (to_f64(ux), to_f64(uy));

    // moments of the validity mask for every product of two basis functions
    let mut mm: Vec<Vec<f64>> = vec![Vec::new(); (MAX_POW + 1) * (MAX_POW + 1)];
    for &(ai, bi) in &BASIS {
        for &(aj, bj) in &BASIS {
            let k = mom(ai + aj, bi + bj);
            if mm[k].is_empty() {
                mm[k] = moment(&mask, ai + aj, bi + bj, &t);
            }
        }
    }
    let ru: Vec<Vec<f64>> = BASIS.iter().map(|&(a, b)| moment(&u0, a, b, &t)).collect();
    let rv: Vec<Vec<f64>> = BASIS.iter().map(|&(a, b)| moment(&v0, a, b, &t)).collect();

    let win_u64 = win as u64;
    let Gradient {
        dudx,
        dudy,
        dvdx,
        dvdy,
    } = &mut out;
    dudx.data
        .par_chunks_mut(w)
        .zip(dudy.data.par_chunks_mut(w))
        .zip(dvdx.data.par_chunks_mut(w))
        .zip(dvdy.data.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, (((r_dudx, r_dudy), r_dvdx), r_dvdy))| {
            for x in 0..w {
                let p = y * w + x;
                let nan = (f32::NAN, f32::NAN, f32::NAN, f32::NAN);
                let count = mm[mom(0, 0)][p].round() as u64;
                let (a, b, c, d) = if window_ok(count, win_u64, min_frac.0, min_frac.1) {
                    let mut m = [[0.0f64; 6]; 6];
                    let mut rhs = [[0.0f64; 2]; 6];
                    for (i, &(ai, bi)) in BASIS.iter().enumerate() {
                        for (j, &(aj, bj)) in BASIS.iter().enumerate() {
                            m[i][j] = mm[mom(ai + aj, bi + bj)][p];
                        }
                        rhs[i] = [ru[i][p], rv[i][p]];
                    }
                    match solve6(&m, &rhs) {
                        Some([sx, sy]) => (sx[0] as f32, sy[0] as f32, sx[1] as f32, sy[1] as f32),
                        None => nan,
                    }
                } else {
                    nan
                };
                r_dudx[x] = a;
                r_dudy[x] = b;
                r_dvdx[x] = c;
                r_dvdy[x] = d;
            }
        });
    out
}

/// Green-Lagrange components `(Exx, Eyy, Exy)` of a displacement gradient.
fn green_lagrange(dudx: f64, dudy: f64, dvdx: f64, dvdy: f64) -> (f64, f64, f64) {
    (
        dudx + 0.5 * (dudx * dudx + dvdx * dvdx),
        dvdy + 0.5 * (dudy * dudy + dvdy * dvdy),
        0.5 * (dudy + dvdx) + 0.5 * (dudx * dudy + dvdx * dvdy),
    )
}

/// Strain measures of a displacement gradient (`strain.strain_fields`); NaN propagates.
/// Computed in f64 from the f32 gradient and stored as f32.
pub fn strain_fields(g: &Gradient) -> Strain {
    strain_fields_with(g, true)
}

/// [`strain_fields`], optionally without the small-strain maps `exx`, `eyy`, `exy`: with
/// `small == false` they are left empty (`h == 0`), not computed; the other five are the same.
pub fn strain_fields_with(g: &Gradient, small: bool) -> Strain {
    let (w, h) = (g.dudx.w, g.dudx.h);
    for other in [&g.dudy, &g.dvdx, &g.dvdy] {
        assert_eq!(
            (other.w, other.h),
            (w, h),
            "strain_fields: gradient components differ in size"
        );
    }
    let field = |f: &(dyn Fn(f64, f64, f64, f64) -> f64 + Sync)| {
        let data: Vec<f32> = (0..w * h)
            .into_par_iter()
            .map(|i| {
                f(
                    f64::from(g.dudx.data[i]),
                    f64::from(g.dudy.data[i]),
                    f64::from(g.dvdx.data[i]),
                    f64::from(g.dvdy.data[i]),
                ) as f32
            })
            .collect();
        Image::from_vec(w, h, data)
    };
    let small_field = |f: &(dyn Fn(f64, f64, f64, f64) -> f64 + Sync)| {
        if small {
            field(f)
        } else {
            Image::from_vec(w, 0, Vec::new())
        }
    };
    Strain {
        exx_gl: field(&|a, b, c, d| green_lagrange(a, b, c, d).0),
        eyy_gl: field(&|a, b, c, d| green_lagrange(a, b, c, d).1),
        exy_gl: field(&|a, b, c, d| green_lagrange(a, b, c, d).2),
        exx: small_field(&|a, _, _, _| a),
        eyy: small_field(&|_, _, _, d| d),
        exy: small_field(&|_, b, c, _| 0.5 * (b + c)),
        gmax: field(&|a, b, c, d| {
            let (exx, eyy, exy) = green_lagrange(a, b, c, d);
            2.0 * (((exx - eyy) / 2.0).powi(2) + exy * exy).sqrt()
        }),
        vol: field(&|a, b, c, d| (1.0 + a) * (1.0 + d) - b * c - 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn window_ok_is_the_half_rule() {
        assert!(window_ok(13, 5, 1, 2)); // 26 >= 25
        assert!(!window_ok(12, 5, 1, 2)); // 24 < 25
        assert!(window_ok(8, 4, 1, 2)); // 16 >= 16: equality passes
        assert!(!window_ok(7, 4, 1, 2));
        assert!(window_ok(0, 0, 1, 2)); // empty window: 0 >= 0
        assert!(window_ok(1, 3, 0, 1)); // min_frac = 0 accepts everything
    }

    #[test]
    fn window_ok_does_not_overflow() {
        assert!(!window_ok(u64::MAX, u64::MAX, u64::MAX, u64::MAX));
        assert!(window_ok(u64::MAX, 1, 1, u64::MAX));
        assert!(window_ok(u64::MAX, u64::MAX, 0, u64::MAX));
    }

    /// Quadratic field `a + bx + cy + dx^2 + exy + fy^2` with coefficients `q`.
    fn quad(x: f64, y: f64, q: &[f64; 6]) -> f64 {
        q[0] + q[1] * x + q[2] * y + q[3] * x * x + q[4] * x * y + q[5] * y * y
    }
    fn quad_dx(x: f64, y: f64, q: &[f64; 6]) -> f64 {
        q[1] + 2.0 * q[3] * x + q[4] * y
    }
    fn quad_dy(x: f64, y: f64, q: &[f64; 6]) -> f64 {
        q[2] + q[4] * x + 2.0 * q[5] * y
    }

    fn field(w: usize, h: usize, q: &[f64; 6]) -> Image<f32> {
        let data = (0..w * h)
            .map(|i| quad((i % w) as f64, (i / w) as f64, q) as f32)
            .collect();
        Image::from_vec(w, h, data)
    }

    /// Valid pixels of the `win` x `win` window around `(x, y)`, cut off at the image border.
    fn valid_count(valid: &[bool], w: usize, h: usize, x: usize, y: usize, win: usize) -> usize {
        let r = win / 2;
        let mut n = 0;
        for yy in y.saturating_sub(r)..=(y + r).min(h - 1) {
            for xx in x.saturating_sub(r)..=(x + r).min(w - 1) {
                n += usize::from(valid[yy * w + xx]);
            }
        }
        n
    }

    #[test]
    fn quadratic_field_is_recovered_exactly_and_border_follows_the_half_rule() {
        let (w, h, win) = (12, 9, 5);
        let (qu, qv) = (
            [0.3, 0.01, -0.02, 1e-3, 2e-3, -1e-3],
            [-0.1, 0.02, 0.005, -2e-3, 1e-3, 3e-3],
        );
        let g = displacement_gradient(&field(w, h, &qu), &field(w, h, &qv), win);
        let all = vec![true; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let ok = 2 * valid_count(&all, w, h, x, y, win) >= win * win;
                assert_eq!(g.dudx.data[i].is_finite(), ok, "({x},{y})");
                if ok {
                    let (fx, fy) = (x as f64, y as f64);
                    for (got, want) in [
                        (g.dudx.data[i], quad_dx(fx, fy, &qu)),
                        (g.dudy.data[i], quad_dy(fx, fy, &qu)),
                        (g.dvdx.data[i], quad_dx(fx, fy, &qv)),
                        (g.dvdy.data[i], quad_dy(fx, fy, &qv)),
                    ] {
                        assert!(
                            (f64::from(got) - want).abs() < 1e-5,
                            "({x},{y}) {got} {want}"
                        );
                    }
                }
            }
        }
        // corners see 3 x 3 = 9 of 25 pixels
        assert!(g.dudx.get(0, 0).is_nan() && g.dvdy.get(w - 1, h - 1).is_nan());
        // the middle of a border row sees 3 x 5 = 15 >= 12.5
        assert!(g.dudx.get(5, 0).is_finite());
    }

    #[test]
    fn nan_in_one_component_invalidates_the_pixel_for_both() {
        let (w, h, win) = (9, 9, 3);
        let q = [0.0, 0.01, 0.02, 0.0, 0.0, 0.0];
        let ux = field(w, h, &q);
        let mut uy = field(w, h, &q);
        uy.set(4, 4, f32::NAN);
        let g = displacement_gradient(&ux, &uy, win);
        // the hole leaves 8 of 9 pixels: still fitted (a linear field needs only 3 of them)
        assert!((g.dudx.get(4, 4) - 0.01).abs() < 1e-6);
        // a 3 x 3 window of which only 4 pixels are valid is rejected (4 < 4.5)
        let mut ux2 = ux.clone();
        let mut uy2 = uy.clone();
        for (x, y) in [(3, 3), (4, 3), (5, 3), (3, 4), (4, 4)] {
            ux2.set(x, y, f32::NAN);
            uy2.set(x, y, 0.0);
        }
        let g2 = displacement_gradient(&ux2, &uy2, win);
        assert!(g2.dudx.get(4, 4).is_nan() && g2.dvdy.get(4, 4).is_nan());
    }

    #[test]
    fn rank_deficient_window_gives_nan_not_garbage() {
        // 3 x 3 window of an image 3 x 3 with the 5 valid pixels on a conic (the cross):
        // 5 >= 4.5 passes the density test but six coefficients are undetermined
        let mut ux = Image::from_vec(3, 3, vec![f32::NAN; 9]);
        let mut uy = ux.clone();
        for (x, y) in [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)] {
            ux.set(x, y, 1.0);
            uy.set(x, y, 1.0);
        }
        let g = displacement_gradient(&ux, &uy, 3);
        assert!(g.dudx.get(1, 1).is_nan() && g.dvdy.get(1, 1).is_nan());
    }

    #[test]
    fn result_does_not_depend_on_the_thread_count() {
        let (w, h) = (23, 17);
        let q = [0.1, 0.02, -0.01, 1e-3, -2e-3, 1e-3];
        let mut ux = field(w, h, &q);
        let uy = field(w, h, &[0.0, -0.01, 0.03, 0.0, 1e-3, 2e-3]);
        for i in (0..w * h).step_by(7) {
            ux.data[i] = f32::NAN;
        }
        let serial = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| displacement_gradient(&ux, &uy, 7));
        let parallel = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| displacement_gradient(&ux, &uy, 7));
        // bit-identical, NaN positions included
        let bits = |g: &Gradient| -> Vec<u32> {
            [&g.dudx, &g.dudy, &g.dvdx, &g.dvdy]
                .iter()
                .flat_map(|im| im.data.iter().map(|v| v.to_bits()))
                .collect()
        };
        assert_eq!(bits(&serial), bits(&parallel));
        assert!(serial.dudx.data.iter().any(|v| v.is_finite()));
    }

    #[test]
    fn empty_image_is_fine() {
        let e = Image::<f32>::new(0, 4);
        let g = displacement_gradient(&e, &e, 5);
        assert_eq!((g.dudx.w, g.dudx.h), (0, 4));
        assert_eq!(strain_fields(&g).vol.data.len(), 0);
    }

    #[test]
    fn strain_fields_formulas() {
        // pure shear-free stretch: F = diag(1 + a, 1 + d)
        let one = |v: f32| Image::from_vec(1, 1, vec![v]);
        let g = Gradient {
            dudx: one(0.1),
            dudy: one(0.0),
            dvdx: one(0.0),
            dvdy: one(-0.2),
        };
        let s = strain_fields(&g);
        let near =
            |im: &Image<f32>, want: f64| assert!((f64::from(im.data[0]) - want).abs() < 1e-7);
        near(&s.exx_gl, 0.1 + 0.5 * 0.01);
        near(&s.eyy_gl, -0.2 + 0.5 * 0.04);
        near(&s.exy_gl, 0.0);
        near(&s.exx, 0.1);
        near(&s.eyy, -0.2);
        near(&s.exy, 0.0);
        near(&s.gmax, 2.0 * ((0.105f64 - (-0.18)) / 2.0).abs());
        near(&s.vol, 1.1 * 0.8 - 1.0);
        // shear: dudy = 0.2 -> exy = 0.1, Exy = 0.1 + 0
        let g = Gradient {
            dudx: one(0.0),
            dudy: one(0.2),
            dvdx: one(0.0),
            dvdy: one(0.0),
        };
        let s = strain_fields(&g);
        near(&s.exy, 0.1);
        near(&s.exy_gl, 0.1);
        near(&s.eyy_gl, 0.5 * 0.04);
        // all four gradients non-zero, so that no term of a formula vanishes and an exchanged
        // pair (dudy and dvdx, dudx and dvdy) shows: E = (F^T F - I) / 2 with
        // F = I + [[dudx, dudy], [dvdx, dvdy]]
        let (a, b, c, d) = (0.1f32, 0.05f32, -0.03f32, -0.2f32);
        let g = Gradient {
            dudx: one(a),
            dudy: one(b),
            dvdx: one(c),
            dvdy: one(d),
        };
        let s = strain_fields(&g);
        let [a, b, c, d] = [a, b, c, d].map(f64::from);
        let c11 = (1.0 + a) * (1.0 + a) + c * c;
        let c12 = (1.0 + a) * b + c * (1.0 + d);
        let c22 = b * b + (1.0 + d) * (1.0 + d);
        let (exx, eyy, exy) = (0.5 * (c11 - 1.0), 0.5 * (c22 - 1.0), 0.5 * c12);
        near(&s.exx_gl, exx);
        near(&s.eyy_gl, eyy);
        near(&s.exy_gl, exy);
        near(&s.exx, a);
        near(&s.eyy, d);
        near(&s.exy, 0.5 * (b + c));
        // the largest shear of E is the difference of its two eigenvalues
        let (mean, radius) = (
            0.5 * (exx + eyy),
            (0.25 * (exx - eyy).powi(2) + exy * exy).sqrt(),
        );
        near(&s.gmax, (mean + radius) - (mean - radius));
        near(&s.vol, (1.0 + a) * (1.0 + d) - b * c - 1.0);
        // NaN propagates through every formula that involves the NaN component (the gradient
        // fields are NaN together, so in practice all of them)
        let g = Gradient {
            dudx: one(f32::NAN),
            dudy: one(0.0),
            dvdx: one(0.0),
            dvdy: one(0.0),
        };
        let s = strain_fields(&g);
        for im in [&s.exx_gl, &s.exy_gl, &s.exx, &s.gmax, &s.vol] {
            assert!(im.data[0].is_nan());
        }
        // Eyy and eyy do not involve dudx
        assert_eq!((s.eyy_gl.data[0], s.eyy.data[0]), (0.0, 0.0));
    }

    proptest! {
        /// Any 5 x 5 window with >= 13 valid pixels determines a quadratic exactly, so the
        /// gradient of a quadratic field is recovered wherever it is available, whatever the
        /// holes, and it is available exactly where the half rule says.
        #[test]
        fn quadratic_gradient_is_exact_with_holes(
            (w, h, codes) in (5usize..=12, 5usize..=12).prop_flat_map(|(w, h)| {
                (Just(w), Just(h), proptest::collection::vec(0u8..100, w * h))
            }),
            q in proptest::array::uniform6(-0.05f64..0.05),
            hole_below in 0u8..40,
        ) {
            let win = 5;
            let mut ux = field(w, h, &q);
            let mut uy = ux.clone();
            let mut valid = vec![true; w * h];
            for (i, &c) in codes.iter().enumerate() {
                if c < hole_below {
                    valid[i] = false;
                    // a hole in either component removes the pixel from both fits
                    if c % 2 == 0 { ux.data[i] = f32::NAN } else { uy.data[i] = f32::NAN }
                }
            }
            let g = displacement_gradient(&ux, &uy, win);
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    let ok = 2 * valid_count(&valid, w, h, x, y, win) >= win * win;
                    prop_assert_eq!(g.dudx.data[i].is_finite(), ok);
                    prop_assert_eq!(g.dvdy.data[i].is_finite(), ok);
                    if ok {
                        let (fx, fy) = (x as f64, y as f64);
                        prop_assert!((f64::from(g.dudx.data[i]) - quad_dx(fx, fy, &q)).abs() < 1e-4);
                        prop_assert!((f64::from(g.dudy.data[i]) - quad_dy(fx, fy, &q)).abs() < 1e-4);
                        prop_assert!((f64::from(g.dvdx.data[i]) - quad_dx(fx, fy, &q)).abs() < 1e-4);
                        prop_assert!((f64::from(g.dvdy.data[i]) - quad_dy(fx, fy, &q)).abs() < 1e-4);
                    }
                }
            }
        }
    }
}

#[cfg(kani)]
mod proofs {
    use super::*;

    #[kani::proof]
    fn k11_window_ok_matches_half_rule() {
        let count: u64 = kani::any();
        let win: u64 = kani::any();
        kani::assume(win >= 1 && win <= 255 && count <= win * win);
        assert_eq!(window_ok(count, win, 1, 2), 2 * count >= win * win);
    }
}
