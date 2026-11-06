//! Small numeric helpers with the behaviour of the numpy / scipy routines `xdic.py` calls
//! (`median_filter`, `label`, `binary_dilation`, `map_coordinates`, `np.median`, `np.linalg`).

/// `np.median` (the mean of the two middle values for an even count); NaN for an empty slice.
pub fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

/// `np.nanmedian` (NaN when no value is finite).
pub fn nanmedian(v: &[f64]) -> f64 {
    let f: Vec<f64> = v.iter().copied().filter(|x| !x.is_nan()).collect();
    median(&f)
}

/// `scipy.ndimage.median_filter(a, size=3, mode="nearest")` of a 1-D array.
pub fn median3_nearest(a: &[f64]) -> Vec<f64> {
    let n = a.len();
    (0..n)
        .map(|i| {
            let mut t = [a[i.saturating_sub(1)], a[i], a[(i + 1).min(n - 1)]];
            t.sort_by(f64::total_cmp);
            t[1]
        })
        .collect()
}

/// `scipy.ndimage.label` of a `w` x `h` mask with the 3 x 3 structure (`diag`, 8-connected) or the
/// default cross (4-connected): the labels (0 = background) and their number. Labels are numbered
/// in the raster order of the first pixel of each component, as scipy does.
pub fn label(mask: &[bool], w: usize, h: usize, diag: bool) -> (Vec<u32>, u32) {
    let mut lab = vec![0u32; w * h];
    let mut n = 0u32;
    let mut stack: Vec<usize> = Vec::new();
    for s in 0..w * h {
        if !mask[s] || lab[s] != 0 {
            continue;
        }
        n += 1;
        lab[s] = n;
        stack.push(s);
        while let Some(p) = stack.pop() {
            let (x, y) = ((p % w) as isize, (p / w) as isize);
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    if (dx == 0 && dy == 0) || (!diag && dx != 0 && dy != 0) {
                        continue;
                    }
                    let (qx, qy) = (x + dx, y + dy);
                    if qx < 0 || qy < 0 || qx >= w as isize || qy >= h as isize {
                        continue;
                    }
                    let q = qy as usize * w + qx as usize;
                    if mask[q] && lab[q] == 0 {
                        lab[q] = n;
                        stack.push(q);
                    }
                }
            }
        }
    }
    (lab, n)
}

/// `binary_dilation(mask, structure=ones((3, 3)), iterations=iters)`.
pub fn dilate3(mask: &[bool], w: usize, h: usize, iters: usize) -> Vec<bool> {
    let mut cur = mask.to_vec();
    for _ in 0..iters {
        let mut next = vec![false; w * h];
        for y in 0..h {
            for x in 0..w {
                if !cur[y * w + x] {
                    continue;
                }
                for yy in y.saturating_sub(1)..(y + 2).min(h) {
                    for xx in x.saturating_sub(1)..(x + 2).min(w) {
                        next[yy * w + xx] = true;
                    }
                }
            }
        }
        cur = next;
    }
    cur
}

/// `map_coordinates(a, [y, x], order=1, mode="nearest")` at one point: bilinear, the image
/// continued by its edge values (the coordinate is clamped to the image).
pub fn bilinear_nearest(a: &[f64], w: usize, h: usize, x: f64, y: f64) -> f64 {
    let xc = x.clamp(0.0, (w - 1) as f64);
    let yc = y.clamp(0.0, (h - 1) as f64);
    let x0 = xc.floor() as usize;
    let y0 = yc.floor() as usize;
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (tx, ty) = (xc - x0 as f64, yc - y0 as f64);
    let at = |yy: usize, xx: usize| a[yy * w + xx];
    (1.0 - ty) * ((1.0 - tx) * at(y0, x0) + tx * at(y0, x1))
        + ty * ((1.0 - tx) * at(y1, x0) + tx * at(y1, x1))
}

/// Inverse of a 3 x 3 matrix (adjugate over determinant); the identity if it is singular.
pub fn inv3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
    let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
    if det == 0.0 || !det.is_finite() {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let id = 1.0 / det;
    [
        [
            c00 * id,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * id,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * id,
        ],
        [
            c01 * id,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * id,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * id,
        ],
        [
            c02 * id,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * id,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * id,
        ],
    ]
}

/// Least squares `min |A x - y|` for the `n x K` matrix `a` (rows), by Householder QR; a
/// numerically zero pivot gives a zero component (the minimum-norm choice of `np.linalg.lstsq`
/// for the rank deficient cases that matter here).
pub fn lstsq<const K: usize>(a: &[[f64; K]], y: &[f64]) -> [f64; K] {
    let n = a.len();
    let mut cols: Vec<Vec<f64>> = (0..K).map(|k| a.iter().map(|r| r[k]).collect()).collect();
    let mut b = y.to_vec();
    let mut scale = 0.0f64;
    for c in &cols {
        scale = scale.max(c.iter().fold(0.0f64, |m, v| m.max(v.abs())));
    }
    for k in 0..K.min(n) {
        let norm = cols[k][k..].iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm == 0.0 {
            continue;
        }
        let alpha = if cols[k][k] > 0.0 { -norm } else { norm };
        let mut v: Vec<f64> = cols[k][k..].to_vec();
        v[0] -= alpha;
        let vn2: f64 = v.iter().map(|t| t * t).sum();
        if vn2 == 0.0 {
            continue;
        }
        for j in k..K {
            let d: f64 = v.iter().zip(&cols[j][k..]).map(|(p, q)| p * q).sum();
            let f = 2.0 * d / vn2;
            for (q, p) in cols[j][k..].iter_mut().zip(&v) {
                *q -= f * p;
            }
        }
        let d: f64 = v.iter().zip(&b[k..]).map(|(p, q)| p * q).sum();
        let f = 2.0 * d / vn2;
        for (q, p) in b[k..].iter_mut().zip(&v) {
            *q -= f * p;
        }
    }
    let mut x = [0.0f64; K];
    for k in (0..K.min(n)).rev() {
        let r = cols[k][k];
        if r.abs() <= 1e-12 * scale.max(1e-300) {
            continue;
        }
        let mut s = b[k];
        for j in k + 1..K {
            s -= cols[j][k] * x[j];
        }
        x[k] = s / r;
    }
    x
}

/// Unit eigenvector of the larger eigenvalue of the symmetric matrix `[[a, b], [b, c]]`, signed
/// as LAPACK's `eigh` returns it for 2 x 2 input (checked against numpy on random matrices):
/// `b < 0`: (-, +); `b > 0`: (+, +) for `a <= c`, (-, -) for `a > c`; `b = 0`: the axis.
pub fn major_axis(a: f64, b: f64, c: f64) -> [f64; 2] {
    if b == 0.0 {
        return if a >= c { [1.0, 0.0] } else { [0.0, 1.0] };
    }
    let th = 0.5 * (2.0 * b).atan2(a - c);
    let (v0, v1) = (th.cos().abs(), th.sin().abs());
    if b < 0.0 {
        [-v0, v1]
    } else if a <= c {
        [v0, v1]
    } else {
        [-v0, -v1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_follows_numpy() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
        assert!(median(&[]).is_nan());
        assert_eq!(nanmedian(&[f64::NAN, 5.0, 1.0]), 3.0);
    }

    #[test]
    fn median3_uses_the_edge_value_outside() {
        assert_eq!(median3_nearest(&[9.0, 1.0, 2.0, 8.0]), [9.0, 2.0, 2.0, 8.0]);
    }

    #[test]
    fn labels_are_numbered_by_the_first_pixel_in_raster_order() {
        let m = [
            true, false, true, //
            false, true, false, //
            false, false, false, //
            true, false, true,
        ];
        let (l8, n8) = label(&m, 3, 4, true);
        assert_eq!(n8, 3);
        assert_eq!(l8, [1, 0, 1, 0, 1, 0, 0, 0, 0, 2, 0, 3]);
        let (l4, n4) = label(&m, 3, 4, false);
        assert_eq!(n4, 5);
        assert_eq!(l4[2], 2);
    }

    #[test]
    fn lstsq_recovers_a_plane_and_ignores_a_zero_column() {
        let a: Vec<[f64; 3]> = (0..20)
            .map(|i| [1.0, i as f64, (i * i % 7) as f64])
            .collect();
        let y: Vec<f64> = a.iter().map(|r| 2.0 + 0.5 * r[1] - 3.0 * r[2]).collect();
        let x = lstsq(&a, &y);
        assert!(
            (x[0] - 2.0).abs() < 1e-10 && (x[1] - 0.5).abs() < 1e-10 && (x[2] + 3.0).abs() < 1e-10
        );
        let z: Vec<[f64; 2]> = (0..5).map(|i| [i as f64, 0.0]).collect();
        let yz: Vec<f64> = (0..5).map(|i| 2.0 * i as f64).collect();
        let xz = lstsq(&z, &yz);
        assert!((xz[0] - 2.0).abs() < 1e-12 && xz[1] == 0.0);
    }

    #[test]
    fn inv3_inverts() {
        let m = [[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]];
        let v = inv3(&m);
        for i in 0..3 {
            for j in 0..3 {
                let s: f64 = (0..3).map(|k| m[i][k] * v[k][j]).sum();
                assert!((s - f64::from(u8::from(i == j))).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn major_axis_signs_follow_lapack() {
        assert_eq!(major_axis(1.0, 0.0, 2.0), [0.0, 1.0]);
        assert_eq!(major_axis(2.0, 0.0, 1.0), [1.0, 0.0]);
        let v = major_axis(1.0, -0.5, 1.5);
        assert!(v[0] < 0.0 && v[1] > 0.0);
        let v = major_axis(2.0, 0.5, 1.0);
        assert!(v[0] < 0.0 && v[1] < 0.0);
    }

    #[test]
    fn bilinear_clamps_to_the_edge() {
        let a = [0.0, 1.0, 2.0, 3.0]; // 2 x 2
        assert_eq!(bilinear_nearest(&a, 2, 2, 0.5, 0.5), 1.5);
        assert_eq!(bilinear_nearest(&a, 2, 2, -3.0, 9.0), 2.0);
    }
}
