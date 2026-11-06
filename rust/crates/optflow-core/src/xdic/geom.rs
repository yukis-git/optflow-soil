//! Crack polylines: the signed distance field, resampling, extension to the border and the small
//! geometric tests (`polyline_fields`, `_resample`, `_extend_to_border`, `near_lines`,
//! `_merge_collinear`, `_vertex_normals` of `xdic.py`).

use rayon::prelude::*;

use super::Poly;

/// Fields of the nearest polyline per pixel (`polyline_fields`): `phi` signed distance (+ on the
/// left normal), `interior` (the nearest point is not beyond an end), the unit normal `(nx, ny)`
/// of the nearest segment, `lid` the index of the nearest polyline (-1 without lines) and `tau`
/// the arc length of the nearest point along it.
#[derive(Clone, Debug)]
pub struct PolyFields {
    pub w: usize,
    pub h: usize,
    pub phi: Vec<f64>,
    pub interior: Vec<bool>,
    pub nx: Vec<f64>,
    pub ny: Vec<f64>,
    pub lid: Vec<i32>,
    pub tau: Vec<f64>,
}

/// The non-degenerate part of a polyline: its vertices, segment vectors and lengths (segments
/// of length <= 1e-9 dropped).
fn clean(p: &[[f64; 2]]) -> (Vec<[f64; 2]>, Vec<[f64; 2]>, Vec<f64>) {
    let mut pts = vec![p[0]];
    let mut d = Vec::new();
    let mut l = Vec::new();
    for k in 1..p.len() {
        let dv = [p[k][0] - p[k - 1][0], p[k][1] - p[k - 1][1]];
        let len = dv[0].hypot(dv[1]);
        if len > 1e-9 {
            pts.push(p[k]);
            d.push(dv);
            l.push(len);
        }
    }
    (pts, d, l)
}

pub fn polyline_fields(lines: &[Poly], w: usize, h: usize) -> PolyFields {
    let n = w * h;
    let mut out = PolyFields {
        w,
        h,
        phi: vec![0.0; n],
        interior: vec![false; n],
        nx: vec![0.0; n],
        ny: vec![0.0; n],
        lid: vec![-1; n],
        tau: vec![0.0; n],
    };
    if w == 0 {
        return out;
    }
    /// Per segment: start, vector, length, left normal, arc length at the start.
    struct Seg {
        p: [f64; 2],
        d: [f64; 2],
        l: f64,
        l2: f64,
        nrm: [f64; 2],
        cum: f64,
    }
    let segs: Vec<(usize, Vec<Seg>)> = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| !line.is_empty())
        .map(|(li, line)| {
            let (p, d, l) = clean(line);
            let mut cum = 0.0;
            let segs = (0..l.len())
                .map(|i| {
                    let s = Seg {
                        p: p[i],
                        d: d[i],
                        l: l[i],
                        l2: l[i] * l[i],
                        nrm: [-d[i][1] / l[i], d[i][0] / l[i]],
                        cum,
                    };
                    cum += l[i];
                    s
                })
                .collect();
            (li, segs)
        })
        .collect();
    // the pixels are independent of one another: per pixel the segments in their order, the
    // first one nearest. `hypot` only when the squared distance does not already rule the
    // segment out (the margin covers the rounding of both)
    let rows = (
        out.phi.par_chunks_mut(w),
        out.interior.par_chunks_mut(w),
        out.nx.par_chunks_mut(w),
        out.ny.par_chunks_mut(w),
        out.lid.par_chunks_mut(w),
        out.tau.par_chunks_mut(w),
    );
    rows.into_par_iter().enumerate().for_each_init(
        || (vec![f64::INFINITY; w], vec![f64::INFINITY; w]),
        |(best, best2), (y, (phi, interior, nx, ny, lid, tau))| {
            best.fill(f64::INFINITY);
            best2.fill(f64::INFINITY);
            let yf = y as f64;
            for (li, line) in &segs {
                let m = line.len();
                for (i, sg) in line.iter().enumerate() {
                    let ry = yf - sg.p[1];
                    for x in 0..w {
                        let xf = x as f64;
                        let rx = xf - sg.p[0];
                        let t_raw = (rx * sg.d[0] + ry * sg.d[1]) / sg.l2;
                        let t = t_raw.clamp(0.0, 1.0);
                        let (ex, ey) = (rx - t * sg.d[0], ry - t * sg.d[1]);
                        if ex * ex + ey * ey > best2[x] {
                            continue;
                        }
                        let dist = ex.hypot(ey);
                        if dist >= best[x] {
                            continue;
                        }
                        let mut s = rx * sg.nrm[0] + ry * sg.nrm[1];
                        // at an inner vertex the sign comes from the pseudo-normal of the two
                        // segments that meet there
                        if i > 0 && t <= 0.0 {
                            let prev = line[i - 1].nrm;
                            let pn = [prev[0] + sg.nrm[0], prev[1] + sg.nrm[1]];
                            s = rx * pn[0] + ry * pn[1];
                        }
                        if i < m - 1 && t >= 1.0 {
                            let next = line[i + 1].nrm;
                            let pn = [sg.nrm[0] + next[0], sg.nrm[1] + next[1]];
                            let (bx, by) = (xf - line[i + 1].p[0], yf - line[i + 1].p[1]);
                            s = bx * pn[0] + by * pn[1];
                        }
                        let mut inner = true;
                        if i == 0 {
                            inner &= t_raw >= 0.0;
                        }
                        if i == m - 1 {
                            inner &= t_raw <= 1.0;
                        }
                        best[x] = dist;
                        best2[x] = dist * dist * (1.0 + 1e-12);
                        phi[x] = if s >= 0.0 { dist } else { -dist };
                        interior[x] = inner;
                        nx[x] = sg.nrm[0];
                        ny[x] = sg.nrm[1];
                        lid[x] = *li as i32;
                        tau[x] = sg.cum + t * sg.l;
                    }
                }
            }
        },
    );
    out
}

/// Points every `step` px of arc length along `p` (both ends included), their unit normals and
/// arc lengths (`_resample`).
pub fn resample(p: &[[f64; 2]], step: f64) -> (Vec<[f64; 2]>, Vec<[f64; 2]>, Vec<f64>) {
    let (pts, d, l) = clean(p);
    let mut cum = vec![0.0f64; l.len() + 1];
    for i in 0..l.len() {
        cum[i + 1] = cum[i] + l[i];
    }
    let total = cum[l.len()];
    let n = ((total / step).ceil() as usize).max(1);
    // np.linspace(0, total, n + 1): k * (total / n), the last value exactly total
    let delta = total / n as f64;
    let s: Vec<f64> = (0..=n)
        .map(|k| if k == n { total } else { k as f64 * delta })
        .collect();
    let mut q = Vec::with_capacity(n + 1);
    let mut nr = Vec::with_capacity(n + 1);
    for &sv in &s {
        // searchsorted(cum, s, side="right") - 1, clipped to the segments
        let idx = cum
            .partition_point(|&c| c <= sv)
            .saturating_sub(1)
            .min(l.len() - 1);
        let t = (sv - cum[idx]) / l[idx];
        q.push([pts[idx][0] + t * d[idx][0], pts[idx][1] + t * d[idx][1]]);
        nr.push([-d[idx][1] / l[idx], d[idx][0] / l[idx]]);
    }
    (q, nr, s)
}

/// Length of a polyline.
pub fn length(p: &[[f64; 2]]) -> f64 {
    p.windows(2)
        .map(|s| (s[1][0] - s[0][0]).hypot(s[1][1] - s[0][1]))
        .sum()
}

/// Extends an end of `p` to the image border along its end tangent when the border is within
/// `reach` px (`_extend_to_border`).
pub fn extend_to_border(p: &[[f64; 2]], w: usize, h: usize, reach: f64) -> Poly {
    let mut p: Poly = p.to_vec();
    let exit_len = |q: [f64; 2], t: [f64; 2]| -> f64 {
        let mut m = f64::INFINITY;
        for (c, dc, lo, hi) in [
            (q[0], t[0], 0.0, w as f64 - 1.0),
            (q[1], t[1], 0.0, h as f64 - 1.0),
        ] {
            if dc > 1e-12 {
                m = m.min((hi - c) / dc);
            } else if dc < -1e-12 {
                m = m.min((lo - c) / dc);
            }
        }
        m
    };
    let k = 4.min(p.len());
    for end in 0..2 {
        let q: Vec<[f64; 2]> = if end == 0 {
            p[..k].to_vec()
        } else {
            p[p.len() - k..].to_vec()
        };
        let b = if end == 0 { p[0] } else { p[p.len() - 1] };
        // tangent of the last k vertices, pointing out of the line (the principal axis)
        let mean = [
            q.iter().map(|v| v[0]).sum::<f64>() / k as f64,
            q.iter().map(|v| v[1]).sum::<f64>() / k as f64,
        ];
        let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
        for v in &q {
            let (a, c) = (v[0] - mean[0], v[1] - mean[1]);
            sxx += a * a;
            sxy += a * c;
            syy += c * c;
        }
        let th = 0.5 * (2.0 * sxy).atan2(sxx - syy);
        let mut t = [th.cos(), th.sin()];
        let out_dir = if end == 0 {
            [q[0][0] - q[k - 1][0], q[0][1] - q[k - 1][1]]
        } else {
            [q[k - 1][0] - q[0][0], q[k - 1][1] - q[0][1]]
        };
        let dot = t[0] * out_dir[0] + t[1] * out_dir[1] + 1e-300;
        if dot < 0.0 {
            t = [-t[0], -t[1]];
        }
        let ln = exit_len(b, t);
        if 0.0 < ln && ln <= reach {
            let qn = [b[0] + (ln + 2.0) * t[0], b[1] + (ln + 2.0) * t[1]];
            if end == 0 {
                p.insert(0, qn);
            } else {
                p.push(qn);
            }
        }
    }
    p
}

/// Joins polylines whose ends face each other across a gap of at most `gap` px along a common
/// direction (`_merge_collinear`).
pub fn merge_collinear(segs: Vec<Poly>, gap: f64, cos_min: f64, off_max: f64) -> Vec<Poly> {
    let unit = |v: [f64; 2]| {
        let n = v[0].hypot(v[1]).max(1e-12);
        [v[0] / n, v[1] / n]
    };
    let mut segs = segs;
    let mut merged = true;
    while merged && segs.len() > 1 {
        merged = false;
        'search: for i in 0..segs.len() {
            for j in 0..segs.len() {
                if i == j {
                    continue;
                }
                for rev in [false, true] {
                    let a = &segs[i];
                    let b: Poly = if rev {
                        segs[j].iter().rev().copied().collect()
                    } else {
                        segs[j].clone()
                    };
                    let al = a[a.len() - 1];
                    let t_a = unit([al[0] - a[0][0], al[1] - a[0][1]]);
                    let bl = b[b.len() - 1];
                    let t_b = unit([bl[0] - b[0][0], bl[1] - b[0][1]]);
                    let v = [b[0][0] - al[0], b[0][1] - al[1]];
                    if v[0].hypot(v[1]) <= gap
                        && t_a[0] * t_b[0] + t_a[1] * t_b[1] >= cos_min
                        && v[0] * t_a[0] + v[1] * t_a[1] > 0.0
                        && (v[0] * t_a[1] - v[1] * t_a[0]).abs() <= off_max
                    {
                        let mut joined = a.clone();
                        joined.extend(b);
                        let mut rest: Vec<Poly> = segs
                            .iter()
                            .enumerate()
                            .filter(|(k, _)| *k != i && *k != j)
                            .map(|(_, p)| p.clone())
                            .collect();
                        rest.push(joined);
                        segs = rest;
                        merged = true;
                        break 'search;
                    }
                }
            }
        }
    }
    segs
}

/// True when most (> 50 %) of polyline `p` (sampled every 2 px) lies within `dist` px of one of
/// the polylines `others` (`near_lines`).
pub fn near_lines(p: &[[f64; 2]], others: &[Poly], dist: f64) -> bool {
    if others.is_empty() {
        return false;
    }
    let (q, _, _) = resample(p, 2.0);
    let mut best = vec![f64::INFINITY; q.len()];
    for o in others {
        for seg in o.windows(2) {
            let (p0, p1) = (seg[0], seg[1]);
            let d = [p1[0] - p0[0], p1[1] - p0[1]];
            let dd = (d[0] * d[0] + d[1] * d[1]).max(1e-12);
            for (b, pt) in best.iter_mut().zip(&q) {
                let tt = (((pt[0] - p0[0]) * d[0] + (pt[1] - p0[1]) * d[1]) / dd).clamp(0.0, 1.0);
                let dist_i = (pt[0] - (p0[0] + tt * d[0])).hypot(pt[1] - (p0[1] + tt * d[1]));
                *b = b.min(dist_i);
            }
        }
    }
    let near = best.iter().filter(|&&b| b <= dist).count();
    (near as f64 / q.len() as f64) > 0.5
}

/// Unit normals at the vertices from the central differences of the vertices (`_vertex_normals`,
/// `np.gradient` along the polyline).
pub fn vertex_normals(p: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let n = p.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = if n == 1 {
            [1.0, 0.0]
        } else if i == 0 {
            [p[1][0] - p[0][0], p[1][1] - p[0][1]]
        } else if i == n - 1 {
            [p[n - 1][0] - p[n - 2][0], p[n - 1][1] - p[n - 2][1]]
        } else {
            [
                (p[i + 1][0] - p[i - 1][0]) / 2.0,
                (p[i + 1][1] - p[i - 1][1]) / 2.0,
            ]
        };
        let m = t[0].hypot(t[1]).max(1e-12);
        out.push([-t[1] / m, t[0] / m]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_distance_interior_and_normal_of_a_vertical_line() {
        // a line pointing down (+y) at x = 10: the left normal (-t_y, t_x) = (-1, 0)
        let f = polyline_fields(&[vec![[10.0, 5.0], [10.0, 25.0]]], 20, 30);
        let q = |x: usize, y: usize| y * 20 + x;
        assert_eq!(f.phi[q(7, 15)], 3.0);
        assert_eq!(f.phi[q(12, 15)], -2.0);
        assert!(f.interior[q(7, 15)] && !f.interior[q(7, 2)] && !f.interior[q(7, 28)]);
        assert_eq!((f.nx[q(7, 15)], f.ny[q(7, 15)]), (-1.0, 0.0));
        assert_eq!(f.tau[q(7, 15)], 10.0);
        assert_eq!(f.lid[q(0, 0)], 0);
    }

    #[test]
    fn no_lines_give_the_empty_fields() {
        let f = polyline_fields(&[], 4, 3);
        assert!(f.lid.iter().all(|&l| l == -1) && f.phi.iter().all(|&p| p == 0.0));
    }

    #[test]
    fn resample_includes_both_ends() {
        let (q, n, s) = resample(&[[0.0, 0.0], [0.0, 10.0]], 4.0);
        assert_eq!(q.len(), 4);
        assert_eq!(q[0], [0.0, 0.0]);
        assert_eq!(q[3], [0.0, 10.0]);
        assert_eq!(n[1], [-1.0, 0.0]);
        assert_eq!(s[3], 10.0);
    }

    #[test]
    fn extension_reaches_the_border() {
        let p = vec![[50.0, 20.0], [50.0, 40.0], [50.0, 60.0]];
        let e = extend_to_border(&p, 100, 80, 36.0);
        assert!(e[0][1] < 0.0 && e.last().unwrap()[1] > 79.0, "{e:?}");
    }

    #[test]
    fn collinear_pieces_are_joined_across_a_gap() {
        let a = vec![[0.0, 0.0], [0.0, 20.0]];
        let b = vec![[0.0, 40.0], [0.0, 60.0]];
        let m = merge_collinear(vec![a, b], 108.0, 0.95, 8.0);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].len(), 4);
    }

    #[test]
    fn near_lines_needs_most_of_the_line_close() {
        let a = vec![[10.0, 0.0], [10.0, 100.0]];
        let b = vec![[12.0, 0.0], [12.0, 100.0]];
        assert!(near_lines(&a, &[b], 5.0));
        assert!(!near_lines(&a, &[vec![[50.0, 0.0], [50.0, 100.0]]], 5.0));
        assert!(!near_lines(&a, &[], 5.0));
    }

    /// The definition of `polyline_fields`: per pixel the first nearest segment in order, without
    /// any shortcut. The fast version must give the same bits.
    fn direct(lines: &[Poly], w: usize, h: usize) -> PolyFields {
        let n = w * h;
        let mut best = vec![f64::INFINITY; n];
        let mut out = PolyFields {
            w,
            h,
            phi: vec![0.0; n],
            interior: vec![false; n],
            nx: vec![0.0; n],
            ny: vec![0.0; n],
            lid: vec![-1; n],
            tau: vec![0.0; n],
        };
        for (li, line) in lines.iter().enumerate() {
            if line.is_empty() {
                continue;
            }
            let (p, d, l) = clean(line);
            let m = l.len();
            let nrm: Vec<[f64; 2]> = (0..m).map(|i| [-d[i][1] / l[i], d[i][0] / l[i]]).collect();
            let mut cum = vec![0.0f64; m + 1];
            for i in 0..m {
                cum[i + 1] = cum[i] + l[i];
            }
            for i in 0..m {
                for y in 0..h {
                    for x in 0..w {
                        let (xf, yf) = (x as f64, y as f64);
                        let (rx, ry) = (xf - p[i][0], yf - p[i][1]);
                        let t_raw = (rx * d[i][0] + ry * d[i][1]) / (l[i] * l[i]);
                        let t = t_raw.clamp(0.0, 1.0);
                        let (ex, ey) = (rx - t * d[i][0], ry - t * d[i][1]);
                        let dist = ex.hypot(ey);
                        let q = y * w + x;
                        if dist >= best[q] {
                            continue;
                        }
                        let mut s = rx * nrm[i][0] + ry * nrm[i][1];
                        if i > 0 && t <= 0.0 {
                            let pn = [nrm[i - 1][0] + nrm[i][0], nrm[i - 1][1] + nrm[i][1]];
                            s = rx * pn[0] + ry * pn[1];
                        }
                        if i < m - 1 && t >= 1.0 {
                            let pn = [nrm[i][0] + nrm[i + 1][0], nrm[i][1] + nrm[i + 1][1]];
                            let (bx, by) = (xf - p[i + 1][0], yf - p[i + 1][1]);
                            s = bx * pn[0] + by * pn[1];
                        }
                        let mut inner = true;
                        if i == 0 {
                            inner &= t_raw >= 0.0;
                        }
                        if i == m - 1 {
                            inner &= t_raw <= 1.0;
                        }
                        best[q] = dist;
                        out.phi[q] = if s >= 0.0 { dist } else { -dist };
                        out.interior[q] = inner;
                        out.nx[q] = nrm[i][0];
                        out.ny[q] = nrm[i][1];
                        out.lid[q] = li as i32;
                        out.tau[q] = cum[i] + t * l[i];
                    }
                }
            }
        }
        out
    }

    #[test]
    fn polyline_fields_have_the_bits_of_the_definition() {
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let cases: Vec<Vec<Poly>> = vec![
            vec![],
            vec![vec![[10.0, 20.0]]],
            // a kinked line, a repeated vertex, and a second line with equidistant pixels between
            vec![
                vec![[3.0, 3.0], [30.0, 10.0], [30.0, 10.0], [41.5, 40.25]],
                vec![[3.0, 30.0], [20.0, 30.0], [20.0, 45.0], [55.0, 45.0]],
            ],
            // parallel lines: ties at every pixel of the middle row
            vec![
                vec![[0.0, 10.0], [59.0, 10.0]],
                vec![[0.0, 20.0], [59.0, 20.0]],
            ],
        ];
        for lines in &cases {
            let (a, b) = (polyline_fields(lines, 60, 50), direct(lines, 60, 50));
            assert_eq!(bits(&a.phi), bits(&b.phi));
            assert_eq!(bits(&a.nx), bits(&b.nx));
            assert_eq!(bits(&a.ny), bits(&b.ny));
            assert_eq!(bits(&a.tau), bits(&b.tau));
            assert_eq!(a.interior, b.interior);
            assert_eq!(a.lid, b.lid);
        }
    }
}
