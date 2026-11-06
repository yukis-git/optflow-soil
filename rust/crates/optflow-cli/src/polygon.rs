//! ROI polygon fill with the pixel rules of Pillow's `ImageDraw.polygon(xy, fill=1)`.
//!
//! The reference rasteriser is Pillow's C scan-line filler, which has no written specification.
//! Its behaviour was reproduced by differential testing against Pillow 13.0 built for arm64. What
//! `tests/polygon.rs` checks against masks Pillow drew (`tests/fixtures/polygon_cases.json`) is
//! 160 random simple polygons and 30 polygons with float vertices at scales 2 to 4 (190 cases),
//! plus the real ROI of `data/example_roi.json` at scales 1, 2 and 4. The larger
//! fuzz run behind the rules below (about 40000 random polygons, simple and self-intersecting)
//! was a one-off and is not committed. What the filler does:
//!
//! * vertices are truncated toward zero to integers (`(int)xy`), so a polygon scaled by `1/scale`
//!   loses the fraction of every vertex;
//! * every non-horizontal edge crosses each scan line `y` of its extent at
//!   `x = fma(y - y0, dx, x0)` in `f32`, with `dx = (x1 - x0) / (y1 - y0)`; the sorted crossings
//!   are paired and each pair `(a, b)` fills `ROUND_UP(a)..=ROUND_DOWN(b)`, both ends inclusive,
//!   where the roundings differ only at an exact `.5`;
//! * an edge ending on a scan line below the last one adds its crossing a second time (so a
//!   vertex shared by two edges that continue on the same side counts twice);
//! * a horizontal edge is drawn as a run;
//! * at a vertex where both edges leave on the same vertical side (a local top, or the bottom of
//!   the last scan line) and both lean to the same horizontal side, one of the two crossings moves
//!   to the pixel next to the nearer edge's position on the adjacent line, which widens the run
//!   there ("sharp tips" are not clipped to a single pixel).
//!
//! The fused multiply-add matches Pillow built for arm64 (clang contracts `a * b + c`); a build
//! without FMA differs only where a crossing is within one `f32` rounding of a pixel boundary.
//!
//! All arithmetic on coordinates is done in `i64`, so any `i32` vertices are safe; the ROI loader
//! keeps vertices within [`MAX_COORD`] so that the `f32` crossings stay meaningful.

use optflow_core::image::Image;

/// Largest coordinate magnitude (full-resolution pixels) a ROI vertex may have; see `load_roi`.
pub const MAX_COORD: f64 = (1u32 << 30) as f64;

/// One polygon edge: first end `(x0, y0)` as given, the bounding box, and `dx` per scan line.
struct Edge {
    x0: i64,
    y0: i64,
    ymin: i64,
    ymax: i64,
    xmin: i64,
    xmax: i64,
    dx: f32,
}

impl Edge {
    fn new((x0, y0): (i32, i32), (x1, y1): (i32, i32)) -> Self {
        let (x0, y0, x1, y1) = (i64::from(x0), i64::from(y0), i64::from(x1), i64::from(y1));
        let dx = if y0 == y1 {
            0.0
        } else {
            (x1 - x0) as f32 / (y1 - y0) as f32
        };
        Self {
            x0,
            y0,
            ymin: y0.min(y1),
            ymax: y0.max(y1),
            xmin: x0.min(x1),
            xmax: x0.max(x1),
            dx,
        }
    }

    fn horizontal(&self) -> bool {
        self.ymin == self.ymax
    }

    /// The edge's x at scan line `y`, as Pillow computes it. The fused multiply-add is deliberate
    /// and exact: `mul_add` rounds once on every platform (a library call where the hardware has
    /// no FMA), so the mask is the same everywhere; it is unrelated to optflow-core's `fma` feature.
    fn x_at(&self, y: i64) -> f32 {
        ((y - self.y0) as f32).mul_add(self.dx, self.x0 as f32)
    }
}

/// `ROUND_UP` of Pillow's `Draw.c`: round half up (toward +infinity for positive values).
fn round_up(f: f32) -> i64 {
    if f >= 0.0 {
        (f + 0.5).floor() as i64
    } else {
        -((f.abs() + 0.5).floor() as i64)
    }
}

/// `ROUND_DOWN` of Pillow's `Draw.c`: round half down.
fn round_down(f: f32) -> i64 {
    if f >= 0.0 {
        (f - 0.5).ceil() as i64
    } else {
        -((f.abs() - 0.5).ceil() as i64)
    }
}

/// Sets `mask[y][x0..=x1]`, clipped to the image; nothing if the run is empty or off-image.
fn hline(mask: &mut Image<bool>, x0: i64, y: i64, x1: i64) {
    if y < 0 || y >= mask.h as i64 {
        return;
    }
    let x0 = x0.max(0);
    let x1 = x1.min(mask.w as i64 - 1);
    if x0 <= x1 {
        let row = y as usize * mask.w;
        mask.data[row + x0 as usize..=row + x1 as usize].fill(true);
    }
}

/// A change of one crossing of a scan line (see the module comment): the edge whose crossing
/// moves, and where to.
struct Tip {
    row: i64,
    edge: usize,
    to: f32,
}

/// Fills the polygon with integer vertices `pts` into a `w` x `h` mask, as Pillow does.
pub fn fill_polygon(w: usize, h: usize, pts: &[(i32, i32)]) -> Image<bool> {
    let mut mask = Image::<bool>::new(w, h);
    let n = pts.len();
    if n == 0 {
        return mask;
    }
    let mut edges: Vec<Edge> = pts.windows(2).map(|p| Edge::new(p[0], p[1])).collect();
    if pts[n - 1] != pts[0] {
        edges.push(Edge::new(pts[n - 1], pts[0]));
    }
    let m = edges.len();
    let height = h as i64;
    let (mut ymin, mut ymax) = (height - 1, 0);
    let mut table = Vec::new(); // the non-horizontal edges
    for (k, e) in edges.iter().enumerate() {
        ymin = ymin.min(e.ymin);
        ymax = ymax.max(e.ymax);
        if e.horizontal() {
            hline(&mut mask, e.xmin, e.ymin, e.xmax);
        } else {
            table.push(k);
        }
    }
    ymin = ymin.max(0);
    ymax = ymax.min(height);

    // vertices where the two edges leave on the same vertical side and lean to the same side
    let mut tips = Vec::new();
    for i in 0..m {
        let (a, b) = (&edges[i], &edges[(i + 1) % m]);
        if a.horizontal() || b.horizontal() {
            continue;
        }
        let (vx, vy) = pts[(i + 1) % n];
        let vy = i64::from(vy);
        let step = if a.ymin == vy && b.ymin == vy {
            1
        } else if a.ymax == vy && b.ymax == vy && vy >= ymax {
            -1 // a bottom vertex counts only on the last scan line
        } else {
            continue;
        };
        let (xa, xb) = (a.x_at(vy + step), b.x_at(vy + step));
        let vxf = vx as f32;
        if xa < vxf && xb < vxf {
            let to = round_up(xa.max(xb)) + 1;
            if (to as f32) < vxf {
                tips.push(Tip {
                    row: vy,
                    edge: i,
                    to: to as f32,
                });
            }
        } else if xa > vxf && xb > vxf {
            let to = round_up(xa.min(xb)) - 1;
            if (to as f32) > vxf {
                tips.push(Tip {
                    row: vy,
                    edge: i,
                    to: to as f32,
                });
            }
        }
    }

    let mut crossings: Vec<(usize, f32)> = Vec::new();
    let mut xs: Vec<f32> = Vec::new();
    for y in ymin..=ymax {
        crossings.clear();
        for &k in &table {
            let e = &edges[k];
            if e.ymin <= y && y <= e.ymax {
                crossings.push((k, e.x_at(y)));
            }
        }
        for t in tips.iter().filter(|t| t.row == y) {
            if let Some(c) = crossings.iter_mut().find(|c| c.0 == t.edge) {
                c.1 = t.to;
            }
        }
        xs.clear();
        xs.extend(crossings.iter().map(|c| c.1));
        for &(k, x) in &crossings {
            if y == edges[k].ymax && y < ymax {
                xs.push(x); // needed to draw consistent polygons
            }
        }
        xs.sort_by(f32::total_cmp);
        for pair in xs.as_chunks::<2>().0 {
            hline(&mut mask, round_up(pair[0]), y, round_down(pair[1]));
        }
    }
    mask
}

/// The ROI mask of `Image.new("1", (w, h))` + `ImageDraw.polygon([(x / scale, y / scale), ...],
/// fill=1)`: `points` are in full-resolution pixels, the mask has the downscaled size.
pub fn polygon_mask(w: usize, h: usize, points: &[(f64, f64)], scale: usize) -> Image<bool> {
    let s = scale as f64;
    let pts: Vec<(i32, i32)> = points
        .iter()
        .map(|&(x, y)| ((x / s) as i32, (y / s) as i32))
        .collect();
    fill_polygon(w, h, &pts)
}
