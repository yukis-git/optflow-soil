//! `fill_polygon` against Pillow's `ImageDraw.polygon(fill=1)` on tens of thousands of
//! pseudo-random simple polygons, as one digest (`tests/fixtures/make_polygon_digest.py` draws the
//! same polygons with Pillow and prints it; its generator is repeated here exactly).
//!
//! The 190 polygons of `polygon_cases.json` localise a failure; these many more reach the rarer
//! corner cases of Pillow's scan-line fill (horizontal edges beside slanted ones, vertices that
//! touch a scan line, runs outside the image) that mutation testing showed that file does not.

use optflow_cli::polygon::fill_polygon;

const POLYGONS: usize = 40000;
const SIZES: [(usize, usize); 3] = [(12, 16), (20, 28), (33, 25)];
const VERTEX_COUNTS: [usize; 9] = [3, 3, 4, 4, 5, 6, 7, 8, 9];
/// Printed by `make_polygon_digest.py` (Pillow 13.0.0).
const PILLOW_DIGEST: u64 = 0x884E_0E5E_DF86_7D42;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + self.below((hi - lo + 1) as usize) as i32
    }
}

type Pt = (i32, i32);

fn orient(a: Pt, b: Pt, c: Pt) -> i64 {
    let (a, b, c) = (
        (i64::from(a.0), i64::from(a.1)),
        (i64::from(b.0), i64::from(b.1)),
        (i64::from(c.0), i64::from(c.1)),
    );
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

fn on_seg(a: Pt, b: Pt, c: Pt) -> bool {
    a.0.min(b.0) <= c.0 && c.0 <= a.0.max(b.0) && a.1.min(b.1) <= c.1 && c.1 <= a.1.max(b.1)
}

fn cross(p1: Pt, p2: Pt, p3: Pt, p4: Pt) -> bool {
    let (d1, d2) = (orient(p3, p4, p1), orient(p3, p4, p2));
    let (d3, d4) = (orient(p1, p2, p3), orient(p1, p2, p4));
    if d1 * d2 < 0 && d3 * d4 < 0 {
        return true;
    }
    (d1 == 0 && on_seg(p3, p4, p1))
        || (d2 == 0 && on_seg(p3, p4, p2))
        || (d3 == 0 && on_seg(p1, p2, p3))
        || (d4 == 0 && on_seg(p1, p2, p4))
}

fn simple(pts: &[Pt]) -> bool {
    let n = pts.len();
    for i in 0..n {
        if pts[..i].contains(&pts[i]) {
            return false;
        }
    }
    for i in 0..n {
        for j in i + 2..n {
            if i == 0 && j == n - 1 {
                continue;
            }
            if cross(pts[i], pts[(i + 1) % n], pts[j], pts[(j + 1) % n]) {
                return false;
            }
        }
    }
    true
}

fn fnv(h: &mut u64, data: &[u8]) {
    for &b in data {
        *h = (*h ^ u64::from(b)).wrapping_mul(0x100_0000_01B3);
    }
}

/// `np.packbits` of a flat boolean mask: eight pixels to a byte, the first in the high bit.
fn packbits(mask: &[bool]) -> Vec<u8> {
    mask.chunks(8)
        .map(|c| {
            c.iter()
                .enumerate()
                .fold(0u8, |acc, (i, &b)| acc | (u8::from(b) << (7 - i)))
        })
        .collect()
}

#[test]
fn many_random_polygons_match_pillow() {
    let mut rng = Lcg(0x5EED_0F901);
    let mut digest = 0xCBF2_9CE4_8422_2325u64;
    let mut accepted = 0;
    while accepted < POLYGONS {
        let (h, w) = SIZES[rng.below(SIZES.len())];
        let n = VERTEX_COUNTS[rng.below(VERTEX_COUNTS.len())];
        let pts: Vec<Pt> = (0..n)
            .map(|_| (rng.range(-3, w as i32 + 2), rng.range(-3, h as i32 + 2)))
            .collect();
        if !simple(&pts) {
            continue;
        }
        let mask = fill_polygon(w, h, &pts);
        let header = [(h >> 8) as u8, h as u8, (w >> 8) as u8, w as u8, n as u8];
        fnv(&mut digest, &header);
        fnv(&mut digest, &packbits(&mask.data));
        accepted += 1;
    }
    assert_eq!(
        digest, PILLOW_DIGEST,
        "fill_polygon differs from Pillow on some of the {POLYGONS} polygons: {digest:#018X}"
    );
}
