//! The geometry of editing the ROI with the pointer: hit tests, what a drag does, and clamping. No
//! window code; all coordinates are full-size image pixels, and vertices stay whole pixels inside
//! the image. The hit radius is given in image pixels by the caller (a constant number of screen
//! pixels divided by the zoom), so handles keep their size on the screen at any zoom.

/// A vertex in full-size image pixels.
pub type P = (f64, f64);

/// What the pointer can grab.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Handle {
    Vertex(usize),
    /// The midpoint of the edge from vertex `i` to the next one.
    Edge(usize),
}

/// A drag under way.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Drag {
    /// A rectangle from this corner.
    NewRect { origin: P },
    /// A vertex, `grab` being the vertex minus the pointer at the press.
    Vertex { i: usize, grab: P },
    /// An edge, as it was at the press: both ends move together.
    Edge { i: usize, a: P, b: P, press: P },
}

/// The shape of the pointer over a handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cursor {
    Grab,
    ResizeHorizontal,
    ResizeVertical,
    ResizeNeSw,
    ResizeNwSe,
}

fn dist(a: P, b: P) -> f64 {
    (a.0 - b.0).hypot(a.1 - b.1)
}

/// Inside the `w` x `h` image, whole pixels.
pub fn clamp(p: P, (w, h): (usize, usize)) -> P {
    (
        p.0.round().clamp(0.0, w.saturating_sub(1) as f64),
        p.1.round().clamp(0.0, h.saturating_sub(1) as f64),
    )
}

/// How many edges a list of vertices has: a line of two has one, a polygon closes.
pub fn edge_count(roi: &[P]) -> usize {
    match roi.len() {
        0 | 1 => 0,
        2 => 1,
        n => n,
    }
}

/// The ends of edge `i`.
pub fn edge(roi: &[P], i: usize) -> (P, P) {
    (roi[i], roi[(i + 1) % roi.len()])
}

pub fn midpoint(roi: &[P], i: usize) -> P {
    let (a, b) = edge(roi, i);
    ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0)
}

/// The handle within `radius` of `p`. A vertex wins over an edge midpoint, the nearer of two wins.
pub fn hit_test(roi: &[P], p: P, radius: f64) -> Option<Handle> {
    let nearest = |it: &mut dyn Iterator<Item = (usize, P)>| -> Option<usize> {
        it.map(|(i, q)| (i, dist(p, q)))
            .filter(|&(_, d)| d <= radius)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    };
    let mut vertices = roi.iter().copied().enumerate();
    if let Some(i) = nearest(&mut vertices) {
        return Some(Handle::Vertex(i));
    }
    let mut mids = (0..edge_count(roi)).map(|i| (i, midpoint(roi, i)));
    nearest(&mut mids).map(Handle::Edge)
}

/// The edge within `radius` of `p`, measured to the segment, and the nearest.
pub fn near_edge(roi: &[P], p: P, radius: f64) -> Option<usize> {
    (0..edge_count(roi))
        .map(|i| {
            let (a, b) = edge(roi, i);
            let (dx, dy) = (b.0 - a.0, b.1 - a.1);
            let l2 = dx * dx + dy * dy;
            let t = if l2 > 0.0 {
                (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / l2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            (i, dist(p, (a.0 + t * dx, a.1 + t * dy)))
        })
        .filter(|&(_, d)| d <= radius)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
}

/// The drag that a press at `p` begins: on a handle it takes that handle; elsewhere it draws a new
/// rectangle if there is no ROI yet, and does nothing if there is one.
pub fn begin_drag(roi: &[P], p: P, radius: f64) -> Option<Drag> {
    match hit_test(roi, p, radius) {
        Some(Handle::Vertex(i)) => Some(Drag::Vertex {
            i,
            grab: (roi[i].0 - p.0, roi[i].1 - p.1),
        }),
        Some(Handle::Edge(i)) => {
            let (a, b) = edge(roi, i);
            Some(Drag::Edge { i, a, b, press: p })
        }
        None if roi.is_empty() => Some(Drag::NewRect { origin: p }),
        None => None,
    }
}

/// The four corners of the rectangle of `a` and `b`, clockwise on the screen from the top left.
pub fn rect(a: P, b: P, size: (usize, usize)) -> Vec<P> {
    let (a, b) = (clamp(a, size), clamp(b, size));
    let (x0, x1) = (a.0.min(b.0), a.0.max(b.0));
    let (y0, y1) = (a.1.min(b.1), a.1.max(b.1));
    vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
}

/// The edge `(a, b)` moved towards `press -> p` along its normal: the translation of both ends is
/// the same, so the edge stays parallel, and it stops where an end would leave the image.
pub fn moved_edge(a: P, b: P, press: P, p: P, (w, h): (usize, usize)) -> (P, P) {
    let (ex, ey) = (b.0 - a.0, b.1 - a.1);
    let l = ex.hypot(ey);
    let (dx, dy) = (p.0 - press.0, p.1 - press.1);
    // a point-like edge has no direction: it follows the pointer
    let (mx, my) = if l > 0.0 {
        let (nx, ny) = (-ey / l, ex / l);
        let t = dx * nx + dy * ny;
        (t * nx, t * ny)
    } else {
        (dx, dy)
    };
    let (mx, my) = (mx.round(), my.round());
    let (max_x, max_y) = (w.saturating_sub(1) as f64, h.saturating_sub(1) as f64);
    let mx = mx.clamp(-a.0.min(b.0), max_x - a.0.max(b.0));
    let my = my.clamp(-a.1.min(b.1), max_y - a.1.max(b.1));
    ((a.0 + mx, a.1 + my), (b.0 + mx, b.1 + my))
}

/// Applies the drag with the pointer at `p`.
pub fn apply(roi: &mut Vec<P>, drag: Drag, p: P, size: (usize, usize)) {
    match drag {
        Drag::NewRect { origin } => *roi = rect(origin, p, size),
        Drag::Vertex { i, grab } => {
            if i < roi.len() {
                roi[i] = clamp((p.0 + grab.0, p.1 + grab.1), size);
            }
        }
        Drag::Edge { i, a, b, press } => {
            if i < roi.len() {
                let (na, nb) = moved_edge(a, b, press, p, size);
                roi[i] = na;
                let j = (i + 1) % roi.len();
                roi[j] = nb;
            }
        }
    }
}

/// A drag ended. A rectangle without an area is no ROI (the pointer came back to where it began).
pub fn finish(roi: &mut Vec<P>, drag: Drag) {
    if let Drag::NewRect { .. } = drag {
        if roi.len() == 4 && (roi[0].0 == roi[1].0 || roi[0].1 == roi[3].1) {
            roi.clear();
        }
    }
}

/// Puts a vertex on edge `i`, at the point of the edge nearest to `p`.
pub fn insert_on_edge(roi: &mut Vec<P>, i: usize, p: P, size: (usize, usize)) {
    let (a, b) = edge(roi, i);
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let l2 = dx * dx + dy * dy;
    let t = if l2 > 0.0 {
        (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / l2).clamp(0.0, 1.0)
    } else {
        0.5
    };
    roi.insert(i + 1, clamp((a.0 + t * dx, a.1 + t * dy), size));
}

/// Removes vertex `i` if more than 3 remain; whether it did.
pub fn delete_vertex(roi: &mut Vec<P>, i: usize) -> bool {
    if roi.len() > 3 && i < roi.len() {
        roi.remove(i);
        true
    } else {
        false
    }
}

fn cross(o: P, a: P, b: P) -> f64 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

fn segments_cross(a: P, b: P, c: P, d: P) -> bool {
    let (d1, d2) = (cross(c, d, a), cross(c, d, b));
    let (d3, d4) = (cross(a, b, c), cross(a, b, d));
    ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
}

/// Whether two edges that share no vertex cross. Such a polygon is still valid for the tracker
/// (the mask fills it by the even-odd rule); the window only says so.
pub fn self_intersects(roi: &[P]) -> bool {
    let n = roi.len();
    if n < 4 {
        return false;
    }
    (0..n).any(|i| {
        (i + 2..n).any(|j| {
            if (j + 1) % n == i {
                return false;
            }
            let (a, b) = edge(roi, i);
            let (c, d) = edge(roi, j);
            segments_cross(a, b, c, d)
        })
    })
}

/// The pointer over a handle.
pub fn cursor(roi: &[P], h: Handle) -> Cursor {
    match h {
        Handle::Vertex(_) => Cursor::Grab,
        Handle::Edge(i) => {
            let (a, b) = edge(roi, i);
            // the edge moves along its normal; the pointer shows that direction
            let (nx, ny) = (-(b.1 - a.1), b.0 - a.0);
            let angle = ny.atan2(nx).to_degrees().rem_euclid(180.0);
            match angle {
                x if !(22.5..157.5).contains(&x) => Cursor::ResizeHorizontal,
                x if (67.5..112.5).contains(&x) => Cursor::ResizeVertical,
                // y points down on the screen: a normal of 45 degrees runs from top left to bottom right
                x if x < 67.5 => Cursor::ResizeNwSe,
                _ => Cursor::ResizeNeSw,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use optflow_cli::polygon::polygon_mask;

    const SIZE: (usize, usize) = (640, 480);

    fn square() -> Vec<P> {
        vec![
            (100.0, 100.0),
            (300.0, 100.0),
            (300.0, 200.0),
            (100.0, 200.0),
        ]
    }

    #[test]
    fn a_drag_on_nothing_makes_a_rectangle_of_four_vertices() {
        let mut roi = vec![];
        let d = begin_drag(&roi, (50.0, 60.0), 5.0).unwrap();
        assert!(matches!(d, Drag::NewRect { .. }));
        apply(&mut roi, d, (10.0, 200.0), SIZE);
        assert_eq!(
            roi,
            vec![(10.0, 60.0), (50.0, 60.0), (50.0, 200.0), (10.0, 200.0)],
            "any direction"
        );
        apply(&mut roi, d, (900.0, -40.0), SIZE);
        assert_eq!(
            roi,
            vec![(50.0, 0.0), (639.0, 0.0), (639.0, 60.0), (50.0, 60.0)],
            "clamped to the image"
        );
        finish(&mut roi, d);
        assert_eq!(roi.len(), 4);
    }

    #[test]
    fn a_rectangle_without_area_is_no_roi() {
        for end in [(50.0, 60.0), (50.0, 300.0), (200.0, 60.0)] {
            let mut roi = vec![];
            let d = begin_drag(&roi, (50.0, 60.0), 5.0).unwrap();
            apply(&mut roi, d, end, SIZE);
            finish(&mut roi, d);
            assert!(roi.is_empty(), "{end:?}");
        }
    }

    #[test]
    fn with_a_roi_only_a_handle_starts_a_drag() {
        let roi = square();
        // inside, and outside, away from any handle
        assert_eq!(begin_drag(&roi, (150.0, 150.0), 5.0), None);
        assert_eq!(begin_drag(&roi, (500.0, 400.0), 5.0), None);
        assert!(matches!(
            begin_drag(&roi, (102.0, 98.0), 5.0),
            Some(Drag::Vertex { i: 0, .. })
        ));
        assert!(matches!(
            begin_drag(&roi, (200.0, 101.0), 5.0),
            Some(Drag::Edge { i: 0, .. })
        ));
        assert!(matches!(
            begin_drag(&roi, (99.0, 150.0), 5.0),
            Some(Drag::Edge { i: 3, .. })
        ));
    }

    #[test]
    fn the_hit_radius_follows_the_zoom_the_caller_gives() {
        let roi = square();
        // 9 screen pixels at two zooms: 0.1 screen px per image px and 2 screen px per image px
        let (far, near) = (9.0 / 0.1, 9.0 / 2.0);
        assert_eq!(hit_test(&roi, (160.0, 100.0), far), Some(Handle::Vertex(0)));
        assert_eq!(hit_test(&roi, (160.0, 100.0), near), None);
        assert_eq!(
            hit_test(&roi, (104.0, 100.0), near),
            Some(Handle::Vertex(0))
        );
        // a vertex wins over a midpoint within reach of both
        let tiny = vec![(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)];
        assert_eq!(hit_test(&tiny, (1.9, 0.0), 3.0), Some(Handle::Vertex(0)));
        assert_eq!(hit_test(&[], (0.0, 0.0), 3.0), None);
    }

    #[test]
    fn a_corner_moves_freely_and_keeps_where_it_was_grabbed() {
        let mut roi = square();
        let d = begin_drag(&roi, (302.0, 203.0), 5.0).unwrap();
        apply(&mut roi, d, (402.0, 253.0), SIZE);
        assert_eq!(roi[2], (400.0, 250.0));
        assert_eq!(roi[1], (300.0, 100.0), "the others stay");
        apply(&mut roi, d, (-100.0, 900.0), SIZE);
        assert_eq!(roi[2], (0.0, 479.0), "clamped");
    }

    #[test]
    fn an_edge_handle_moves_the_edge_in_parallel() {
        // the top edge, the pointer also going sideways: only the normal part counts
        let mut roi = square();
        let d = begin_drag(&roi, (200.0, 100.0), 5.0).unwrap();
        apply(&mut roi, d, (260.0, 40.0), SIZE);
        assert_eq!(&roi[..2], &[(100.0, 40.0), (300.0, 40.0)]);
        assert_eq!(&roi[2..], &square()[2..], "the others stay");
        // the closing edge, from vertex 3 to vertex 0
        let mut roi = square();
        let d = begin_drag(&roi, (100.0, 150.0), 5.0).unwrap();
        apply(&mut roi, d, (70.0, 400.0), SIZE);
        assert_eq!((roi[3], roi[0]), ((70.0, 200.0), (70.0, 100.0)));
        // a slanted edge stays parallel, also where it hits the border
        let tri = vec![(100.0, 100.0), (500.0, 300.0), (120.0, 400.0)];
        let (a, b) = edge(&tri, 0);
        let mid = midpoint(&tri, 0);
        for p in [(320.0, 180.0), (900.0, -900.0), (0.0, 900.0)] {
            let (na, nb) = moved_edge(a, b, mid, p, SIZE);
            assert_eq!((nb.0 - na.0, nb.1 - na.1), (b.0 - a.0, b.1 - a.1), "{p:?}");
            for q in [na, nb] {
                assert!((0.0..=639.0).contains(&q.0) && (0.0..=479.0).contains(&q.1));
            }
        }
    }

    #[test]
    fn near_the_border_an_edge_stops_with_its_shape_kept() {
        let mut roi = square();
        let d = begin_drag(&roi, (200.0, 100.0), 5.0).unwrap();
        apply(&mut roi, d, (200.0, -500.0), SIZE);
        assert_eq!(&roi[..2], &[(100.0, 0.0), (300.0, 0.0)]);
    }

    #[test]
    fn vertices_are_inserted_on_the_edge_and_deleted_above_three() {
        let mut roi = square();
        assert_eq!(near_edge(&roi, (250.0, 103.0), 5.0), Some(0));
        assert_eq!(near_edge(&roi, (250.0, 150.0), 5.0), None);
        insert_on_edge(&mut roi, 0, (250.0, 103.0), SIZE);
        assert_eq!(roi.len(), 5);
        assert_eq!(roi[1], (250.0, 100.0));
        // on the closing edge it goes last
        let mut roi = square();
        insert_on_edge(&mut roi, 3, (100.0, 120.0), SIZE);
        assert_eq!(roi[4], (100.0, 120.0));
        assert!(delete_vertex(&mut roi, 4));
        assert!(delete_vertex(&mut roi, 0));
        assert_eq!(roi.len(), 3);
        assert!(!delete_vertex(&mut roi, 0), "three vertices remain");
        assert!(!delete_vertex(&mut square(), 9));
    }

    #[test]
    fn crossing_edges_are_found_and_shared_vertices_do_not_count() {
        assert!(!self_intersects(&square()));
        assert!(!self_intersects(&square()[..3]));
        let bow = vec![(0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0)];
        assert!(self_intersects(&bow));
        // a notch, as in the real data
        let notch = vec![
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (60.0, 100.0),
            (50.0, 40.0),
            (40.0, 100.0),
            (0.0, 100.0),
        ];
        assert!(!self_intersects(&notch));
    }

    /// The core fills a self-intersecting polygon by the even-odd rule, without failing: nothing
    /// to refuse, a note in the window is enough.
    #[test]
    fn the_mask_of_a_crossed_polygon_is_filled_even_odd() {
        let bow = vec![(10.0, 10.0), (90.0, 90.0), (90.0, 10.0), (10.0, 90.0)];
        let m = polygon_mask(100, 100, &bow, 1);
        let at = |x: usize, y: usize| m.data[y * 100 + x];
        assert!(at(80, 50), "the right lobe");
        assert!(at(20, 50), "the left lobe");
        assert!(!at(50, 20), "the top is outside");
        assert!(!at(50, 80), "the bottom is outside");
    }

    #[test]
    fn the_pointer_shows_the_direction_of_the_move() {
        let roi = square();
        assert_eq!(cursor(&roi, Handle::Vertex(1)), Cursor::Grab);
        assert_eq!(cursor(&roi, Handle::Edge(0)), Cursor::ResizeVertical);
        assert_eq!(cursor(&roi, Handle::Edge(1)), Cursor::ResizeHorizontal);
        assert_eq!(cursor(&roi, Handle::Edge(2)), Cursor::ResizeVertical);
        assert_eq!(cursor(&roi, Handle::Edge(3)), Cursor::ResizeHorizontal);
        // an edge down and to the right has a normal up and to the right
        let slant = vec![(0.0, 0.0), (10.0, 10.0), (0.0, 20.0)];
        assert_eq!(cursor(&slant, Handle::Edge(0)), Cursor::ResizeNeSw);
    }
}
