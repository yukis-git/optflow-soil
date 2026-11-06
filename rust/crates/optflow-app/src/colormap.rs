//! Colour scales for the overlay.

/// Stops of a perceptually ordered scale (viridis, sampled).
const VIRIDIS: [[u8; 3]; 9] = [
    [68, 1, 84],
    [71, 44, 122],
    [59, 81, 139],
    [44, 113, 142],
    [33, 144, 141],
    [39, 173, 129],
    [92, 200, 99],
    [170, 220, 50],
    [253, 231, 37],
];

/// Stops of a diverging scale (blue, white, red) for signed quantities.
const DIVERGING: [[u8; 3]; 5] = [
    [33, 102, 172],
    [146, 197, 222],
    [247, 247, 247],
    [244, 165, 130],
    [178, 24, 43],
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scale {
    Sequential,
    Diverging,
}

fn lerp_stops(stops: &[[u8; 3]], t: f32) -> [u8; 3] {
    let t = if t.is_nan() { 0.0 } else { t.clamp(0.0, 1.0) };
    let x = t * (stops.len() - 1) as f32;
    let i = (x.floor() as usize).min(stops.len() - 2);
    let f = x - i as f32;
    let mut out = [0u8; 3];
    for c in 0..3 {
        out[c] = (stops[i][c] as f32 * (1.0 - f) + stops[i + 1][c] as f32 * f).round() as u8;
    }
    out
}

/// The colour of `v` on `scale` for the range `lo..=hi` (clamped).
pub fn color(scale: Scale, v: f32, lo: f32, hi: f32) -> [u8; 3] {
    let t = if hi > lo { (v - lo) / (hi - lo) } else { 0.5 };
    match scale {
        Scale::Sequential => lerp_stops(&VIRIDIS, t),
        Scale::Diverging => lerp_stops(&DIVERGING, t),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ends_and_middle() {
        assert_eq!(color(Scale::Sequential, 0.0, 0.0, 1.0), VIRIDIS[0]);
        assert_eq!(color(Scale::Sequential, 9.0, 0.0, 1.0), VIRIDIS[8]);
        assert_eq!(color(Scale::Diverging, 0.0, -1.0, 1.0), DIVERGING[2]);
        assert_eq!(color(Scale::Diverging, f32::NAN, -1.0, 1.0), DIVERGING[0]);
    }

    #[test]
    fn degenerate_range_does_not_panic() {
        let _ = color(Scale::Sequential, 1.0, 1.0, 1.0);
    }
}
