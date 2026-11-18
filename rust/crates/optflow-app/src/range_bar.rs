//! The logic of the bar with two handles of "2. 範囲と領域": where a frame lies on the bar, which
//! handle a click takes, and what a handle may do. No window code (the widget is in `main.rs`).
//!
//! The bar spans frame 0 to `count - 1`. The left handle is the start frame; the right handle is
//! the end frame shown inclusive, that is the stored exclusive end minus 1 (`stop = end + 1`; a
//! stored 0 is "to the last frame"). Moving a handle never crosses the other one: the end stays at
//! least one `step` after the start, so two frames are still computed.

/// Which handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Handle {
    Start,
    End,
}

/// The bar can be used: a source of one frame, or of unknown length, has nothing to slide.
pub fn usable(count: Option<usize>) -> bool {
    count.is_some_and(|n| n >= 2)
}

/// The start frame as drawn: inside the bar, whatever the number field holds.
pub fn start_shown(start: i64, count: usize) -> i64 {
    start.clamp(0, count.max(1) as i64 - 1)
}

/// The end frame (inclusive) as drawn: the stored end minus 1, 0 meaning the last frame, and
/// inside the bar.
pub fn end_shown(stop: i64, count: usize) -> i64 {
    let last = count.max(1) as i64 - 1;
    if stop <= 0 {
        last
    } else {
        (stop - 1).clamp(0, last)
    }
}

/// The exclusive end to store for an end frame shown inclusive.
pub fn stop_from_end(end: i64) -> i64 {
    end.max(0) + 1
}

/// The frame under horizontal position `x` of a bar that runs from `left` over `width`
/// (frame 0 at `left`, the last frame at `left + width`), the nearest frame, inside the bar.
pub fn frame_at(x: f32, left: f32, width: f32, count: usize) -> i64 {
    let last = count.max(1) as i64 - 1;
    if width <= 0.0 || last == 0 {
        return 0;
    }
    let f = ((x - left) / width * last as f32).round();
    (f as i64).clamp(0, last)
}

/// The horizontal position of a frame on the same bar.
pub fn x_of(frame: i64, left: f32, width: f32, count: usize) -> f32 {
    let last = count.max(1) as i64 - 1;
    if last == 0 {
        return left;
    }
    left + width * frame.clamp(0, last) as f32 / last as f32
}

/// The handle a click on the track at `frame` takes: the nearer one; at equal distance the one on
/// the side of the click, the start when the click is exactly between them.
pub fn nearer(frame: i64, start: i64, end: i64) -> Handle {
    let (ds, de) = ((frame - start).abs(), (frame - end).abs());
    if ds < de || (ds == de && frame <= start) {
        Handle::Start
    } else if de < ds {
        Handle::End
    } else if frame < end {
        Handle::Start
    } else {
        Handle::End
    }
}

/// The start frame when its handle is moved to `frame`: at most `step` before the end, never
/// below 0 (and never above the end when the bar is too short for the step).
pub fn move_start(frame: i64, end: i64, step: i64) -> i64 {
    frame.clamp(0, (end - step.max(1)).max(0))
}

/// The end frame (inclusive) when its handle is moved to `frame`: at least `step` after the start,
/// at most the last frame (the start plus a step if the bar is too short).
pub fn move_end(frame: i64, start: i64, step: i64, count: usize) -> i64 {
    let last = count.max(1) as i64 - 1;
    frame.clamp((start + step.max(1)).min(last), last)
}

/// Moves `handle` to `frame`: the new `(start, end)` frames, end inclusive.
pub fn moved(
    handle: Handle,
    frame: i64,
    start: i64,
    end: i64,
    step: i64,
    count: usize,
) -> (i64, i64) {
    match handle {
        Handle::Start => (move_start(frame, end, step), end),
        Handle::End => (start, move_end(frame, start, step, count)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn x_and_frame_map_both_ways_across_the_track() {
        // 101 frames over 200 px starting at x = 10
        assert_eq!(frame_at(10.0, 10.0, 200.0, 101), 0);
        assert_eq!(frame_at(210.0, 10.0, 200.0, 101), 100);
        assert_eq!(frame_at(110.0, 10.0, 200.0, 101), 50);
        // outside the track: the first and the last frame
        assert_eq!(frame_at(-50.0, 10.0, 200.0, 101), 0);
        assert_eq!(frame_at(900.0, 10.0, 200.0, 101), 100);
        // the nearest frame (2 px per frame)
        assert_eq!(frame_at(13.1, 10.0, 200.0, 101), 2);
        assert_eq!(frame_at(12.9, 10.0, 200.0, 101), 1);
        for f in [0, 1, 37, 99, 100] {
            assert_eq!(frame_at(x_of(f, 10.0, 200.0, 101), 10.0, 200.0, 101), f);
        }
        // a long video: several frames per pixel, still the whole range
        assert_eq!(frame_at(300.0, 0.0, 300.0, 6143), 6142);
        assert_eq!(x_of(6142, 0.0, 300.0, 6143), 300.0);
        // a frame beyond the bar is drawn at its end
        assert_eq!(x_of(500, 10.0, 200.0, 101), 210.0);
        assert_eq!(x_of(-5, 10.0, 200.0, 101), 10.0);
    }

    #[test]
    fn the_end_is_shown_inclusive_and_stored_exclusive() {
        // stop 41 (exclusive) is frame 40 on the bar and comes back as 41
        assert_eq!(end_shown(41, 100), 40);
        assert_eq!(stop_from_end(40), 41);
        // 0 is "to the last frame"
        assert_eq!(end_shown(0, 100), 99);
        assert_eq!(stop_from_end(99), 100);
        // a number field beyond the source is drawn at the bar's end, start too
        assert_eq!(end_shown(250, 100), 99);
        assert_eq!(start_shown(250, 100), 99);
        assert_eq!(start_shown(-3, 100), 0);
    }

    #[test]
    fn a_click_on_the_track_takes_the_nearer_handle() {
        assert_eq!(nearer(10, 20, 80), Handle::Start);
        assert_eq!(nearer(70, 20, 80), Handle::End);
        assert_eq!(nearer(0, 20, 80), Handle::Start);
        assert_eq!(nearer(99, 20, 80), Handle::End);
        // exactly between: the start
        assert_eq!(nearer(50, 20, 80), Handle::Start);
        // on a handle
        assert_eq!(nearer(80, 20, 80), Handle::End);
        assert_eq!(nearer(20, 20, 80), Handle::Start);
        // both on the same frame (a bar too short for a gap): the side of the click decides
        assert_eq!(nearer(3, 5, 5), Handle::Start);
        assert_eq!(nearer(9, 5, 5), Handle::End);
    }

    #[test]
    fn handles_cannot_cross_and_keep_two_frames() {
        // step 1: the start stops one before the end, the end one after the start
        assert_eq!(move_start(99, 50, 1), 49);
        assert_eq!(move_start(-7, 50, 1), 0);
        assert_eq!(move_start(30, 50, 1), 30);
        assert_eq!(move_end(0, 20, 1, 100), 21);
        assert_eq!(move_end(150, 20, 1, 100), 99);
        assert_eq!(move_end(60, 20, 1, 100), 60);
        // step 5: end >= start + 5 so that two frames are computed
        assert_eq!(move_start(48, 50, 5), 45);
        assert_eq!(move_end(22, 20, 5, 100), 25);
        // moved() leaves the other handle alone
        assert_eq!(moved(Handle::Start, 70, 10, 50, 1, 100), (49, 50));
        assert_eq!(moved(Handle::End, 5, 10, 50, 1, 100), (10, 11));
        assert_eq!(moved(Handle::End, 70, 10, 50, 1, 100), (10, 70));
        // the result always leaves two computed frames
        for (h, f) in [
            (Handle::Start, 0),
            (Handle::Start, 99),
            (Handle::End, 0),
            (Handle::End, 99),
        ] {
            let (a, b) = moved(h, f, 30, 60, 3, 100);
            assert!(b - a >= 3, "{h:?} {f}: {a} {b}");
        }
    }

    #[test]
    fn a_short_source_is_clamped_and_one_frame_has_no_bar() {
        assert!(usable(Some(2)));
        assert!(!usable(Some(1)));
        assert!(!usable(Some(0)));
        assert!(!usable(None));
        // two frames: the start can only be 0, the end only 1
        assert_eq!(move_start(1, 1, 1), 0);
        assert_eq!(move_end(0, 0, 1, 2), 1);
        assert_eq!(moved(Handle::Start, 1, 0, 1, 1, 2), (0, 1));
        assert_eq!(moved(Handle::End, 0, 0, 1, 1, 2), (0, 1));
        // a step longer than the source cannot be satisfied: the handles stay inside the bar
        assert_eq!(move_start(9, 3, 50), 0);
        assert_eq!(move_end(0, 0, 50, 4), 3);
        // frame 0 of a one-frame source
        assert_eq!(frame_at(80.0, 0.0, 100.0, 1), 0);
        assert_eq!(x_of(0, 5.0, 100.0, 1), 5.0);
        assert_eq!(end_shown(0, 1), 0);
    }
}
