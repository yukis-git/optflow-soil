//! The frames a run computes: the start, end and interval of "2. 範囲と領域", the line that says
//! which frames they are, and the checks that refuse a range that cannot run.
//!
//! The end is exclusive, as in the command line (`--stop`): 0 is the end of the source.

/// The source frames of the range, if they can be told: `count` frames in the source (unknown for a
/// video whose length ffprobe did not give; then only a range with an end can be listed).
pub fn picked(count: Option<usize>, start: i64, stop: i64, step: i64) -> Option<Vec<usize>> {
    let end = if stop > 0 {
        Some(stop)
    } else {
        count.map(|n| n as i64)
    }?;
    let len = count.map_or(end.max(0) as usize, |n| n.min(end.max(0) as usize));
    let stop = (stop > 0).then_some(stop);
    Some(optflow_cli::io::slice_indices(
        len,
        start.max(0),
        stop,
        step.max(1),
    ))
}

/// Why the range cannot run, in Japanese; `Ok` when it can or when the source is too unknown to
/// tell (the run itself then reports what it finds).
pub fn check(count: Option<usize>, start: i64, stop: i64, step: i64) -> Result<(), String> {
    if stop > 0 && stop <= start {
        return Err("終了フレームは開始フレームより後にしてください。".into());
    }
    if let Some(n) = count {
        if start >= n as i64 {
            return Err(format!(
                "開始フレームが最後のフレーム({})を超えています。",
                n.saturating_sub(1)
            ));
        }
        if stop > n as i64 {
            return Err(format!(
                "終了フレームが最後のフレームの次({n})を超えています。"
            ));
        }
    }
    match picked(count, start, stop, step) {
        Some(p) if p.len() < 2 => Err(
            "計算するフレームが 2 枚以上になるように、開始・終了・間隔を選んでください。".into(),
        ),
        _ => Ok(()),
    }
}

/// `1.50` seconds of source frame `n` at `fps` frames per second.
fn secs(n: i64, fps: f64) -> String {
    format!("{:.2}", n as f64 / fps)
}

/// The line under the range fields: 「計算するフレーム: N 枚(a 〜 b、間隔 s、t1 〜 t2 秒)」, `a`
/// and `b` being the first and the last source frame computed. A range that cannot run gives the
/// reason instead.
pub fn text(count: Option<usize>, start: i64, stop: i64, step: i64, fps: f64) -> String {
    let (start, step) = (start.max(0), step.max(1));
    let fps = if fps > 0.0 { fps } else { 1.0 };
    if let Err(why) = check(count, start, stop, step) {
        return why;
    }
    match picked(count, start, stop, step) {
        Some(p) if p.len() >= 2 => {
            let (a, b) = (p[0] as i64, p[p.len() - 1] as i64);
            format!(
                "計算するフレーム: {} 枚({a} 〜 {b}、間隔 {step}、{} 〜 {} 秒)",
                p.len(),
                secs(a, fps),
                secs(b, fps)
            )
        }
        _ => format!(
            "計算するフレーム: {start} 〜 最後まで(間隔 {step}、{} 秒から)",
            secs(start, fps)
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_line_names_the_frames_and_the_times() {
        // 100 frames at 25 fps, all of them
        assert_eq!(
            text(Some(100), 0, 0, 1, 25.0),
            "計算するフレーム: 100 枚(0 〜 99、間隔 1、0.00 〜 3.96 秒)"
        );
        // every 5th frame from 10 up to, not including, 51: 10, 15, ..., 50
        assert_eq!(
            text(Some(100), 10, 51, 5, 25.0),
            "計算するフレーム: 9 枚(10 〜 50、間隔 5、0.40 〜 2.00 秒)"
        );
        // the last frame computed is not the end number when the step does not meet it
        assert_eq!(
            text(Some(100), 0, 10, 4, 1.0),
            "計算するフレーム: 3 枚(0 〜 8、間隔 4、0.00 〜 8.00 秒)"
        );
        // a video of unknown length, with and without an end
        assert_eq!(
            text(None, 3, 8, 1, 2.0),
            "計算するフレーム: 5 枚(3 〜 7、間隔 1、1.50 〜 3.50 秒)"
        );
        assert_eq!(
            text(None, 3, 0, 2, 2.0),
            "計算するフレーム: 3 〜 最後まで(間隔 2、1.50 秒から)"
        );
        // a frame rate that is not a number to divide by counts as 1
        assert!(text(Some(10), 0, 0, 1, 0.0).contains("0.00 〜 9.00 秒"));
    }

    #[test]
    fn a_range_that_cannot_run_is_refused_with_the_reason() {
        assert!(check(Some(10), 0, 0, 1).is_ok());
        assert!(check(Some(10), 0, 10, 1).is_ok());
        assert!(check(None, 0, 0, 1).is_ok());
        let why = |c, a, b, s| check(c, a, b, s).unwrap_err();
        assert!(why(Some(10), 5, 5, 1).contains("終了フレームは開始フレームより後"));
        assert!(why(Some(10), 6, 3, 1).contains("終了フレームは開始フレームより後"));
        assert!(why(Some(10), 0, 11, 1).contains("最後のフレームの次(10)"));
        assert!(why(Some(10), 10, 0, 1).contains("最後のフレーム(9)"));
        assert!(why(None, 4, 4, 1).contains("終了フレーム"));
        // one frame is no sequence
        assert!(why(Some(10), 9, 0, 1).contains("2 枚以上"));
        assert!(why(Some(10), 0, 9, 9).contains("2 枚以上"));
        assert!(why(None, 2, 3, 1).contains("2 枚以上"));
        // the line gives the reason in place of the frames
        assert_eq!(text(Some(10), 5, 5, 1, 1.0), why(Some(10), 5, 5, 1));
    }

    #[test]
    fn the_bar_handles_give_the_range_the_line_names() {
        use crate::range_bar::{end_shown, moved, stop_from_end, Handle};
        // start at 12, end handle on frame 40 (stored 41), step 4
        let (start, end) = moved(Handle::End, 40, 12, 99, 4, 100);
        let stop = stop_from_end(end);
        assert_eq!((start, stop), (12, 41));
        assert_eq!(end_shown(stop, 100), 40);
        assert_eq!(picked(Some(100), start, stop, 4).unwrap().last(), Some(&40));
        assert_eq!(
            text(Some(100), start, stop, 4, 10.0),
            "計算するフレーム: 8 枚(12 〜 40、間隔 4、1.20 〜 4.00 秒)"
        );
        // a start handle dragged to the right edge stops short of the end, so the range can run
        let (start, _) = moved(Handle::Start, 99, start, end, 4, 100);
        assert_eq!(start, 36);
        assert!(check(Some(100), start, stop, 4).is_ok());
    }
}
