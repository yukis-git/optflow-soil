//! Video input through `ffmpeg` / `ffprobe` (no linked decoder).
//!
//! Python reads video with OpenCV: BGR24 frames converted to RGB and then to grey with the
//! luminance formula. The same input is made here by asking ffmpeg for `rgb24` frames and applying
//! the same formula, so a video and the same frames saved as images agree with the Python tool.
//!
//! A stream with a display rotation (a phone video) is turned upright as OpenCV does it: ffprobe
//! gives the rotation, ffmpeg is told `-noautorotate`, and a `transpose` / `hflip,vflip` filter
//! makes a quarter or a half turn. The frame size read from the pipe and the filter come from this
//! one decision. Any other angle is ignored, as OpenCV ignores it (ffmpeg's own autorotate would
//! rotate the picture by it).
//!
//! Frames are passed through with `-fps_mode passthrough`, which exists from FFmpeg 5.1; older
//! versions (recognised through `ffmpeg -version`) get the equivalent `-vsync passthrough`. A build
//! not named by a release number (a development build, named by a revision or a date) is taken to
//! be current; a date-named build from before FFmpeg 5.1 would need `-vsync`.

use std::io::Read;
use std::path::Path;
use std::process::{Child, ChildStdout, Command, Stdio};
use std::thread::JoinHandle;

use anyhow::{anyhow, bail, Result};

use crate::io::{check_range, luminance, lut8, Gray};

/// The program `tool` (`ffmpeg`, `ffprobe`) as a command: the copy next to the executable if there
/// is one (a Windows install that ships `ffmpeg.exe` beside the app), else the one on `PATH`.
/// On Windows the child gets no console window.
pub fn tool_command(tool: &str) -> Command {
    #[cfg_attr(not(windows), allow(unused_mut))]
    let mut cmd = Command::new(tool_path(tool));
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

/// Path of `tool` next to the running executable (with `.exe` on Windows) if it exists there,
/// else the bare name, which the OS looks up on `PATH`.
pub fn tool_path(tool: &str) -> std::path::PathBuf {
    let file = if cfg!(windows) {
        format!("{tool}.exe")
    } else {
        tool.to_string()
    };
    if let Some(beside) = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(|d| d.join(&file)))
    {
        if beside.is_file() {
            return beside;
        }
    }
    std::path::PathBuf::from(tool)
}

/// Whether both `ffmpeg` and `ffprobe` can be started.
pub fn tools_available() -> bool {
    ["ffmpeg", "ffprobe"].iter().all(|t| {
        tool_command(t)
            .arg("-version")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    })
}

/// Size, frame rate and (if the container says) frame count of a video, as [`iter_video`]
/// delivers its frames.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VideoMeta {
    pub w: usize,
    pub h: usize,
    pub fps: f64,
    pub frames: Option<usize>,
}

/// Frame rate from ffprobe's `r_frame_rate` (`30000/1001`).
fn parse_rate(v: &str) -> Option<f64> {
    let (n, d) = v.trim().split_once('/')?;
    let (n, d) = (n.parse::<f64>().ok()?, d.parse::<f64>().ok()?);
    (d > 0.0 && n > 0.0).then(|| n / d)
}

/// [`VideoMeta`] of the first video stream of `path`.
pub fn probe_meta(path: &Path) -> Result<VideoMeta> {
    let (w, h) = probe_size(path)?;
    let out = tool_command("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate,nb_frames",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(path)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| spawn_error("ffprobe", &e))?;
    let (mut fps, mut frames) = (None, None);
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        match line.split_once('=') {
            Some(("r_frame_rate", v)) => fps = parse_rate(v),
            Some(("nb_frames", v)) => frames = v.trim().parse::<usize>().ok(),
            _ => {}
        }
    }
    Ok(VideoMeta {
        w,
        h,
        fps: fps.unwrap_or(30.0),
        frames,
    })
}

fn spawn_error(tool: &str, e: &std::io::Error) -> anyhow::Error {
    if e.kind() == std::io::ErrorKind::NotFound {
        anyhow!("{tool} not found on PATH (needed for video input)")
    } else {
        anyhow!("cannot run {tool}: {e}")
    }
}

/// How the frames of a stream with a display rotation are turned upright. ffmpeg would do this by
/// itself (`autorotate`), but the frame size we read has to come from the very same decision, so
/// ffmpeg is told not to ([`iter_video`] passes `-noautorotate`) and the turn is asked for by a
/// filter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Turn {
    None,
    /// A quarter turn clockwise.
    Clockwise,
    /// Half a turn.
    Half,
    /// A quarter turn counterclockwise.
    CounterClockwise,
}

impl Turn {
    /// The turn for the `rotation` of ffprobe's display matrix: the angle in degrees by which the
    /// display turns the frame counterclockwise, so -90 (a phone held upright) is a quarter turn
    /// clockwise. The angle is rounded to whole degrees, as ffmpeg and OpenCV do; only the quarter
    /// turns are applied. OpenCV, the reference, leaves a stream with any other angle as it is.
    fn from_rotation(rotation: f64) -> Turn {
        // `as` saturates and maps NaN to 0: neither is a quarter turn
        match (-rotation).rem_euclid(360.0).round() as u32 {
            90 => Turn::Clockwise,
            180 => Turn::Half,
            270 => Turn::CounterClockwise,
            _ => Turn::None,
        }
    }

    /// The ffmpeg filter that makes the turn.
    fn filter(self) -> Option<&'static str> {
        match self {
            Turn::None => None,
            Turn::Clockwise => Some("transpose=clock"),
            Turn::Half => Some("hflip,vflip"),
            Turn::CounterClockwise => Some("transpose=cclock"),
        }
    }

    /// Whether width and height trade places.
    fn swaps_sides(self) -> bool {
        matches!(self, Turn::Clockwise | Turn::CounterClockwise)
    }
}

/// The `-vf` argument: the turn (first, where ffmpeg's own autorotate would put it) and then the
/// conversion to RGB. BT.601 whatever the stream is tagged with, nearest chroma and accurate
/// rounding: the settings (found by comparison) that give OpenCV's BGR frames bit for bit for 8-bit
/// 4:4:4 and 4:2:0 video and for even-sized MJPEG; other pixel formats differ by at most about 3
/// grey levels. format=rgb24 makes this filter do the YUV to RGB step.
fn video_filter(turn: Turn) -> String {
    let convert = "scale=in_color_matrix=bt601:flags=neighbor+accurate_rnd,format=rgb24";
    match turn.filter() {
        Some(f) => format!("{f},{convert}"),
        None => convert.to_string(),
    }
}

/// The width and height of the first video stream as stored, and the turn that makes it upright,
/// from the text `ffprobe` printed. `None` without a size.
fn parse_probe(text: &str) -> Option<(usize, usize, Turn)> {
    let (mut w, mut h, mut rotation) = (None, None, 0.0f64);
    for line in text.lines() {
        match line.split_once('=') {
            Some(("width", v)) => w = v.trim().parse::<usize>().ok(),
            Some(("height", v)) => h = v.trim().parse::<usize>().ok(),
            Some(("rotation", v)) => rotation = v.trim().parse().unwrap_or(0.0),
            _ => {}
        }
    }
    match (w, h) {
        (Some(w), Some(h)) if w > 0 && h > 0 => Some((w, h, Turn::from_rotation(rotation))),
        _ => None,
    }
}

/// Width and height of the first video stream as stored, and the turn that makes it upright.
fn probe(path: &Path) -> Result<(usize, usize, Turn)> {
    let out = tool_command("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:stream_side_data=rotation",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(path)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| spawn_error("ffprobe", &e))?;
    if !out.status.success() {
        bail!("cannot open video {}", path.display());
    }
    parse_probe(&String::from_utf8_lossy(&out.stdout))
        .ok_or_else(|| anyhow!("no video stream in {}", path.display()))
}

/// Width and height of the first video stream as [`iter_video`] delivers its frames: turned
/// upright, so the sides of a stream with a quarter turn display rotation swap.
pub fn probe_size(path: &Path) -> Result<(usize, usize)> {
    let (w, h, turn) = probe(path)?;
    Ok(if turn.swaps_sides() { (h, w) } else { (w, h) })
}

/// The largest major version taken for a release number. FFmpeg's are single digits; a longer
/// number is a date (`2023-08-14-git-...`, `20240101-...`).
const MAX_RELEASE_MAJOR: u32 = 99;

/// The `(major, minor)` of the first line of `ffmpeg -version` (`ffmpeg version 6.1.1 Copyright
/// ...`, also `n7.0` and `4.4.2-0ubuntu0.22.04.1`); `None` for a build that is not named by a
/// release: a development build is named by a revision (`N-109044-g1b2c3d4`) or by its date.
fn parse_ffmpeg_version(text: &str) -> Option<(u32, u32)> {
    let line = text.lines().next()?;
    let word = line
        .strip_prefix("ffmpeg version ")?
        .split_whitespace()
        .next()?;
    let word = word.strip_prefix(['n', 'N']).unwrap_or(word);
    let number_len = |t: &str| t.find(|c: char| !c.is_ascii_digit()).unwrap_or(t.len());
    let major_len = number_len(word);
    let major = word[..major_len]
        .parse()
        .ok()
        .filter(|&m| m <= MAX_RELEASE_MAJOR)?;
    // the minor version is the number after a dot ("5.1", "5.1.2"); "4-rc" has none
    let minor = word[major_len..]
        .strip_prefix('.')
        .map(|r| &r[..number_len(r)])
        .and_then(|m| m.parse().ok())
        .unwrap_or(0);
    Some((major, minor))
}

/// The option that passes every frame through unchanged: `-fps_mode` replaced `-vsync` in FFmpeg
/// 5.1. An unknown version (a development build) gets the current one.
fn passthrough_option(version: Option<(u32, u32)>) -> [&'static str; 2] {
    match version {
        Some(v) if v < (5, 1) => ["-vsync", "passthrough"],
        _ => ["-fps_mode", "passthrough"],
    }
}

/// Runs `ffmpeg -version`, which reports a missing ffmpeg by name, and returns its version.
fn ffmpeg_version() -> Result<Option<(u32, u32)>> {
    let out = tool_command("ffmpeg")
        .arg("-version")
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .map_err(|e| spawn_error("ffmpeg", &e))?;
    Ok(parse_ffmpeg_version(&String::from_utf8_lossy(&out.stdout)))
}

/// The frames `start, start + step, ...` below `stop` of a video, as grey levels. The iterator
/// ends after an error item. Dropping it stops ffmpeg.
pub struct VideoFrames {
    child: Child,
    /// `None` once ffmpeg has been shut down.
    stdout: Option<ChildStdout>,
    stderr: Option<JoinHandle<String>>,
    w: usize,
    h: usize,
    buf: Vec<u8>,
    lut: [f64; 256],
    idx: i64,
    start: i64,
    stop: Option<i64>,
    step: i64,
    finished: bool,
}

/// Opens `path` for reading frames; the range rules are those of `iter_video`.
pub fn iter_video(path: &Path, start: i64, stop: Option<i64>, step: i64) -> Result<VideoFrames> {
    check_range(start, step)?;
    let version = ffmpeg_version()?;
    let (w, h, turn) = probe(path)?;
    let (w, h) = if turn.swaps_sides() { (h, w) } else { (w, h) };
    let mut child = tool_command("ffmpeg")
        // the turn is made by the filter below, from the same probe that gave the frame size
        .args(["-v", "error", "-nostdin", "-noautorotate", "-i"])
        .arg(path)
        .args(["-map", "0:v:0", "-an", "-sn", "-dn", "-vf"])
        .arg(video_filter(turn))
        .args(passthrough_option(version))
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| spawn_error("ffmpeg", &e))?;
    let stdout = child.stdout.take();
    let mut stderr = child.stderr.take().expect("stderr was piped");
    // drained on a thread so that a chatty ffmpeg can never block on a full pipe
    let stderr = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = stderr.read_to_end(&mut bytes);
        String::from_utf8_lossy(&bytes).into_owned()
    });
    Ok(VideoFrames {
        child,
        stdout,
        stderr: Some(stderr),
        w,
        h,
        buf: vec![0; w * h * 3],
        lut: lut8(),
        idx: 0,
        start,
        stop,
        step,
        finished: false,
    })
}

impl VideoFrames {
    /// Stops ffmpeg (if it still runs) and reaps it.
    fn shut_down(&mut self) {
        let _ = self.child.kill();
        // close our end of the output before anything waits for ffmpeg or for its stderr
        self.stdout = None;
        let _ = self.child.wait();
        if let Some(t) = self.stderr.take() {
            let _ = t.join();
        }
    }

    /// Reads the next raw frame into `buf`: `Some(Ok(()))`, or `None` at the end of the stream,
    /// or an error (ffmpeg failed).
    fn read_raw(&mut self) -> Option<Result<()>> {
        let stdout = self.stdout.as_mut()?; // None: shut down already
        match stdout.read_exact(&mut self.buf) {
            Ok(()) => Some(Ok(())),
            // the end of the stream (a trailing partial frame is dropped, as a decoder would)
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => self.end_of_stream(),
            Err(e) => {
                self.shut_down();
                Some(Err(anyhow!("cannot read frames from ffmpeg: {e}")))
            }
        }
    }

    /// The raw frame in `buf` as grey levels.
    fn gray(&self) -> Gray {
        let lut = &self.lut;
        let data = self
            .buf
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| luminance(lut[p[0] as usize], lut[p[1] as usize], lut[p[2] as usize]))
            .collect();
        Gray {
            w: self.w,
            h: self.h,
            data,
        }
    }

    /// ffmpeg closed its output: an error if it failed, otherwise the iteration is over.
    fn end_of_stream(&mut self) -> Option<Result<()>> {
        // the output is complete, so waiting does not hang
        self.stdout = None;
        let stderr = self.stderr.take();
        let status = self.child.wait().ok();
        let text = stderr
            .map(|t| t.join().unwrap_or_default())
            .unwrap_or_default();
        if status.is_some_and(|s| s.success()) {
            return None;
        }
        let detail = text.trim();
        Some(Err(anyhow!(
            "ffmpeg failed to decode the video{}",
            if detail.is_empty() {
                String::new()
            } else {
                format!(": {detail}")
            }
        )))
    }
}

impl Iterator for VideoFrames {
    type Item = Result<Gray>;

    fn next(&mut self) -> Option<Self::Item> {
        while !self.finished {
            if self.stop.is_some_and(|s| self.idx >= s) {
                self.finished = true;
                self.shut_down();
                return None;
            }
            match self.read_raw() {
                Some(Ok(())) => {}
                Some(Err(e)) => {
                    self.finished = true;
                    return Some(Err(e));
                }
                None => {
                    self.finished = true;
                    return None;
                }
            }
            let k = self.idx;
            self.idx += 1;
            if k >= self.start && (k - self.start) % self.step == 0 {
                return Some(Ok(self.gray()));
            }
        }
        None
    }
}

impl Drop for VideoFrames {
    fn drop(&mut self) {
        if self.stderr.is_some() {
            self.shut_down(); // not reaped yet: stop ffmpeg
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffmpeg_versions_are_read_from_the_first_line() {
        let cases: [(&str, Option<(u32, u32)>); 16] = [
            (
                "ffmpeg version 9.0.2 Copyright (c) 2000-2026 the FFmpeg developers\nbuilt with",
                Some((9, 0)),
            ),
            ("ffmpeg version 5.1 Copyright (c) 2000-2022", Some((5, 1))),
            ("ffmpeg version 5.1.2-3 Copyright", Some((5, 1))),
            ("ffmpeg version n6.0 Copyright", Some((6, 0))),
            (
                "ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021",
                Some((4, 4)),
            ),
            ("ffmpeg version 4.0-rc1 Copyright", Some((4, 0))),
            ("ffmpeg version 2.8.22 Copyright", Some((2, 8))),
            // development builds are named by a revision
            ("ffmpeg version N-109044-g1b2c3d4 Copyright", None),
            ("ffmpeg version git-2024-01-01-1b2c3d4 Copyright", None),
            // ... or by their date, which is not a release number
            (
                "ffmpeg version 2021-10-12-git-abc1234 Copyright (c) 2000-2021",
                None,
            ),
            (
                "ffmpeg version 2023-08-14-git-bd4ec8bf5e-full_build-www.gyan.dev Copyright",
                None,
            ),
            ("ffmpeg version 20240101-1b2c3d4 Copyright", None),
            ("ffmpeg version N-109044-g1b2c3d4-20221116 Copyright", None),
            // builds of a release carry a suffix after its number
            (
                "ffmpeg version 7.0.2-static https://johnvansickle.com/ffmpeg/ built with gcc",
                Some((7, 0)),
            ),
            (
                "ffmpeg version 6.1-essentials_build-www.gyan.dev Copyright",
                Some((6, 1)),
            ),
            ("ffmpeg version 99.0 Copyright", Some((99, 0))),
        ];
        for (text, want) in cases {
            assert_eq!(parse_ffmpeg_version(text), want, "{text}");
        }
        // a date-named build is "unknown" and gets the current option, not a major of 2021
        let dated = parse_ffmpeg_version("ffmpeg version 2023-08-14-git-bd4ec8bf5e Copyright");
        assert_eq!(passthrough_option(dated), ["-fps_mode", "passthrough"]);
        assert_eq!(parse_ffmpeg_version(""), None);
        assert_eq!(parse_ffmpeg_version("not ffmpeg"), None);
    }

    #[test]
    fn the_display_rotation_picks_the_turn_that_makes_the_frame_upright() {
        // ffprobe's `rotation` is the counterclockwise angle the display applies: -90 (a phone held
        // upright) is a quarter turn clockwise
        let turn = Turn::from_rotation;
        assert_eq!(turn(0.0), Turn::None);
        assert_eq!(turn(-90.0), Turn::Clockwise);
        assert_eq!(turn(270.0), Turn::Clockwise);
        assert_eq!(turn(90.0), Turn::CounterClockwise);
        assert_eq!(turn(-270.0), Turn::CounterClockwise);
        assert_eq!(turn(180.0), Turn::Half);
        assert_eq!(turn(-180.0), Turn::Half);
        assert_eq!(turn(360.0), Turn::None);
        // whole degrees are rounded as ffmpeg does; an angle that is not a quarter turn is ignored, as
        // OpenCV (the Python reference) ignores it
        assert_eq!(turn(-89.6), Turn::Clockwise);
        assert_eq!(turn(90.4), Turn::CounterClockwise);
        for other in [
            45.0,
            -45.0,
            89.0,
            -91.0,
            179.0,
            1.0,
            f64::NAN,
            f64::INFINITY,
        ] {
            assert_eq!(turn(other), Turn::None, "{other}");
        }
        assert_eq!(Turn::None.filter(), None);
        assert_eq!(Turn::Clockwise.filter(), Some("transpose=clock"));
        assert_eq!(Turn::CounterClockwise.filter(), Some("transpose=cclock"));
        assert_eq!(Turn::Half.filter(), Some("hflip,vflip"));
        assert!(Turn::Clockwise.swaps_sides() && Turn::CounterClockwise.swaps_sides());
        assert!(!Turn::None.swaps_sides() && !Turn::Half.swaps_sides());
    }

    #[test]
    fn the_probe_text_gives_the_stored_size_and_the_turn() {
        let p = parse_probe;
        assert_eq!(
            p("width=1920\nheight=1080\n"),
            Some((1920, 1080, Turn::None))
        );
        // the rotation line follows the size lines (stream_side_data)
        assert_eq!(
            p("width=1080\nheight=1920\nrotation=-90\n"),
            Some((1080, 1920, Turn::Clockwise))
        );
        assert_eq!(
            p("width=1080\nheight=1920\nrotation=90.000000\n"),
            Some((1080, 1920, Turn::CounterClockwise))
        );
        assert_eq!(
            p("width=64\nheight=48\nrotation=-180\n"),
            Some((64, 48, Turn::Half))
        );
        // an unreadable rotation is none
        assert_eq!(
            p("width=64\nheight=48\nrotation=N/A\n"),
            Some((64, 48, Turn::None))
        );
        // a size of 0 in either dimension is no size
        for text in [
            "width=0\nheight=48\n",
            "width=64\nheight=0\n",
            "width=0\nheight=0\n",
        ] {
            assert_eq!(p(text), None, "{text:?}");
        }
        assert_eq!(p("width=1\nheight=1\n"), Some((1, 1, Turn::None)));
        assert_eq!(p("height=48\n"), None);
        assert_eq!(p(""), None);
    }

    #[test]
    fn the_filter_chain_turns_first_and_then_converts() {
        let plain = "scale=in_color_matrix=bt601:flags=neighbor+accurate_rnd,format=rgb24";
        assert_eq!(video_filter(Turn::None), plain);
        assert_eq!(
            video_filter(Turn::Clockwise),
            format!("transpose=clock,{plain}")
        );
        assert_eq!(video_filter(Turn::Half), format!("hflip,vflip,{plain}"));
    }

    #[test]
    fn old_ffmpeg_gets_vsync_and_new_or_unknown_gets_fps_mode() {
        for old in [(2, 8), (4, 4), (5, 0)] {
            assert_eq!(passthrough_option(Some(old)), ["-vsync", "passthrough"]);
        }
        for new in [Some((5, 1)), Some((6, 0)), Some((9, 0)), None] {
            assert_eq!(passthrough_option(new), ["-fps_mode", "passthrough"]);
        }
    }
}
