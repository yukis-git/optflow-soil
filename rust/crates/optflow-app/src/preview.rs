//! Showing a chosen source frame of the input before a run: the frame is read on a thread of its
//! own (a video frame is found by decoding up to it, which can take seconds), so the window never
//! waits for it. Display only: what a run reads never passes through here.

use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

use crate::messages;
use crate::source::{Gray8, Source};
use crate::view::shrink_gray;

/// What the frame is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Purpose {
    /// The frame under the slider.
    Slider,
    /// The start frame: the reference the ROI is drawn on.
    Start,
}

struct Request {
    source: Source,
    n: i64,
    factor: usize,
    purpose: Purpose,
}

/// A frame that was read: the picture, and it reduced for a texture.
pub struct Loaded {
    pub n: i64,
    pub purpose: Purpose,
    pub gray: Arc<Gray8>,
    pub small: (usize, usize, Vec<u8>),
    /// Found by a time seek: not the frame to use as the reference of a run.
    pub seeked: bool,
}

/// The frame-reading thread. Of the requests of one purpose only the latest counts: while a slider
/// moves, the frames in between are skipped. It ends with this value.
pub struct FrameWorker {
    tx: Sender<Request>,
    pub rx: Receiver<Result<Loaded, (Purpose, i64, String)>>,
}

/// Reads frame `n` of `source` for a texture reduced by `factor`.
pub fn load(source: &Source, n: i64, factor: usize, purpose: Purpose) -> anyhow::Result<Loaded> {
    // only the slider preview may use the seek; the start frame is the run's reference
    let seeked = purpose == Purpose::Slider && source.is_video && source.seekable.is_some();
    let gray = match purpose {
        Purpose::Slider => source.frame_preview(n)?,
        Purpose::Start => source.frame(n)?,
    };
    let small = shrink_gray(&gray, factor.max(1));
    Ok(Loaded {
        n,
        purpose,
        gray: Arc::new(gray),
        small,
        seeked,
    })
}

impl FrameWorker {
    pub fn start(repaint: Arc<dyn Fn() + Send + Sync>) -> FrameWorker {
        let (tx, jobs) = channel::<Request>();
        let (done, rx) = channel();
        std::thread::spawn(move || {
            let mut next: Option<Request> = None;
            loop {
                let first = match next.take() {
                    Some(r) => r,
                    None => match jobs.recv() {
                        Ok(r) => r,
                        Err(_) => return,
                    },
                };
                // the latest of each purpose; the start frame goes first
                let (mut start, mut slider) = (None, None);
                for r in std::iter::once(first).chain(jobs.try_iter()) {
                    match r.purpose {
                        Purpose::Start => start = Some(r),
                        Purpose::Slider => slider = Some(r),
                    }
                }
                let job = match (start, slider) {
                    (Some(a), b) => {
                        next = b;
                        a
                    }
                    (None, Some(b)) => b,
                    (None, None) => continue,
                };
                let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    load(&job.source, job.n, job.factor, job.purpose)
                }));
                let out = match out {
                    Ok(Ok(l)) => Ok(l),
                    Ok(Err(e)) => Err((job.purpose, job.n, messages::describe(&e))),
                    Err(_) => Err((
                        job.purpose,
                        job.n,
                        "フレームを表示できません(内部エラー)。".to_string(),
                    )),
                };
                if done.send(out).is_err() {
                    return;
                }
                repaint();
            }
        });
        FrameWorker { tx, rx }
    }

    pub fn request(&self, source: &Source, n: i64, factor: usize, purpose: Purpose) {
        let _ = self.tx.send(Request {
            source: source.clone(),
            n,
            factor,
            purpose,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    fn scratch(name: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("optflow-app-prev-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write_images(dir: &Path, n: usize) {
        for i in 0..n {
            let im = image::GrayImage::from_fn(16, 12, |x, y| {
                image::Luma([(x * 10 + y + 7 * i as u32) as u8])
            });
            im.save(dir.join(format!("f{i:03}.png"))).unwrap();
        }
    }

    #[test]
    fn the_worker_reads_a_frame_off_the_calling_thread() {
        let d = scratch("worker");
        write_images(&d, 5);
        let s = Source::open(&d).unwrap();
        let w = FrameWorker::start(Arc::new(|| {}));
        w.request(&s, 3, 1, Purpose::Slider);
        let l = w.rx.recv_timeout(Duration::from_secs(20)).unwrap().unwrap();
        assert_eq!((l.n, l.purpose), (3, Purpose::Slider));
        assert_eq!(l.gray.data, s.frame(3).unwrap().data);
        assert_eq!((l.small.0, l.small.1), (16, 12));
        // a frame that is not there is an error with its number, not a hang
        let e = scratch("empty");
        let missing = Source {
            path: e.to_string_lossy().into_owned(),
            ..s.clone()
        };
        w.request(&missing, 1, 1, Purpose::Start);
        let r = w.rx.recv_timeout(Duration::from_secs(20)).unwrap();
        assert!(matches!(r, Err((Purpose::Start, 1, _))));
        let _ = std::fs::remove_dir_all(&d);
        let _ = std::fs::remove_dir_all(&e);
    }

    #[test]
    fn of_one_purpose_the_latest_request_wins_and_the_start_frame_is_never_dropped() {
        let d = scratch("coalesce");
        write_images(&d, 6);
        let s = Source::open(&d).unwrap();
        let w = FrameWorker::start(Arc::new(|| {}));
        // hold the worker on a first request, then queue several
        w.request(&s, 0, 1, Purpose::Slider);
        for n in 1..4 {
            w.request(&s, n, 1, Purpose::Slider);
        }
        w.request(&s, 5, 1, Purpose::Start);
        let mut got = vec![];
        while let Ok(r) = w.rx.recv_timeout(Duration::from_millis(1500)) {
            let l = r.unwrap();
            got.push((l.purpose, l.n));
        }
        // the start frame arrives, and the slider ends on its latest frame
        assert!(got.contains(&(Purpose::Start, 5)), "{got:?}");
        assert_eq!(
            got.iter().rfind(|g| g.0 == Purpose::Slider),
            Some(&(Purpose::Slider, 3))
        );
        let _ = std::fs::remove_dir_all(&d);
    }
}
