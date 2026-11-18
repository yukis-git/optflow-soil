//! Showing frames of the input before a run: a frame is read on a thread of its own (a video frame
//! is found by decoding up to it, or by a seek, which can take seconds), so the window never waits
//! for it. Display only: what a run reads never passes through here.

use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

use crate::messages;
use crate::source::{Gray8, Source};
use crate::view::shrink_gray;

/// What the frame is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Purpose {
    /// The thumbnail of the start frame, under the range bar.
    ThumbStart,
    /// The thumbnail of the end frame.
    ThumbEnd,
    /// The start frame: the reference the ROI is drawn on.
    Start,
}

impl Purpose {
    /// Where the purpose stands in the queue of the reading thread: the start frame first.
    fn slot(self) -> usize {
        match self {
            Purpose::Start => 0,
            Purpose::ThumbStart => 1,
            Purpose::ThumbEnd => 2,
        }
    }

    /// A frame for a thumbnail, which may be found by a seek.
    pub fn is_thumb(self) -> bool {
        self != Purpose::Start
    }
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

/// The frame-reading threads. Of the requests of one purpose only the latest counts: while a handle
/// of the range bar moves, the frames in between are skipped. It ends with this value.
pub struct FrameWorker {
    tx_start: Sender<Request>,
    tx_thumb: Sender<Request>,
    pub rx: Receiver<Result<Loaded, (Purpose, i64, String)>>,
}

/// Reads frame `n` of `source` for a texture reduced by `factor`.
pub fn load(source: &Source, n: i64, factor: usize, purpose: Purpose) -> anyhow::Result<Loaded> {
    // only a thumbnail may use the seek; the start frame is the run's reference
    let seeked = purpose.is_thumb() && source.is_video && source.seekable.is_some();
    let gray = if purpose.is_thumb() {
        source.frame_preview(n)?
    } else {
        source.frame(n)?
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
    /// Two threads: one for the start frame (a video is decoded up to it, seconds on a long one)
    /// and one for the thumbnails, so a thumbnail never waits for the start frame.
    pub fn start(repaint: Arc<dyn Fn() + Send + Sync>) -> FrameWorker {
        let (done, rx) = channel();
        let spawn = |done: Sender<_>, repaint: Arc<dyn Fn() + Send + Sync>| {
            let (tx, jobs) = channel::<Request>();
            std::thread::spawn(move || {
                // the latest request of each purpose
                let mut pending: [Option<Request>; 3] = [None, None, None];
                loop {
                    if pending.iter().all(Option::is_none) {
                        match jobs.recv() {
                            Ok(r) => {
                                let i = r.purpose.slot();
                                pending[i] = Some(r);
                            }
                            Err(_) => return,
                        }
                    }
                    for r in jobs.try_iter() {
                        let i = r.purpose.slot();
                        pending[i] = Some(r);
                    }
                    let Some(job) = pending.iter_mut().find_map(Option::take) else {
                        continue;
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
            tx
        };
        let tx_start = spawn(done.clone(), repaint.clone());
        let tx_thumb = spawn(done, repaint);
        FrameWorker {
            tx_start,
            tx_thumb,
            rx,
        }
    }

    pub fn request(&self, source: &Source, n: i64, factor: usize, purpose: Purpose) {
        let tx = if purpose.is_thumb() {
            &self.tx_thumb
        } else {
            &self.tx_start
        };
        let _ = tx.send(Request {
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
        w.request(&s, 3, 1, Purpose::ThumbStart);
        let l = w.rx.recv_timeout(Duration::from_secs(20)).unwrap().unwrap();
        assert_eq!((l.n, l.purpose), (3, Purpose::ThumbStart));
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
        w.request(&s, 0, 1, Purpose::ThumbStart);
        for n in 1..4 {
            w.request(&s, n, 1, Purpose::ThumbStart);
        }
        w.request(&s, 5, 1, Purpose::Start);
        // the two thumbnails are kept apart: each has its own latest
        w.request(&s, 2, 1, Purpose::ThumbEnd);
        w.request(&s, 4, 1, Purpose::ThumbEnd);
        let mut got = vec![];
        while let Ok(r) = w.rx.recv_timeout(Duration::from_millis(1500)) {
            let l = r.unwrap();
            got.push((l.purpose, l.n));
        }
        // the start frame arrives, and the slider ends on its latest frame
        assert!(got.contains(&(Purpose::Start, 5)), "{got:?}");
        assert_eq!(
            got.iter().rfind(|g| g.0 == Purpose::ThumbStart),
            Some(&(Purpose::ThumbStart, 3))
        );
        assert_eq!(
            got.iter().rfind(|g| g.0 == Purpose::ThumbEnd),
            Some(&(Purpose::ThumbEnd, 4))
        );
        let _ = std::fs::remove_dir_all(&d);
    }
}
