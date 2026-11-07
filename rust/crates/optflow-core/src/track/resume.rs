//! Saving and restoring the state of a running sequence, so that a stopped run continues with the
//! bits of an uninterrupted one. Only what cannot be recomputed goes into the file:
//!
//! * the reference and the previous frame are re-read from the input by the caller; the checksums
//!   of the frames kept here prove that the input gave the same bits;
//! * `Hybrid::u_prev` is the composition of `u0r` with the previous direct leg wherever the pixel
//!   was direct-good (`good_prev`), so only the other pixels are stored; its checksum is kept too;
//! * `Hybrid::prev_adj` is not stored when it is the previous direct leg (the reference is `k - 1`);
//! * `Pair::low` and the times of the previous pair are never read after their own frame.
//!
//! A `crack_model = "xdic"` run carries crack lines between legs and is not saved.

use super::*;
use crate::checkpoint::{sum_f32, CheckpointError, Data, Reader};
use std::io::{Read, Seek};

/// Sections of the state. Applications may use ids from [`SECTION_APP`] on.
const SCALARS: u32 = 10;
const U0R_X: u32 = 11;
const U0R_Y: u32 = 12;
const C0R: u32 = 13;
const PAIR_UX: u32 = 14;
const PAIR_UY: u32 = 15;
const PAIR_CONF: u32 = 16;
const ANCH: u32 = 20;
const U_PREV_X: u32 = 21;
const U_PREV_Y: u32 = 22;
const NFB: u32 = 23;
const GOOD_PREV: u32 = 24;
const ADJ_X: u32 = 25;
const ADJ_Y: u32 = 26;

/// The first section id that this module does not use.
pub const SECTION_APP: u32 = 100;

const ADJ_NONE: u64 = 0;
const ADJ_STORED: u64 = 1;
const ADJ_IS_PAIR: u64 = 2;

/// The numbers a checkpoint is about: which frames must be read back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunHead {
    /// Frames taken after frame 0.
    pub k: usize,
    /// The reference frame.
    pub ref_idx: usize,
    /// Height and width of what is tracked.
    pub shape: (usize, usize),
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// The state of a sequence between two frames, ready to be written.
pub struct RunState<'a> {
    run: &'a Run,
    pair: &'a Pair,
    scalars: Vec<u64>,
}

impl RunState<'_> {
    pub fn head(&self) -> RunHead {
        RunHead {
            k: self.run.k,
            ref_idx: self.run.ref_idx,
            shape: self.run.shape,
        }
    }

    /// The sections to pass to [`crate::checkpoint::write_container`].
    pub fn sections(&self) -> Vec<(u32, Data<'_>)> {
        // every field is named, so that a new one cannot be forgotten here
        let Run {
            k: _,
            shape: _,
            reference: _,
            ref_idx: _,
            u0r,
            c0r,
            prev: _,
            prev_pair: _,
            hybrid,
        } = self.run;
        let Pair {
            raw_ux,
            raw_uy,
            conf,
            low: _,
            coarse: _,
            fine: _,
            xleg: _,
        } = self.pair;
        let mut s = vec![
            (SCALARS, Data::U64(&self.scalars)),
            (U0R_X, Data::F32(&u0r.0.data)),
            (U0R_Y, Data::F32(&u0r.1.data)),
            (C0R, Data::F64(c0r)),
            (PAIR_UX, Data::F32(&raw_ux.data)),
            (PAIR_UY, Data::F32(&raw_uy.data)),
            (PAIR_CONF, Data::F32(&conf.data)),
        ];
        if let Some(hy) = hybrid {
            let hybrid::Hybrid {
                anch,
                u_prev,
                nfb,
                good_prev,
                prev_adj,
            } = hy;
            s.extend([
                (ANCH, Data::Bool(anch)),
                (GOOD_PREV, Data::Bool(good_prev)),
                (U_PREV_X, Data::MaskedF32(&u_prev.0.data, good_prev)),
                (U_PREV_Y, Data::MaskedF32(&u_prev.1.data, good_prev)),
                (NFB, Data::U32(nfb)),
            ]);
            if self.scalars[5] == ADJ_STORED {
                let (ax, ay) = prev_adj.as_ref().expect("stored means present");
                s.extend([(ADJ_X, Data::F32(&ax.data)), (ADJ_Y, Data::F32(&ay.data))]);
            }
        }
        s
    }
}

/// A state read back, the frames it needs still to come.
pub struct SavedRun {
    head: RunHead,
    scalars: Vec<u64>,
    u0r: (Image<f32>, Image<f32>),
    c0r: Vec<f64>,
    pair: (Image<f32>, Image<f32>, Image<f32>),
    hybrid: Option<hybrid::Hybrid>,
}

impl SavedRun {
    /// The numbers of a checkpoint, without reading its arrays.
    pub fn head<R: Read + Seek>(r: &mut Reader<R>) -> Result<RunHead, CheckpointError> {
        let s = r.u64s(SCALARS)?;
        if s.len() != SCALAR_COUNT {
            return Err(CheckpointError::Damaged("the scalars".into()));
        }
        Ok(RunHead {
            k: s[0] as usize,
            ref_idx: s[1] as usize,
            shape: (s[2] as usize, s[3] as usize),
        })
    }

    /// Reads the arrays. Every checksum of the container is checked.
    pub fn read<R: Read + Seek>(r: &mut Reader<R>) -> Result<SavedRun, CheckpointError> {
        let head = SavedRun::head(r)?;
        let scalars = r.u64s(SCALARS)?;
        let (h, w) = head.shape;
        let n = w * h;
        let im = |v: Vec<f32>| Image::from_vec(w, h, v);
        let u0r = (im(r.f32s(U0R_X, n)?), im(r.f32s(U0R_Y, n)?));
        let c0r = r.f64s(C0R, n)?;
        let pair = (
            im(r.f32s(PAIR_UX, n)?),
            im(r.f32s(PAIR_UY, n)?),
            im(r.f32s(PAIR_CONF, n)?),
        );
        let hybrid = if scalars[4] == 1 {
            let (cx, cy) = compose(&u0r.0, &u0r.1, &pair.0, &pair.1);
            let good_prev = r.bools(GOOD_PREV, n)?;
            let ux = r.masked_f32s(U_PREV_X, &cx.data, &good_prev)?;
            let uy = r.masked_f32s(U_PREV_Y, &cy.data, &good_prev)?;
            if sum_f32(&ux) != scalars[8] || sum_f32(&uy) != scalars[9] {
                return Err(CheckpointError::Mismatch(
                    "the recomputed displacement differs from the saved one",
                ));
            }
            let prev_adj = match scalars[5] {
                ADJ_NONE => None,
                ADJ_STORED => Some((im(r.f32s(ADJ_X, n)?), im(r.f32s(ADJ_Y, n)?))),
                ADJ_IS_PAIR => Some((pair.0.clone(), pair.1.clone())),
                _ => return Err(CheckpointError::Damaged("the adjacent leg".into())),
            };
            Some(hybrid::Hybrid {
                anch: r.bools(ANCH, n)?,
                u_prev: (im(ux), im(uy)),
                nfb: r.u32s(NFB, n)?,
                good_prev,
                prev_adj,
            })
        } else {
            None
        };
        Ok(SavedRun {
            head,
            scalars,
            u0r,
            c0r,
            pair,
            hybrid,
        })
    }

    /// The running state, given the reference frame and the previous frame as the caller re-read
    /// them; they must have the bits they had.
    fn into_run(self, reference: Image<f32>, prev: Image<f32>) -> Result<Run, CheckpointError> {
        if shape_of(&reference) != self.head.shape || shape_of(&prev) != self.head.shape {
            return Err(CheckpointError::Mismatch("the size of the frames changed"));
        }
        if sum_f32(&reference.data) != self.scalars[6] {
            return Err(CheckpointError::Mismatch("the reference frame changed"));
        }
        if sum_f32(&prev.data) != self.scalars[7] {
            return Err(CheckpointError::Mismatch("the previous frame changed"));
        }
        let (raw_ux, raw_uy, conf) = self.pair;
        Ok(Run {
            k: self.head.k,
            shape: self.head.shape,
            reference,
            ref_idx: self.head.ref_idx,
            u0r: self.u0r,
            c0r: self.c0r,
            prev,
            prev_pair: Some(Pair {
                raw_ux,
                raw_uy,
                conf,
                low: Image::new(0, 0),
                coarse: 0.0,
                fine: 0.0,
                xleg: None,
            }),
            hybrid: self.hybrid,
        })
    }
}

/// k, ref_idx, h, w, hybrid, adjacent leg, sum of the reference, of the previous frame, of `u_prev`.
const SCALAR_COUNT: usize = 10;

/// A sequence whose state can be saved between two frames, or that continues a saved one. As an
/// iterator it is [`track_sequence_with`], item for item.
pub struct Resumable<'a, I> {
    seq: Sequence<'a, I>,
}

/// [`track_sequence_with`] that can be saved between frames; with `restored` it continues that
/// state, and `frames` must yield the frames after the previous one.
pub fn resumable_sequence<'a, I>(
    engine: &'a dyn FineEngine,
    frames: I,
    cfg: &'a Config,
    mask: Option<&'a Image<bool>>,
    opts: StrainOpts,
    restored: Option<(SavedRun, Image<f32>, Image<f32>)>,
) -> Result<Resumable<'a, I>, CheckpointError>
where
    I: Iterator<Item = Image<f32>> + 'a,
{
    let state = match restored {
        None => State::Start,
        Some((saved, reference, prev)) => {
            if saved.hybrid.is_some() != cfg.is_hybrid() {
                return Err(CheckpointError::Mismatch("the sequence mode changed"));
            }
            if cfg.is_xdic() {
                return Err(CheckpointError::Mismatch("a crack run cannot be resumed"));
            }
            if mask.is_some_and(|m| shape_of(m) != saved.head.shape) {
                return Err(CheckpointError::Mismatch("the ROI changed"));
            }
            State::Running(Box::new(saved.into_run(reference, prev)?))
        }
    };
    Ok(Resumable {
        seq: Sequence {
            engine,
            frames,
            cfg,
            mask,
            opts,
            state,
        },
    })
}

impl<I> Resumable<'_, I> {
    /// The state after the last item, if it can be saved: `None` before the first frame, after an
    /// error and for a crack run.
    pub fn state(&self) -> Option<RunState<'_>> {
        if self.seq.cfg.is_xdic() {
            return None;
        }
        let State::Running(run) = &self.seq.state else {
            return None;
        };
        let pair = run.prev_pair.as_ref()?;
        let adj = match run.hybrid.as_ref().map(|hy| hy.prev_adj.as_ref()) {
            Some(Some((ax, ay))) => {
                if bits_equal(&ax.data, &pair.raw_ux.data)
                    && bits_equal(&ay.data, &pair.raw_uy.data)
                {
                    ADJ_IS_PAIR
                } else {
                    ADJ_STORED
                }
            }
            _ => ADJ_NONE,
        };
        let (sum_ux, sum_uy) = match &run.hybrid {
            Some(hy) => (sum_f32(&hy.u_prev.0.data), sum_f32(&hy.u_prev.1.data)),
            None => (0, 0),
        };
        let scalars = vec![
            run.k as u64,
            run.ref_idx as u64,
            run.shape.0 as u64,
            run.shape.1 as u64,
            u64::from(run.hybrid.is_some()),
            adj,
            sum_f32(&run.reference.data),
            sum_f32(&run.prev.data),
            sum_ux,
            sum_uy,
        ];
        debug_assert_eq!(scalars.len(), SCALAR_COUNT);
        Some(RunState { run, pair, scalars })
    }
}

impl<I: Iterator<Item = Image<f32>>> Iterator for Resumable<'_, I> {
    type Item = Result<(usize, TrackResult), TrackError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.seq.next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::write_container;
    use std::io::Cursor;

    /// Noise that drifts and decorrelates, so that the reference moves and pixels fall back.
    fn frames(n: usize, w: usize, h: usize) -> Vec<Image<f32>> {
        let (pw, ph) = (w + 40, h + 40);
        let a = crate::interp::gauss_blur(&crate::image::pseudo_random_image(pw, ph, 7), 1.5);
        let b = crate::interp::gauss_blur(&crate::image::pseudo_random_image(pw, ph, 8), 1.5);
        (0..n)
            .map(|i| {
                let (dx, dy) = (i * 2, i);
                let mix = (i as f64 * 0.11).min(0.8);
                let data = (0..h)
                    .flat_map(|y| (0..w).map(move |x| (x, y)))
                    .map(|(x, y)| {
                        let j = (y + dy) * pw + x + dx;
                        ((1.0 - mix) * a.data[j] + mix * b.data[j]) as f32
                    })
                    .collect();
                Image::from_vec(w, h, data)
            })
            .collect()
    }

    fn cfg(seq_mode: &str) -> Config {
        Config {
            seq_mode: seq_mode.into(),
            ref_update_conf: 0.6,
            outer_iters: 2,
            ..Config::default()
        }
    }

    fn result_bits(r: &TrackResult) -> Vec<u32> {
        let mut v = vec![];
        for im in [
            &r.ux,
            &r.uy,
            &r.conf,
            &r.raw_ux,
            &r.raw_uy,
            &r.strain.gmax,
            &r.strain.vol,
            &r.strain.exx_gl,
        ] {
            v.extend(im.data.iter().map(|x| x.to_bits()));
        }
        if let Some(fb) = &r.fb {
            v.extend(fb.data.iter().map(|&x| x as u32));
        }
        v
    }

    /// Saves after frame `stop`, restores into a new sequence and compares every later result.
    fn check(mode: &str, mask: Option<&Image<bool>>, stops: &[usize]) -> (Vec<usize>, Vec<u64>) {
        let (n, w, h) = (10, 128, 96);
        let all = frames(n, w, h);
        let cfg = cfg(mode);
        let opts = StrainOpts::default();
        let whole: Vec<(usize, TrackResult)> =
            track_sequence_on(&CpuEngine, all.clone().into_iter(), &cfg, mask)
                .map(Result::unwrap)
                .collect();
        let refs: Vec<usize> = whole
            .iter()
            .map(|(_, r)| r.timing.reference.unwrap())
            .collect();
        let mut adj_kinds = vec![];
        for &stop in stops {
            let mut a =
                resumable_sequence(&CpuEngine, all.clone().into_iter(), &cfg, mask, opts, None)
                    .unwrap();
            assert!(
                a.state().is_none(),
                "nothing to save before the first frame"
            );
            for _ in 0..stop {
                a.next().unwrap().unwrap();
            }
            let state = a.state().unwrap();
            let head = state.head();
            assert_eq!(head.k, stop);
            adj_kinds.push(state.scalars[5]);
            let mut file = Cursor::new(Vec::new());
            write_container(&mut file, &state.sections()).unwrap();
            let bytes = file.into_inner();
            let mut r = Reader::open(Cursor::new(&bytes)).unwrap();
            assert_eq!(SavedRun::head(&mut r).unwrap(), head);
            let saved = SavedRun::read(&mut r).unwrap();
            let rest = all[stop + 1..].to_vec();
            let b = resumable_sequence(
                &CpuEngine,
                rest.into_iter(),
                &cfg,
                mask,
                opts,
                Some((saved, all[head.ref_idx].clone(), all[stop].clone())),
            )
            .unwrap();
            let mut seen = 0;
            for item in b {
                let (k, res) = item.unwrap();
                let (wk, want) = &whole[k - 1];
                assert_eq!(k, *wk);
                assert_eq!(
                    result_bits(&res),
                    result_bits(want),
                    "{mode}: stopped at {stop}, frame {k}"
                );
                assert_eq!(res.timing.reference, want.timing.reference);
                assert_eq!(
                    res.timing.mean_conf.map(f64::to_bits),
                    want.timing.mean_conf.map(f64::to_bits)
                );
                seen += 1;
            }
            assert_eq!(seen, n - 1 - stop);
        }
        (refs, adj_kinds)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn a_restored_chain_continues_with_the_same_bits() {
        let (refs, _) = check("chain", None, &[1, 2, 5, 8, 9]);
        assert!(refs.iter().any(|&r| r > 0), "the reference moved: {refs:?}");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn a_restored_hybrid_continues_with_the_same_bits() {
        let stops: Vec<usize> = (1..=9).collect();
        let (refs, adj) = check("hybrid", None, &stops);
        assert!(refs.iter().any(|&r| r > 0), "the reference moved: {refs:?}");
        // the state was saved with the adjacent leg in its own arrays and as the direct leg
        assert!(adj.contains(&ADJ_STORED), "{adj:?}");
        assert!(
            adj.contains(&ADJ_IS_PAIR) || adj.contains(&ADJ_NONE),
            "{adj:?}"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn a_restored_hybrid_with_a_roi_continues_with_the_same_bits() {
        let (w, h) = (128, 96);
        let data = (0..w * h)
            .map(|i| (20..110).contains(&(i % w)) && (10..85).contains(&(i / w)))
            .collect();
        let mask = Image::from_vec(w, h, data);
        check("hybrid", Some(&mask), &[1, 3, 7]);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn other_frames_than_the_saved_ones_are_refused() {
        let all = frames(4, 128, 96);
        let cfg = cfg("chain");
        let opts = StrainOpts::default();
        let mut a = resumable_sequence(&CpuEngine, all.clone().into_iter(), &cfg, None, opts, None)
            .unwrap();
        a.next().unwrap().unwrap();
        a.next().unwrap().unwrap();
        let state = a.state().unwrap();
        let mut file = Cursor::new(Vec::new());
        write_container(&mut file, &state.sections()).unwrap();
        let read = || {
            let bytes = file.get_ref().clone();
            SavedRun::read(&mut Reader::open(Cursor::new(bytes)).unwrap()).unwrap()
        };
        let cont = |saved, reference, prev| {
            resumable_sequence(
                &CpuEngine,
                std::iter::empty(),
                &cfg,
                None,
                opts,
                Some((saved, reference, prev)),
            )
            .err()
            .map(|e| e.to_string())
        };
        assert_eq!(cont(read(), all[0].clone(), all[2].clone()), None);
        assert!(cont(read(), all[1].clone(), all[2].clone()).is_some());
        assert!(cont(read(), all[0].clone(), all[3].clone()).is_some());
    }

    #[test]
    fn comparing_legs_goes_by_bits() {
        assert!(bits_equal(&[f32::NAN, 1.0], &[f32::NAN, 1.0]));
        assert!(!bits_equal(&[f32::NAN], &[f32::from_bits(0x7FC0_0001)]));
        assert!(!bits_equal(&[0.0], &[-0.0]));
    }
}
