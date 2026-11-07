//! Checkpoints of a sequence run, for the desktop app: when to save, the atomic write, and what
//! identifies the run. The command line never uses them: `run_with` passes none, and there is no
//! flag for them.
//!
//! A checkpoint is one file in the run folder. It is written beside its final name, flushed to
//! the disk and renamed, so that a crash or a power cut during a write leaves the previous one in
//! place. The result folders of the frames it covers are flushed first: once a checkpoint is in
//! place, everything it says has been written is on the disk. A run that finishes removes it.

use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use optflow_core::checkpoint::{write_container, CheckpointError, Data, Reader};
use optflow_core::track::resume::{Resumable, RunHead, SavedRun, SECTION_APP};

/// The name of the checkpoint in a run folder.
pub const FILE: &str = "checkpoint.ofck";
/// The name it is written under.
pub const TMP: &str = "checkpoint.ofck.tmp";

const META: u32 = SECTION_APP;
const EXTRA: u32 = SECTION_APP + 1;

/// Key and value pairs in length-prefixed binary form: what identifies the run. The values are the
/// caller's; this module only stores them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Meta(pub Vec<(String, String)>);

impl Meta {
    pub fn get(&self, key: &str) -> Option<&str> {
        self.0
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for (k, v) in &self.0 {
            for s in [k, v] {
                out.extend_from_slice(&(s.len() as u32).to_le_bytes());
                out.extend_from_slice(s.as_bytes());
            }
        }
        out
    }

    pub fn decode(mut b: &[u8]) -> Option<Meta> {
        let take = |b: &mut &[u8]| -> Option<String> {
            let (n, rest) = b.split_first_chunk::<4>()?;
            let n = u32::from_le_bytes(*n) as usize;
            let s = rest.get(..n)?;
            *b = &rest[n..];
            String::from_utf8(s.to_vec()).ok()
        };
        let mut pairs = Vec::new();
        while !b.is_empty() {
            let k = take(&mut b)?;
            let v = take(&mut b)?;
            pairs.push((k, v));
        }
        Some(Meta(pairs))
    }
}

/// When to save during a run. A save is due when `interval_s` have passed since the last one and
/// the last write would cost at most `max_cost` of the time since then.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Policy {
    pub interval_s: f64,
    pub max_cost: f64,
}

impl Default for Policy {
    fn default() -> Policy {
        Policy {
            interval_s: 300.0,
            max_cost: 0.05,
        }
    }
}

impl Policy {
    /// Seconds to wait after a write that took `last_write_s`.
    pub fn wait_s(&self, last_write_s: f64) -> f64 {
        if self.max_cost > 0.0 {
            self.interval_s.max(last_write_s / self.max_cost)
        } else {
            f64::INFINITY
        }
    }
}

/// A finished save.
#[derive(Clone, Copy, Debug)]
pub struct SaveInfo {
    pub k: usize,
    pub bytes: u64,
    pub secs: f64,
}

/// The checkpoints of one run.
pub struct Checkpointing {
    pub out: PathBuf,
    pub policy: Policy,
    pub meta: Meta,
    /// Continue from the checkpoint in `out`.
    pub resume: bool,
    since: Instant,
    last_write_s: f64,
    /// Frames up to this one are known to be on the disk.
    synced: usize,
}

impl Checkpointing {
    pub fn new(out: PathBuf, policy: Policy, meta: Meta, resume: bool) -> Checkpointing {
        Checkpointing {
            out,
            policy,
            meta,
            resume,
            since: Instant::now(),
            last_write_s: 0.0,
            synced: 0,
        }
    }

    pub fn path(&self) -> PathBuf {
        self.out.join(FILE)
    }

    /// Whether a save is due now.
    pub fn due(&self) -> bool {
        self.since.elapsed().as_secs_f64() >= self.policy.wait_s(self.last_write_s)
    }

    pub(crate) fn resumed_at(&mut self, k: usize) {
        self.synced = k;
        self.since = Instant::now();
    }

    /// Saves the state of `seq`. `Ok(None)`: this run cannot be saved (a crack run).
    pub fn save<I>(&mut self, seq: &Resumable<'_, I>, extra: &[u8]) -> Result<Option<SaveInfo>> {
        let t = Instant::now();
        let Some(state) = seq.state() else {
            return Ok(None);
        };
        let k = state.head().k;
        self.sync_results(k)?;
        let meta = self.meta.encode();
        let mut sections = vec![(META, Data::Bytes(&meta)), (EXTRA, Data::Bytes(extra))];
        sections.extend(state.sections());
        let tmp = self.out.join(TMP);
        let bytes = write_durably(&tmp, &self.path(), &sections)
            .with_context(|| format!("cannot write {}", tmp.display()))?;
        self.synced = k;
        let secs = t.elapsed().as_secs_f64();
        self.last_write_s = secs;
        self.since = Instant::now();
        Ok(Some(SaveInfo { k, bytes, secs }))
    }

    /// Flushes the results of the frames since the last save to the disk.
    fn sync_results(&self, k: usize) -> Result<()> {
        if self.synced == 0 {
            for name in ["config.json", "crop.json", "roi.json"] {
                let p = self.out.join(name);
                if p.exists() {
                    sync_file(&p)?;
                }
            }
        }
        for j in self.synced + 1..=k {
            let dir = self.out.join(format!("frame_{j:05}"));
            for e in fs::read_dir(&dir).with_context(|| format!("cannot read {}", dir.display()))? {
                sync_file(&e?.path())?;
            }
            sync_dir(&dir);
        }
        sync_dir(&self.out);
        Ok(())
    }

    /// The run finished: nothing is left to resume.
    pub(crate) fn finished(&self) -> Result<()> {
        for name in [FILE, TMP] {
            let p = self.out.join(name);
            if p.exists() {
                fs::remove_file(&p).with_context(|| format!("cannot remove {}", p.display()))?;
            }
        }
        Ok(())
    }
}

fn sync_file(p: &Path) -> Result<()> {
    let f = OpenOptions::new()
        .write(true)
        .open(p)
        .with_context(|| format!("cannot open {}", p.display()))?;
    f.sync_all()
        .with_context(|| format!("cannot flush {}", p.display()))
}

/// Flushes a directory entry (the rename). Windows cannot open a directory for this and has
/// nothing to do here.
fn sync_dir(dir: &Path) {
    #[cfg(unix)]
    if let Ok(d) = File::open(dir) {
        let _ = d.sync_all();
    }
    #[cfg(not(unix))]
    let _ = dir;
}

/// Writes the container under `tmp`, flushes it and renames it to `path`. On an error the previous
/// file at `path` is untouched and `tmp` is removed.
pub fn write_durably(tmp: &Path, path: &Path, sections: &[(u32, Data)]) -> Result<u64> {
    let write = || -> Result<u64> {
        let mut f = File::create(tmp)?;
        let bytes = write_container(&mut f, sections)?;
        f.sync_all()?;
        Ok(bytes)
    };
    match write() {
        Ok(bytes) => {
            fs::rename(tmp, path)?;
            if let Some(dir) = path.parent() {
                sync_dir(dir);
            }
            Ok(bytes)
        }
        Err(e) => {
            // never a directory left in the way by a test; a failed removal changes nothing
            let _ = fs::remove_file(tmp);
            Err(e)
        }
    }
}

/// What a checkpoint says about itself, read without its arrays.
#[derive(Clone, Debug)]
pub struct Summary {
    pub meta: Meta,
    pub head: RunHead,
    pub bytes: u64,
}

fn open(path: &Path) -> Result<Reader<File>, CheckpointError> {
    Reader::open(File::open(path)?)
}

/// The summary of the checkpoint at `path`: a truncated file or a damaged table is an error here,
/// a damaged array only when it is read.
pub fn summary(path: &Path) -> Result<Summary, CheckpointError> {
    let bytes = fs::metadata(path)?.len();
    let mut r = open(path)?;
    let raw = r.bytes(META)?;
    let meta = Meta::decode(&raw).ok_or_else(|| CheckpointError::Damaged("the metadata".into()))?;
    let head = SavedRun::head(&mut r)?;
    Ok(Summary { meta, head, bytes })
}

/// The caller's bytes saved with the checkpoint.
pub fn extra(path: &Path) -> Result<Vec<u8>, CheckpointError> {
    open(path)?.bytes(EXTRA)
}

/// The tracker state of the checkpoint at `path`.
pub(crate) fn state(path: &Path) -> Result<SavedRun, CheckpointError> {
    SavedRun::read(&mut open(path)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_round_trips_and_a_bad_one_is_refused() {
        let m = Meta(vec![
            ("source".into(), "/データ/写真 1".into()),
            ("empty".into(), String::new()),
        ]);
        let b = m.encode();
        assert_eq!(Meta::decode(&b), Some(m.clone()));
        assert_eq!(m.get("source"), Some("/データ/写真 1"));
        assert_eq!(m.get("no"), None);
        assert_eq!(Meta::decode(&b[..b.len() - 1]), None);
        assert_eq!(Meta::decode(&[]), Some(Meta::default()));
    }

    #[test]
    fn the_write_cost_stretches_the_interval_never_shortens_it() {
        let p = Policy::default();
        assert_eq!(p.wait_s(0.0), 300.0);
        assert_eq!(p.wait_s(1.0), 300.0);
        assert_eq!(p.wait_s(15.0), 300.0);
        assert_eq!(p.wait_s(30.0), 600.0);
        let off = Policy {
            interval_s: 300.0,
            max_cost: 0.0,
        };
        assert!(off.wait_s(1.0).is_infinite());
    }

    #[test]
    fn a_failed_write_leaves_the_previous_checkpoint() {
        let dir = crate::unit_scratch("ckpt_atomic");
        fs::create_dir_all(&dir).unwrap();
        let (tmp, path) = (dir.join(TMP), dir.join(FILE));
        let meta = Meta(vec![("a".into(), "b".into())]).encode();
        write_durably(&tmp, &path, &[(META, Data::Bytes(&meta))]).unwrap();
        let before = fs::read(&path).unwrap();
        assert!(!tmp.exists());
        // the temporary name cannot be created: a directory stands there
        fs::create_dir_all(&tmp).unwrap();
        assert!(write_durably(&tmp, &path, &[(META, Data::Bytes(&[]))]).is_err());
        assert_eq!(fs::read(&path).unwrap(), before);
        fs::remove_dir_all(&tmp).unwrap();
        // a leftover of a crash is replaced by the next write
        fs::write(&tmp, b"half a file").unwrap();
        let bytes = write_durably(&tmp, &path, &[(META, Data::Bytes(&meta))]).unwrap();
        assert_eq!(bytes, before.len() as u64);
        assert!(!tmp.exists());
    }
}
