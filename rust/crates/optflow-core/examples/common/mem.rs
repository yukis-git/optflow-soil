//! Counting global allocator for memory profiles (`bench_pair --mem`, the CLI's `mem_cli`
//! example): the bytes allocated now, the peak, and the peak per pipeline stage and per tile,
//! sampled at the marks of `optflow_core::prof` (stage boundaries, tiles). The window between two
//! marks that lies outside every stage is named after the marks around it.
//!
//! With `MEM_TRACE=1` the blocks of at least 1 MiB are also tracked: which stage and tile
//! allocated them and a backtrace, and the set alive at the global peak is reported (taken when a
//! large block raises the peak, so the small blocks allocated after it are not in the list).
//!
//! The counts are the bytes the program asks for (`Layout::size`), not the pages the system
//! gives it: untouched pages of a zeroed block count here but not in the RSS, and the allocator's
//! own overhead and cached pages count there but not here. On macOS the process's physical
//! footprint (what `/usr/bin/time -l` reports as "peak memory footprint") is read at every mark
//! as well, so the window in which its lifetime maximum was reached is known.

#![allow(dead_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::backtrace::Backtrace;
use std::cell::Cell;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use optflow_core::prof::{self, Mark, Stage};

/// The counting allocator: `System` plus the counters of this module.
pub struct Counting;

/// Counting is on (from [`start`]); before that the allocator only forwards to `System`, so a
/// benchmark that never calls `start` is not slowed down. A block allocated before `start` and
/// freed after it is subtracted without having been added: the count saturates at 0 instead of
/// wrapping, but is low by that block, so the inputs of a profile are built after `start`.
static ON: AtomicBool = AtomicBool::new(false);
static CUR: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
/// Peak since the last mark.
static WIN: AtomicUsize = AtomicUsize::new(0);
/// Context (index into the label table) and tile (-1: none) of the running window, and those
/// of the global peak.
static CTX: AtomicUsize = AtomicUsize::new(0);
static TILE: AtomicI64 = AtomicI64::new(-1);
static PEAK_CTX: AtomicUsize = AtomicUsize::new(0);
static PEAK_TILE: AtomicI64 = AtomicI64::new(-1);
static TRACE: AtomicBool = AtomicBool::new(false);
const BIG: usize = 1 << 20;

thread_local! {
    /// Set while this module allocates for itself, so it does not trace its own blocks.
    static BUSY: Cell<bool> = const { Cell::new(false) };
}

#[derive(Clone)]
struct Block {
    size: usize,
    ctx: usize,
    tile: i64,
    bt: Arc<Backtrace>,
}

#[derive(Default)]
struct Trace {
    live: HashMap<usize, Block>,
    /// The live large blocks when a large block last raised the peak, and the bytes then.
    at_peak: Vec<Block>,
    at_peak_bytes: usize,
}

static TRACED: Mutex<Option<Trace>> = Mutex::new(None);

/// Window statistics per context label.
#[derive(Default, Clone)]
struct Stat {
    windows: usize,
    peak: usize,
    seconds: f64,
    /// The largest lifetime maximum of the footprint that was reached in a window of this label.
    fp_reached: u64,
}

#[cfg(target_os = "macos")]
extern "C" {
    fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u64) -> i32;
    fn getpid() -> i32;
}

/// `(physical footprint now, its lifetime maximum)` of this process in bytes (`rusage_info_v4`:
/// `ri_phys_footprint`, `ri_lifetime_max_phys_footprint`); zeros where unknown.
pub fn footprint() -> (u64, u64) {
    #[cfg(target_os = "macos")]
    {
        // rusage_info_v4 is 37 u64 after the 16-byte uuid (2 u64); room to spare
        let mut buf = [0u64; 64];
        // SAFETY: RUSAGE_INFO_V4 (4) writes a `struct rusage_info_v4` (296 bytes) into `buf`,
        // which holds 512.
        let rc = unsafe { proc_pid_rusage(getpid(), 4, buf.as_mut_ptr()) };
        if rc == 0 {
            return (buf[9], buf[30]);
        }
    }
    (0, 0)
}

struct State {
    labels: Vec<String>,
    index: HashMap<String, usize>,
    stats: BTreeMap<usize, Stat>,
    /// Peak per tile (over every window inside the tile).
    tiles: BTreeMap<usize, usize>,
    /// Stage being run, the last mark, and when the window started.
    stage: Option<Stage>,
    last: String,
    since: Instant,
    /// (seconds since start, bytes now, window peak, label, footprint now, footprint maximum)
    /// at every mark.
    timeline: Vec<(f64, usize, usize, usize, u64, u64)>,
    t0: Instant,
    /// Lifetime maximum of the footprint at the last mark, and the label and tile of the window
    /// in which it was last raised.
    fp_max: u64,
    fp_ctx: usize,
    fp_tile: i64,
}

/// The window statistics since the last [`start`].
static STATE: Mutex<Option<State>> = Mutex::new(None);

fn busy<T>(f: impl FnOnce() -> T) -> T {
    BUSY.with(|b| {
        let was = b.replace(true);
        let out = f();
        b.set(was);
        out
    })
}

fn added(size: usize) {
    if !ON.load(Relaxed) {
        return;
    }
    let now = CUR.fetch_add(size, Relaxed) + size;
    WIN.fetch_max(now, Relaxed);
    if PEAK.fetch_max(now, Relaxed) < now {
        PEAK_CTX.store(CTX.load(Relaxed), Relaxed);
        PEAK_TILE.store(TILE.load(Relaxed), Relaxed);
    }
}

fn removed(size: usize) {
    if ON.load(Relaxed) {
        let _ = CUR.fetch_update(Relaxed, Relaxed, |c| Some(c.saturating_sub(size)));
    }
}

fn track(ptr: *mut u8, size: usize) {
    if size < BIG || !TRACE.load(Relaxed) || BUSY.with(Cell::get) {
        return;
    }
    busy(|| {
        let bt = Arc::new(Backtrace::force_capture());
        let mut g = TRACED.lock().unwrap();
        let t = g.get_or_insert_with(Trace::default);
        t.live.insert(
            ptr as usize,
            Block {
                size,
                ctx: CTX.load(Relaxed),
                tile: TILE.load(Relaxed),
                bt,
            },
        );
        let now = CUR.load(Relaxed);
        if now > t.at_peak_bytes {
            t.at_peak_bytes = now;
            t.at_peak = t.live.values().cloned().collect();
        }
    });
}

fn untrack(ptr: *mut u8, size: usize) {
    if size < BIG || !TRACE.load(Relaxed) || BUSY.with(Cell::get) {
        return;
    }
    busy(|| {
        if let Some(t) = TRACED.lock().unwrap().as_mut() {
            t.live.remove(&(ptr as usize));
        }
    });
}

// SAFETY: every call is forwarded to `System` with the same arguments; the counters only
// observe the sizes.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            added(layout.size());
            track(p, layout.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            added(layout.size());
            track(p, layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        untrack(ptr, layout.size());
        unsafe { System.dealloc(ptr, layout) };
        removed(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            untrack(ptr, layout.size());
            if new_size >= layout.size() {
                added(new_size - layout.size());
            } else {
                removed(layout.size() - new_size);
            }
            track(p, new_size);
        }
        p
    }
}

fn label_id(st: &mut State, label: &str) -> usize {
    if let Some(&i) = st.index.get(label) {
        return i;
    }
    st.labels.push(label.to_string());
    st.index.insert(label.to_string(), st.labels.len() - 1);
    st.labels.len() - 1
}

/// Starts the profile: installs the probe, resets the peaks and the window statistics (a second
/// call starts a new profile), and turns on the tracing of large blocks if `MEM_TRACE=1`. The
/// footprint maximum is that of the whole process and cannot be reset: profile one input per
/// process for footprint figures.
pub fn start() {
    busy(|| {
        let now = Instant::now();
        let mut st = State {
            labels: Vec::new(),
            index: HashMap::new(),
            stats: BTreeMap::new(),
            tiles: BTreeMap::new(),
            stage: None,
            last: "start".into(),
            since: now,
            timeline: Vec::new(),
            t0: now,
            fp_max: footprint().1,
            fp_ctx: 0,
            fp_tile: -1,
        };
        let id = label_id(&mut st, "(before the first stage)");
        CTX.store(id, Relaxed);
        *STATE.lock().unwrap() = Some(st);
        prof::set_probe(probe);
        if std::env::var("MEM_TRACE").is_ok_and(|v| v == "1") {
            TRACE.store(true, Relaxed);
        }
    });
    TILE.store(-1, Relaxed);
    PEAK_CTX.store(CTX.load(Relaxed), Relaxed);
    PEAK_TILE.store(-1, Relaxed);
    ON.store(true, Relaxed);
    reset_peak();
}

/// Sets the peak (global, and of the large blocks) to the bytes allocated now.
pub fn reset_peak() {
    let now = CUR.load(Relaxed);
    PEAK.store(now, Relaxed);
    WIN.store(now, Relaxed);
    busy(|| {
        if let Some(t) = TRACED.lock().unwrap().as_mut() {
            t.at_peak_bytes = now;
            t.at_peak = t.live.values().cloned().collect();
        }
    });
}

/// Names the current window `label` (a phase of the harness itself, e.g. "embed").
pub fn phase(label: &str) {
    boundary(None, label.to_string(), None);
}

fn stage_label(s: Stage) -> String {
    s.name().to_string()
}

fn boundary(stage: Option<Option<Stage>>, next_label: String, tile: Option<i64>) {
    busy(|| {
        let mut guard = STATE.lock().unwrap();
        let Some(st) = guard.as_mut() else { return };
        let now = CUR.load(Relaxed);
        let win = WIN.swap(now, Relaxed);
        let ctx = CTX.load(Relaxed);
        let secs = st.since.elapsed().as_secs_f64();
        let e = st.stats.entry(ctx).or_default();
        e.windows += 1;
        e.peak = e.peak.max(win);
        e.seconds += secs;
        let tile_now = TILE.load(Relaxed);
        if tile_now >= 0 {
            let t = st.tiles.entry(tile_now as usize).or_default();
            *t = (*t).max(win);
        }
        let (fp_now, fp_max) = footprint();
        if fp_max > st.fp_max {
            st.fp_max = fp_max;
            st.fp_ctx = ctx;
            st.fp_tile = tile_now;
            let e = st.stats.entry(ctx).or_default();
            e.fp_reached = e.fp_reached.max(fp_max);
        }
        let t = st.t0.elapsed().as_secs_f64();
        st.timeline.push((t, now, win, ctx, fp_now, fp_max));
        if let Some(s) = stage {
            st.stage = s;
        }
        if let Some(t) = tile {
            TILE.store(t, Relaxed);
        }
        let id = label_id(st, &next_label);
        CTX.store(id, Relaxed);
        st.since = Instant::now();
    });
}

fn probe(m: Mark) {
    match m {
        Mark::Enter(s) => boundary(Some(Some(s)), stage_label(s), None),
        Mark::Exit(s) => {
            let next = format!("(after {})", s.name());
            boundary(Some(None), next, None);
        }
        Mark::TileStart(k) => {
            let next = "(tile start)".to_string();
            boundary(None, next, Some(k as i64));
        }
        Mark::TileEnd(_) => boundary(None, "(after a tile)".to_string(), Some(-1)),
    }
}

fn mb(b: usize) -> f64 {
    b as f64 / 1e6
}

/// Without generic arguments: `a::b<c::d>::e` is `a::b::e`.
fn plain(name: &str) -> String {
    let mut out = String::new();
    let mut depth = 0usize;
    for ch in name.chars() {
        match ch {
            '<' => depth += 1,
            '>' => depth = depth.saturating_sub(1),
            _ if depth == 0 => out.push(ch),
            _ => {}
        }
    }
    let parts: Vec<&str> = out.split("::").filter(|p| !p.is_empty()).collect();
    parts[parts.len().saturating_sub(2)..].join("::")
}

/// The innermost `n` frames of `bt` in the sources of this workspace's crates (file:line and
/// function, from the line tables; a build without them shows only function names), without
/// this module's own frames: where a block was allocated.
fn frames(bt: &Backtrace, n: usize) -> String {
    let text = format!("{bt}");
    let mut out: Vec<String> = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        let line = line.trim();
        // "N: name", then optionally "at path:line:column"
        let Some((_, name)) = line.split_once(": ") else {
            continue;
        };
        let at = match lines.peek() {
            Some(l) if l.trim().starts_with("at ") => Some(
                lines
                    .next()
                    .unwrap()
                    .trim()
                    .trim_start_matches("at ")
                    .to_string(),
            ),
            _ => None,
        };
        let Some(file) = at else { continue };
        let ours = file.contains("/crates/optflow-")
            && !file.contains("examples/common/mem.rs")
            && !name.contains("__rust_");
        if !ours {
            continue;
        }
        let short: Vec<&str> = file.rsplit('/').take(1).collect();
        let place = short[0].rsplit_once(':').map_or(short[0], |(a, _)| a);
        out.push(format!("{place} {}", plain(name)));
        if out.len() == n {
            break;
        }
    }
    out.join(" <- ")
}

/// The profile so far, as text: the global peak and where it was reached, the peak per window
/// label and per tile, and (with `MEM_TRACE=1`) the large blocks alive at the peak.
pub fn report() -> String {
    // close the running window
    boundary(None, "(after the report)".to_string(), None);
    busy(|| {
        let mut out = String::new();
        let guard = STATE.lock().unwrap();
        let st = guard.as_ref().expect("mem::start was called");
        let peak = PEAK.load(Relaxed);
        let lab = |i: usize| st.labels.get(i).cloned().unwrap_or_default();
        let pt = PEAK_TILE.load(Relaxed);
        let _ = writeln!(
            out,
            "MEM peak {:.1} MB in \"{}\"{}; now {:.1} MB",
            mb(peak),
            lab(PEAK_CTX.load(Relaxed)),
            if pt >= 0 {
                format!(" (tile {pt})")
            } else {
                String::new()
            },
            mb(CUR.load(Relaxed))
        );
        let _ = writeln!(
            out,
            "MEM footprint peak {:.1} MB, last raised in \"{}\"{}",
            st.fp_max as f64 / 1e6,
            lab(st.fp_ctx),
            if st.fp_tile >= 0 {
                format!(" (tile {})", st.fp_tile)
            } else {
                String::new()
            }
        );
        let _ = writeln!(
            out,
            "MEM per window label: peak MB, windows, seconds, footprint maximum reached MB"
        );
        let mut rows: Vec<(&usize, &Stat)> = st.stats.iter().collect();
        rows.sort_by_key(|(_, s)| std::cmp::Reverse(s.peak));
        for (i, s) in rows {
            let _ = writeln!(
                out,
                "MEM   {:<34} {:9.1} {:6} {:9.2} {:9.1}",
                lab(*i),
                mb(s.peak),
                s.windows,
                s.seconds,
                s.fp_reached as f64 / 1e6
            );
        }
        if !st.tiles.is_empty() {
            let _ = write!(out, "MEM per tile peak MB:");
            for (k, p) in &st.tiles {
                let _ = write!(out, " {k}:{:.0}", mb(*p));
            }
            let _ = writeln!(out);
        }
        if let Some(t) = TRACED.lock().unwrap().as_ref() {
            let mut blocks = t.at_peak.clone();
            blocks.sort_by_key(|b| std::cmp::Reverse(b.size));
            let total: usize = blocks.iter().map(|b| b.size).sum();
            let _ = writeln!(
                out,
                "MEM large blocks alive at the peak ({:.1} MB then): {} blocks, {:.1} MB",
                mb(t.at_peak_bytes),
                blocks.len(),
                mb(total)
            );
            // group equal (size, label, tile, place)
            let mut groups: Vec<(usize, usize, String, i64, String)> = Vec::new();
            for b in &blocks {
                let place = frames(&b.bt, 3);
                match groups
                    .iter_mut()
                    .find(|g| g.1 == b.size && g.2 == lab(b.ctx) && g.3 == b.tile && g.4 == place)
                {
                    Some(g) => g.0 += 1,
                    None => groups.push((1, b.size, lab(b.ctx), b.tile, place)),
                }
            }
            for (n, size, label, tile, place) in groups {
                let _ = writeln!(
                    out,
                    "MEM   {n:3} x {:9.1} MB = {:9.1} MB  {label}{}  {place}",
                    mb(size),
                    mb(n * size),
                    if tile >= 0 {
                        format!(" tile {tile}")
                    } else {
                        String::new()
                    }
                );
            }
        }
        out
    })
}

/// The timeline: `MEMT seconds now_MB window_peak_MB footprint_MB footprint_max_MB label` at
/// every mark (the label of the window that ends there).
pub fn timeline() -> String {
    busy(|| {
        let guard = STATE.lock().unwrap();
        let st = guard.as_ref().expect("mem::start was called");
        let mut out = String::new();
        for &(t, now, win, ctx, fp, fpm) in &st.timeline {
            let _ = writeln!(
                out,
                "MEMT {t:9.3} {:9.1} {:9.1} {:9.1} {:9.1} {}",
                mb(now),
                mb(win),
                fp as f64 / 1e6,
                fpm as f64 / 1e6,
                st.labels[ctx]
            );
        }
        out
    })
}

/// Bytes allocated now and the peak since [`start`].
pub fn now_and_peak() -> (usize, usize) {
    (CUR.load(Relaxed), PEAK.load(Relaxed))
}
