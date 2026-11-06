# optflow (Rust)

Rust port of the Python reference `python/optflow_ref`: phase-based, per-pixel displacement and strain
tracking for soil images. The Python code is the authority for every algorithm detail; the Rust code
must give the same results (displacement RMS difference <= 1e-3 px, strain <= 1e-5).

Acceptance criteria of the port (CLI, tiling, verification): see "Verification" below.

Layout:

| path | content |
|---|---|
| `crates/optflow-core` | the library: no I/O, no OS calls, `#![forbid(unsafe_code)]` |
| `crates/optflow-cli` | the `optflow` binary: images, ROI/mask files, config JSON, video through ffmpeg, `.npy` output |
| `crates/optflow-app` | the desktop app (eframe): pick images or a video, an ROI and settings, run, view and export the fields |
| `crates/optflow-gpu` | the Metal backend of the fine stage (macOS only; see "GPU (Metal, macOS)") |
| `scripts/` | `check.sh` (full verification), `check_test.sh` (tests of it), generators of reference data |

## Build

The toolchain is pinned in `rust-toolchain.toml` (Rust 1.98.1, with clippy and rustfmt); rustup
installs it on first use.

```sh
cd rust
cargo build --release          # binary: target/release/optflow
```

## Command line

`optflow` takes the same arguments and writes the same files as `python -m optflow_ref`. `optflow --help`
lists every option. Errors are one `error: ...` line on stderr and exit status 2.

```sh
# two images; the ROI is a polygon in full-resolution pixels, the images are downscaled 4x
optflow pair --ref a.png --def b.png --roi roi.json --scale 4 --out out/pair

# the ROI as an image file (.npy bool/uint8 or PNG, nonzero = inside), for ROIs of several pieces
optflow pair --ref a.png --def b.png --mask roi_mask.png --out out/pair

# a video: every 5th frame from frame 0, tracked against an automatically updated reference
optflow seq --frames clip.mp4 --start 0 --step 5 --scale 2 --out out/seq

# a directory or a glob of images (plain string order), with a changed setting
optflow seq --frames 'frames/*.png' --set sigma_w=6 --out out/seq
```

- `pair` writes `ux uy conf Exx Eyy Exy gmax vol` (`.npy`, float32), `valid` (uint8), `timing.json`,
  `config.json` and `crop.json` into `--out`; `seq` writes the same into one directory per frame
  (`frame_00000`, ...). `--raw` adds `raw_ux.npy` and `raw_uy.npy`.
- With `--roi` or `--mask` only the ROI's bounding box plus a margin is processed (`--crop auto`, the
  default; `--crop off` processes the whole image). Outputs keep the full image size, NaN outside.
- Configuration: `--config file.json` and/or `--set key=value` (repeatable). Defaults are those of the
  Python `Config`; unknown keys are errors. The keys, their meanings and limits are documented in `python/optflow_ref/config.py`.
- `--set crack_model=xdic` (keys `xdic_RB`, `xdic_iters`) detects crack lines as jumps of the baseline
  field and solves the pixels near them per side of the line (experimental; port of `python/optflow_ref/xdic.py`). With a
  crack line the output also has `crack.npy` (bool, the pixels within 1 px of a line), `crack_nx.npy`,
  `crack_ny.npy`, `cod.npy`, `csd.npy` (opening and slip on those pixels) and `crack_lines.json` (the
  lines, the bands and rejected lines, the COD / CSD samples along each line; Rust only). CPU only
  (`--device gpu` is refused), images of at most 3840 x 2160 pixels in area (no tiles: the baseline
  iterations take the whole image, about 1.1 GB per million pixels; use an ROI above that). The constants of the method are together in `crates/optflow-core/src/xdic/params.rs`;
  `timing.json` has `xdic_detect` and `xdic`, `bench_pair --files A.npy,B.npy --xdic` the stages.
  The Python stage-1/2 crack keys (`crack_split`, `crack_out`, ...) are not adopted and stay rejected.
- An image with a side longer than 1024 px gets its fine stage in tiles of 1024 px, so that memory stays
  bounded (spec section 2.2).

### ffmpeg and ffprobe

Only video input (`seq --frames clip.mp4`) needs them; both must be on `PATH` (images need neither). A
missing tool is reported as `error: ffmpeg not found on PATH (needed for video input)`.

- Pixels: ffmpeg is asked for 8-bit `rgb24` frames (BT.601, nearest chroma, accurate rounding) and the
  tool applies the luminance `0.299 R + 0.587 G + 0.114 B`, as Python does with OpenCV's frames. For 8-bit
  4:4:4 and 4:2:0 video and for even-sized MJPEG this equals OpenCV's frames bit for bit; other pixel
  formats differ by up to about 3 grey levels. Video of more than 8 bits is quantised to 8 bits (as in
  Python).
- Version handling: frames are passed through without dropping or duplicating (`-fps_mode passthrough`,
  FFmpeg 5.1 and later; `-vsync passthrough` before that). The version is read from the first line of
  `ffmpeg -version`. A build that is not named by a release number (a development build named by a
  revision or a date) is taken to be current; a date-named build from before FFmpeg 5.1 would need
  `-vsync`.
- Rotation: the frame shape and the pixel orientation come from one decision, the rotation ffprobe
  reads. ffmpeg gets `-noautorotate` before `-i`, and the turn is the first filter (ffprobe's rotation
  is counterclockwise degrees: -90 and 270 give `transpose=clock`, 90 gives `transpose=cclock`, ±180
  gives `hflip,vflip`; only 90 and 270 swap width and height). An angle that is not a multiple of a
  quarter turn is ignored, as cv2 does. ffprobe truncates to whole degrees, so a stored 89.6 is not
  turned (cv2 would turn it); real files carry exact quarter turns.

## Input limits

- Grey 32-bit float TIFF is not supported (the image crate cannot decode it): `error: <path>:
  unsupported pixel type: …`, exit 2. Python's reference reads such files.
- A decoded image may take at most 2 GiB (a 26 Mpx RGB 16-bit frame needs about 160 MB); a larger or
  forged header gives `error: <path>: image too large to read: …` instead of an allocation failure.
- `track_pair` and `track_sequence` reject an `Image` whose data is not `w * h` values long
  (`TrackError::DataLength`; Rust only, a NumPy array cannot be malformed that way).

## The `fma` feature

`cargo build --release -p optflow-cli --features fma` turns `c + a*b` in the hot loops into a fused
multiply-add. It is faster on aarch64 (-8 % time at 10 threads, -12 % at one thread, Apple M4) but it
changes the results by about 1e-7 px, so it is **off by default and non-additive**: no dependent crate
may enable it implicitly (cargo unifies features across the graph). Only `optflow-cli`'s own opt-in
`fma` feature forwards to `optflow-core/fma`. On a default x86-64 Windows build `mul_add` is a slow
library call. `check.sh` builds, lints and tests it separately. Spec section 2.4 gives the rule for
deviations from bit-identical results.

## GPU (Metal, macOS)

The fine stage (the part that takes most of the time) can run on the Metal GPU:

```sh
cd rust
cargo build --release -p optflow-cli --features gpu
target/release/optflow pair --ref a.png --def b.png --out out --device gpu
```

`--device cpu` (the default) is the CPU path, bit for bit as without the feature. A default build
(without `--features gpu`) does not build `optflow-gpu`; `--device gpu` there stops with `error:
--device gpu needs a macOS build with --features gpu` (exit 2). A GPU failure (no Metal device, a window
too large for the device, a failed command buffer) also stops with exit 2; the program never falls back to
the CPU silently. `timing.json` of a GPU run has `"device": "gpu"` after `coarse`.

- **Accuracy** (GPU against CPU, same input and settings; GPU spec section 5, two tiers, rev 3).
  `python rust/scripts/compare_outputs.py CPU_OUT GPU_OUT [--frame N] --tier plain|robust` (numpy only;
  prints both sets of metrics, judges the chosen tier, exit 1 on fail; `--frame N` picks a frame of a `seq` output).
  *plain* (synthetic cases and 512^2): ux, uy RMS difference <= 1e-5 px over the pixels finite in both;
  conf RMS <= 1e-5 over all pixels; Exx, Eyy, Exy, gmax, vol RMS <= 1e-6 over the pixels finite in both;
  validity masks (ux finite) agree on >= 99.9 %. *robust* (real full frames and sequences), over the
  pixels finite in both: ux, uy RMS of the 99.9 % smallest |diff| <= 1e-5 and p99 |diff| <= 1e-5; fraction
  with |du| > 1e-4 <= 0.5 %; conf RMS <= 2e-5; strain RMS of the 99.9 % smallest <= 2e-6; mask agreement
  >= 99.9 %. Reason: on the real full frame about 40 bistable spots at the rims of invalid regions flip
  between two self-consistent solutions under any change that is not bit-identical (the CPU itself, with
  sub-ulp input noise, misses the plain limits as much as the GPU does). The GPU output is deterministic
  (no floating-point atomics, fixed summation order). The shaders are compiled with `MTLMathMode::Safe`
  (no fast-math).
- **Limits**: macOS only. Only the fine stage runs on the GPU; the coarse estimate, the strain and the
  mask and fill of the pre-warp run on the CPU (its blur, `blur_field`, runs on the GPU). Windows or tiles whose work buffers exceed the device's `maxBufferLength` or
  recommended working set are refused before any window is refined (for `pair` before tracking starts;
  for `seq` when frame 1, the first one tracked, arrives). On full frames the GPU
  run uses about 1.9x the CPU's memory with one lane (max RSS 3.14 GiB against 1.67 GiB on 6240x4160, quiet; 3.84 GiB / 2.28 GiB before the memory work) and about 2.6x with the default two lanes (4.0 to 4.48 GiB).
- **Tile prefetch**: `OPTFLOW_TILE_PREFETCH` applies to both `--device cpu` and `--device gpu`. On full frames the start of tile k+1 (window, crop, the GPU reference spectrum) is prepared on one
  thread while tile k is refined. It is on by default, changes no bit of the output and gains about 1 to 2 %;
  `OPTFLOW_TILE_PREFETCH=0` turns it off (it keeps one prepared tile in memory; max RSS stays under 4.1 GiB on the full frame). On the CPU it adds a thread, and the `coarse`/`fine` times of `timing.json` are then attributed with the overlap.
- **GPU lanes**: `OPTFLOW_GPU_LANES` (1 or 2, default 2; `--device gpu` with the tile prefetch on) is how many tiles the GPU engine refines at once, each on its own thread with its own buffers and banks (one device, queue and buffer pool), so one tile's CPU work between its command buffers (the pre-warp rule, the copies, the launches) overlaps the other's GPU work. The cores are joined in the order of one at a time and the output is the same bit for bit; `OPTFLOW_GPU_LANES=1` restores one tile at a time, and so does the memory profile of the examples (its probe attributes the memory between one tile's marks to that tile). On the quiet machine two lanes made the full frame 7.4 to 7.8 % faster (25.1-25.3 s -> 23.3-23.4 s) and the 720p sequence 5.3 to 6.2 % faster; they hold a second tile's GPU working set (full frame max RSS 3.1-3.3 -> 4.0-4.5 GiB, sequence 1.40 -> 1.58 GiB).
- **Benchmark**: `bench_gpu` times a speckle pair on both engines (`--sizes`, `--runs`, `--warmup`,
  `--label`, `--shift`, `--max-disp` as in `bench_pair`; `--gpu-only`) and prints a `RESULT label=...
  device=gpu|cpu n=... total=... <stage>=...` line per run.

  ```sh
  cargo run --release -p optflow-gpu --example bench_gpu -- --sizes 512 --runs 3 --compare-cpu
  cargo run --release -p optflow-gpu --example bench_gpu -- --sizes 1024 --runs 1 --gpu-only --mem
  ```

  `--compare-cpu` also compares the GPU result with the CPU result (default threads) with the measures
  above and prints a `GATE` line (exit status 1 on FAIL). `--mem` prints the largest Metal
  `currentAllocatedSize` and the peak footprint of the process. `OPTFLOW_GPU_PROFILE=1` and
  `OPTFLOW_GPU_KERNELS=1` add the GPU time per stage and per kernel (they add waits, so the wall
  time changes). They act on `optflow --device gpu` too, which then runs slower. The synthetic benchmark of the Python side takes the device too:
  `python/bench/run_synth.py --impl rust --device gpu` (label `rust-gpu`).

- **Speed** (GPU against CPU with 10 threads; Apple M4, 2026-10-27 and 2026-10-30):

| case | CPU [s] | GPU [s] | ratio | max memory (CPU / GPU) | load average before -> after (CPU; GPU) |
|---|---|---|---|---|---|
| 512^2 pair (`bench_gpu`, min of 3), 2026-10-27 | 0.393 | 0.257 | 1.53 | process peak footprint 626 MiB, Metal allocation 294 MiB (GPU) | 2.34 -> 3.27 |
| full frame 6240x4160, 35 tiles (CLI pair, wall), 2026-10-27 (before the speed rounds) | 54.35 | 41.64 | 1.30 | 2.28 GiB / 3.86 GiB (max RSS) | 4.05 -> 9.43; 4.78 -> 4.49 |
| 1280x720 sequence, 20 steps (CLI seq, wall), 2026-10-27 (before the speed rounds) | 33.28 | 19.90 | 1.67 | 0.79 GiB / 1.41 GiB (max RSS) | 1.71 -> 6.48; 4.32 -> 3.46 |
| full frame, quiet machine, after the speed rounds | 54.35 (as above) | 26.61 | 2.04 | 2.28 GiB / 3.84 GiB | 3.31 -> 3.13 (GPU) |
| 1280x720 sequence, 20 steps, quiet machine, after the speed rounds | 33.28 (as above) | 16.57 | 2.01 | 0.79 GiB / 1.41 GiB (before) | 3.31 -> 3.13 (GPU) |
| full frame, quiet machine, after the memory work | 52.84 | 25.47 | 2.07 | 1.67 GiB / 3.14 GiB | 1.4 -> 9.0 |
| 1280x720 sequence, 20 steps, quiet machine, after the memory work | 31.89 | 16.23 | 1.96 | 0.77 GiB / 1.40 GiB | 1.4 -> 6.5 |
| 512^2 pair, quiet machine, after the speed rounds | 0.393 (as above) | 0.211 | 1.86 | | |
| full frame, CPU and GPU alternating, loaded machine (reference only) | 82.62, 75.18 | 33.15, 27.46 | 2.49, 2.74 | | GPU 3.66 -> 6.64, 3.49 -> 4.82; CPU 6.64 -> 14.69, 4.82 -> 12.18 |
| 1280x720 sequence, alternating, loaded machine (reference only) | 46.11, 44.06 | 18.65, 17.99 | 2.47, 2.45 | | GPU 3.86 -> 4.28, 3.55 -> 3.28; CPU 4.28 -> 10.87, 3.28 -> 9.19 |

The first three rows were measured on 2026-10-27 at a4d9894 with several other jobs on the machine (background
load about 4-5); the ratio is the CPU time over the GPU time. The quiet-machine rows use the GPU times of the
last quiet gate run (load 3.1 to 3.3, commit f98b990) against the CPU times of the first three rows. The
alternating rows are CPU and GPU runs back to back on a machine that other jobs kept at load 4 to 14: the ratio
is more robust to that load than the absolute times, but the CPU side suffers more from it (reference only).
The speed rounds (GPU speed work) took
the full frame from 41.6 to 26.6 s (-36 %), the sequence from 19.9 to 16.6 s (-17 %) and 512^2 from 0.257 to
0.211 s (-18 %). The rounds were stopped when each further step gained less than 5 %; a hybrid CPU+GPU tile split
and a pipelined sequence were tried and not kept (spec rows 10 to 12). Full-frame GPU outputs of two runs are
bit-identical.

  The synthetic suite gives the same verdicts as the CPU (`rust-gpu`: 28 OK / 0 NG / 13 record-only,
  41 of 41 cases as `rust`). Accuracy against the CPU passes the plain tier at 512^2 and on sequence
  frames 1 and 10 (frame 20 only misses the all-pixel conf RMS: two pixels flip across the match-border
  threshold). The real full frame fails the plain tier (ux RMS 5.4e-5 px, strain RMS 8e-6 to 1.3e-5;
  median |dux| 0, 99 % below 2e-6 px, the RMS is carried by 0.08 % of the pixels) but passes the robust
  tier (ux, uy 99.9 % RMS 1.35e-6 / 2.34e-6, p99 1.9e-6 / 4.3e-6, strain 99.9 % RMS <= 7.3e-7, conf 9.3e-6,
  mask agreement 0.9999985), as do sequence frames 1, 10 and 20; see the GPU spec, sections 5 and 8.

## Verification: `scripts/check.sh`

```sh
rust/scripts/check.sh --quick   # fmt, clippy -D warnings, cargo test --release
rust/scripts/check.sh           # also: fma build (clippy + tests), gpu crate and feature, Kani, Miri
```

- Tools that are not installed are skipped with a `skipped: ...` line; a tool that is installed and fails
  fails the script.
- The **gpu** stage (clippy and tests of `optflow-gpu` and of `optflow-cli --features gpu`) runs only on
  macOS with a Metal device (`system_profiler SPDisplaysDataType` mentions Metal); elsewhere the script
  prints `skipped: optflow-gpu (needs macOS with a Metal device)`. `--quick` does not run it. The
  `--workspace` test of the fma stage excludes `optflow-gpu` (its tests need a Metal device).
- **Miri** (`rustup toolchain install nightly --component miri`) runs the `optflow-core` library tests,
  about 30 minutes. The flags (tree borrows, `-Zmiri-no-extra-rounding-error`, `-Zmiri-ignore-leaks`, few
  proptest cases) are explained in the script; a caller's `MIRIFLAGS` replaces all of them. Tests that
  Miri cannot finish are `#[cfg_attr(miri, ignore)]`, each with its reason.
- **Kani** (`cargo install --locked kani-verifier && cargo kani setup`) runs **one harness at a time
  through a memory-cap wrapper** with a 6 GB cap: `MEMCAP_GB=6 MEMCAP_TAG=kani $MEMCAP_WRAPPER cargo kani
  --harness <name>`. `MEMCAP_WRAPPER` is an executable of yours that runs its arguments under a memory limit;
  it has no default. The harness names are those after `#[kani::proof]` in `crates/optflow-core/src`.
  Never run a bare `cargo kani`: one harness once grew to 87 GB. Without the wrapper the script prints
  `skipped: kani needs a memory-cap wrapper (...)` and does not run Kani at all.
- `scripts/check_test.sh` tests `check.sh` in a few seconds with a fake `cargo` and a fake wrapper on
  `PATH`: tools present or missing, a failing tool, `--quick`, wrapper present (each harness runs alone
  through it) and missing.

## Tests and fixtures

`cd rust && cargo test --release` runs unit tests, proptest properties and the parity tests
(`crates/optflow-core/tests/parity_*.rs`, `tile_equivalence.rs`, ...) that compare with the Python
reference. Tests that need a real `ffmpeg` skip themselves when it is missing; those that use a fake
`ffmpeg` shell script are Unix only.

Fixtures are generated, never edited by hand:

| fixture | generator | notes |
|---|---|---|
| `crates/optflow-core/tests/fixtures/` (`*.npy`, `meta.json`, `config_cases.json`) | `python/bench/make_parity_fixtures.py` (run with `python/.venv`; re-run whenever `python/optflow_ref` changes) | float32; **budget 8 MB for the whole directory** (raised from 5 MB for the hybrid-sequence fixtures `hyb*`), now 7.1 MB |
| `crates/optflow-core/tests/fixtures/xdic/` (`xd_*.npy` u16 images and f32/bool maps, `xdic_meta.json`) | `python/bench/make_xdic_fixtures.py` (re-run whenever `xdic.py` or `track.py` changes) | crack_model xdic; own directory, 4.3 MB |
| `crates/optflow-core/tests/atan2_mpmath.txt` | `scripts/gen_atan2_reference.py` (needs mpmath) | correctly rounded `atan2` values; outside the 5 MB budget |
| `crates/optflow-core/src/config/unicode_cn.rs` | `scripts/gen_unicode_cn.py` | unassigned code points of the reference Python's Unicode database |
| `crates/optflow-cli/tests/fixtures/images/`, `decode_expected.json` | `make_decode_fixtures.py` there | what Python's `read_image` returns |
| `crates/optflow-cli/tests/fixtures/polygon_cases.json` | `make_polygon_fixtures.py` there | Pillow's polygon masks |

Timing and memory: `cargo run --release -p optflow-core --example bench_pair -- --sizes 512` (the
options are in its header comment); the Python benchmark harness (`python/bench`; `python -m
bench.report` from `python/`) compares Python and Rust and writes `results/REPORT.md`.

## Platform notes: Windows

Developed on macOS (Apple silicon). The Windows build is compiled and tested in CI (GitHub Actions); the GUI has not been verified on real Windows hardware. Known hazards for Windows:

- Path and glob handling: `\` is a separator only when built for Windows (`cfg(windows)`, unit-tested
  with both separator sets on macOS, never run on Windows); case-insensitive matching as in Python on
  Windows is not implemented.
- Text outputs (`timing.json`, `config.json`, `crop.json`) are written with LF, Python's text mode on
  Windows writes CRLF, so byte-for-byte comparisons with Python output differ.
- Line endings in a checkout: the root `.gitattributes` marks binaries and keeps the fixture JSON, the
  `atan2` table and the shell scripts at LF; other text files still follow `core.autocrlf`.
- The benchmark harness (`python/bench`) is Unix only: `/usr/bin/time -l`, `os.getloadavg`, `resource`.
- Several CLI tests are `#[cfg(unix)]` (fake ffmpeg shell scripts, non-UTF-8 arguments, closed stdout).
- ffmpeg is started as `ffmpeg` / `ffprobe`, which finds `.exe` files only: `.cmd`/`.bat` shims of
  package managers are not found.
- Numerics differ in the last bits: the platform's `exp`, `sin`, `cos`, `ln` and the FFT's SIMD path are
  not those of macOS. The parity tolerances are far larger, but nothing checks bit equality across
  platforms (`atan2` is portable by design).
- Memory figures in the reports are macOS footprints with its allocator; Windows' heap behaves
  differently.
- JSON files saved with a UTF-8 byte order mark (Notepad, PowerShell 5) are rejected (`invalid config
  JSON: expected value at line 1 column 1`); save them without BOM.
