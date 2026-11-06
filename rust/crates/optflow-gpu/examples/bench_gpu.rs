//! Speed of the GPU fine stage against the CPU on the speckle pair of the tests.
//!
//!     cargo run --release -p optflow-gpu --example bench_gpu -- [--sizes 512,1024] [--runs 3]
//!         [--warmup 1] [--label NAME] [--shift TX,TY] [--max-disp D] [--gpu-only]
//!         [--compare-cpu] [--mem]
//!
//! (`--size N` is `--sizes N`.) Every run also prints one `RESULT label=NAME device=gpu|cpu n=N
//! total=S <stage>=S ...` line (wall seconds per pipeline stage), like `bench_pair`.
//!
//! `--compare-cpu` compares the last GPU result with the last CPU result (CPU at the default
//! thread count) on the same pair, with the measures of the GPU spec section 5: ux, uy RMS of the
//! difference over the pixels finite in both, conf RMS over all pixels, the strain fields' RMS
//! over the pixels finite in both, the agreement of the validity masks (ux finite); and prints a
//! `GATE` line (pass or FAIL, and the process exits with status 1 on FAIL). The largest
//! differences are printed too, and not judged.
//!
//! `--mem` prints the largest Metal `currentAllocatedSize` seen (sampled every millisecond
//! during the GPU runs) and the peak physical footprint of the process (both engines' runs, so
//! combine it with `--gpu-only` for the GPU alone).
//!
//! Each run times `track_pair` and, inside it, the fine stage (the engine's `refine_raw`) on the
//! GPU and then on the CPU, back to back; the minimum of the runs is printed. `RAYON_NUM_THREADS`
//! sets the CPU's threads. With `OPTFLOW_GPU_PROFILE=1` the stages run in command buffers of their
//! own and their GPU time is printed; with `OPTFLOW_GPU_KERNELS=1` every dispatch does, and the
//! GPU time per kernel is printed. Both change the wall time (more waits). The GPU times are
//! the minimum over the runs, per stage or kernel (the GPU is shared and its clock varies), as
//! is the wall time per pipeline stage (`optflow_core::prof`).
//!
//! On other systems the example builds and does nothing, so that the crate's targets build on
//! every OS.

#[cfg(target_os = "macos")]
#[path = "bench_gpu/mac.rs"]
mod mac;

#[cfg(target_os = "macos")]
fn main() {
    mac::main()
}

#[cfg(not(target_os = "macos"))]
fn main() {}
