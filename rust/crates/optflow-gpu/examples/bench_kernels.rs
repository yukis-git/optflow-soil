//! GPU time of the fine stage's kernels on a 512^2 window (synthetic data), each repeated in one
//! command buffer so that the GPU clock and other users of the GPU disturb it less than the
//! single dispatches of a pipeline run.
//!
//!     cargo run --release -p optflow-gpu --example bench_kernels -- [--size 512] [--reps 10]
//!         [--batches 5]
//!
//! Prints, per kernel group, the minimum over the batches of the GPU time per repetition.
//!
//! On other systems the example builds and does nothing, so that the crate's targets build on
//! every OS.

#[cfg(target_os = "macos")]
#[path = "bench_kernels/mac.rs"]
mod mac;

#[cfg(target_os = "macos")]
fn main() {
    mac::main()
}

#[cfg(not(target_os = "macos"))]
fn main() {}
