//! Phase-based dense displacement and strain measurement (port of python/optflow_ref).
#![forbid(unsafe_code)]

mod atan2;
#[doc(hidden)]
pub mod checkpoint;
pub mod compare;
pub mod config;
pub mod edt;
pub mod engine;
pub mod fft;
pub mod gabor;
#[doc(hidden)]
pub mod gpu_model;
#[doc(hidden)]
pub mod gpu_support;
pub mod image;
pub mod interp;
// per-iteration trace of the fine stage (OPTFLOW_ITER_TRACE); not part of the API
#[doc(hidden)]
pub mod iter_trace;
mod madd;
pub mod npy;
pub mod phaseflow;
pub mod poc;
mod prewarp;
// stage timers for the benchmarks (examples/bench_pair.rs); not part of the API
#[doc(hidden)]
pub mod prof;
mod select;
pub mod strain;
pub mod tile;
pub mod track;
pub mod xdic;
