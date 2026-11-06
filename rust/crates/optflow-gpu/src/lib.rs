#![cfg(target_os = "macos")]
//! The Metal backend of optflow: device, shader library, buffer pool and batched dispatch.

pub mod device;
pub mod engine;
pub mod fft;
pub mod kernels;

pub use engine::GpuEngine;

use optflow_core::engine::EngineError;

#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("no Metal device available for --device gpu")]
    NoDevice,
    #[error("Metal shader compilation failed: {0}")]
    Compile(String),
    #[error("GPU command failed: {0}")]
    Command(String),
    #[error(
        "the GPU needs {need} bytes for a {w}x{h} window, more than the device allows ({limit})"
    )]
    TooLarge {
        need: u64,
        limit: u64,
        w: usize,
        h: usize,
    },
}

impl From<GpuError> for EngineError {
    fn from(e: GpuError) -> Self {
        EngineError(e.to_string())
    }
}

/// Name of the default Metal device.
pub fn device_name() -> Result<String, GpuError> {
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};
    let dev = MTLCreateSystemDefaultDevice().ok_or(GpuError::NoDevice)?;
    Ok(dev.name().to_string())
}
