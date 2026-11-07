//! The choice of the device of the fine stage: what this build offers and what the machine has.

use optflow_cli::args::Device;

/// Whether this build can offer the GPU at all: a macOS build with the `gpu` feature. Elsewhere
/// (Windows, a plain build) the window has no such choice and every run is on the CPU.
pub const BUILT: bool = cfg!(all(feature = "gpu", target_os = "macos"));

/// What the machine offers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Gpu {
    /// The look for a Metal device has not finished (it runs on a thread of its own so that the
    /// window opens at once).
    Checking,
    /// The name of the device.
    Ready(String),
    /// Why not, in Japanese.
    Unavailable(String),
}

impl Gpu {
    pub fn usable(&self) -> bool {
        matches!(self, Gpu::Ready(_))
    }
}

/// The result of looking for a Metal device (see [`optflow_cli::run::gpu_probe`]), as a state.
pub fn probe() -> Gpu {
    match optflow_cli::run::gpu_probe() {
        Ok(name) => Gpu::Ready(name),
        Err(e) => Gpu::Unavailable(unavailable_text(&e)),
    }
}

/// The reason the GPU cannot be used, in Japanese; the original text follows when it is not the
/// known one.
pub fn unavailable_text(e: &str) -> String {
    if e.contains("no Metal device") {
        "この Mac では Metal 対応の GPU が見つかりませんでした。CPU で解析します。".into()
    } else if e.contains("no GPU backend") {
        "このアプリには GPU の機能が入っていません。CPU で解析します。".into()
    } else {
        format!("GPU を使えません。CPU で解析します。({e})")
    }
}

/// `CPU` or `GPU` for the notes.
pub fn label(d: Device) -> &'static str {
    match d {
        Device::Cpu => "CPU",
        Device::Gpu => "GPU",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_reasons_are_japanese_and_keep_the_unknown_ones() {
        assert!(unavailable_text("no Metal device available for --device gpu").contains("Metal"));
        assert!(unavailable_text("this build has no GPU backend (x)").contains("入っていません"));
        assert!(unavailable_text("weird").contains("(weird)"));
    }

    #[test]
    fn a_build_without_the_feature_reports_no_gpu() {
        if !BUILT {
            assert!(matches!(probe(), Gpu::Unavailable(_)));
        }
    }
}
