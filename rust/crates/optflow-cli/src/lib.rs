//! `optflow`: command-line front end of `optflow-core` (the Rust port of `python -m optflow_ref`).
#![forbid(unsafe_code)]

pub mod args;
pub mod checkpoint;
pub mod crop;
pub mod glob;
pub mod io;
pub mod mask;
pub mod polygon;
pub mod run;
pub mod settings;
pub mod tavg;
pub mod video;

use std::fmt;
use std::io::Write;
use std::process::ExitCode;

/// Status of a run that ends because stdout was closed (`optflow ... | head -1`): the one a shell
/// reports for a process killed by SIGPIPE (128 + 13).
const EXIT_STDOUT_CLOSED: u8 = 141;

/// Stdout is a closed pipe: the run stops without a message (see [`say`]).
#[derive(Debug)]
pub struct StdoutClosed;

impl fmt::Display for StdoutClosed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("stdout is closed")
    }
}

impl std::error::Error for StdoutClosed {}

/// Writes `text` to stdout and flushes. `print!` panics when the write fails, which a closed pipe
/// makes it do; here that case is the [`StdoutClosed`] error and other failures get a message.
pub fn say(text: &str) -> anyhow::Result<()> {
    let mut out = std::io::stdout().lock();
    out.write_all(text.as_bytes())
        .and_then(|()| out.flush())
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::BrokenPipe {
                anyhow::Error::new(StdoutClosed)
            } else {
                anyhow::anyhow!("cannot write to stdout: {e}")
            }
        })
}

/// Runs the tool on the arguments after the program name. Every user-facing failure prints
/// `error: <message>` to stderr and returns status 2; a closed stdout ends the run quietly with
/// status 141.
pub fn main_with(args: &[String]) -> ExitCode {
    let result = match args::parse(args) {
        Ok(args::Parsed::Help) => say(args::USAGE),
        Ok(args::Parsed::Run(cli)) => run::run(&cli),
        Err(msg) => {
            eprintln!("{}", args::USAGE.lines().next().unwrap_or_default());
            eprintln!("error: {msg}");
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) if e.downcast_ref::<StdoutClosed>().is_some() => ExitCode::from(EXIT_STDOUT_CLOSED),
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(2)
        }
    }
}

/// A fresh empty directory under the system temp directory for one unit test (the test removes it
/// at its end); the process id keeps concurrent runs apart.
#[cfg(test)]
pub(crate) fn unit_scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("optflow-cli-unit-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The error of a closed stdout names itself (`main_with` swallows it, but anything that
    /// prints the error chain would show it).
    #[test]
    fn stdout_closed_displays_its_cause() {
        assert_eq!(StdoutClosed.to_string(), "stdout is closed");
        let e = anyhow::Error::new(StdoutClosed);
        assert!(e.downcast_ref::<StdoutClosed>().is_some());
        assert_eq!(format!("{e:#}"), "stdout is closed");
    }
}
