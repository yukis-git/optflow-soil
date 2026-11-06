//! The usage errors of the `optflow` binary against those of the Python reference's argparse, on
//! command lines that fail in the parser: the same message and the same status. Every line here
//! must fail (or ask for help) in the parser: none may run the tool, which would write files into
//! `python/`.
//!
//! The test needs a Python with the reference installed, `$OPTFLOW_PYTHON` (`rust/scripts/check.sh`
//! sets it to the repository's `python/.venv/bin/python` when that exists). Without one it prints
//! a "skipped" line (visible with `-- --nocapture`, and in `check.sh`'s own note) and passes.

use std::path::PathBuf;
use std::process::Command;

/// The Python and the directory to run it in, or the reason there is none.
fn python() -> Result<(PathBuf, PathBuf), String> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../python");
    let Some(exe) = std::env::var_os("OPTFLOW_PYTHON").map(PathBuf::from) else {
        return Err("OPTFLOW_PYTHON not set".to_string());
    };
    let ok = Command::new(&exe)
        .current_dir(&dir)
        .args(["-c", "import optflow_ref.__main__"])
        .output()
        .is_ok_and(|o| o.status.success());
    if ok {
        Ok((exe, dir))
    } else {
        Err(format!(
            "{} cannot import optflow_ref from {}",
            exe.display(),
            dir.display()
        ))
    }
}

/// The message of the last line of `stderr` (`prefix: error: message`) and the exit status. The
/// quotes around the names in "(choose from ...)" depend on the Python version: left out.
fn message(stderr: &[u8], status: Option<i32>) -> (Option<String>, Option<i32>) {
    let text = String::from_utf8_lossy(stderr);
    let msg = text
        .lines()
        .last()
        .and_then(|l| l.split_once("error: "))
        .map(|(_, m)| match m.split_once("(choose from ") {
            Some((head, list)) => format!("{head}(choose from {}", list.replace('\'', "")),
            None => m.to_string(),
        });
    (msg, status)
}

/// Command lines that fail in the parser or ask for help (never one that runs).
const LINES: &[&[&str]] = &[
    // no command, a wrong one, options before it
    &[],
    &[""],
    &["run"],
    &["-1"],
    &["-x"],
    &["-xy"],
    &["--bogus"],
    &["--bogus", "pair"],
    &["-x", "pair"],
    &["--ref", "a", "pair"],
    &["--bogus", "pair", "--ref", "a", "--def", "b", "--out", "o"],
    &["-x", "-y", "seq", "--frames", "d", "--out", "o", "--zzz"],
    // `--` before the command: the next token is the command; `--` after it: the rest is extra
    &["--"],
    &["-x", "--"],
    &["-x", "-y", "--"],
    &["--", "-x"],
    &["--", "-h"],
    &["--", "--"],
    &["--", "--", "seq"],
    &["--", "--h=1"],
    &["-x", "--", "-y"],
    &["--", "seq", "--out", "o"],
    &["--", "pair"],
    &["--", "pair", "-x"],
    &["-x", "--", "seq", "--out", "o"],
    &["-x", "--", "seq", "--frames", "d", "--out", "o", "--", "y"],
    &["--", "seq", "--frames", "d", "--out", "o", "--", "y"],
    &["pair", "--", "--ref", "a"],
    &["pair", "--ref", "--", "--def", "b", "--out", "o"],
    &["seq", "--", "--out", "o"],
    &["seq", "--frames", "d", "--", "--out", "o"],
    &["seq", "--out", "o", "--"],
    &["seq", "--out", "o", "--", "x"],
    &["seq", "--frames", "d", "--out", "o", "--"],
    &["seq", "--frames", "d", "--out", "o", "--", "x"],
    &["seq", "--frames", "d", "--out", "o", "--", "--", "x"],
    &["seq", "--frames", "d", "--out", "o", "x", "--", "y"],
    &["seq", "--frames", "d", "--out", "o", "-x", "--", "y"],
    &["seq", "--frames", "d", "--out", "o", "--start", "--"],
    &["seq", "--frames", "d", "--out", "o", "--start", "--", "1"],
    &["seq", "--frames", "d", "--out", "o", "--start=--"],
    // options of the command
    &["pair", "-x"],
    &["pair", "-1"],
    &["pair", "--ref", "--def", "b", "--out", "o"],
    &["pair", "--ref", "a", "--def", "b", "--out", "o", "--bogus"],
    &["pair", "--ref", "a", "--def", "b", "--out", "o", "-x"],
    &["pair", "--ref", "a", "--def", "b", "--out", "o", "extra"],
    &["seq", "--frames", "d", "--start", "-1.", "--out", "o"],
    &["seq", "--frames", "d", "--start", "x", "--out", "o"],
    &["seq", "--frames", "d", "--out", "o", "--crop", "maybe"],
    &["seq", "--frames", "d", "--out", "o", "--s", "1"],
    &["seq", "--frames", "d", "--out", "o", "--png=1"],
    &["seq", "--out", "o"],
    // a blank in the value of an attached option (`--out=my dir`) or of a help token
    &["seq", "--frames=a b", "--out", "o", "--zzz"],
    &["seq", "--fr=a b", "--out", "o", "--zzz"],
    &["seq", "--frames=a b", "--out=o p", "--zzz"],
    &["seq", "--frames a", "--out", "o", "--zzz"],
    &["seq", "--frames a=b", "--out", "o", "--zzz"],
    &["seq", "--png=a b", "--frames", "d", "--out", "o"],
    &["seq", "--start=1 2", "--frames", "d", "--out", "o"],
    &["seq", "--frames", "d", "--out", "o", "--scale=2 x"],
    &["seq", "--frames", "d", "--out", "o", "--crop=a b"],
    &["seq", "--out", "o", "--s=1"],
    &["seq", "--out", "o", "--s=a b"],
    &["seq", "--out", "o", "--s a"],
    &["seq", "--st=1 2", "--frames", "d", "--out", "o"],
    &["pair", "--r=x y", "--def", "d", "--out", "o"],
    &["seq", "--ref=a b", "--frames", "d", "--out", "o"],
    &["--he=a b"],
    &["-h=a b"],
    &["--h=a b", "seq"],
    &["-h=1 2", "seq"],
    &["--help=1 2"],
    &["--bogus=a b"],
    &["-x y"],
    &["-x=a b", "seq"],
    &["--help x=1"],
    &["--hel x=1"],
    &["seq", "--out", "o", "--help x"],
    &["seq", "--out", "o", "--help x=1"],
    &["-hx y"],
    &["-h x"],
    &["-hello world"],
    &["-hx=1 2"],
    &["seq", "--out", "o", "-h x"],
    // help, and help given a value
    &["-h"],
    &["-x", "-h"],
    &["--he"],
    &["pair", "-h"],
    &["seq", "--out", "o", "--help"],
    &["-hx"],
    &["-hh"],
    &["-h1"],
    &["-hx=1"],
    &["-hx", "--h=1"],
    &["-h", "--", "seq"],
    &["pair", "-hx"],
    &["seq", "--out", "o", "-hx"],
    &["--h=1"],
    &["--help=1"],
    &["--he=1"],
    &["--help="],
    &["--h="],
    &["-h=1"],
    &["-h="],
    &["-h=x=y"],
    &["-x", "--h=1"],
    &["--h=1", "-x"],
    &["--h=1", "-hx"],
    &["--h=1", "--"],
    &["--help=", "seq"],
    &["pair", "--h=1"],
    &["pair", "-h=1"],
    &["pair", "--ref", "a", "--def", "b", "--out", "o", "--h=2"],
    &["seq", "--out", "o", "--help=1"],
    &["seq", "--out", "o", "--he=1"],
    &["seq", "--out", "o", "-h="],
];

#[test]
fn usage_errors_match_argparse() {
    let (exe, dir) = match python() {
        Ok(found) => found,
        Err(why) => {
            eprintln!("skipped: {why}: the argparse parity test did not run");
            return;
        }
    };
    for args in LINES {
        let rust = Command::new(env!("CARGO_BIN_EXE_optflow"))
            .args(*args)
            .output()
            .unwrap();
        let py = Command::new(&exe)
            .current_dir(&dir)
            .args(["-m", "optflow_ref"])
            .args(*args)
            .output()
            .unwrap();
        let want = message(&py.stderr, py.status.code());
        let got = message(&rust.stderr, rust.status.code());
        assert_eq!(got, want, "arguments {args:?}");
        if want.1 == Some(0) {
            // help: the usage goes to stdout
            assert!(String::from_utf8_lossy(&rust.stdout).starts_with("usage:"));
            assert!(String::from_utf8_lossy(&py.stdout).starts_with("usage:"));
        }
    }
}
