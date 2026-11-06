#!/usr/bin/env bash
# Full local verification. Usage: rust/scripts/check.sh [--quick]
# Without --quick it also builds, lints and tests the non-default `fma` feature and (on macOS with a
# Metal device) the `gpu` crate and feature, and runs Kani and Miri; a tool that is not installed is skipped with a message, one that is installed and fails
# fails the script. (rust/scripts/check_test.sh tests this.)
# Kani runs one harness at a time through a memory-cap wrapper (MEMCAP_WRAPPER: an executable that runs
# its arguments under a memory limit, reading MEMCAP_GB and MEMCAP_TAG) with a 6 GB cap, because one
# harness once used 87 GB. Without the wrapper Kani is skipped, never run unguarded.
set -euo pipefail
cd "$(dirname "$0")/.."
# The CLI's argparse parity test (crates/optflow-cli/tests/argparse_parity.rs) compares the usage
# errors of the binary with the Python reference and skips itself without a Python that has it:
# use the repository's venv when there is one (OPTFLOW_VENV_PYTHON overrides its path; a given
# OPTFLOW_PYTHON wins), and say so when the test will be skipped.
venv_python="${OPTFLOW_VENV_PYTHON:-$PWD/../python/.venv/bin/python}"
if [[ -z "${OPTFLOW_PYTHON:-}" && -x "$venv_python" ]]; then
  export OPTFLOW_PYTHON="$venv_python"
fi
if [[ -z "${OPTFLOW_PYTHON:-}" ]]; then
  echo "skipped: OPTFLOW_PYTHON not set and no $venv_python: the argparse parity test of the CLI will not run"
fi
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --release
if [[ "${1:-}" != "--quick" ]]; then
  # `fma` is off by default and changes results (about 1e-7 px), so the runs above do not cover
  # it: lint and test it separately.
  cargo clippy --all-targets --features optflow-cli/fma -- -D warnings
  # optflow-gpu is a workspace member but builds only on macOS and needs a Metal device: the gpu
  # stage below tests it.
  cargo test --release --workspace --exclude optflow-gpu --features optflow-cli/fma
  # The Metal backend builds only on macOS and runs only with a Metal device.
  if [[ "$(uname)" == "Darwin" ]] && system_profiler SPDisplaysDataType 2>/dev/null | grep -q "Metal"; then
    cargo clippy -p optflow-gpu --all-targets -- -D warnings
    cargo clippy -p optflow-cli --all-targets --features gpu -- -D warnings
    cargo test --release -p optflow-gpu
    cargo test --release -p optflow-cli --features gpu
  else
    echo "skipped: optflow-gpu (needs macOS with a Metal device)"
  fi
  MEMCAP_WRAPPER="${MEMCAP_WRAPPER:-}"
  if ! cargo kani --version >/dev/null 2>&1; then
    echo "skipped: cargo kani not installed (cargo install --locked kani-verifier && cargo kani setup)"
  elif [[ ! -x "$MEMCAP_WRAPPER" ]]; then
    echo "skipped: kani needs a memory-cap wrapper (MEMCAP_WRAPPER='$MEMCAP_WRAPPER' is not an executable file; set MEMCAP_WRAPPER=<path>)"
  else
    # The harness names: the function after each #[kani::proof] (other attributes, such as
    # #[kani::unwind(n)], may sit in between).
    harnesses="$(find crates/optflow-core/src -name '*.rs' -exec awk '
      /#\[kani::proof\]/ { proof = 1; next }
      proof && /^[[:space:]]*(pub[[:space:]]+)?fn[[:space:]]+[A-Za-z0-9_]+/ {
        sub(/^[[:space:]]*(pub[[:space:]]+)?fn[[:space:]]+/, ""); sub(/[^A-Za-z0-9_].*/, ""); print; proof = 0
      }' {} + | sort)"
    if [[ -z "$harnesses" ]]; then
      echo "error: no #[kani::proof] harness found under crates/optflow-core/src" >&2
      exit 1
    fi
    for h in $harnesses; do
      echo "kani: $h"
      (cd crates/optflow-core && MEMCAP_GB=6 MEMCAP_TAG=kani "$MEMCAP_WRAPPER" cargo kani --harness "$h")
    done
  fi
  if cargo +nightly miri --version >/dev/null 2>&1; then
    # Miri on the lib tests of optflow-core, about 30 minutes. The flags:
    #   tree-borrows: rayon's crossbeam-epoch trips Stacked Borrows (a known false positive)
    #   no-extra-rounding-error: Miri otherwise perturbs exp/log/... by a few ULP, which breaks
    #     exact comparisons (kernel symmetry, equal results for any thread count, log2 of 2^k)
    #   ignore-leaks: crossbeam-epoch keeps pointers cast to integers, which the leak check reports
    #   the PROPTEST variables: no regression files (no file access under Miri) and few cases
    # Tests that Miri cannot finish in reasonable time (full pipelines on images of 128 pixels and
    # more, each over 30 minutes) or cannot run exactly are #[cfg_attr(miri, ignore)], each with
    # its reason next to it.
    # A MIRIFLAGS set by the caller replaces all of these flags, the env-forward ones included.
    export PROPTEST_DISABLE_FAILURE_PERSISTENCE=1 PROPTEST_CASES=4
    flags="-Zmiri-tree-borrows -Zmiri-no-extra-rounding-error -Zmiri-ignore-leaks"
    flags="$flags -Zmiri-env-forward=PROPTEST_DISABLE_FAILURE_PERSISTENCE -Zmiri-env-forward=PROPTEST_CASES"
    MIRIFLAGS="${MIRIFLAGS:-$flags}" cargo +nightly miri test -p optflow-core --lib
  else
    echo "skipped: cargo miri not installed (rustup toolchain install nightly --component miri)"
  fi
fi
