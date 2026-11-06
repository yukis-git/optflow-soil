#!/usr/bin/env bash
# Tests of check.sh with a fake `cargo` and a fake memory-cap wrapper: a tool that is not installed is skipped
# with a message, a Miri, Kani or fma run that fails fails the script, Kani runs one harness at a
# time through the wrapper (and is skipped, never run unguarded, without it), --quick runs none of them.
# Usage: rust/scripts/check_test.sh
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

cat >"$work/cargo" <<'EOF'
#!/usr/bin/env bash
# FAKE_KANI / FAKE_MIRI: ok (default), missing or fail. FAKE_KANI=fail fails the harness named in
# FAKE_KANI_FAIL_ON (any harness if it is unset). FAKE_FMA=fail fails the fma runs.
echo "$*" >>"$FAKE_LOG"
# the Python that the CLI's argparse parity test would get
[[ "$*" == "test --release" ]] && echo "OPTFLOW_PYTHON=${OPTFLOW_PYTHON:-}" >>"$FAKE_ENV_LOG"
case "$*" in
  *"--features optflow-cli/fma"*)
    [[ "${FAKE_FMA:-ok}" == fail ]] && { echo "error: fma build failed" >&2; exit 1; } ;;
  "kani --version")
    [[ "${FAKE_KANI:-ok}" == missing ]] && { echo "error: no such command: \`kani\`" >&2; exit 101; }
    echo "cargo-kani 0.0" ;;
  "kani --harness "*)
    if [[ "${FAKE_KANI:-ok}" == fail && ( -z "${FAKE_KANI_FAIL_ON:-}" || "$*" == *"$FAKE_KANI_FAIL_ON" ) ]]; then
      echo "kani: verification failed" >&2; exit 1
    fi ;;
  "kani"*)
    echo "unexpected: cargo $*" >&2; exit 1 ;;
  "+nightly miri --version")
    [[ "${FAKE_MIRI:-ok}" == missing ]] && { echo "error: cargo-miri is not installed" >&2; exit 1; }
    echo "miri 0.0" ;;
  "+nightly miri test"*)
    [[ "${FAKE_MIRI:-ok}" == fail ]] && { echo "error: Undefined Behavior: fake" >&2; exit 1; } ;;
esac
exit 0
EOF
chmod +x "$work/cargo"

# The fake memory-cap wrapper records its environment and arguments, then runs the command like the real one.
cat >"$work/memcap.sh" <<'EOF'
#!/usr/bin/env bash
echo "MEMCAP_GB=${MEMCAP_GB:-} MEMCAP_TAG=${MEMCAP_TAG:-} $*" >>"$FAKE_MEMCAP_LOG"
exec "$@"
EOF
chmod +x "$work/memcap.sh"

# The number of Kani harnesses in the sources: check.sh must run every one of them.
harness_count="$(grep -rho '#\[kani::proof\]' "$here/../crates/optflow-core/src" | wc -l | tr -d ' ')"

failures=0
# run NAME EXPECTED_STATUS [--quick]  (FAKE_KANI / FAKE_MIRI / FAKE_FMA / MEMCAP_WRAPPER come from the environment)
run() {
  local name="$1" want="$2"
  shift 2
  : >"$work/log"
  : >"$work/memcap_log"
  : >"$work/env_log"
  FAKE_LOG="$work/log" FAKE_MEMCAP_LOG="$work/memcap_log" FAKE_ENV_LOG="$work/env_log" \
    PATH="${EXTRA_PATH:+$EXTRA_PATH:}$work:$PATH" \
    bash "$here/check.sh" "$@" >"$work/out" 2>&1
  local got=$?
  if [[ $got -ne $want ]]; then
    echo "FAIL $name: exit status $got, wanted $want"
    sed 's/^/    | /' "$work/out"
    failures=$((failures + 1))
    return 1
  fi
  return 0
}
expect_out() { # NAME PATTERN
  grep -q -- "$2" "$work/out" || { echo "FAIL $1: output lacks '$2'"; sed 's/^/    | /' "$work/out"; failures=$((failures + 1)); }
}
expect_ran() { # NAME PATTERN
  grep -q -- "$2" "$work/log" || { echo "FAIL $1: cargo was not called with '$2'"; failures=$((failures + 1)); }
}
expect_not_ran() { # NAME PATTERN
  grep -q -- "$2" "$work/log" && { echo "FAIL $1: cargo was called with '$2'"; failures=$((failures + 1)); }
}
expect_count() { # NAME FILE PATTERN WANTED
  local n
  n="$(grep -c -- "$3" "$2")"
  [[ "$n" -eq "$4" ]] || { echo "FAIL $1: '$3' occurs $n times in $(basename "$2"), wanted $4"; failures=$((failures + 1)); }
}
# Every harness ran once, alone and through the wrapper with the 6 GB cap, and Kani never ran as a whole.
expect_kani_per_harness() { # NAME
  expect_count "$1" "$work/log" '^kani --harness [a-z0-9_]*$' "$harness_count"
  expect_count "$1" "$work/memcap_log" \
    '^MEMCAP_GB=6 MEMCAP_TAG=kani cargo kani --harness [a-z0-9_]*$' "$harness_count"
  expect_ran "$1" "^kani --harness k6_grid_windows_inside_image$"
  expect_not_ran "$1" "^kani$"
}

if [[ "$harness_count" -lt 1 ]]; then
  echo "FAIL: no #[kani::proof] harness found in the sources"
  failures=$((failures + 1))
fi

FAKE_KANI=ok FAKE_MIRI=ok FAKE_FMA=ok MEMCAP_WRAPPER="$work/memcap.sh"
export FAKE_KANI FAKE_MIRI FAKE_FMA MEMCAP_WRAPPER

name="all tools present"
run "$name" 0 && {
  expect_ran "$name" "^clippy --all-targets --features optflow-cli/fma -- -D warnings$"
  expect_ran "$name" "^test --release --workspace --exclude optflow-gpu --features optflow-cli/fma$"
  expect_kani_per_harness "$name"
  expect_ran "$name" "^+nightly miri test -p optflow-core --lib$"
}

name="memory-cap wrapper missing: kani is skipped, not run unguarded, and miri still runs"
MEMCAP_WRAPPER="$work/no-such-memcap.sh" run "$name" 0 && {
  expect_out "$name" "skipped: kani needs a memory-cap wrapper (MEMCAP_WRAPPER='$work/no-such-memcap.sh"
  expect_not_ran "$name" "^kani --harness"
  expect_not_ran "$name" "^kani$"
  expect_ran "$name" "^+nightly miri test"
}

name="MEMCAP_WRAPPER unset: kani is skipped"
MEMCAP_WRAPPER="" run "$name" 0 && {
  expect_out "$name" "skipped: kani needs a memory-cap wrapper"
  expect_not_ran "$name" "^kani --harness"
}

name="kani missing is skipped and miri still runs"
FAKE_KANI=missing run "$name" 0 && {
  expect_out "$name" "skipped: cargo kani not installed"
  expect_not_ran "$name" "^kani --harness"
  expect_ran "$name" "^+nightly miri test"
}

name="miri missing is skipped"
FAKE_MIRI=missing run "$name" 0 && {
  expect_out "$name" "skipped: cargo miri not installed"
  expect_kani_per_harness "$name"
}

name="both missing"
FAKE_KANI=missing FAKE_MIRI=missing run "$name" 0 && {
  expect_out "$name" "skipped: cargo kani not installed"
  expect_out "$name" "skipped: cargo miri not installed"
}

name="a failing miri run fails the script"
FAKE_MIRI=fail run "$name" 1 && expect_out "$name" "Undefined Behavior"

name="a failing kani harness fails the script and stops the run"
FAKE_KANI=fail FAKE_KANI_FAIL_ON=k2_taps_in_bounds_for_any_coordinate run "$name" 1 && {
  expect_out "$name" "verification failed"
  expect_not_ran "$name" "miri test"
}

name="a failing fma run fails the script"
FAKE_FMA=fail run "$name" 1 && {
  expect_out "$name" "fma build failed"
  expect_not_ran "$name" "kani --harness"
  expect_not_ran "$name" "miri test"
}

name="--quick runs no fma build, kani or miri"
run "$name" 0 --quick && {
  expect_not_ran "$name" "fma"
  expect_not_ran "$name" "kani"
  expect_not_ran "$name" "miri"
}

# The Python of the argparse parity test: the repository's venv when it exists, a given one wins, and
# a missing one is said so. (OPTFLOW_VENV_PYTHON stands in for python/.venv/bin/python.)
printf '#!/bin/sh\n' >"$work/venv-python"
chmod +x "$work/venv-python"
expect_python() { # NAME WANTED
  grep -qx "OPTFLOW_PYTHON=$2" "$work/env_log" \
    || { echo "FAIL $1: cargo test got '$(cat "$work/env_log")', wanted OPTFLOW_PYTHON=$2"; failures=$((failures + 1)); }
}

name="OPTFLOW_PYTHON is the repository's venv Python when there is one"
OPTFLOW_PYTHON="" OPTFLOW_VENV_PYTHON="$work/venv-python" run "$name" 0 --quick && {
  expect_python "$name" "$work/venv-python"
  grep -q "skipped: OPTFLOW_PYTHON" "$work/out" && { echo "FAIL $name: said it skips"; failures=$((failures + 1)); }
}

name="a given OPTFLOW_PYTHON wins over the venv"
OPTFLOW_PYTHON=/given/python OPTFLOW_VENV_PYTHON="$work/venv-python" run "$name" 0 --quick && expect_python "$name" /given/python

name="no Python: the script says that the argparse parity test is skipped"
OPTFLOW_PYTHON="" OPTFLOW_VENV_PYTHON="$work/no-such-python" run "$name" 0 --quick && {
  expect_out "$name" "skipped: OPTFLOW_PYTHON not set and no $work/no-such-python"
  expect_python "$name" ""
}

# The gpu stage, with a fake uname and system_profiler (first in PATH) so that it does not depend on
# the machine: a Mac with Metal runs it, a machine without Metal or another OS skips it.
mkdir -p "$work/fake-mac" "$work/fake-nometal" "$work/fake-linux"
printf '#!/bin/sh\necho Darwin\n' >"$work/fake-mac/uname"
printf '#!/bin/sh\necho "Metal Support: Metal 4"\n' >"$work/fake-mac/system_profiler"
cp "$work/fake-mac/uname" "$work/fake-nometal/uname"
printf '#!/bin/sh\necho "Chipset Model: none"\n' >"$work/fake-nometal/system_profiler"
printf '#!/bin/sh\necho Linux\n' >"$work/fake-linux/uname"
cp "$work/fake-mac/system_profiler" "$work/fake-linux/system_profiler"
chmod +x "$work"/fake-*/*

name="gpu stage on a Mac with Metal"
EXTRA_PATH="$work/fake-mac" run "$name" 0 && {
  expect_ran "$name" "^clippy -p optflow-gpu --all-targets -- -D warnings$"
  expect_ran "$name" "^clippy -p optflow-cli --all-targets --features gpu -- -D warnings$"
  expect_ran "$name" "^test --release -p optflow-gpu$"
  expect_ran "$name" "^test --release -p optflow-cli --features gpu$"
}

name="gpu stage skipped without a Metal device"
EXTRA_PATH="$work/fake-nometal" run "$name" 0 && {
  expect_out "$name" "skipped: optflow-gpu"
  expect_not_ran "$name" "-p optflow-gpu"
  expect_not_ran "$name" "features gpu"
}

name="gpu stage skipped on another OS"
EXTRA_PATH="$work/fake-linux" run "$name" 0 && {
  expect_out "$name" "skipped: optflow-gpu"
  expect_not_ran "$name" "-p optflow-gpu"
  expect_not_ran "$name" "features gpu"
}

name="--quick runs no gpu stage"
EXTRA_PATH="$work/fake-mac" run "$name" 0 --quick && expect_not_ran "$name" "optflow-gpu"

# optflow-gpu is a workspace member: the --workspace run must exclude it on every machine.
name="the workspace test excludes optflow-gpu"
EXTRA_PATH="$work/fake-linux" run "$name" 0 && {
  expect_ran "$name" "--workspace"
  if grep -- "--workspace" "$work/log" | grep -qv -- "--exclude optflow-gpu"; then
    echo "FAIL $name: a --workspace run does not exclude optflow-gpu"
    failures=$((failures + 1))
  fi
}

if [[ $failures -eq 0 ]]; then
  echo "check_test.sh: all scenarios passed"
else
  echo "check_test.sh: $failures failure(s)"
  exit 1
fi
