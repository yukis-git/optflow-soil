use std::process::ExitCode;

#[cfg(target_os = "macos")]
mod large_alloc;

#[cfg(target_os = "macos")]
#[global_allocator]
static ALLOC: large_alloc::LargeMmap = large_alloc::LargeMmap;

fn main() -> ExitCode {
    let mut args = Vec::new();
    for (i, arg) in std::env::args_os().skip(1).enumerate() {
        match arg.into_string() {
            Ok(text) => args.push(text),
            Err(raw) => {
                eprintln!("error: argument {} is not valid UTF-8: {raw:?}", i + 1);
                return ExitCode::from(2);
            }
        }
    }
    optflow_cli::main_with(&args)
}
