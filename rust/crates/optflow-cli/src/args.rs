//! Command-line parsing (std only) with the behaviour of the `argparse` parser of
//! `python -m optflow_ref`: `--name value` or `--name=value`, unique prefixes of option names,
//! negative numbers as values, and argparse's error texts.

use std::path::PathBuf;

pub const USAGE: &str = "\
usage: optflow {pair,seq} ...

  pair --ref FILE --def FILE [common options]
  seq  --frames VIDEO|DIR|GLOB [--start N] [--stop N] [--step N] [common options]

common options:
  --roi FILE      JSON with {\"polygon\": [[x, y], ...]} in full-resolution pixels
  --mask FILE     instead of --roi: a .npy (bool or uint8) or PNG file, nonzero = inside, with the
                  shape of the scaled image
  --scale N       integer downscale factor (box average), default 1
  --crop auto|off with --roi/--mask: auto (default) tracks only the ROI's bounding box plus a margin
                  (the outputs keep the full size, NaN outside); off tracks the whole image
  --crop-margin N crop margin in pixels of the scaled image (default: max_disp + 4 sigma_w + guard
                  + 32 + the coarse window footprint (poc_win/2 + grid_step) * 2^levels, doubled
                  for seq)
  --device cpu|gpu  fine stage on the CPU (default) or the Metal GPU (build with --features gpu)
  --config FILE   configuration JSON
  --set K=V       override one configuration value (repeatable), e.g. --set sigma_w=6
  --out DIR       output directory (required)
  --png           accepted for compatibility; prints a warning and writes nothing
  --raw           also write raw_ux.npy and raw_uy.npy (the displacement before the confidence
                  threshold, the ROI and the guard)
  -h, --help      show this help
";

/// What to track.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Pair {
        reference: PathBuf,
        deformed: PathBuf,
    },
    Seq {
        frames: String,
        start: i64,
        stop: Option<i64>,
        step: i64,
    },
}

/// A parsed command line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cli {
    pub command: Command,
    pub roi: Option<PathBuf>,
    pub mask: Option<PathBuf>,
    pub scale: i64,
    pub crop: CropMode,
    pub crop_margin: Option<i64>,
    pub device: Device,
    pub config: Option<PathBuf>,
    pub set: Vec<String>,
    pub out: PathBuf,
    pub png: bool,
    pub raw: bool,
}

/// `--crop`: whether `--roi` also crops the images to the ROI's bounding box plus a margin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CropMode {
    Auto,
    Off,
}

/// `--device`: where the fine stage runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Gpu,
}

/// What the command line asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Parsed {
    Run(Box<Cli>),
    Help,
}

const COMMON: [&str; 11] = [
    "--roi",
    "--mask",
    "--scale",
    "--crop",
    "--crop-margin",
    "--device",
    "--config",
    "--set",
    "--out",
    "--png",
    "--raw",
];
const PAIR: [&str; 2] = ["--ref", "--def"];
const SEQ: [&str; 4] = ["--frames", "--start", "--stop", "--step"];

/// argparse's test for "this token is an option, not a value".
fn looks_like_option(s: &str) -> bool {
    if !s.starts_with('-') || s.len() < 2 || s.contains(' ') {
        return false;
    }
    !is_negative_number(s)
}

/// `^-\d+$|^-\d*\.\d+$`
fn is_negative_number(s: &str) -> bool {
    let Some(rest) = s.strip_prefix('-') else {
        return false;
    };
    let digits = |t: &str| t.bytes().all(|b| b.is_ascii_digit());
    match rest.split_once('.') {
        None => !rest.is_empty() && digits(rest),
        Some((int, frac)) => digits(int) && !frac.is_empty() && digits(frac),
    }
}

fn parse_int(opt: &str, text: &str) -> Result<i64, String> {
    match text.trim().parse::<i128>() {
        Ok(v) => Ok(v.clamp(i64::MIN as i128, i64::MAX as i128) as i64),
        Err(_) => Err(format!(
            "argument {opt}: invalid int value: {}",
            crate::settings::py_repr(text)
        )),
    }
}

/// How argparse reads an option-like `token` that is the help option, or `None` if it is not:
/// `Ok` for a request for help, `Err` with the message for help given a value.
///
/// * `-h`, `--help`, and any unambiguous abbreviation of the long option (`--h`, `--he`, `--hel`)
///   ask for help; so does a short option followed by letters (`-hx`, `-hh`: argparse takes them
///   for `-h` and more short options, and the help is printed before it looks at the rest);
/// * a value attached with `=` is an error, for the long and the short form alike (`--help=1`,
///   `--h=`, `-h=1`; the value is everything after the first `=`).
fn help_token(token: &str) -> Option<Result<(), String>> {
    let ignored = |value: &str| {
        Err(format!(
            "argument -h/--help: ignored explicit argument {}",
            crate::settings::py_repr(value)
        ))
    };
    if let Some(long) = token.strip_prefix("--") {
        let (name, value) = match long.split_once('=') {
            Some((name, value)) => (name, Some(value)),
            None => (long, None),
        };
        if name.is_empty() || !"help".starts_with(name) {
            return None;
        }
        return Some(value.map_or(Ok(()), ignored));
    }
    let rest = token.strip_prefix("-h")?;
    Some(match rest.strip_prefix('=') {
        Some(value) => ignored(value),
        None => Ok(()),
    })
}

/// Parses everything after the program name.
///
/// As argparse does: option-like tokens before the command are unrecognized arguments (reported
/// last, after the errors of the command's own parser), except the help option, which ends the
/// parse at once ([`help_token`]); a `--` ends the options, and the next token is the command; the
/// first token that is not option-like is the command. After the command's own `--` every token
/// is an unrecognized argument, `--` included.
pub fn parse(args: &[String]) -> Result<Parsed, String> {
    let mut unrecognized = Vec::new();
    let mut i = 0;
    while i < args.len() && (looks_like_option(&args[i]) || help_token(&args[i]).is_some()) {
        if args[i] == "--" {
            i += 1;
            break;
        }
        if let Some(help) = help_token(&args[i]) {
            return help.map(|()| Parsed::Help);
        }
        unrecognized.push(args[i].clone());
        i += 1;
    }
    let Some(cmd) = args.get(i) else {
        return Err("the following arguments are required: cmd".into());
    };
    i += 1;
    let is_pair = match cmd.as_str() {
        "pair" => true,
        "seq" => false,
        other => {
            return Err(format!(
                "argument cmd: invalid choice: {} (choose from pair, seq)",
                crate::settings::py_repr(other)
            ));
        }
    };
    let specific: &[&str] = if is_pair { &PAIR } else { &SEQ };
    let options: Vec<&str> = specific.iter().chain(COMMON.iter()).copied().collect();

    let (mut reference, mut deformed, mut frames) = (None, None, None);
    let (mut start, mut stop, mut step) = (0, None, 1);
    let (mut roi, mut scale, mut config, mut out) = (None, 1, None, None);
    let (mut crop, mut crop_margin, mut mask) = (CropMode::Auto, None, None);
    let mut device = Device::Cpu;
    let mut set = Vec::new();
    let (mut png, mut raw) = (false, false);

    while i < args.len() {
        let token = &args[i];
        i += 1;
        if token == "--" {
            unrecognized.extend(args[i - 1..].iter().cloned());
            break;
        }
        if let Some(help) = help_token(token) {
            return help.map(|()| Parsed::Help);
        }
        // A single-dash token is not an option of the command (only `-h`, above). A long one is
        // looked up by name, blanks or not: `--out=my dir` is the option `--out` with the value
        // `my dir` (argparse looks for the option before it looks for blanks), and a token that
        // names no option, such as `--out my dir`, is an unrecognized argument below.
        if !token.starts_with("--") {
            unrecognized.push(token.clone());
            continue;
        }
        let (name, inline) = match token.split_once('=') {
            Some((n, v)) => (n, Some(v.to_string())),
            None => (token.as_str(), None),
        };
        let opt: &str = if let Some(exact) = options.iter().find(|o| **o == name) {
            exact
        } else {
            let hits: Vec<&&str> = options.iter().filter(|o| o.starts_with(name)).collect();
            match hits.as_slice() {
                [one] => one,
                [] => {
                    unrecognized.push(token.clone());
                    continue;
                }
                many => {
                    let list: Vec<&str> = many.iter().map(|o| **o).collect();
                    return Err(format!(
                        "ambiguous option: {token} could match {}",
                        list.join(", ")
                    ));
                }
            }
        };
        if opt == "--png" || opt == "--raw" {
            if let Some(v) = inline {
                return Err(format!(
                    "argument {opt}: ignored explicit argument {}",
                    crate::settings::py_repr(&v)
                ));
            }
            if opt == "--png" {
                png = true;
            } else {
                raw = true;
            }
            continue;
        }
        let value = match inline {
            Some(v) => v,
            None => match args.get(i) {
                Some(next) if !looks_like_option(next) => {
                    i += 1;
                    next.clone()
                }
                _ => return Err(format!("argument {opt}: expected one argument")),
            },
        };
        match opt {
            "--ref" => reference = Some(PathBuf::from(value)),
            "--def" => deformed = Some(PathBuf::from(value)),
            "--frames" => frames = Some(value),
            "--start" => start = parse_int(opt, &value)?,
            "--stop" => stop = Some(parse_int(opt, &value)?),
            "--step" => step = parse_int(opt, &value)?,
            "--roi" => roi = Some(PathBuf::from(value)),
            "--mask" => mask = Some(PathBuf::from(value)),
            "--scale" => scale = parse_int(opt, &value)?,
            "--crop" => {
                crop = match value.as_str() {
                    "auto" => CropMode::Auto,
                    "off" => CropMode::Off,
                    _ => {
                        return Err(format!(
                            "argument --crop: invalid choice: {} (choose from auto, off)",
                            crate::settings::py_repr(&value)
                        ))
                    }
                }
            }
            "--crop-margin" => crop_margin = Some(parse_int(opt, &value)?),
            "--device" => {
                device = match value.as_str() {
                    "cpu" => Device::Cpu,
                    "gpu" => Device::Gpu,
                    _ => {
                        return Err(format!(
                            "argument --device: invalid choice: {} (choose from cpu, gpu)",
                            crate::settings::py_repr(&value)
                        ))
                    }
                }
            }
            "--config" => config = Some(PathBuf::from(value)),
            "--set" => set.push(value),
            "--out" => out = Some(PathBuf::from(value)),
            _ => unreachable!("option {opt} is in the table but not handled"),
        }
    }

    let mut missing = Vec::new();
    if is_pair {
        if reference.is_none() {
            missing.push("--ref");
        }
        if deformed.is_none() {
            missing.push("--def");
        }
    } else if frames.is_none() {
        missing.push("--frames");
    }
    if out.is_none() {
        missing.push("--out");
    }
    if !missing.is_empty() {
        return Err(format!(
            "the following arguments are required: {}",
            missing.join(", ")
        ));
    }
    if !unrecognized.is_empty() {
        return Err(format!(
            "unrecognized arguments: {}",
            unrecognized.join(" ")
        ));
    }
    let command = match (reference, deformed, frames) {
        (Some(reference), Some(deformed), _) if is_pair => Command::Pair {
            reference,
            deformed,
        },
        (_, _, Some(frames)) => Command::Seq {
            frames,
            start,
            stop,
            step,
        },
        _ => unreachable!("required arguments were checked"),
    };
    Ok(Parsed::Run(Box::new(Cli {
        command,
        roi,
        mask,
        scale,
        crop,
        crop_margin,
        device,
        config,
        set,
        out: out.expect("--out was checked"),
        png,
        raw,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(line: &str) -> Result<Parsed, String> {
        let args: Vec<String> = line.split_whitespace().map(String::from).collect();
        parse(&args)
    }

    fn cli(line: &str) -> Cli {
        match p(line) {
            Ok(Parsed::Run(c)) => *c,
            other => panic!("{line}: {other:?}"),
        }
    }

    #[test]
    fn pair_with_all_options() {
        let c = cli("pair --ref a.png --def=b.png --roi r.json --scale 2 --config c.json --set a=1 --set b=2 --out o --png");
        assert_eq!(
            c.command,
            Command::Pair {
                reference: "a.png".into(),
                deformed: "b.png".into()
            }
        );
        assert_eq!(c.roi.unwrap().to_str(), Some("r.json"));
        assert_eq!((c.scale, c.png), (2, true));
        assert_eq!(c.set, ["a=1", "b=2"]);
        assert_eq!(c.out.to_str(), Some("o"));
    }

    #[test]
    fn mask_and_raw_options() {
        let c = cli("pair --ref a --def b --out o");
        assert_eq!((c.mask, c.raw), (None, false));
        let c = cli("seq --frames v --out o --mask m.npy --raw");
        assert_eq!(c.mask.unwrap().to_str(), Some("m.npy"));
        assert!(c.raw);
        assert_eq!(
            p("pair --ref a --def b --out o --raw=1").unwrap_err(),
            "argument --raw: ignored explicit argument '1'"
        );
        assert_eq!(
            p("pair --ref a --def b --out o --r").unwrap_err(),
            "ambiguous option: --r could match --ref, --roi, --raw"
        );
    }

    #[test]
    fn crop_options() {
        let c = cli("pair --ref a --def b --out o");
        assert_eq!((c.crop, c.crop_margin), (CropMode::Auto, None));
        let c = cli("pair --ref a --def b --out o --crop off --crop-margin=-7");
        assert_eq!((c.crop, c.crop_margin), (CropMode::Off, Some(-7)));
        let c = cli("seq --frames v --out o --crop=auto --crop-m 40");
        assert_eq!((c.crop, c.crop_margin), (CropMode::Auto, Some(40)));
        assert_eq!(
            p("pair --ref a --def b --out o --crop maybe").unwrap_err(),
            "argument --crop: invalid choice: 'maybe' (choose from auto, off)"
        );
        assert_eq!(
            p("pair --ref a --def b --out o --crop-margin x").unwrap_err(),
            "argument --crop-margin: invalid int value: 'x'"
        );
        assert_eq!(
            p("pair --ref a --def b --out o --cr off").unwrap_err(),
            "ambiguous option: --cr could match --crop, --crop-margin"
        );
    }

    #[test]
    fn seq_defaults_and_negative_numbers() {
        let c = cli("seq --frames v.mp4 --out o");
        assert_eq!(
            c.command,
            Command::Seq {
                frames: "v.mp4".into(),
                start: 0,
                stop: None,
                step: 1
            }
        );
        assert_eq!(c.scale, 1);
        let c = cli("seq --frames d --start -1 --stop -3 --step=2 --out o --scale -2");
        assert_eq!(
            c.command,
            Command::Seq {
                frames: "d".into(),
                start: -1,
                stop: Some(-3),
                step: 2
            }
        );
        assert_eq!(c.scale, -2);
    }

    /// argparse's `^-\d+$|^-\d*\.\d+$`: digits, or optional digits, a dot and at least one digit.
    #[test]
    fn negative_numbers_are_digits_with_an_optional_fraction() {
        for s in ["-1", "-12", "-0", "-1.5", "-.5", "-0.25", "-10.05"] {
            assert!(is_negative_number(s), "{s}");
            assert!(!looks_like_option(s), "{s}");
        }
        for s in [
            "-", "-.", "-1.", "-1.x", "-x.5", "-1.5.2", "-a", "-1e3", "-1 ", "1", "1.5", "",
        ] {
            assert!(!is_negative_number(s), "{s:?}");
        }
        // as values: "-1." is an option-like token for argparse, so --start has no argument
        assert!(looks_like_option("-1."));
        assert!(looks_like_option("-1.x"));
    }

    /// argparse's `_parse_optional`: a token is option-like if it starts with `-`, has a second
    /// character, has no blank, and is not a negative number.
    #[test]
    fn option_like_tokens() {
        for s in [
            "-x", "-xy", "--ref", "--ref=a", "-1x", "-a.5", "-1.", "-1e3",
        ] {
            assert!(looks_like_option(s), "{s:?}");
        }
        for s in [
            "", "-", "a", "ab", "x-y", "a b", "-a b", "- 1", "-1", "-12", "-1.5", "-.5",
        ] {
            assert!(!looks_like_option(s), "{s:?}");
        }
    }

    #[test]
    fn options_before_the_command_are_unrecognized_arguments_reported_last() {
        // the texts argparse of the Python reference prints (python -m optflow_ref ...; the test
        // `usage_errors_match_argparse` compares them with the live reference)
        assert_eq!(
            p("-x").unwrap_err(),
            "the following arguments are required: cmd"
        );
        assert_eq!(
            p("--bogus -y").unwrap_err(),
            "the following arguments are required: cmd"
        );
        assert_eq!(
            p("--bogus pair").unwrap_err(),
            "the following arguments are required: --ref, --def, --out"
        );
        assert_eq!(
            p("--bogus pair --ref a --def b --out o").unwrap_err(),
            "unrecognized arguments: --bogus"
        );
        assert_eq!(
            p("-x -y seq --frames d --out o --zzz").unwrap_err(),
            "unrecognized arguments: -x -y --zzz"
        );
        // a token that is not option-like is the command, whatever it is
        assert_eq!(
            p("-1").unwrap_err(),
            "argument cmd: invalid choice: '-1' (choose from pair, seq)"
        );
        assert_eq!(
            p("--ref a pair").unwrap_err(),
            "argument cmd: invalid choice: 'a' (choose from pair, seq)"
        );
        // help ends the parse at once, however it is spelled and whatever precedes it; `-h` with
        // more letters is `-h` and more short options (argparse prints the help first)
        for line in [
            "-h",
            "--help",
            "--he",
            "-x -h",
            "--bogus --h pair --ref",
            "-hx",
            "-hh",
            "-h1",
            "-hx=1",
            "-hx --h=1",
            "-h -- seq",
            "pair -hx",
            "seq --out o -hx",
        ] {
            assert_eq!(p(line), Ok(Parsed::Help), "{line}");
        }
    }

    /// A value attached to the help option is an error, in every spelling and wherever it is, with
    /// the text argparse prints (everything after the first `=`, as a Python `repr`); the first
    /// help token decides.
    #[test]
    fn help_with_a_value_is_an_ignored_explicit_argument() {
        let ignored = |v: &str| format!("argument -h/--help: ignored explicit argument {v}");
        for (line, value) in [
            ("--h=1", "'1'"),
            ("--help=1", "'1'"),
            ("--he=1", "'1'"),
            ("--help=", "''"),
            ("--h=", "''"),
            ("-h=1", "'1'"),
            ("-h=", "''"),
            ("-h=x=y", "'x=y'"),
            ("-x --h=1", "'1'"),
            ("--h=1 -x", "'1'"),
            ("--h=1 -hx", "'1'"),
            ("-x -h=1", "'1'"),
            ("--help= seq", "''"),
            ("--h=1 --", "'1'"),
            ("pair --h=1", "'1'"),
            ("pair -h=1", "'1'"),
            ("pair -h=x=y", "'x=y'"),
            ("pair --ref a --def b --out o --h=2", "'2'"),
            ("seq --out o --help=1", "'1'"),
            ("seq --out o --he=1", "'1'"),
            ("seq --out o -h=", "''"),
        ] {
            assert_eq!(p(line).unwrap_err(), ignored(value), "{line}");
        }
        // not the help option: `--helpx=1` and `--hx=1` are unknown options
        assert_eq!(
            p("--helpx=1 pair --ref a --def b --out o").unwrap_err(),
            "unrecognized arguments: --helpx=1"
        );
    }

    /// Tokens of the command line that have blanks in them (the arguments are not split here).
    fn pv(tokens: &[&str]) -> Result<Parsed, String> {
        let args: Vec<String> = tokens.iter().map(|t| t.to_string()).collect();
        parse(&args)
    }

    /// A blank in an `--option=value` token does not make it a positional argument: argparse
    /// looks for the option (exact name or abbreviation) before it looks for blanks, so
    /// `--out=my dir` is the option `--out` with the value `my dir`. A token whose name part is no
    /// option, or has a blank, stays a positional; the help option is found the same way.
    #[test]
    fn a_blank_in_the_value_of_an_attached_option_is_fine() {
        // a value with blanks reaches the option: the error that follows is about something else
        let seq = |extra: &[&str]| {
            let mut line = vec!["seq"];
            line.extend(extra);
            pv(&line).unwrap_err()
        };
        assert_eq!(
            seq(&["--frames=a b", "--out=o p", "--zzz"]),
            "unrecognized arguments: --zzz"
        );
        assert_eq!(
            seq(&["--fr=a b", "--out", "o", "--zzz"]),
            "unrecognized arguments: --zzz"
        );
        assert_eq!(
            seq(&["--set=lam=1 2", "--frames", "d", "--out", "o", "--zzz"]),
            "unrecognized arguments: --zzz"
        );
        assert_eq!(
            seq(&["--png=a b", "--frames", "d", "--out", "o"]),
            "argument --png: ignored explicit argument 'a b'"
        );
        assert_eq!(
            seq(&["--start=1 2", "--frames", "d", "--out", "o"]),
            "argument --start: invalid int value: '1 2'"
        );
        assert_eq!(
            seq(&["--scale=2 x", "--frames", "d", "--out", "o"]),
            "argument --scale: invalid int value: '2 x'"
        );
        assert_eq!(
            seq(&["--crop=a b", "--frames", "d", "--out", "o"]),
            "argument --crop: invalid choice: 'a b' (choose from auto, off)"
        );
        // an abbreviation that fits several options is ambiguous, and the message has the token
        assert_eq!(
            seq(&["--out", "o", "--s=1"]),
            "ambiguous option: --s=1 could match --start, --stop, --step, --scale, --set"
        );
        assert_eq!(
            seq(&["--out", "o", "--s=a b"]),
            "ambiguous option: --s=a b could match --start, --stop, --step, --scale, --set"
        );
        assert_eq!(
            seq(&["--st=1 2", "--frames", "d", "--out", "o"]),
            "ambiguous option: --st=1 2 could match --start, --stop, --step"
        );
        assert_eq!(
            pv(&["pair", "--r=x y", "--def", "d", "--out", "o"]).unwrap_err(),
            "ambiguous option: --r=x y could match --ref, --roi, --raw"
        );
        // a name that is no option of the command, or has a blank in it, is a positional
        assert_eq!(
            seq(&["--ref=a b", "--frames", "d", "--out", "o"]),
            "unrecognized arguments: --ref=a b"
        );
        assert_eq!(
            seq(&["--frames a", "--out", "o", "--zzz"]),
            "the following arguments are required: --frames"
        );
        assert_eq!(
            seq(&["--frames a=b", "--out", "o", "--zzz"]),
            "the following arguments are required: --frames"
        );
        assert_eq!(
            seq(&["--out", "o", "--s a"]),
            "the following arguments are required: --frames"
        );
        // the help option, with blanks anywhere after its name
        let ignored = |v: &str| format!("argument -h/--help: ignored explicit argument {v}");
        for (tokens, want) in [
            (&["--he=a b"][..], ignored("'a b'")),
            (&["-h=a b"], ignored("'a b'")),
            (&["--h=a b", "seq"], ignored("'a b'")),
            (&["-h=1 2", "seq"], ignored("'1 2'")),
            (&["--help=1 2"], ignored("'1 2'")),
            (
                &["--bogus=a b"],
                "argument cmd: invalid choice: '--bogus=a b' (choose from pair, seq)".into(),
            ),
            (
                &["-x y"],
                "argument cmd: invalid choice: '-x y' (choose from pair, seq)".into(),
            ),
            (
                &["-x=a b", "seq"],
                "argument cmd: invalid choice: '-x=a b' (choose from pair, seq)".into(),
            ),
            (
                &["--help x=1"],
                "argument cmd: invalid choice: '--help x=1' (choose from pair, seq)".into(),
            ),
            (
                &["--hel x=1"],
                "argument cmd: invalid choice: '--hel x=1' (choose from pair, seq)".into(),
            ),
            (
                &["seq", "--out", "o", "--help x"],
                "the following arguments are required: --frames".into(),
            ),
            (
                &["seq", "--out", "o", "--help x=1"],
                "the following arguments are required: --frames".into(),
            ),
        ] {
            assert_eq!(pv(tokens).unwrap_err(), want, "{tokens:?}");
        }
        for tokens in [
            &["-hx y"][..],
            &["-h x"],
            &["-hello world"],
            &["-hx=1 2"],
            &["seq", "--out", "o", "-h x"],
        ] {
            assert_eq!(pv(tokens), Ok(Parsed::Help), "{tokens:?}");
        }
    }

    /// `--` ends the options. Before the command: the next token is the command, whatever it
    /// looks like. After it: every token is an unrecognized argument, `--` included, in order.
    #[test]
    fn a_double_dash_ends_the_options() {
        let invalid =
            |t: &str| format!("argument cmd: invalid choice: {t} (choose from pair, seq)");
        let none = "the following arguments are required: cmd";
        // the command is the token after `--`, whatever it looks like
        for (line, token) in [
            ("-- -x", "'-x'"),
            ("-- -h", "'-h'"),
            ("-- --", "'--'"),
            ("-- -- seq", "'--'"),
            ("-x -- -y", "'-y'"),
            ("-x -y -- -z", "'-z'"),
            ("-- --h=1", "'--h=1'"),
        ] {
            assert_eq!(p(line).unwrap_err(), invalid(token), "{line}");
        }
        for (line, want) in [
            ("--", none),
            ("-x --", none),
            ("-x -y --", none),
            (
                "-- seq --out o",
                "the following arguments are required: --frames",
            ),
            (
                "-- pair",
                "the following arguments are required: --ref, --def, --out",
            ),
            (
                "-- pair -x",
                "the following arguments are required: --ref, --def, --out",
            ),
            (
                "-x -- seq --out o",
                "the following arguments are required: --frames",
            ),
            (
                "pair -- --ref a",
                "the following arguments are required: --ref, --def, --out",
            ),
            (
                "pair -- --ref a --def b --out o",
                "the following arguments are required: --ref, --def, --out",
            ),
            (
                "seq -- --out o",
                "the following arguments are required: --frames, --out",
            ),
            (
                "seq --frames d -- --out o",
                "the following arguments are required: --out",
            ),
            (
                "seq --out o --",
                "the following arguments are required: --frames",
            ),
            (
                "seq --out o -- x",
                "the following arguments are required: --frames",
            ),
            ("seq --frames d --out o --", "unrecognized arguments: --"),
            (
                "seq --frames d --out o -- x",
                "unrecognized arguments: -- x",
            ),
            (
                "seq --frames d --out o -- --x",
                "unrecognized arguments: -- --x",
            ),
            (
                "seq --frames d --out o -- -- x",
                "unrecognized arguments: -- -- x",
            ),
            (
                "seq --frames d --out o x -- y",
                "unrecognized arguments: x -- y",
            ),
            (
                "seq --frames d --out o -x -- y",
                "unrecognized arguments: -x -- y",
            ),
            (
                "-- seq --frames d --out o -- y",
                "unrecognized arguments: -- y",
            ),
            (
                "-x -- seq --frames d --out o -- y",
                "unrecognized arguments: -x -- y",
            ),
            (
                "-x -y seq --frames d --out o -z -- y",
                "unrecognized arguments: -x -y -z -- y",
            ),
            // `--` is no value for an option, but an attached value may be anything
            (
                "pair --ref -- --def b --out o",
                "argument --ref: expected one argument",
            ),
            (
                "seq --frames d --out o --start --",
                "argument --start: expected one argument",
            ),
            (
                "seq --frames d --out o --start -- 1",
                "argument --start: expected one argument",
            ),
            (
                "seq --frames d --out o --start=--",
                "argument --start: invalid int value: '--'",
            ),
        ] {
            assert_eq!(p(line).unwrap_err(), want, "{line}");
        }
        // a command after `--` runs as it would without it
        let c = cli("-- seq --frames d --out o");
        assert_eq!(c.out, std::path::PathBuf::from("o"));
        assert_eq!(p("-- seq --frames d --out o"), p("seq --frames d --out o"));
    }

    #[test]
    fn an_option_does_not_take_an_option_as_value() {
        assert_eq!(
            p("pair --ref --def b --out o").unwrap_err(),
            "argument --ref: expected one argument"
        );
        assert_eq!(
            p("pair --ref a --def b --out --png").unwrap_err(),
            "argument --out: expected one argument"
        );
        assert_eq!(
            p("seq --frames d --start -1. --out o").unwrap_err(),
            "argument --start: expected one argument"
        );
    }

    #[test]
    fn unique_prefixes_are_accepted() {
        let c = cli("pair --re a --def b --ou o --sc 3");
        assert_eq!(c.scale, 3);
        let e = p("pair --ref a --def b --out o --s 3").unwrap_err();
        assert!(e.starts_with("ambiguous option: --s could match "), "{e}");
    }

    #[test]
    fn errors_have_argparse_texts() {
        assert_eq!(
            p("").unwrap_err(),
            "the following arguments are required: cmd"
        );
        assert_eq!(
            p("pair --out o").unwrap_err(),
            "the following arguments are required: --ref, --def"
        );
        assert_eq!(
            p("seq").unwrap_err(),
            "the following arguments are required: --frames, --out"
        );
        assert_eq!(
            p("pair --ref a --def b --out").unwrap_err(),
            "argument --out: expected one argument"
        );
        assert_eq!(
            p("pair --ref a --def b --out o --scale x").unwrap_err(),
            "argument --scale: invalid int value: 'x'"
        );
        assert_eq!(
            p("pair --ref a --def b --out o --tile 5").unwrap_err(),
            "unrecognized arguments: --tile 5"
        );
        assert_eq!(
            p("pair --ref a --def b --out o --start 1").unwrap_err(),
            "unrecognized arguments: --start 1"
        );
        assert_eq!(
            p("run").unwrap_err(),
            "argument cmd: invalid choice: 'run' (choose from pair, seq)"
        );
        assert_eq!(
            p("pair --ref a --def b --out o --png=1").unwrap_err(),
            "argument --png: ignored explicit argument '1'"
        );
    }

    #[test]
    fn help_is_recognised() {
        assert_eq!(p("--help"), Ok(Parsed::Help));
        assert_eq!(p("pair -h"), Ok(Parsed::Help));
        assert_eq!(p("seq --frames x --help"), Ok(Parsed::Help));
    }
}
