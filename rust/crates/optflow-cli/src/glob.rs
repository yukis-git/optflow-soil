//! `glob.glob` of Python (non-recursive, hidden names only when the pattern asks for them) and
//! `fnmatch` on POSIX (case-sensitive), for `--frames` patterns and directories.

use std::fs;
use std::path::Path;

fn has_magic(s: &str) -> bool {
    s.contains(['*', '?', '['])
}

/// The characters that separate the components of a path: `/`, and on Windows `\` as well, as
/// `os.path` has it there. Elsewhere a backslash is an ordinary character of a file name (on macOS
/// and Linux `a\b.png` is one file).
#[cfg(windows)]
pub const SEPARATORS: &[char] = &['/', '\\'];
#[cfg(not(windows))]
pub const SEPARATORS: &[char] = &['/'];

/// `os.path.split` for the separator set `seps` (single-byte characters): the head without its
/// trailing separators (unless it is all separators) and the tail.
fn split_with<'a>(path: &'a str, seps: &[char]) -> (&'a str, &'a str) {
    let i = path.rfind(seps).map_or(0, |i| i + 1);
    let (head, tail) = path.split_at(i);
    let trimmed = head.trim_end_matches(seps);
    (if trimmed.is_empty() { head } else { trimmed }, tail)
}

/// `os.path.split` of this platform.
fn split(path: &str) -> (&str, &str) {
    split_with(path, SEPARATORS)
}

/// The last component of `path` (`os.path.split(path)[1]`) for the separator set `seps`.
pub fn basename_with<'a>(path: &'a str, seps: &[char]) -> &'a str {
    split_with(path, seps).1
}

/// The last component of `path` on this platform.
pub fn basename(path: &str) -> &str {
    basename_with(path, SEPARATORS)
}

/// `os.path.join(dir, name)` for the separator set `seps` and the separator `sep` it adds: none is
/// added after a directory that ends with a separator already.
fn join_with(dir: &str, name: &str, seps: &[char], sep: char) -> String {
    if dir.is_empty() {
        name.to_string()
    } else if dir.ends_with(seps) {
        format!("{dir}{name}")
    } else {
        format!("{dir}{sep}{name}")
    }
}

/// `os.path.join` of this platform.
pub fn join(dir: &str, name: &str) -> String {
    join_with(dir, name, SEPARATORS, std::path::MAIN_SEPARATOR)
}

fn lexists(path: &str) -> bool {
    fs::symlink_metadata(path).is_ok()
}

/// `fnmatch.fnmatchcase(name, pattern)`: `*`, `?`, `[seq]`, `[!seq]`.
pub fn fnmatch(name: &str, pattern: &str) -> bool {
    let name: Vec<char> = name.chars().collect();
    let pat: Vec<char> = pattern.chars().collect();
    match_from(&name, &pat)
}

/// Parses the `[...]` set starting at `pat[0] == '['`; returns the set test and the pattern index
/// after the `]`, or `None` if the bracket is not closed (then `[` is a literal).
fn class(pat: &[char]) -> Option<(impl Fn(char) -> bool + '_, usize)> {
    let mut j = 1;
    let negate = pat.get(j) == Some(&'!');
    if negate {
        j += 1;
    }
    let body_start = j;
    if pat.get(j) == Some(&']') {
        j += 1;
    }
    while j < pat.len() && pat[j] != ']' {
        j += 1;
    }
    if j >= pat.len() {
        return None;
    }
    let body = &pat[body_start..j];
    let test = move |c: char| {
        let mut k = 0;
        let mut hit = false;
        while k < body.len() {
            if k + 2 < body.len() && body[k + 1] == '-' {
                hit |= body[k] <= c && c <= body[k + 2];
                k += 3;
            } else {
                hit |= body[k] == c;
                k += 1;
            }
        }
        hit != negate
    };
    Some((test, j + 1))
}

fn match_from(name: &[char], pat: &[char]) -> bool {
    match pat.first() {
        None => name.is_empty(),
        Some('*') => (0..=name.len()).any(|k| match_from(&name[k..], &pat[1..])),
        Some('?') => !name.is_empty() && match_from(&name[1..], &pat[1..]),
        Some('[') => match class(pat) {
            Some((test, next)) => {
                !name.is_empty() && test(name[0]) && match_from(&name[1..], &pat[next..])
            }
            None => name.first() == Some(&'[') && match_from(&name[1..], &pat[1..]),
        },
        Some(&c) => name.first() == Some(&c) && match_from(&name[1..], &pat[1..]),
    }
}

/// Entry names of `dir` (`""` is the current directory); `dirs_only` keeps directories.
fn list(dir: &str, dirs_only: bool) -> Vec<String> {
    let Ok(entries) = fs::read_dir(if dir.is_empty() { "." } else { dir }) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter(|e| !dirs_only || e.path().is_dir())
        .filter_map(|e| e.file_name().into_string().ok())
        .collect()
}

fn glob_inner(pattern: &str, dirs_only: bool) -> Vec<String> {
    let (dirname, basename) = split(pattern);
    if !has_magic(pattern) {
        let found = if basename.is_empty() {
            Path::new(dirname).is_dir()
        } else {
            lexists(pattern)
        };
        return if found {
            vec![pattern.to_string()]
        } else {
            vec![]
        };
    }
    let dirs = if dirname.is_empty() {
        vec![String::new()]
    } else if has_magic(dirname) {
        glob_inner(dirname, true)
    } else {
        vec![dirname.to_string()]
    };
    let mut out = Vec::new();
    for d in dirs {
        if has_magic(basename) {
            let hidden_ok = basename.starts_with('.');
            for name in list(&d, dirs_only) {
                if (hidden_ok || !name.starts_with('.')) && fnmatch(&name, basename) {
                    out.push(join(&d, &name));
                }
            }
        } else {
            let p = join(&d, basename);
            if lexists(&p) {
                out.push(p);
            }
        }
    }
    out
}

/// The paths matching `pattern`, in directory order (sort them for a stable result).
pub fn glob(pattern: &str) -> Vec<String> {
    glob_inner(pattern, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnmatch_examples() {
        assert!(fnmatch("img_01.png", "img_*.png"));
        assert!(fnmatch("img_01.png", "img_0?.png"));
        assert!(!fnmatch("img_01.png", "img_?.png"));
        assert!(fnmatch("a1", "a[0-9]"));
        assert!(!fnmatch("ab", "a[0-9]"));
        assert!(fnmatch("ab", "a[!0-9]"));
        assert!(fnmatch("a]", "a[]]"));
        assert!(fnmatch("a[", "a["));
        assert!(!fnmatch("A.png", "a.png"));
        assert!(fnmatch("", "*"));
    }

    /// Every member of a set counts (not just the first), and the `!` that negates a set is not
    /// itself a member of it.
    #[test]
    fn fnmatch_sets_test_every_member_and_negation_excludes_the_marker() {
        for c in ["a", "b", "c"] {
            assert!(fnmatch(c, "[abc]"), "{c}");
            assert!(!fnmatch(c, "[!abc]"), "{c}");
        }
        assert!(!fnmatch("d", "[abc]"));
        assert!(fnmatch("d", "[!abc]"));
        // "!" and "[" are not in `abc`, so the negated set takes them
        assert!(fnmatch("!", "[!abc]"));
        assert!(fnmatch("[", "[!abc]"));
        // members after a range, and a range after a member
        assert!(fnmatch("x", "[a-cx]") && fnmatch("b", "[xa-c]") && fnmatch("x", "[xa-c]"));
        assert!(!fnmatch("d", "[a-cx]"));
        // a "!" that is not first is a member
        assert!(fnmatch("!", "[a!]") && fnmatch("a", "[a!]"));
        // a "-" last is a member, not the start of a range
        assert!(fnmatch("-", "[a-]") && fnmatch("a", "[a-]") && !fnmatch("b", "[a-]"));
        assert!(!fnmatch("-", "[!a-]") && fnmatch("b", "[!a-]"));
        // a "]" first is a member, and the set goes on to the next "]"
        assert!(fnmatch("]", "[]a]") && fnmatch("a", "[]a]") && !fnmatch("b", "[]a]"));
        assert!(fnmatch("b]", "[!a]]") && !fnmatch("a]", "[!a]]") && fnmatch("b", "[!]a]"));
        assert!(!fnmatch("]", "[!]a]") && !fnmatch("a", "[!]a]"));
        // a "[" that no "]" closes is a literal "[" and only that
        assert!(fnmatch("[", "[") && !fnmatch("a", "[") && !fnmatch("ab", "[b"));
        assert!(fnmatch("[b", "[b") && fnmatch("[!", "[!") && !fnmatch("a", "[!"));
    }

    #[test]
    fn split_like_os_path() {
        assert_eq!(split("a/b/c"), ("a/b", "c"));
        assert_eq!(split("c"), ("", "c"));
        assert_eq!(split("a/"), ("a", ""));
        assert_eq!(split("/c"), ("/", "c"));
        assert_eq!(split("a//b"), ("a", "b"));
    }

    /// The separator sets of POSIX and of Windows, so that both are tested on every platform.
    const POSIX: &[char] = &['/'];
    const WINDOWS: &[char] = &['/', '\\'];

    #[test]
    fn a_backslash_separates_only_where_the_separator_set_says_so() {
        // POSIX: a backslash is a character of a file name
        assert_eq!(split_with("a\\b/c", POSIX), ("a\\b", "c"));
        assert_eq!(split_with("a\\b", POSIX), ("", "a\\b"));
        assert_eq!(split_with("a\\", POSIX), ("", "a\\"));
        assert_eq!(split_with("a\\/b", POSIX), ("a\\", "b"));
        // Windows: both separate, and trailing ones are dropped from the head like slashes
        assert_eq!(split_with("a\\b/c", WINDOWS), ("a\\b", "c"));
        assert_eq!(split_with("a/b\\c", WINDOWS), ("a/b", "c"));
        assert_eq!(split_with("a\\b", WINDOWS), ("a", "b"));
        assert_eq!(split_with("a\\", WINDOWS), ("a", ""));
        assert_eq!(split_with("\\c", WINDOWS), ("\\", "c"));
        assert_eq!(split_with("a\\\\/b", WINDOWS), ("a", "b"));
        assert_eq!(split_with("c", WINDOWS), ("", "c"));
        // the last component, for the file extension
        assert_eq!(basename_with("dir\\clip.mp4", POSIX), "dir\\clip.mp4");
        assert_eq!(basename_with("dir\\clip.mp4", WINDOWS), "clip.mp4");
        assert_eq!(basename_with("a/b/c.png", POSIX), "c.png");
        assert_eq!(basename_with("a/b/", WINDOWS), "");
        // joining: a directory that ends with a separator of the set gets none added
        assert_eq!(join_with("a\\", "f", POSIX, '/'), "a\\/f");
        assert_eq!(join_with("a\\", "f", WINDOWS, '\\'), "a\\f");
        assert_eq!(join_with("a/", "f", WINDOWS, '\\'), "a/f");
        assert_eq!(join_with("a", "f", WINDOWS, '\\'), "a\\f");
        assert_eq!(join_with("a", "f", POSIX, '/'), "a/f");
        assert_eq!(join_with("", "f", WINDOWS, '\\'), "f");
    }

    #[test]
    fn this_platforms_separators_are_the_posix_ones_except_on_windows() {
        assert_eq!(SEPARATORS, if cfg!(windows) { WINDOWS } else { POSIX });
        assert_eq!(
            std::path::MAIN_SEPARATOR,
            if cfg!(windows) { '\\' } else { '/' }
        );
    }

    #[test]
    fn glob_lists_matches_and_skips_hidden() {
        let dir = crate::unit_scratch("glob");
        fs::create_dir_all(dir.join("sub")).unwrap();
        for f in ["a1.png", "a2.png", ".a3.png", "b.txt", "sub/c.png"] {
            fs::write(dir.join(f), b"").unwrap();
        }
        let d = dir.to_str().unwrap();
        // matches found by listing a directory are joined with the platform's separator (as
        // `os.path.join` does); a pattern without wildcards is returned as it was written
        let s = std::path::MAIN_SEPARATOR;
        let mut got = glob(&format!("{d}/a*.png"));
        got.sort();
        assert_eq!(got, [format!("{d}{s}a1.png"), format!("{d}{s}a2.png")]);
        assert_eq!(glob(&format!("{d}/.a*.png")), [format!("{d}{s}.a3.png")]);
        assert_eq!(
            glob(&format!("{d}/*/c.png")),
            [format!("{d}{s}sub{s}c.png")]
        );
        assert_eq!(glob(&format!("{d}/b.txt")), [format!("{d}/b.txt")]);
        assert!(glob(&format!("{d}/nothing")).is_empty());
        assert!(glob(&format!("{d}/nothing_*")).is_empty());
        let _ = fs::remove_dir_all(&dir);
    }
}
