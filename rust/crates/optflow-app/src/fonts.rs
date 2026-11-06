//! Finding a system font with Japanese glyphs (egui ships none).

use std::path::PathBuf;

/// Candidate font files in order of preference, for the platform this is built for.
pub fn candidates() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = Vec::new();
    if cfg!(windows) {
        let root = std::env::var_os("WINDIR")
            .or_else(|| std::env::var_os("SystemRoot"))
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"C:\Windows"));
        for f in [
            "meiryo.ttc",
            "YuGothM.ttc",
            "YuGothR.ttc",
            "msgothic.ttc",
            "BIZ-UDGothicR.ttc",
        ] {
            v.push(root.join("Fonts").join(f));
        }
    } else if cfg!(target_os = "macos") {
        for f in [
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        ] {
            v.push(PathBuf::from(f));
        }
    } else {
        for f in [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        ] {
            v.push(PathBuf::from(f));
        }
    }
    v
}

/// The first candidate that exists and is readable, with its bytes. `None`: run without CJK
/// glyphs (the text then shows boxes, but the app works).
pub fn load_first(paths: &[PathBuf]) -> Option<(PathBuf, Vec<u8>)> {
    paths
        .iter()
        .find_map(|p| std::fs::read(p).ok().map(|b| (p.clone(), b)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn falls_back_to_the_next_candidate() {
        let dir = std::env::temp_dir().join(format!("optflow-app-font-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ok = dir.join("b.ttf");
        std::fs::write(&ok, b"font").unwrap();
        let got = load_first(&[dir.join("missing.ttc"), ok.clone()]).unwrap();
        assert_eq!(got.0, ok);
        assert_eq!(got.1, b"font");
        assert!(load_first(&[dir.join("none")]).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn there_are_candidates() {
        assert!(!candidates().is_empty());
    }
}
