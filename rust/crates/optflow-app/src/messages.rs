//! Readable Japanese messages for the failures a user can meet.

use optflow_cli::run::Cancelled;

pub const FFMPEG_MISSING: &str = "動画の読み込みには ffmpeg が必要ですが見つかりません。\n\
ffmpeg.exe と ffprobe.exe を、このアプリ(optflow-app.exe)と同じフォルダに置くか、PATH の通った場所に入れてください。\n\
(連番の画像ファイルなら ffmpeg なしで使えます。画像フォルダを選んでください。)";

/// A Japanese message for `e` (the original English text follows in parentheses when the error is
/// not one of the known ones, so nothing is hidden).
pub fn describe(e: &anyhow::Error) -> String {
    if e.downcast_ref::<Cancelled>().is_some() {
        return "中止しました。".into();
    }
    let full = format!("{e:#}");
    let known = |needle: &str| full.contains(needle);
    if known("takes images of at most") {
        return format!(
            "亀裂モデルは約 830 万画素(3840x2160)までの画像にしか使えません(100 万画素あたり約 1.1 GB のメモリも使います)。ROI を狭く取るか、縮小率を上げる(詳細設定)か、亀裂モデルをオフにしてください。\n({full})"
        );
    }
    if known("not found on PATH") || known("cannot run ffmpeg") || known("cannot run ffprobe") {
        return FFMPEG_MISSING.into();
    }
    if known("no images found") {
        return format!("画像が見つかりません。画像(png/tif/jpg/bmp)の入ったフォルダを選んでください。\n({full})");
    }
    if known("need at least two frames") || known("no frames to track") {
        return format!(
            "追跡できるフレームが2枚未満です。開始・終了・間隔を確認してください。\n({full})"
        );
    }
    if known("cannot open video") || known("no video stream") || known("ffmpeg failed") {
        return format!("動画を開けませんでした。ファイルが壊れているか、対応していない形式の可能性があります。\n({full})");
    }
    if known("cannot create")
        || known("cannot write")
        || known("Permission denied")
        || known("access is denied")
    {
        return format!("出力フォルダに書き込めません。別のフォルダを選んでください。\n({full})");
    }
    if known("cannot decode") || known("unsupported") {
        return format!("画像を読み込めませんでした。\n({full})");
    }
    format!("エラーが発生しました。\n({full})")
}

/// The message of a run that stopped because the GPU engine failed (it did not start, or it failed
/// in the fine stage). The run is never moved to the CPU by itself: the results would then come
/// from two devices. `has_checkpoint`: a checkpoint of this run is in the folder.
pub fn gpu_failed(e: &anyhow::Error, has_checkpoint: bool) -> String {
    let mut text = String::from(
        "GPU での計算に失敗したため、解析を止めました。CPU に自動で切り替えることはしません(結果を出した装置が途中で変わるのを避けるためです)。\n",
    );
    if has_checkpoint {
        text.push_str(
            "保存済みの再開用データは消していません。GPU のまま「再開」でもう一度試せます(GPU の再開用データは CPU では使えません)。CPU で解析するときは、計算装置を CPU にして最初から解析してください。\n",
        );
    } else {
        text.push_str(
            "もう一度 GPU で試すか、「3. 設定」の計算装置を CPU にして解析してください。\n",
        );
    }
    text.push_str(&format!("({e:#})"));
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;

    #[test]
    fn crack_size_limit_is_explained() {
        let m = describe(&anyhow!(
            "crack_model 'xdic' takes images of at most 8294400 pixels (it holds about 1.1 GB per million pixels), got 4000x3000"
        ));
        assert!(m.contains("ROI") && m.contains("縮小率"));
    }

    #[test]
    fn a_gpu_failure_says_what_was_not_done() {
        let e = anyhow!("GPU command failed: x");
        let with = gpu_failed(&e, true);
        assert!(with.contains("自動で切り替えることはしません"));
        assert!(with.contains("消していません") && with.contains("(GPU command failed: x)"));
        let without = gpu_failed(&e, false);
        assert!(without.contains("計算装置を CPU") && !without.contains("消していません"));
    }

    #[test]
    fn known_errors_are_japanese() {
        assert!(describe(&anyhow!(
            "ffmpeg not found on PATH (needed for video input)"
        ))
        .contains("ffmpeg.exe"));
        assert!(describe(&anyhow!("no images found for x")).contains("画像が見つかりません"));
        assert_eq!(describe(&anyhow::Error::new(Cancelled)), "中止しました。");
        assert!(describe(&anyhow!("weird")).contains("(weird)"));
    }
}
