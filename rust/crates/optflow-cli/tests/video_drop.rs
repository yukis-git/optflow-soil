//! Dropping a `VideoFrames` iterator before its end stops ffmpeg and reaps it, so that a process
//! that is not finished decoding does not outlive the iterator. A fake `ffmpeg` writes two frames
//! and then stays alive, as ffmpeg does while its output pipe is full. The test sets `PATH` for
//! its process, so it is the only test of this file.
#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::Command;

use optflow_cli::video::iter_video;

const W: usize = 8;
const H: usize = 6;

fn script(dir: &Path, name: &str, body: &str) {
    let path = dir.join(name);
    std::fs::write(&path, format!("#!/bin/sh\n{body}")).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

fn is_alive(pid: &str) -> bool {
    Command::new("sh")
        .args(["-c", &format!("kill -0 {pid} 2>/dev/null")])
        .status()
        .unwrap()
        .success()
}

#[test]
fn dropping_the_frames_early_stops_ffmpeg() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("video_drop");
    let _ = std::fs::remove_dir_all(&dir);
    let bin = dir.join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    script(
        &bin,
        "ffprobe",
        &format!("printf 'width={W}\\nheight={H}\\n'\n"),
    );
    script(
        &bin,
        "ffmpeg",
        r#"if [ "$1" = "-version" ]; then
  echo "ffmpeg version 7.1.1 Copyright (c) the FFmpeg developers"
  exit 0
fi
echo $$ > "$FAKE_PIDFILE"
cat "$FAKE_FRAMES"
exec sleep 60
"#,
    );
    // two frames of grey levels 40 and 200
    let mut frames = vec![40u8; W * H * 3];
    frames.extend(vec![200u8; W * H * 3]);
    std::fs::write(dir.join("frames.raw"), frames).unwrap();
    let video = dir.join("v.mp4");
    std::fs::write(&video, b"x").unwrap();
    let pidfile = dir.join("ffmpeg.pid");
    std::env::set_var("FAKE_PIDFILE", &pidfile);
    std::env::set_var("FAKE_FRAMES", dir.join("frames.raw"));
    std::env::set_var("PATH", format!("{}:/usr/bin:/bin", bin.display()));

    // a script written a moment ago can be "busy" while the harness forks
    let mut opened = iter_video(&video, 0, None, 1);
    for _ in 0..5 {
        if !opened
            .as_ref()
            .err()
            .is_some_and(|e| e.to_string().contains("busy"))
        {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
        opened = iter_video(&video, 0, None, 1);
    }
    let mut frames = opened.unwrap();
    let first = frames.next().expect("a frame").expect("decoded");
    assert_eq!((first.w, first.h, first.data.len()), (W, H, W * H));
    let pid = std::fs::read_to_string(&pidfile)
        .unwrap()
        .trim()
        .to_string();
    assert!(is_alive(&pid), "the fake ffmpeg should be running");

    // the second frame was never asked for: ffmpeg is still going
    drop(frames);
    let alive = is_alive(&pid);
    if alive {
        // do not leave the sleeping process behind
        let _ = Command::new("kill").arg(&pid).status();
    }
    assert!(!alive, "ffmpeg (pid {pid}) outlived the dropped iterator");
}
