## v0.4.2 の変更
- **時間平均 `tavg` の先頭・末尾のずれを直しました:** `tavg`(アプリの「推奨」は 3)は各フレームを前後 `tavg` 枚との平均にします。動画の先頭と末尾では片側のフレームがなく、平均が時間的に片寄って(`tavg` 3 で最大 1.5 フレーム)、先頭の平均フレームは全フレームの基準なので、**動画の先頭か末尾が動いているとき変位が実際より小さく(遅れて)出ていました**。v0.4.2 では、片側になる窓でも同じ枚数を平均しますが、平均の前に隣のフレームを中心のフレームに位置合わせします(`tavg_align` の dense モデルと同じ粗い流れを使います)。位置合わせに失敗したとき、また画像が位相窓より小さいときは従来どおりの平均です。窓が両側にそろうフレームと、`tavg_align` を自分で指定した実行は変わりません。
- **影響を受ける人:** `tavg` を使う実行(アプリの「推奨」)で、計算する最初か最後のフレームが入力動画の先頭か末尾に近く、そのときすでに動いていた場合だけです。`--start` が `tavg` 以上で、動画の末尾に届かない範囲なら結果は同じです。`tavg` なし(CLI の既定、アプリの「標準」)の出力は v0.4.1 とバイト単位で同じです。
- **確かめたこと:** 合成のフーチング(60 フレーム)の最終フレームの変位誤差 RMS は、推奨設定で 0.382 px から 0.146 px になりました(CLI の既定は 0.123 px)。実データ(1920×1080、推奨設定、15 フレーム間隔の 39 結果)の有効画素の割合は、フレーム 0 から: 平均 0.763 → 0.761、最小 0.607 → 0.607。動画の末尾まで: 平均 0.589 → 0.589、最小 0.539 → 0.538。フレーム 600 から: 同じ。位置合わせの追加時間は、先頭か末尾の窓 1 つにつき約 0.5 秒(1920×1080、CPU)です。
- アプリの画面、出力の形式、コマンドラインのオプションは v0.4.1 から変えていません。再開用データには版が入っているので、v0.4.1 で中止した run は v0.4.2 では再開できません(最初からやり直します)。

## v0.4.1 の変更
- **開始・終了フレームはバーで選びます:** 「2. 範囲と領域」の「確認するフレーム」のスライダーと「このフレームを開始に/終了に」のボタンをやめ、フレーム 0 から最後までの 1 本のバーに 2 つの取っ手(左が開始、右が終了)を置きました。取っ手のドラッグ、バーのクリック(近い方の取っ手が動く)、フォーカスした取っ手の ←/→ キーで動かせます。取っ手は交差せず、計算するフレームが 2 枚以上残る位置までしか動きません。数値欄(開始・終了・間隔)はそのまま使え、バーと連動します。終了の数値は従来どおり「含まない」番号で、右の取っ手はその 1 つ前のフレームです。
- **サムネイル:** バーの上に、開始と終了のフレームの縮小画像(「開始 a(t 秒)」「終了 b(t 秒)」、開始には ROI の枠)を並べます。別スレッドで読むのでドラッグ中も止まらず、6143 フレームの動画で後ろのフレーム(1843、4299)の縮小画像が約 0.5 秒で出ました(Mac、固定フレームレートの動画を時刻で直接読む)。大きい画像は、開始の取っ手を止めて約 0.3 秒後に開始フレームへ切り替わります(動画の後ろのフレームは先頭から読むので、同じ動画で約 8 秒)。左のパネルの既定の幅を 350 から 500 に広げました(縮小画像を 2 枚並べるため。幅が足りないときは縦に並びます)。
- **動画で書き出し:** 「5. 書き出し」の「動画で書き出す…」で、表示中の run(中止した run、以前の版の run も)の計算した結果を MP4 にします。結果 1 つが 1 コマで、画面と同じ絵(表示中の量を最初のフレームに重ね、ROI の枠)の下に、色の範囲(値と単位)と「frame 番号 t=秒」を入れます。色の範囲は動画全体で 1 つ(「自動」は全結果から決める)。ffmpeg に 1 コマずつ送るので PNG を大量に書かず、メモリは 1 結果分です。エンコーダーは libx264(crf 18)、なければ h264_videotoolbox(Mac)、なければ mpeg4。再生速度は既定で元のフレームレート ÷ 間隔で、「再生速度(fps)」で変えられます。進み具合(「動画を書き出し中… k / N」)と「中止」(書きかけのファイルを消す)があります。完了時の自動書き出しはしません。実データ(1920×1080、39 結果、gmax)で書き出しは 2.9 秒、5.5 MB でした(Mac、libx264)。動画の文字は ASCII だけです。
- 追跡・出力・再開用データの指紋・CLI は v0.4.0 から変えていません。

### 仕様の一覧(v0.4.2)
| 項目 | 内容 | 根拠 |
|---|---|---|
| 配布物と OS | Windows(64 ビット)`optflow-windows.zip`、Mac(Apple Silicon)`optflow-macos-arm64.zip`。インストーラーなし、署名なし | 本ノートの「ダウンロードと起動」、README |
| 入力 | 画像列(png、jpg、tif、bmp。ディレクトリ/グロブ)、2 枚の画像、動画(mp4、mov、avi、mkv、m4v。ffmpeg と ffprobe が別に必要)。復号した 1 枚は最大 2 GiB | `optflow-cli/src/io.rs`、`rust/README.md` |
| 計算装置 | CPU(既定)。Mac 版のみ GPU(Metal)を選べる。Windows に GPU はない。亀裂モデルは CPU のみ | README、`rust/README.md` |
| 計算モード | `seq_mode`: chain(CLI の既定)/ hybrid。アプリの「推奨」= hybrid、`tavg` 3、`prewarp_freeze_at` 3、`ref_update_conf` 0.3。「標準」= 既定値 | `optflow-core/src/config.rs`、README |
| 既定の追跡の条件 | 格子 16 px、位相窓 64 px、最大変位 32 px、波長 8 px × (0.6, 1.0, 1.6)、ひずみ窓 15 | `metrics.json` の `tracker_config`(下の実測) |
| 主な出力 | `ux`、`uy`、`conf`、`Exx`、`Eyy`、`Exy`、`gmax`、`vol`、`valid`、`fb`(.npy)、結果の CSV・PNG・MP4(動画で書き出す)、`first_last/`(gmax・ux・uy・conf の PNG、CSV、`summary.txt`) | README、v0.4.0 のノート |
| 計算範囲 | 開始・終了(含まない)・間隔、ROI(多角形またはマスク)、縮小率 `scale`。計算するフレームが 2 枚未満の範囲は実行前に断る | README、`range.rs` |
| 信頼度フィルター | ON(既定)/ OFF(ROI 内の全画素を出力。追跡・`conf`・`valid` は不変) | README |
| 中止と再開 | 中止は現在のフレームが終わってから。自動保存(既定 5 分)。入力・ROI・設定・アプリの版・計算装置が同じときだけ再開でき、結果は中断しない場合とバイト単位で同じ | README |
| ライセンス | PolyForm Noncommercial 1.0.0(非商用に限る) | `LICENSE` |
| 速度の目安 | 1 結果あたり、1920×1080(ROI 1720×850、推奨設定): CPU 7.8〜9.7 秒、GPU 3.5〜4.2 秒。6240×4160(ROI 5000×2500、推奨設定): CPU 113〜124 秒、GPU 42〜46 秒(Apple M4、他のジョブが動く中での参考値)。最大メモリは 1920×1080 で CPU 約 1.7 GB / GPU 約 4.0 GB、6240×4160 で CPU 約 7.7 GB / GPU 約 8.0 GB | v0.4.0 のノートの「既知の制限」、`rust/README.md` |

### 精度の目安(CPU、CLI の既定の設定 = chain、`tavg` なし。「推奨設定」の行だけ v0.4.2 で実測、他は v0.4.1 で実測し v0.4.2 でも変わりません)
精度は画像のテクスチャ、ノイズ、変形の大きさに依存します。下の値は合成画像で真値と比べたもので、実データの精度ではありません。
| 項目 | 条件 | 値 | 実行したコマンド |
|---|---|---|---|
| 変位の精度(一様な変形) | 512×512 の合成画像 2 枚(1 組)、ノイズなし。平行移動 0.5 px(+ 0.3 px)、一様ひずみ 0.001、0.01、0.1、単純せん断 0.1、回転 0.5〜5 度 | 変位誤差 RMS 3.4e-4 px(平行移動)、1.3e-4 / 2.9e-4 / 9.3e-4 px(ひずみ 0.001 / 0.01 / 0.1)、6.9e-4 px(せん断 0.1)、2.5〜2.9e-4 px(回転)。p99 は最大 2.3e-3 px | `python -m bench.run_synth --impl rust --pairs-only --no-save-fields`(`python/`、`shift_0.5`、`uniform_*`、`shear_0.1`、`rot_*`) |
| ひずみの精度(一様な変形) | 同上 | ひずみ誤差 RMS(Exx / gmax): ひずみ 0.001 で 8.7e-6 / 9.6e-6、0.01 で 1.7e-5 / 3.0e-5、0.1 で 9.4e-5 / 1.2e-4。せん断 0.1 の gmax 8.8e-5 | 同上 |
| ノイズへの強さ | 同じ一様ひずみ 0.01、8 ビット量子化、ガウスノイズ σ = 0.005 / 0.01 / 0.02(輝度 0〜1) | 変位 RMS 0.035 / 0.069 / 0.14 px、gmax RMS 3.2e-3 / 6.0e-3 / 1.2e-2 | 同上(`noise_*`) |
| 変位の精度(沈下するフーチング) | 1280×720、60 フレーム、沈下 30 px(s/B = 0.2)、ROI 内、H.264 の動画、最終フレーム | 変位誤差 RMS 0.123 px、p99 0.683 px、最大 1.12 px。沈下量 20 px 付近(フレーム 39)で RMS 0.095 px、10 px 付近(フレーム 20)で 0.032 px。せん断帯の中(最大 gmax の 40 % 以上の画素)は 0.437 px、帯の外は 0.107 px | `python -m demo.run_demo --no-animation`(`python/`、`summary.md` の 4 と 6、`python/demo/README.md` の 0.123 px と一致) |
| ひずみの精度(沈下するフーチング) | 同上、最終フレーム(真値の gmax は最大 約 2.3、追跡が有効な画素の 99.9 パーセンタイルで約 1.1) | ひずみ誤差 RMS 帯の外 / 帯の中: Exx 0.009 / 0.046、Eyy 0.013 / 0.194、Exy 0.007 / 0.047、gmax 0.023 / 0.243、vol 0.003 / 0.015。フレーム 20 は gmax 0.004 / 0.026 | 同上 |
| 追跡率 | 同上 | ROI に対する有効画素の割合は最小 0.852(最終フレーム)、ROI の縁から 12.2 px のガードの外では最小 0.942。一様な変形の合成画像(512×512)では 1.000。実データ(1920×1080 の動画、ROI あり、推奨設定、最後の結果): 0.932 | 同上、`results` の v0.4.0 のノート |
| 推奨設定での同じ試験 | 上と同じフーチング、`--set seq_mode=hybrid --set tavg=3 --set prewarp_freeze_at=3 --set ref_update_conf=0.3` | 最終フレームの変位誤差 RMS 0.146 px、p99 0.777 px、gmax RMS 帯の外 0.024 / 中 0.254、有効画素の割合は最小 0.854(ガードの外で 0.944)。この合成の動きは 1 フレームあたり約 0.5 px と速く、動画の先頭から全速で動きます(v0.4.1 では先頭・末尾の時間平均の片寄りで RMS 0.382 px でした)。フレーム数を変えた同じ沈下 30 px では、30 フレーム(1 フレーム約 1 px)で RMS 0.160 px、120 フレームで 0.138 px(CLI の既定は 0.123 / 0.124 px)| 上の `run_demo` に `--set` を付けたもの |
| GPU と CPU の差 | 同じ入力・同じ設定。実データ(1920×1080 の動画 6 枚、6240×4160 の写真 4 枚、ROI あり、推奨設定) | ROI 内の有効率の差は最大 0.0003 ポイント。変位の差は 1080p で p99 が最大 約 4e-5 px、一部の画素で約 0.01 px、gmax で約 0.015。512×512 の合成画像では、真値との誤差の差は最大 2.25e-8 px | `rust/README.md`(2026-11-07/08 の測定。GPU 仕様のロバスト基準は未達) |

## v0.4.2 changes
- **Fixed the lag of the time average `tavg` at the start and end of a video:** `tavg` (3 in the app's "推奨" preset) averages every frame with `tavg` frames on either side. At the first and last frames of a video one side is missing, so the average was off-centre in time (by up to 1.5 frames for `tavg` 3), and because the averaged first frame is the reference of every frame, **displacements came out smaller than the truth (lagging) whenever the video was already moving at its start or end**. In v0.4.2 a window that is one-sided still averages the same number of frames, but first aligns the neighbours to the centre frame (the dense coarse flow of `tavg_align`). When the alignment fails, or the image is smaller than the phase window, the plain mean is used as before. Frames whose window is complete, and runs where you set `tavg_align` yourself, are unchanged.
- **Who is affected:** only runs with `tavg` (the app's "推奨") whose first or last computed frame is near the start or end of the input video and was already moving there. A range that starts at or after frame `tavg` and ends before the end of the video gives the same results. Without `tavg` (command-line default, the app's "標準") the outputs are byte-identical to v0.4.1.
- **Checked:** on the synthetic footing (60 frames) the last-frame displacement RMS of the recommended settings goes from 0.382 px to 0.146 px (command-line default: 0.123 px). On real data (1920×1080, recommended settings, 39 results every 15 frames) the valid fraction from frame 0 is mean 0.763 → 0.761, min 0.607 → 0.607; to the end of the video mean 0.589 → 0.589, min 0.539 → 0.538; from frame 600 identical. The alignment costs about 0.5 s per start or end window (1920×1080, CPU).
- The app's screens, the output formats and the command-line options are unchanged from v0.4.1. The resume data contains the version, so a run stopped in v0.4.1 cannot be resumed in v0.4.2 (start it again).

## v0.4.1 changes
- **Start and end frames on one bar:** the "確認するフレーム" slider and the "このフレームを開始に/終了に" buttons of step 2 are gone. One bar from frame 0 to the last frame carries two handles (left: start, right: end). Drag a handle, click the bar (the nearer handle goes there), or use the left/right arrow keys on a focused handle. The handles never cross and stop where at least two frames are still computed. The number fields (start, end, step) still work and stay in sync with the bar. The end number is still exclusive; the right handle is the frame before it.
- **Thumbnails:** above the bar, small pictures of the start and end frames ("開始 a(t 秒)", "終了 b(t 秒)", the ROI outline on the start one). They are read on a separate thread, so dragging never stalls; on a 6143-frame video the thumbnails of late frames (1843, 4299) appeared in about 0.5 s (Mac, constant-frame-rate video read by a time seek). The big picture switches to the start frame about 0.3 s after the start handle stops (a late frame of a video is decoded from the start: about 8 s on the same video). The default width of the left panel went from 350 to 500 so that two thumbnails fit side by side (they stack when the panel is narrower).
- **Video export:** "動画で書き出す…" in step 5 writes the computed results of the run on show (also a stopped run, or a run of an older version) as an MP4. One result is one video frame: the canvas picture (the quantity on show over the first frame, the ROI outline) with a strip holding the colour bar (range and unit) and "frame N t=seconds". One colour range for the whole video ("auto" is computed over all results). Frames are piped to ffmpeg one at a time, so no pile of PNGs and one result in memory. Encoder: libx264 (crf 18), else h264_videotoolbox (Mac), else mpeg4. The playback speed defaults to source frame rate / step and can be changed in "再生速度(fps)". There is progress ("動画を書き出し中… k / N") and a cancel button (the partial file is deleted). Nothing is written automatically when a run finishes. On real data (1920x1080, 39 results, gmax) the export took 2.9 s and 5.5 MB (Mac, libx264). The text in the video is ASCII only.
- Tracking, outputs, the fingerprint of the resume data and the command line are unchanged from v0.4.0.

### Specifications (v0.4.2)
| Item | Content | Source |
|---|---|---|
| Builds and OS | Windows (64-bit) `optflow-windows.zip`, Mac (Apple Silicon) `optflow-macos-arm64.zip`; no installer, unsigned | "Download and run" below, README |
| Input | image sequence (png, jpg, tif, bmp; directory or glob), two images, video (mp4, mov, avi, mkv, m4v; needs `ffmpeg` and `ffprobe` installed separately); one decoded image at most 2 GiB | `optflow-cli/src/io.rs`, `rust/README.md` |
| Device | CPU (default); GPU (Metal) only in the Mac build; no GPU on Windows; the crack model is CPU only | README, `rust/README.md` |
| Sequence modes | `seq_mode`: chain (CLI default) / hybrid. The app's "推奨" preset = hybrid, `tavg` 3, `prewarp_freeze_at` 3, `ref_update_conf` 0.3; "標準" = defaults | `optflow-core/src/config.rs`, README |
| Default tracker | grid 16 px, phase window 64 px, max displacement 32 px, wavelength 8 px x (0.6, 1.0, 1.6), strain window 15 | `tracker_config` in `metrics.json` (measurement below) |
| Main outputs | `ux`, `uy`, `conf`, `Exx`, `Eyy`, `Exy`, `gmax`, `vol`, `valid`, `fb` (.npy), CSV, PNG and MP4 (video export) of the results, `first_last/` (PNGs of gmax, ux, uy, conf, CSV, `summary.txt`) | README, v0.4.0 notes |
| Range | start, end (exclusive), step, ROI (polygon or mask), downscale `scale`; a range with fewer than 2 computed frames is refused before the run | README, `range.rs` |
| Confidence filter | on (default) / off (every ROI pixel is written; tracking, `conf`, `valid` unchanged) | README |
| Stop and resume | stops after the current frame; autosave (every 5 min by default); resumes only with the same input, ROI, settings, app version and device, and then the result files are byte-identical to an uninterrupted run | README |
| License | PolyForm Noncommercial 1.0.0 (noncommercial use only) | `LICENSE` |
| Speed (reference) | per result, 1920x1080 (ROI 1720x850, recommended settings): CPU 7.8-9.7 s, GPU 3.5-4.2 s; 6240x4160 (ROI 5000x2500, recommended settings): CPU 113-124 s, GPU 42-46 s (Apple M4, with other jobs running). Peak memory: 1920x1080 CPU about 1.7 GB / GPU about 4.0 GB; 6240x4160 CPU about 7.7 GB / GPU about 8.0 GB | "Known limits" of the v0.4.0 notes, `rust/README.md` |

### Accuracy (CPU, command-line default settings = chain, no `tavg`; only the recommended-settings row was measured for v0.4.2, the others for v0.4.1 and unchanged in v0.4.2)
Accuracy depends on texture, noise and the size of the deformation. The values below are against the truth of synthetic images, not accuracy on real data.
| Item | Conditions | Value | Command run |
|---|---|---|---|
| Displacement accuracy (uniform deformations) | 512x512 synthetic pair, no noise: translation 0.5 px (+ 0.3 px), uniform strain 0.001, 0.01, 0.1, simple shear 0.1, rotation 0.5-5 deg | displacement error RMS 3.4e-4 px (translation), 1.3e-4 / 2.9e-4 / 9.3e-4 px (strain 0.001 / 0.01 / 0.1), 6.9e-4 px (shear 0.1), 2.5-2.9e-4 px (rotation); p99 at most 2.3e-3 px | `python -m bench.run_synth --impl rust --pairs-only --no-save-fields` (in `python/`; cases `shift_0.5`, `uniform_*`, `shear_0.1`, `rot_*`) |
| Strain accuracy (uniform deformations) | same | strain error RMS (Exx / gmax): 8.7e-6 / 9.6e-6 at strain 0.001, 1.7e-5 / 3.0e-5 at 0.01, 9.4e-5 / 1.2e-4 at 0.1; gmax 8.8e-5 at shear 0.1 | same |
| Noise | uniform strain 0.01, 8-bit, Gaussian noise sigma = 0.005 / 0.01 / 0.02 (intensity 0 to 1) | displacement RMS 0.035 / 0.069 / 0.14 px, gmax RMS 3.2e-3 / 6.0e-3 / 1.2e-2 | same (`noise_*`) |
| Displacement accuracy (settling footing) | 1280x720, 60 frames, settlement 30 px (s/B = 0.2), inside the ROI, H.264 video, last frame | displacement error RMS 0.123 px, p99 0.683 px, max 1.12 px; RMS 0.095 px at about 20 px settlement (frame 39) and 0.032 px at about 10 px (frame 20); inside the shear band (pixels with at least 40 % of the largest gmax) 0.437 px, outside 0.107 px | `python -m demo.run_demo --no-animation` (in `python/`; `summary.md` sections 4 and 6; matches the 0.123 px in `python/demo/README.md`) |
| Strain accuracy (settling footing) | same, last frame (true gmax up to about 2.3, about 1.1 at the 99.9th percentile of the tracked pixels) | strain error RMS outside / inside the band: Exx 0.009 / 0.046, Eyy 0.013 / 0.194, Exy 0.007 / 0.047, gmax 0.023 / 0.243, vol 0.003 / 0.015; frame 20: gmax 0.004 / 0.026 | same |
| Tracking rate | same | valid fraction of the ROI at least 0.852 (last frame), at least 0.942 outside the 12.2 px edge guard; 1.000 on the 512x512 uniform cases. Real data (1920x1080 video, ROI, recommended settings, last result): 0.932 | same; v0.4.0 notes |
| Same test with the recommended settings | the footing above with `--set seq_mode=hybrid --set tavg=3 --set prewarp_freeze_at=3 --set ref_update_conf=0.3` | last-frame displacement RMS 0.146 px, p99 0.777 px, gmax RMS 0.024 outside / 0.254 inside the band, valid fraction at least 0.854 (0.944 outside the guard). The scene is fast (about 0.5 px per frame) and moves at full speed from the first frame (v0.4.1 gave RMS 0.382 px because the time average was off-centre at the ends). The same 30 px settlement over 30 frames (about 1 px per frame) gives RMS 0.160 px, over 120 frames 0.138 px (command-line default: 0.123 / 0.124 px) | the `run_demo` above with `--set` |
| GPU against CPU | same input and settings; real data (a 1920x1080 video, 6 frames, 6240x4160 photos, 4 frames, ROI, recommended settings) | valid fraction differs by at most 0.0003 points; displacement differs by at most about 4e-5 px at p99 on 1080p, about 0.01 px at a few pixels, about 0.015 in gmax; on the 512x512 synthetic suite the error against the truth differs by at most 2.25e-8 px. The GPU spec's robust tier was not met | `rust/README.md` (measured 2026-11-07/08) |

## v0.4.0 の変更
- **計算するフレームの選択:** 「2. 範囲と領域」のスライダーで入力の任意のフレームを画像に表示でき(ROI は重ねたまま)、「このフレームを開始に」「このフレームを終了に」と「計算するフレーム: N 枚(…)」の行で範囲を決められます。成り立たない範囲は実行前に断ります。
- **信頼度フィルターの ON/OFF:** 「3. 設定」で OFF にすると、信頼度が低い画素も含め ROI 内の全画素の変位とひずみを出力します。追跡、`conf`、`valid.npy`、`fb.npy`、有効率は変わりません。既定は ON で、このとき出力は v0.3.0 とバイト単位で同じです(`timing.json` の時間の値を除く)。実データ(1920×1080 の動画の先頭 6 結果、ROI あり、推奨設定)の最後の結果では、ROI 内で有効率は 93.2 %(ON/OFF とも)、変位が出る画素は ON で 93.2 %、OFF で 100 % でした。
- **最初→最後の分布:** 完了した run に `first_last/`(gmax・ux・uy・conf の PNG、最後の結果の CSV、`summary.txt`)を書き、「5. 書き出し」のボタンで任意の run について書き出せます。
- 再開用データには信頼度フィルターの設定が入ります。v0.3.0 までの再開用データは使えません(版が違うため、これまでどおり)。

## v0.4.0 changes
- **Choosing the frames to compute:** a slider in step 2 shows any frame of the input on the canvas (the ROI stays drawn), with "このフレームを開始に" / "このフレームを終了に" and a line "計算するフレーム: N 枚(…)". A range that cannot run is refused before the run.
- **Confidence filter on/off:** off in step 3 writes the displacement and strain for every ROI pixel, low confidence or not. Tracking, `conf`, `valid.npy`, `fb.npy` and the valid fraction do not change. The default is on, and then the output is byte-identical to v0.3.0 (except the seconds in `timing.json`). On real data (the first 6 results of a 1920x1080 video, ROI, recommended settings) in the last result, the valid fraction was 93.2 % (on and off), and the share of ROI pixels with a displacement was 93.2 % on and 100 % off.
- **First to last:** a finished run gets `first_last/` (PNGs of gmax, ux, uy, conf, the CSV of the last result, `summary.txt`), and a button in step 5 writes the same for any run on show.
- The resume data holds the confidence-filter setting. Resume data of v0.3.0 and earlier cannot be used (another app version, as before).

**ライセンス：PolyForm Noncommercial 1.0.0（非商用に限る）** / License: noncommercial use only.

## ダウンロードと起動
インストーラーはありません。zip を展開して実行します。

| OS | ファイル |
|---|---|
| Windows（64 ビット） | `optflow-windows.zip` |
| Mac（Apple Silicon） | `optflow-macos-arm64.zip` |

**Windows：** zip を展開し、`optflow-app.exe` を起動します。警告が出たら「詳細情報」→「実行」を選びます（署名していないためです）。

**Mac：** zip を展開し、ターミナルで次を実行します（署名していないため、実行の許可が必要です）。
```
cd ~/Downloads/optflow-macos
xattr -dr com.apple.quarantine .
./optflow-app
```

動画を解析するときだけ `ffmpeg` と `ffprobe` が必要です。アプリと同じフォルダーに置くか、PATH を通してください。

## アプリの操作
- ROI は画像上のドラッグで作り、角と辺の中央の点をドラッグして変更できます。
- 解析中はフレームが終わるごとに分布を表示します。
- 「中止」は現在のフレームが終わってから止まり、途中経過を保存します。解析中も定期的に自動保存します。同じ出力先を開くと「再開」できます。再開した結果のファイルは、中断しなかった場合とバイト単位で同じです(`timing.json` の時間の値を除く)。亀裂モデルでは再開できません。

## 既知の制限
- **Windows の画面は実機で未確認です。** OpenGL 2.0 以上が必要で、リモートデスクトップや仮想マシンでは開かないことがあります。
- **Mac の画面は、プログラムからの入力で一連の操作を確認しました。** 実際のマウスの操作感は未確認です。
- 「中止」は処理中のフレームが終わるまで待ちます。6240×4160 の全面では 2〜6 分かかることがあります。
- 再開は、中断時と同じアプリの版でだけできます。アプリを更新すると、以前の版で中断した解析は再開できません。
- 長い動画での再開にかかる時間は未測定です。動画は先頭から読み直すため、位置が後ろほど時間がかかります。
- `scale` が 2 以上のとき、解析中の分布は画像に重ねて表示されません。
- 亀裂モデルは試験機能で、既定はオフです。判定基準は未検証です。
- Mac 版のアプリでは「3. 設定」の「計算装置」で CPU と GPU(Metal)を選べます。既定は CPU です。GPU の結果は CPU の結果と最後の桁までは一致せず、GPU の使用メモリは CPU より多くなります(1920×1080 の連番で最大約 4.0 GB、CPU は約 1.7 GB。6240×4160 では CPU 約 7.7 GB、GPU 約 8.0 GB)。再開用データは計算した装置でしか再開できません。GPU が失敗したときは CPU に自動で切り替えず、止まります。亀裂モデルは CPU のみです。Windows 版に GPU はありません。実データ(1920×1080 の動画から 6 枚、6240×4160 の写真から 4 枚、ROI あり、推奨設定)で GPU と CPU の結果を比べると、ROI 内の有効率の差は最大 0.0003 ポイントでしたが、GPU 仕様の「ロバスト」基準は満たしませんでした(1080p は全フレームで ux/uy の p99 などが限度 1e-5 px を超え、p99 の差は最大でも約 4e-5 px、6240×4160 は信頼度の RMS だけ超過)。そのため既定は CPU です。一部の画素では変位で約 0.01 px、gmax で約 0.015 の差が出るので、局所的な gmax を厳密に比べるときは CPU を勧めます。速さは、アプリの 1 結果あたりで 1920×1080(ROI 1720×850、推奨設定)が CPU 7.8〜9.7 秒に対し GPU 3.5〜4.2 秒、6240×4160(ROI 5000×2500、推奨設定)が CPU 113〜124 秒に対し GPU 42〜46 秒でした(Apple M4、他のジョブが動く中での参考値)。

---

## Download and run
No installer: unzip and run. **Windows:** `optflow-windows.zip`, run `optflow-app.exe`; at the warning choose "More info" then "Run anyway" (the build is unsigned). **Mac (Apple Silicon):** `optflow-macos-arm64.zip`, then in a terminal: `cd ~/Downloads/optflow-macos && xattr -dr com.apple.quarantine . && ./optflow-app`. Video input needs `ffmpeg` and `ffprobe` beside the app or on `PATH`.

**In the app:** drag on the image to make the ROI, then drag corners and edge midpoints to change it. The distribution of each finished frame is shown during a run. "中止" stops after the current frame and saves the progress, progress is also saved periodically, and opening the same output folder offers "再開". A resumed run's output files are byte-identical to an uninterrupted run (except the seconds in `timing.json`). Crack-model runs cannot be resumed.

**Known limits:** the app window has not been checked on real Windows hardware (CI runs it headless). Windows needs OpenGL 2.0+ and may not open under remote desktop or in a VM. On Mac the full flow was driven with programmatic input; real mouse feel is unchecked. "中止" waits for the current frame, which can take 2-6 minutes on a whole 6240x4160 image. A run resumes only with the app version that started it, so updating the app means earlier stopped runs cannot be resumed. Resume time on long videos is unmeasured; the decoder reads from the start, so it grows with position. At `scale` 2 or more the live map is not overlaid on the image. The crack model is experimental and off by default. In the Mac app, "計算装置" (settings, step 3) chooses CPU (default) or GPU (Metal); Windows has no GPU. GPU results agree with the CPU only to within the tolerances of the GPU spec, not bit for bit; a stopped run resumes only on the device that saved it; a GPU failure stops the run and never switches to the CPU; the crack model is CPU only; peak memory was up to about 4.0 GB on GPU against about 1.7 GB on CPU at 1920x1080, and about 8.0 against 7.7 GB at 6240x4160. On real data (a 1920x1080 video, 6240x4160 photos, ROI, recommended settings) the valid fraction differed by at most 0.0003 percentage points, but the GPU spec's robust tier was not met (1080p: ux/uy p99 and others over the 1e-5 px limit on every frame, p99 difference up to about 4e-5 px; 6240x4160: only the confidence RMS), so the default stays CPU. Speed could only be measured on a Mac busy with other jobs, so it was not re-measured; the figures below are from the earlier Mac command line measurement. With the same settings, seconds per frame were 8.6 on CPU and 3.4 on GPU at 1920x1080, and 113 on CPU and 44 on GPU at 6240x4160 (measured on an Apple M4).
