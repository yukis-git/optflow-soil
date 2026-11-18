# optflow-soil

**日本語** | [English](#english)

## 日本語

土の模型実験の画像・動画から、画素ごとの変位とひずみを位相(Gabor フィルタの局所位相)で追跡するソフトウェア。
Python の参照実装(`python/optflow_ref`)、Rust 製のコマンドライン(`rust/crates/optflow-cli`)、Apple Silicon の
Metal GPU 版(`optflow-gpu`)、デスクトップアプリ(`optflow-app`、eframe/egui)から成ります。

> **ライセンス: [PolyForm Noncommercial License 1.0.0](LICENSE)(非商用に限る)。** 商用利用はできません。
> 依存ライブラリのライセンスは [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)。

### 何をするか
- 参照フレームの ROI(多角形またはマスク)内の土を、後続フレームまで画素単位で追跡し、変位 `ux, uy`、信頼度 `conf`、
  ひずみ `Exx, Eyy, Exy`、最大せん断ひずみ `gmax`、体積ひずみ `vol` を `.npy` で出力します。
- 入力は画像列(ディレクトリ/グロブ)、2 枚の画像、動画(ffmpeg が必要)。
- Rust 版は Python 参照実装と同じ引数・同じファイルを出力するよう作られ、パリティテストで比較しています。

### 必要なもの
- Rust(`rust/rust-toolchain.toml` で固定。rustup が初回に取得)
- Python 3.9 以上(3.9 と 3.12 で検証。Rust CLI の argparse 互換テストは Python 3.12 を使います)と `python/requirements.txt`(参照実装、デモ、テスト用)
- 動画を読むときだけ `ffmpeg` と `ffprobe`(`PATH` 上。同梱はしません)
- GPU 版: Apple Silicon の Mac(Metal)のみ

### ダウンロードして使う(ビルド不要)
[リリースページ](https://github.com/yukis-git/optflow-soil/releases/latest)から zip をダウンロードして展開します。インストーラーはありません。
- **Windows(64 ビット):** `optflow-windows.zip` を展開し、`optflow-app.exe` を起動します。警告が出たら「詳細情報」→「実行」を選びます(署名していないためです)。
- **Mac(Apple Silicon):** `optflow-macos-arm64.zip` を展開し、ターミナルで `cd ~/Downloads/optflow-macos && xattr -dr com.apple.quarantine . && ./optflow-app` を実行します。
- 動画を解析するときだけ `ffmpeg` と `ffprobe` が必要です。アプリと同じフォルダーに置くか、`PATH` を通してください。

### インストールとビルド(ソースから)
```sh
git clone https://github.com/yukis-git/optflow-soil.git && cd optflow-soil
cd rust && cargo build --release -p optflow-cli -p optflow-app   # target/release/optflow, optflow-app
# GPU(macOS のみ): cargo build --release -p optflow-cli -p optflow-app --features optflow-cli/gpu,optflow-app/gpu
cd ../python && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

### CLI クイックスタート(合成画像、手元のデータ不要)
リポジトリのルートから:
```sh
python/.venv/bin/python - <<'PY'
import os, sys
sys.path.insert(0, "python")
import numpy as np
from PIL import Image
from optflow_ref import synth
os.makedirs("qs/frames", exist_ok=True)
pat = synth.make_pattern(256, 256, seed=1)
for k in range(5):  # フレームごとに 0.5 % の一様な伸びと平行移動
    f = synth.uniform_strain(0.005 * k, 0.0, (128, 128), t=(0.8 * k, 0.0))
    img = synth.degrade(synth.render(pat, 256, 256, f), seed=k, noise=0.005)
    Image.fromarray((img * 255).round().astype(np.uint8)).save(f"qs/frames/f{k:02d}.png")
PY
rust/target/release/optflow seq --frames 'qs/frames/*.png' --out qs/out
ls qs/out/frame_00004        # ux.npy uy.npy conf.npy Exx.npy ... valid.npy
```
実際のデータでは `--roi roi.json`(`{"polygon": [[x, y], ...]}`、フル解像度の画素)、`--scale 4`(整数のダウンスケール)、
`--set key=value`(設定の上書き)を使います。`optflow --help` と [rust/README.md](rust/README.md) に全オプションがあります。

### 推奨設定と注意
```sh
optflow seq --frames clip.mp4 --roi roi.json --scale 2 --step 5 \
    --set seq_mode=hybrid --set tavg=3 --set prewarp_freeze_at=3 --out out/
```
- 推奨は `seq_mode=hybrid`、`tavg=3`、`prewarp_freeze_at=3`。**CLI の既定値は従来どおり `seq_mode=chain`** のままです(上のオプションを付けたときだけ変わります)。デスクトップアプリの「推奨」プリセットはこの組に `ref_update_conf=0.3` を加えたものです。
- `ref_update_conf`(参照フレームを更新する信頼度のしきい値)は**データ依存**です。2 本のテスト動画で 1 つの値が両方に合うことはなく、0.3 と 0.5 を使い分けました。自動で決める方式は採用していません。
  参照の切り替え回数と、静止領域のひずみのノイズを見て決めてください。
- 評価は誤差だけでなく**有効画素の割合の時間変化**も合わせて見てください(追跡できていない画素を除いた誤差は良く見えます)。

### Python クイックスタート
```sh
cd python
.venv/bin/python -m optflow_ref seq --frames '../qs/frames/*.png' --out ../qs/out_py
.venv/bin/python -m pytest          # テスト
```
Python 版は参照実装で遅いです(Rust 版が実用向け)。

### 合成デモ(私有データ不要)
```sh
cd python && .venv/bin/python -m demo.run_demo --size small --out ../demo_out
```
フーチングが砂に沈む合成動画を作り、ROI を設定して追跡し、真値と比較して `summary.md`、`overview.png` などを出力します。動画の生成に `ffmpeg` が必要です。
`--size small` は動作確認用で精度は低いです。詳細は [python/demo/README.md](python/demo/README.md)。

### アプリの使い方
`rust/target/release/optflow-app` を起動し、画像列または動画を開き、ROI を指定し、プリセット(推奨/標準)を選んで実行します。
- **ROI:** 画像上をドラッグすると四角形を作ります。作った後は角と辺の中央の点をドラッグして変更でき、辺のダブルクリックで頂点を追加、頂点の右クリックで削除します(クリックで頂点を順に置く多角形も作れます)。
- **解析中の表示:** フレームが終わるごとに分布(gmax、ux、uy、信頼度)を表示します。表示は結果のファイルを変えません。
- **中止と再開:** 「中止」は現在のフレームが終わってから止まり、途中経過を保存します。解析中も定期的に自動保存します(既定 5 分ごと、書き込みが実行時間の約 5 % を超えないように間隔を延ばします)。同じ出力先を開くと「再開」が出ます。入力、ROI、設定、アプリの版がすべて同じときだけ再開でき、再開した結果のファイルは中断しなかった場合とバイト単位で同じです(`timing.json` の時間の値を除く)。再開用データは `run_NNN/checkpoint.ofck` で、完了すると消えます。亀裂モデルでは再開できません。
- **計算するフレーム:** 「2. 範囲と領域」の、フレーム 0 から最後までのバーの 2 つの取っ手で開始フレームと終了フレームを決めます(左が開始、右が終了)。取っ手をドラッグするか、バーの上をクリックすると近い方の取っ手がそこへ動き、フォーカスした取っ手は ←/→ で 1 フレームずつ動きます。取っ手は交差せず、計算するフレームが 2 枚以上残る(終了 ≥ 開始 + 間隔)位置までしか動きません。バーの上に、開始と終了のフレームの縮小画像(「開始 a(t 秒)」「終了 b(t 秒)」、開始のほうには ROI の枠)が並びます。別スレッドで読むので、ドラッグ中も画面は止まりません。固定フレームレートの動画では時刻で直接読むため、後ろのフレームの縮小画像も約 0.5 秒で出ます(6143 フレームの動画で測定)。大きい画像は、開始の取っ手を止めて約 0.3 秒後に開始フレームに切り替わります(動画の後ろのフレームは先頭から読むので数秒かかります)。数値欄(開始・終了・間隔)はそのまま使え、バーと連動します。数値欄がフレーム数の範囲外のときも、バーは端に寄せて描くだけで設定は変えず、実行前に理由を示します。終了の数値は従来どおり「含まない」番号で、バーの右の取っ手はその 1 つ前のフレームです。「計算するフレーム: N 枚(a 〜 b、間隔 s、t1 〜 t2 秒)」の行で、実際に計算するフレームを確認できます。
- **信頼度フィルター:** 「3. 設定」のチェックボックス。ON(既定)は従来どおり、信頼度が低い画素などを除いて出力します。OFF は、ROI 内の全画素の変位(`raw_ux`、`raw_uy` と同じ、最初のフレームに合成した変位)とそこから求めたひずみを出力します。追跡、`conf.npy`、`valid.npy`(信頼度を通った画素。ON と同じ内容)、`fb.npy`、有効率は変わりません。画面・CSV・PNG は ROI 内の全画素を示し、時系列の gmax の p50/p95/p99 は出力した全画素で計算します。OFF の run には `confidence_filter.json` が残ります。亀裂モデルでは ON のままです。
- **最初→最後の分布:** 解析が完了すると `run_NNN/first_last/` に、最後の結果(最初の計算フレームからの変化)の gmax・ux・uy・conf の PNG(最初のフレームに重ねたもの)、`first_last_grid10px.csv`(最後の結果の行)、`summary.txt` を書きます。「5. 書き出し」の「最初→最後の分布を書き出す…」で、表示中の run について同じものを好きなフォルダーに書き出せます(中止した run では計算できた最後のフレームについて書き、`summary.txt` に記します)。
- **動画で書き出す:** 「5. 書き出し」の「動画で書き出す…」で、表示中の run(完了した run も中止した run も、以前の版の run も)の計算した結果を MP4 にします。結果 1 つが動画の 1 コマで、画面と同じ絵(表示中の量を最初のフレームに重ね、ROI の枠を描く)の下に、色の範囲(値と単位)と「frame 番号 t=秒」を入れます。色の範囲は動画全体で 1 つで、画面の範囲を使います(「自動」のときは全結果から決めるので、時間を通して色を比べられます)。ffmpeg(入力に使うもの)へ 1 コマずつ送るため、PNG を大量に書かず、メモリも 1 結果分だけです。エンコーダーは libx264(crf 18)、なければ macOS の h264_videotoolbox、なければ mpeg4 を使い、yuv420p で縦横を偶数にそろえます。再生速度は既定で「元のフレームレート ÷ 間隔」で、「再生速度(fps)」で変えられます。別スレッドで「動画を書き出し中… k / N」と進み、「中止」で書きかけのファイルを消します。完了時に自動では書き出しません。縮小率 2 以上の run は入力画像と大きさが違うため書き出せません。動画の文字は ASCII(数字と量の名前)だけです。
結果の表示(ux, uy, ひずみ、gmax など)、統計、CSV・PNG の書き出しができます。画面は日本語です。
`--headless-smoke OUT` で、窓を開かずに合成画像列で一巡する自己確認ができます。

### 既知の制限
- **ひび割れモデル**(`--set crack_model=xdic`)は**実験的**で、既定ではオフ。ひびの判定しきい値は検証されていません。CPU のみ(`--device gpu` は拒否)。
- **GPU 版は Mac の Metal のみ**(Windows に GPU はありません)。アプリでは GPU 付きでビルドした Mac 版(`cargo build --release -p optflow-app --features gpu`)の「3. 設定」で「計算装置」を選びます。既定は CPU。GPU は CPU と完全には一致せず、推奨設定の実データでは GPU 仕様のロバスト基準を満たしませんでした(`rust/README.md` 参照)。再開は同じ装置でしかできず、失敗時は CPU に切り替えず止まります。
- **Windows の GUI は実機で未検証。** CI(GitHub Actions)でのビルド・ヘッドレス確認のみ。OpenGL 2.0 以上が必要で、リモートデスクトップや仮想マシンでは開かないことがあります。
- macOS 用のバイナリは署名していません(初回は Gatekeeper の警告が出ます)。
- 動画の読み込みには ffmpeg が必要です。
- 精度の数値は、リポジトリ内のテストと上のデモで再現できるものだけを信頼してください。実データでの精度は条件(テクスチャ、照明、変形の大きさ)に依存します。

### ライセンスと引用
PolyForm Noncommercial License 1.0.0。非商用の目的(個人、教育、研究機関などの非営利の利用)に限り使用・改変・再配布できます。
全文は [LICENSE](LICENSE)。商用利用を希望する場合は、著作権者に別途ご相談ください。
引用: [CITATION.cff](CITATION.cff)(GitHub の "Cite this repository")。Sakata, Yuki, optflow-soil, 2026.

---

## English

Dense per-pixel displacement and strain tracking of soil from model-test images and video, using local Gabor phase.
It contains a Python reference implementation (`python/optflow_ref`), a Rust command line (`rust/crates/optflow-cli`),
a Metal GPU backend for Apple Silicon (`optflow-gpu`) and a desktop app (`optflow-app`, eframe/egui).

> **License: [PolyForm Noncommercial License 1.0.0](LICENSE). Noncommercial use only.** Dependencies: see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

### What it does
- Tracks the soil inside an ROI (polygon or mask) of a reference frame to later frames, per pixel, and writes displacement
  `ux, uy`, confidence `conf`, strain `Exx, Eyy, Exy`, maximum shear `gmax` and volumetric strain `vol` as `.npy` files.
- Input: an image sequence (directory/glob), two images, or a video (needs ffmpeg).
- The Rust CLI takes the same arguments and writes the same files as the Python reference; parity tests compare them.

### Requirements
- Rust (pinned in `rust/rust-toolchain.toml`; rustup fetches it on first use)
- Python 3.9+ (verified on 3.9 and 3.12; the Rust CLI argparse-parity test uses Python 3.12) with `python/requirements.txt` (reference implementation, demo, tests)
- `ffmpeg` and `ffprobe` on `PATH`, only for video input (not bundled)
- GPU backend: Apple Silicon Mac (Metal) only

### Download and run (no build needed)
Download a zip from the [releases page](https://github.com/yukis-git/optflow-soil/releases/latest) and unzip it. There is no installer.
- **Windows (64-bit):** `optflow-windows.zip`, then run `optflow-app.exe`. At the warning choose "More info" then "Run anyway" (the build is unsigned).
- **Mac (Apple Silicon):** `optflow-macos-arm64.zip`, then in a terminal: `cd ~/Downloads/optflow-macos && xattr -dr com.apple.quarantine . && ./optflow-app`.
- Video input needs `ffmpeg` and `ffprobe` beside the app or on `PATH`.

### Install and build (from source)
```sh
git clone https://github.com/yukis-git/optflow-soil.git && cd optflow-soil
cd rust && cargo build --release -p optflow-cli -p optflow-app   # target/release/optflow, optflow-app
# GPU (macOS only): cargo build --release -p optflow-cli -p optflow-app --features optflow-cli/gpu,optflow-app/gpu
cd ../python && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

### CLI quick start (synthetic images, no data needed)
Use the script in the Japanese section above (it writes `qs/frames/*.png`), then, from the repository root:
```sh
rust/target/release/optflow seq --frames 'qs/frames/*.png' --out qs/out
ls qs/out/frame_00004
```
With real data add `--roi roi.json` (`{"polygon": [[x, y], ...]}` in full-resolution pixels), `--scale 4` (integer
downscale) and `--set key=value` (setting overrides). `optflow --help` and [rust/README.md](rust/README.md) list everything.

### Recommended configuration and caveats
```sh
optflow seq --frames clip.mp4 --roi roi.json --scale 2 --step 5 \
    --set seq_mode=hybrid --set tavg=3 --set prewarp_freeze_at=3 --out out/
```
- Recommended: `seq_mode=hybrid`, `tavg=3`, `prewarp_freeze_at=3`. **The CLI defaults remain `seq_mode=chain`**; they change only when you pass these options.
  The app's "recommended" preset is this set plus `ref_update_conf=0.3`.
- `ref_update_conf` (the confidence threshold for moving the reference frame) is **data dependent**. No single value suited both
  test videos (0.3 and 0.5 were used), and an automatic choice was tried but not adopted. Check the number of reference switches and the
  strain noise in a static region.
- Judge a setting by the **valid fraction over time** as well as the error: an error computed only over the pixels that were still tracked looks better than it is.

### Python quick start
```sh
cd python
.venv/bin/python -m optflow_ref seq --frames '../qs/frames/*.png' --out ../qs/out_py
.venv/bin/python -m pytest
```
The Python version is the reference and is slow; use the Rust CLI for real work.

### Synthetic demo (no private data)
```sh
cd python && .venv/bin/python -m demo.run_demo --size small --out ../demo_out
```
Synthesises a footing settling into sand, sets an ROI, tracks, compares with the known truth and writes `summary.md`, `overview.png`, ... (needs `ffmpeg` to encode the video).
`--size small` is a smoke test with low accuracy. See [python/demo/README.md](python/demo/README.md) (Japanese).

### Using the app
Run `rust/target/release/optflow-app`, open an image sequence or video, set the ROI, choose a preset (recommended/standard) and run.
- **ROI:** drag on the image to make a rectangle. Then drag corners and edge midpoints to change it, double-click an edge to add a vertex, right-click a vertex to delete it (clicking vertices one by one also draws a polygon).
- **During a run:** the distribution of each finished frame is shown (gmax, ux, uy, confidence). Showing it does not change the output files.
- **Stop and resume:** "中止" stops after the current frame and saves the progress. Progress is also saved periodically during a run (every 5 minutes by default, less often if the write would cost more than about 5 % of the run time). Opening the same output folder offers "再開". A run resumes only with the same input, ROI, settings and app version, and its output files are byte-identical to an uninterrupted run (except the seconds in `timing.json`). The resume data is `run_NNN/checkpoint.ofck`; it is removed when the run finishes. Crack-model runs cannot be resumed.
- **Frames to compute:** in step 2 a bar from frame 0 to the last frame has two handles, the start frame (left) and the end frame (right). Drag a handle, or click the bar and the nearer handle goes there; a focused handle moves one frame with the left/right arrow keys. The handles never cross and stop where at least two frames are still computed (end >= start + step). Above the bar two thumbnails show the start and end frames ("開始 a(t 秒)", "終了 b(t 秒)", with the ROI outline on the start one). They are read on a separate thread, so the window does not freeze while dragging. For a constant-frame-rate video they are read by a time seek, so a late frame appears in about 0.5 s (measured on a 6143-frame video). The big picture switches to the start frame about 0.3 s after the start handle stops (a late frame of a video is decoded from the start, so it takes seconds). The number fields (start, end, step) still work and stay in sync with the bar. A number outside the frame range is drawn at the end of the bar without changing the setting, and the run is refused with the reason. The end number is still exclusive, and the right handle is the frame before it. The line "計算するフレーム: N 枚(a 〜 b, …)" says which frames will be computed.
- **Confidence filter:** a checkbox in step 3. On (default) is the earlier behaviour. Off writes, for every pixel of the ROI, the unmasked displacement (the same as `raw_ux`, `raw_uy`, composed to the first frame in a sequence) and the strain computed from it. Tracking, `conf.npy`, `valid.npy` (the pixels that pass the filter, the same as with it on), `fb.npy` and the valid fraction do not change. The view, CSV and PNG show all ROI pixels; the gmax p50/p95/p99 of the time series are computed over everything written. A run with the filter off leaves `confidence_filter.json`. The crack model keeps it on.
- **First to last:** a finished run writes `run_NNN/first_last/`: PNGs of gmax, ux, uy and conf of the last result (the change from the first computed frame) over the first frame, `first_last_grid10px.csv` (the rows of the last result) and `summary.txt`. "最初→最後の分布を書き出す…" in step 5 writes the same for the run on show into a chosen folder (for a stopped run, for the last frame it computed, and `summary.txt` says so).
- **Video export:** "動画で書き出す…" in step 5 writes the computed results of the run on show (finished or stopped, also a run of an older version) as an MP4. One result is one video frame: the same picture as the canvas (the quantity on show over the first frame, the ROI outline) with a strip below it holding the colour bar (range and unit) and "frame N t=seconds". The colour range is one for the whole video, the one on show (with "auto" it is computed over all the results, so colours compare across time). Frames are piped to the same ffmpeg the app uses for input one at a time: no pile of PNG files, and only one result in memory. The encoder is libx264 (crf 18), else h264_videotoolbox on macOS, else mpeg4; the output is yuv420p with even sides. The playback speed defaults to source frame rate / step and can be changed in "再生速度(fps)". It runs on a separate thread with "動画を書き出し中… k / N" and a cancel button that deletes the partial file. It is not written automatically when a run finishes. A run with scale 2 or more cannot be exported (its maps are not the size of the input image). The text in the video is ASCII only (digits and the name of the quantity).
It shows the fields (ux, uy, strains, gmax), statistics, and exports CSV and PNG. The interface is in Japanese.
`--headless-smoke OUT` runs a self-check on a synthetic sequence without opening a window.

### Known limits
- The **crack model** (`--set crack_model=xdic`) is **experimental**, off by default, its decision threshold is not validated, and it is CPU only (`--device gpu` is refused).
- **GPU: Mac with Metal only** (Windows has no GPU). In the app, build with `cargo build --release -p optflow-app --features gpu` and choose "計算装置" (device) in step 3; the default is CPU. GPU results differ slightly from the CPU and, with the recommended settings on real data, did not meet the GPU spec's robust tier (see `rust/README.md`), a run resumes only on the device that saved it, and a GPU failure stops the run instead of switching to the CPU.
- **The Windows GUI is not verified on real hardware**: it is only built and smoke-tested headless in CI. It needs OpenGL 2.0+ and may not open under remote desktop or in a VM.
- The macOS binary is unsigned (Gatekeeper will warn).
- Video input needs ffmpeg.
- Trust accuracy numbers only where a test in this repository or the demo reproduces them; accuracy on real data depends on texture, lighting and deformation size.

### License and citation
PolyForm Noncommercial License 1.0.0: use, modification and redistribution for noncommercial purposes only (full text in [LICENSE](LICENSE)).
For commercial use, contact the copyright holder. Cite via [CITATION.cff](CITATION.cff): Sakata, Yuki, optflow-soil, 2026.
