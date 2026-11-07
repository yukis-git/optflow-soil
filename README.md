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
# GPU(macOS のみ): cargo build --release -p optflow-cli --features gpu
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
結果の表示(ux, uy, ひずみ、gmax など)、統計、CSV・PNG の書き出しができます。画面は日本語です。
`--headless-smoke OUT` で、窓を開かずに合成画像列で一巡する自己確認ができます。

### 既知の制限
- **ひび割れモデル**(`--set crack_model=xdic`)は**実験的**で、既定ではオフ。ひびの判定しきい値は検証されていません。CPU のみ(`--device gpu` は拒否)。
- **GPU 版は Mac の Metal のみ。**
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
# GPU (macOS only): cargo build --release -p optflow-cli --features gpu
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
It shows the fields (ux, uy, strains, gmax), statistics, and exports CSV and PNG. The interface is in Japanese.
`--headless-smoke OUT` runs a self-check on a synthetic sequence without opening a window.

### Known limits
- The **crack model** (`--set crack_model=xdic`) is **experimental**, off by default, its decision threshold is not validated, and it is CPU only (`--device gpu` is refused).
- **GPU: Mac with Metal only.**
- **The Windows GUI is not verified on real hardware**: it is only built and smoke-tested headless in CI. It needs OpenGL 2.0+ and may not open under remote desktop or in a VM.
- The macOS binary is unsigned (Gatekeeper will warn).
- Video input needs ffmpeg.
- Trust accuracy numbers only where a test in this repository or the demo reproduces them; accuracy on real data depends on texture, lighting and deformation size.

### License and citation
PolyForm Noncommercial License 1.0.0: use, modification and redistribution for noncommercial purposes only (full text in [LICENSE](LICENSE)).
For commercial use, contact the copyright holder. Cite via [CITATION.cff](CITATION.cff): Sakata, Yuki, optflow-soil, 2026.
