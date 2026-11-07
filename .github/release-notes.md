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
