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

## 既知の制限
- **Windows の画面は実機で未確認です。** OpenGL 2.0 以上が必要で、リモートデスクトップや仮想マシンでは開かないことがあります。
- **Mac のアプリの画面も、このビルドでは確認していません。** 自動テストでは画面を開かずに動作を確かめています。
- 亀裂モデルは試験機能で、既定はオフです。判定基準は未検証です。
- GPU での計算は、Mac のコマンドライン版（`--device gpu`）だけです。アプリは CPU で動きます。

---

## Download and run
No installer: unzip and run. **Windows:** `optflow-windows.zip`, run `optflow-app.exe`; at the warning choose "More info" then "Run anyway" (the build is unsigned). **Mac (Apple Silicon):** `optflow-macos-arm64.zip`, then in a terminal: `cd ~/Downloads/optflow-macos && xattr -dr com.apple.quarantine . && ./optflow-app`. Video input needs `ffmpeg` and `ffprobe` beside the app or on `PATH`.

**Known limits:** the app window has not been checked on real Windows hardware or from this macOS build (CI runs it headless). Windows needs OpenGL 2.0+ and may not open under remote desktop or in a VM. The crack model is experimental and off by default. GPU computing is available only in the Mac command line (`--device gpu`).
