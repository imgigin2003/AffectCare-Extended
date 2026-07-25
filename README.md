# 🎙️ AffectCare — Extended

> **A distress signal shouldn't need a button.**

AffectCare is a CNN+LSTM audio classifier that listens for vocal distress — screams, cries for help, panic — and flags it, designed around a single non-negotiable rule: **missing a real emergency is worse than a false alarm.**

**🔗 Live demo: [affectcare-extended.pages.dev](https://affectcare-extended.pages.dev/)**

**This _Extended_ edition wraps the original command-line model in a web app that runs the model entirely in your browser** — record or upload a clip, get a verdict live, with a **Model Insights** tab that visualizes the model's performance and the exact spectrogram each clip produces. There is **no backend**: the PyTorch model is exported to ONNX and the whole `librosa` preprocessing pipeline is reimplemented in JavaScript, so the app is a pure static site on Cloudflare Pages. Nothing you record ever leaves your device.

---

## ✨ What makes this interesting

| | |
| --- | --- |
| **Zero backend** | Prediction, spectrogram, and metrics all run client-side. No server to host, sleep, or pay for. |
| **PyTorch → ONNX** | The trained model runs in-browser via `onnxruntime-web`, verified to match PyTorch to ~1e-6. |
| **librosa → JavaScript** | The full MFCC pipeline (STFT, Slaney mel filterbank, DCT-II) reimplemented in JS and **validated against librosa to ~0 error**. |
| **Private by design** | Audio is decoded and scored locally; refreshing the page wipes everything. No uploads, no storage, no DB. |

---

## 🏗️ Architecture

Everything is a static asset served by Cloudflare Pages. The visitor's browser does all the compute.

```
                Cloudflare Pages  (static hosting)
   ┌──────────────────────────────────────────────────────┐
   │  React + Vite UI                                       │
   │    ├─ best_model.onnx     ← runs via onnxruntime-web   │
   │    ├─ scaler.json         ← StandardScaler params      │
   │    ├─ metrics.json        ← Insights: test metrics     │
   │    └─ loss_history.json    ← Insights: training loss   │
   │                                                        │
   │  In-browser pipeline:                                  │
   │  mic/upload → decode → resample 22.05k → trim →        │
   │  MFCC (JS, librosa-faithful) → StandardScaler → ONNX   │
   └──────────────────────────────────────────────────────┘
             no network calls after load · no server
```

The `ML-Backend/` folder is the **training + export** side: it trains the model and produces the four browser assets via `export_onnx.py`. It isn't part of the live app.

---

## 🔬 The hard part: making the browser match Python exactly

A model is only as good as the features it's fed. The Python pipeline uses `librosa` to turn audio into a `13 × 173` MFCC "image"; to run in the browser without a server, that had to be reproduced in JavaScript **bit-for-bit** — otherwise predictions drift.

[`frontend/src/utils/mfcc.js`](frontend/src/utils/mfcc.js) is a faithful port of `librosa.feature.mfcc` with every default matched: STFT `n_fft=2048`, `hop=512`, periodic Hann window, center zero-padding; a 128-band **Slaney** mel filterbank with Slaney normalization; `power_to_db` with a global `top_db=80` floor; and an **orthonormal DCT-II** over the mel axis.

It's not trusted — it's **verified**. The whole chain (JS MFCC → `scaler.json` → `best_model.onnx`) was run in Node against the Python model on real signals:

```
tone440   JS=0.0866  PY=0.0866   Δ=5.7e-8
noise     JS=0.2015  PY=0.2015   Δ=0
chirp     JS=0.1348  PY=0.1348   Δ=1.1e-7
silence   JS=0.0770  PY=0.0770   Δ=1.4e-17
```

The browser reproduces the Python model to **~1e-7**. What you see in the browser is exactly what the trained model would say.

---

## 🖥️ The web app

### Detect tab
- **Record** — captures raw mic PCM via the Web Audio API (with browser auto-gain / noise-suppression **disabled**, so the signal isn't artificially loudened) and encodes a WAV in-browser.
- **Upload** — `.wav` / `.mp3`, decoded locally.
- **Analyze** — runs the in-browser pipeline and shows a verdict card: green **No distress** or red **Distress detected**, following the higher-probability class, with a confidence meter and the raw distress probability.
- **Dark / light** — theme toggle, persisted, honoring the OS preference.

### Model Insights tab
- **Stat tiles** — Recall, Precision, F1, from the held-out test split.
- **Confusion matrix** — rendered live from the real counts, themed, with each cell labeled (✓ Caught / ✕ Missed / false alarm / true negative) so meaning never rests on color alone.
- **Training-loss curve** — a live SVG line chart of the 40-epoch loss, with a hover crosshair.
- **This clip** — the clip's **waveform** and the exact **MFCC spectrogram** the model reads, both rendered in-browser.

> ℹ️ The confusion matrix and loss curve are **model-level** (they describe the trained model); the waveform and spectrogram are **per-clip**.

---

# 🧠 The ML core

> Refer to the main Repo
### 🎙️ [AffectCare](https://github.com/imgigin2003/AffectCare)
