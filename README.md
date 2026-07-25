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

The model itself — unchanged from the original AffectCare project, under [`ML-Backend/`](ML-Backend/).

## Why CNN + LSTM

Raw audio is just a long list of pressure values. The pipeline first converts each clip into an **MFCC spectrogram** — a 2D "image" of the sound (frequency × time).

- **CNN** — scans that image for local spatial patterns (the _texture_ of a scream vs. a hum)
- **LSTM** — reads the CNN's output as a sequence, capturing how those patterns unfold _over time_

A CNN sees shapes but not sequence; an LSTM needs something structured to read. Together they cover both.

## Why recall over precision

A **missed distress signal** (false negative) can cost a life; a **false alarm** costs a caregiver a few minutes. So training deliberately optimizes for catching every emergency:

- `BCEWithLogitsLoss` with `pos_weight=1.5` — penalizes missed distress harder than false alarms
- A **low decision threshold (0.15)** — even moderate suspicion can trigger an alert
- **F1-based early stopping**, not raw recall

### The trap I hit and fixed
Saving the "best" checkpoint by **recall alone** produced a model that flagged almost everything — 96% recall, but 72/80 normal sounds triggered false alarms. Switching the checkpoint criterion to **F1** fixed it. The single most important debugging lesson of the project.

## Results

Threshold = 0.15, held-out 20% test split (160 files):

| Metric | Score |
| --- | --- |
| Recall (emergencies caught) | **88.75%** |
| Precision (real alarms) | 68.93% |
| F1 Score | 77.60% |

|                     | Predicted normal   | Predicted distress |
| ------------------- | ------------------ | ------------------ |
| **Actual normal**   | 48 (true negative) | 32 (false alarm)   |
| **Actual distress** | 9 (missed)         | 71 (caught)        |

## Dataset

~800 clips, balanced 50/50:

| Class | Source |
| --- | --- |
| `distress` | Kaggle [Human Scream Dataset](https://www.kaggle.com/datasets/whats2000/human-screaming-detection-dataset) |
| `normal` | [ESC-50](https://github.com/karolpiczak/ESC-50) environmental sounds + calm speech |

**The siren confusion** — a dataset audit flagged ESC-50 siren clips (`-42`) as acoustically close to screams: a real case of spurious correlation, not a hypothetical.

**The whisper test** — a genuine recorded whisper was correctly flagged as distress, but at only **35% confidence** vs ~97% for loud screams. The `distress` class skews toward _loud_ vocalizations, so the model learned "loud + sharp = distress" more than the deeper qualities of fear. (This is also why the app disables mic auto-gain — AGC inflates quiet speech and trips that bias.)

## Key design decisions
- **`n_mfcc=13`, not 40** — 40 let the model memorize the ~640 training clips (loss → 0.03 but test recall _fell_); 13 generalized better. A real bias-variance tradeoff.
- **Gradient clipping + a fixed seed** — tamed non-reproducible recall swings (76%–97%) caused by exploding-gradient loss spikes.
- **Deep-copying the best checkpoint** — `state_dict()` returns a live reference; without `copy.deepcopy()` the checkpoint silently became the final overfit model.

---

## 📁 Project structure

```
AffectCare-Extended/
├── frontend/                   # the deployed app (React + Vite → Cloudflare Pages)
│   ├── public/                 # best_model.onnx, scaler.json, metrics.json, loss_history.json
│   └── src/
│       ├── utils/
│       │   ├── mfcc.js          # librosa-faithful MFCC (validated to ~1e-7)
│       │   ├── audio.js         # decode → resample → trim → fix_length
│       │   ├── api.js           # in-browser ONNX inference + static data
│       │   ├── spectrogram.js   # MFCC → magma PNG, client-side
│       │   └── wav.js           # PCM → WAV encoder
│       ├── hooks/               # useRecorder (raw PCM), useTheme
│       └── components/          # ResultCard, ModelInsights, ConfusionMatrix, LossCurve, …
├── ML-Backend/                 # training + export (not part of the live app)
│   ├── src/                     # model.py, train.py, preprocessing.py, best_model.pth, scaler.pkl …
│   ├── export_onnx.py          # → frontend/public/{best_model.onnx, scaler.json, metrics, loss}
│   ├── gen_metrics.py          # computes held-out metrics.json
│   └── predict.py              # original single-file CLI inference
└── backend/                    # legacy Node/Express variant that served the model over HTTP
                                #   (kept for reference; the live app is fully client-side)
```

---

## ⚙️ Run & deploy

### Run the app locally
```bash
cd frontend
npm install
npm run dev            # → http://localhost:5173
```

### Re-export the browser assets (after retraining)
```bash
cd ML-Backend
python3.12 -m venv src/venv && source src/venv/bin/activate
pip install --only-binary=:all: -r src/requirements.txt onnx onnxruntime
python gen_metrics.py     # refresh metrics.json
python export_onnx.py     # writes best_model.onnx + scaler.json + static JSON into frontend/public/
```

### Deploy (Cloudflare Pages)
Connect the repo and set:
- **Root directory:** `frontend`
- **Build command:** `npm run build`  (a `prebuild` step copies the ONNX Runtime `.wasm`/`.mjs` from `node_modules`)
- **Output directory:** `dist`

Every push to `main` auto-builds and deploys. That's the entire ops story — no server.

---

## 🔭 Future work
- **Small dataset (~800 files)** — enough to demonstrate the pipeline, not production generalization.
- **Loud-bias** — quiet/whispered distress scores lower than loud; more quiet samples would help.
- **Out-of-distribution audio** — music and real-room noise weren't in training, so the model can misjudge them; it was trained on screams vs. isolated environmental sounds.
- **Recall-first false-alarm rate** is high for real deployment — acceptable for a portfolio demo.

---

## 📝 Notes

Built as a solo learning project. The model and training loop were written and debugged personally (non-reproducible results, a `state_dict()` reference bug, an overfitting spiral); the Extended edition adds a fully client-side web app on top — porting the model to ONNX and `librosa` to JavaScript, and **proving** the browser matches Python before shipping. The gaps above are named on purpose.

---

_Built with PyTorch, Librosa, React, ONNX Runtime Web — and a lot of confusion matrices._ 🧧
