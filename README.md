# 🎙️ AffectCare — Extended

> **A distress signal shouldn't need a button.**

AffectCare is a CNN+LSTM audio classifier that listens for vocal distress — screams, cries for help, panic — and flags it, designed around a single non-negotiable rule: **missing a real emergency is worse than a false alarm.**

**This repo is the _Extended_ edition.** It takes the original command-line model and wraps it in a full-stack web app: a **React + Vite** interface where you can **record or upload** a clip and get a verdict live, backed by a **Node/Express (MVC)** API that bridges to the same Python ML pipeline — plus a **Model Insights** tab that visualizes both the model's overall performance and the exact spectrogram each clip produces. No database; uploaded audio is deleted the moment it's scored.

---

## ✨ What's new in the Extended edition

|           | Original (CLI)               | Extended (this repo)                                                                          |
| --------- | ---------------------------- | --------------------------------------------------------------------------------------------- |
| Interface | `python predict.py file.wav` | Web UI — record **or** upload, in the browser                                                 |
| Result    | one line of stdout           | verdict card + confidence, dark/light themed                                                  |
| Insight   | static PNGs on disk          | live **Model Insights** tab (metrics, confusion matrix, loss curve, per-clip MFCC + waveform) |
| Serving   | none                         | Node/Express **MVC** API calling the Python model                                             |
| Privacy   | —                            | clips deleted right after scoring; refresh wipes state; **no DB**                             |

The deep-learning core is unchanged — the same `best_model.pth`, `scaler.pkl`, and preprocessing pipeline described further down. The extension is everything _around_ it.

---

## 🏗️ Architecture

Three processes, one job. The browser never talks to Python directly — Express is the bridge.

```
┌──────────────┐   POST /api/*   ┌───────────────┐   execFile    ┌──────────────────┐
│  React + Vite │ ─────────────▶ │  Node/Express  │ ───────────▶ │  Python  (Torch,  │
│   UI  :5173   │                │   MVC  :5001   │   argv        │  Librosa)  ML     │
│               │ ◀───────────── │                │ ◀─────────── │  predict_api.py   │
└──────────────┘     JSON        └───────────────┘  JSON/stdout  │  visualize_api.py │
     record / upload                  no DB, no state            │  gen_metrics.py   │
     dark · light                     deletes clip after scoring └──────────────────┘
```

- **Frontend (`frontend/`)** — Vite dev server on `:5173`, proxies `/api` → `:5001`.
- **Backend (`backend/`)** — Express on `:5001`, MVC layout, shells out to Python via `execFile`.
- **ML core (`ML-Backend/`)** — the original project; three thin JSON wrappers expose it to Node.

---

## 🖥️ The web app

### Detect tab

- **Record** — captures microphone audio as raw PCM through the Web Audio API and encodes it to a WAV **in the browser** (see [`frontend/src/utils/wav.js`](frontend/src/utils/wav.js)). This sidesteps `MediaRecorder`'s WebM/Opus output, which `librosa` can't read without `ffmpeg` — the server always receives a clean, decodable WAV.
- **Upload** — accepts `.wav` / `.mp3`.
- **Analyze** — sends the clip to `POST /api/predict` and renders a **verdict card**: _Distress detected_ / _No distress_, with a confidence meter shown **relative to the verdict** (a clip the model is sure is normal reads as high confidence, not a misleadingly low number).
- **Dark / light** — a theme toggle persisted to `localStorage`, honoring the OS preference on first visit.

### Model Insights tab

Two kinds of visualization, because they answer different questions:

**Model performance** — the same for every clip, because it describes the _model_, measured live from the held-out test split:

- **Stat tiles** — Recall, Precision, F1 (from [`ML-Backend/results/metrics.json`](ML-Backend/results/metrics.json), computed by `gen_metrics.py`).
- **Confusion matrix** — rendered live in the browser from the real counts (not a static image), so it themes correctly and labels each cell (✓ Caught / ✕ Missed / false alarm / true negative) — the good/bad meaning never rests on color alone.
- **Training-loss curve** — a live SVG line chart from the actual 40-epoch `loss_history.json`, with a hover crosshair.

**This clip** — recomputed per upload:

- **Waveform** — drawn client-side from the audio blob.
- **MFCC spectrogram** — the _exact_ `13 × 173` image the CNN reads, rendered by `visualize_api.py` and returned as a base64 PNG.

> ℹ️ A confusion matrix and loss curve are **model-level, not per-clip** — a single upload has no ground-truth label to score against and no training history. So those describe the model; the waveform and spectrogram are what's genuinely computed per clip.

---

## 🔌 API

| Method | Route                     | Purpose                            | Returns                                          |
| ------ | ------------------------- | ---------------------------------- | ------------------------------------------------ |
| `POST` | `/api/predict`            | Score one clip                     | `{ label, confidence, threshold }`               |
| `POST` | `/api/visualize`          | Render the clip's MFCC spectrogram | `{ mfcc_png, stats }`                            |
| `GET`  | `/api/model/metrics`      | Cached held-out test metrics       | `{ recall, precision, f1, confusion_matrix, … }` |
| `GET`  | `/api/model/loss-history` | Per-epoch training loss            | `{ loss: number[] }`                             |
| `GET`  | `/api/health`             | Liveness check                     | `{ status: "ok" }`                               |

Both upload routes accept a single `audio` multipart field, write it to `backend/uploads/`, run the model, and **delete the file in a `finally` block** — nothing is retained.

---

## 🧱 Backend — MVC layout

```
backend/
├── server.js                         # app entry, mounts routes, error handler
└── src/
    ├── config/config.js              # ports, paths, resolves the Python venv
    ├── middleware/upload.js          # multer: .wav/.mp3 only, 20 MB cap
    ├── routes/
    │   ├── prediction.routes.js      # /api/predict, /api/health
    │   └── insights.routes.js        # /api/visualize, /api/model/*
    ├── controllers/
    │   ├── prediction.controller.js  # handle upload → predict → delete
    │   └── insights.controller.js    # metrics, loss history, visualize
    └── models/
        ├── prediction.model.js       # execFile → predict_api.py
        └── visualization.model.js    # execFile → visualize_api.py
```

The backend invokes the ML pipeline through the Python interpreter at `ML-Backend/src/venv/bin/python` by default; override with the `PYTHON_BIN` env var. The three bridge scripts (`predict_api.py`, `visualize_api.py`, `gen_metrics.py`) each print **one JSON object to stdout** so Node can parse them reliably.

## ⚛️ Frontend — structure

```
frontend/src/
├── App.jsx                    # tabs, clip state, orchestration
├── hooks/
│   ├── useTheme.js            # dark/light, persisted
│   └── useRecorder.js         # Web Audio PCM capture → WAV
├── utils/
│   ├── api.js                 # fetch helpers for every endpoint
│   └── wav.js                 # PCM → 16-bit WAV encoder
└── components/
    ├── ThemeToggle.jsx
    ├── ResultCard.jsx         # verdict + confidence meter
    ├── ModelInsights.jsx      # the whole Insights tab
    ├── ConfusionMatrix.jsx    # live 2×2 from real counts
    ├── LossCurve.jsx          # live SVG line chart
    └── Waveform.jsx           # canvas waveform from the blob
```

---

# 🧠 The ML core

Everything below documents the model itself — unchanged from the original AffectCare project, now living under [`ML-Backend/`](ML-Backend/).

## Why CNN + LSTM

Raw audio is just a long list of pressure values — no shape a neural network can learn from. The pipeline first converts each clip into an **MFCC spectrogram**: a 2D "image" of the sound, with frequency on one axis and time on the other.

- **CNN** — scans that image for local spatial patterns (the _texture_ of a scream vs. a hum)
- **LSTM** — reads the CNN's output as a sequence, capturing how those patterns unfold _over time_ across the clip, not just a single frame

Neither alone is enough: a CNN sees shapes but not sequence; an LSTM needs something structured to read. Together they cover both.

## Why recall over precision

In elderly care, a **missed distress signal** (false negative) can cost a life. A **false alarm** (false positive) costs a caregiver a few minutes checking on nothing. Those two mistakes are not equally bad — so this project deliberately optimizes for **catching every possible emergency**, even at the cost of more false alarms:

- `BCEWithLogitsLoss` with `pos_weight=1.5` — penalizes missed distress harder than false alarms during training
- A **low decision threshold (0.15)**, not the default 0.5 — even a moderate suspicion triggers an alert
- **F1-based early stopping**, not accuracy or raw recall — this matters more than it sounds (see below)

### The trap I hit and fixed

Early on, saving the "best" checkpoint by **recall alone** produced a model that flagged almost everything as distress — 96% recall, but 72 out of 80 normal sounds triggered false alarms. Technically high-recall, practically useless — like a car alarm that goes off when a leaf lands on it.

Switching the checkpoint criterion to **F1 score** (which only rewards models that balance recall _and_ precision) fixed this and produced a genuinely usable result. This was the single most important debugging lesson of the project.

## Results

Final model, threshold = 0.15, evaluated on a held-out 20% test split (160 files):

| Metric                      | Score      |
| --------------------------- | ---------- |
| Recall (emergencies caught) | **88.75%** |
| Precision (real alarms)     | 68.93%     |
| F1 Score                    | 77.60%     |

**Confusion matrix** (the Insights tab renders this live from the same numbers):

|                     | Predicted normal   | Predicted distress |
| ------------------- | ------------------ | ------------------ |
| **Actual normal**   | 48 (true negative) | 32 (false alarm)   |
| **Actual distress** | 9 (missed)         | 71 (caught)        |

9 missed emergencies, 32 false alarms, out of 80 real distress clips and 80 real normal clips. Not perfect — but honest, reproducible, and built through a deliberate, defensible process rather than a lucky threshold guess.

## Dataset

~800 audio clips, balanced 50/50 across two classes:

| Class      | Source                                                                                                                                                         |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `distress` | Human scream / distress vocalization clips from the Kaggle [Human Scream Dataset](https://www.kaggle.com/datasets/whats2000/human-screaming-detection-dataset) |
| `normal`   | [ESC-50](https://github.com/karolpiczak/ESC-50) (ambient/household/urban sounds) + supplementary calm speech clips                                             |

### The siren confusion

A dataset audit (`src/audit_dataset.py`) flagged several ESC-50 "normal" clips with scream-level energy — most were legitimately loud sounds (sirens, chainsaws, trains), but the **siren clips (`-42` category) stood out**: sustained, high-pitched, high-energy bursts acoustically close to a scream. A real, observed case of spurious correlation in a real dataset, not a hypothetical.

### The whisper test

`predict.py` was tested against a genuine recorded whisper (not a volume-scaled scream) simulating someone quietly calling for help. Result: correctly flagged as distress, but at **35% confidence** — well above 0.15, but meaningfully weaker than the ~97% given to loud screams. The `distress` class likely skews toward _loud_ vocalizations, so the model learned "loud + sharp = distress" more strongly than the deeper acoustic qualities of fear itself.

## Key design decisions

**`n_mfcc=13`, not 40.** I tested both. 40 coefficients gave the model more room to memorize the ~640 training clips instead of generalizing (loss dropped to 0.03, but test recall _fell_). At this dataset size, 13 generalized better — a direct, observed example of the bias-variance tradeoff.

**Gradient clipping + a seeded run.** Early runs showed wild, non-reproducible recall swings (76%–97%) with identical settings — `torch.manual_seed(42)` wasn't locking down every randomness source, and a mid-training loss spike (a classic exploding-gradient symptom) was throwing results around. `torch.nn.utils.clip_grad_norm_` plus a confirmed fixed seed made results trustworthy enough to compare across experiments.

**Deep-copying the "best" checkpoint.** `model.state_dict()` returns a live reference, not a frozen snapshot — an early checkpointing bug silently "restored" the final overfit model instead of the actual best one. `copy.deepcopy()` fixed it.

---

## 📁 Project structure

```
AffectCare-Extended/
├── ML-Backend/                 # the deep-learning core (original project)
│   ├── data/processed/         # cached MFCC arrays (X.npy, y.npy)
│   ├── src/
│   │   ├── preprocessing.py     # load → trim → fix_length → MFCC
│   │   ├── audit_dataset.py     # flags mislabeled/high-energy clips
│   │   ├── prepare_data.py      # train/test split + scaling
│   │   ├── model.py             # CNN+LSTM architecture
│   │   ├── train.py             # training loop, F1-based early stopping
│   │   ├── evaluate.py          # metrics + saves results/ plots
│   │   ├── best_model.pth        # trained weights
│   │   └── scaler.pkl            # fitted StandardScaler (needed for inference)
│   ├── results/                # confusion-matrix.png, loss_curve.png, metrics.json
│   ├── predict.py              # single-file CLI inference
│   ├── predict_api.py          # JSON bridge → /api/predict
│   ├── visualize_api.py        # JSON bridge → /api/visualize (MFCC PNG)
│   └── gen_metrics.py          # caches test metrics → results/metrics.json
├── backend/                    # Node/Express MVC API  (:5001)
└── frontend/                   # React + Vite UI        (:5173)
```

---

## ⚙️ Installation & running

You'll run three things. **ML deps first**, then backend, then frontend.

### 1. Python ML core

```bash
cd ML-Backend
python3 -m venv src/venv
source src/venv/bin/activate        # Windows: src\venv\Scripts\activate
pip install -r src/requirements.txt

# (first run only) cache the test metrics the Insights tab reads
python gen_metrics.py

# sanity check the CLI
python predict.py path/to/audio.wav
```

```
→ Prediction: Distress
   87.00% distress · 13.00% normal
```

### 2. Backend API

```bash
cd backend
npm install
npm run dev            # → http://localhost:5001
```

By default the API calls `ML-Backend/src/venv/bin/python`. If your interpreter lives elsewhere, set `PYTHON_BIN=/path/to/python`.

### 3. Frontend

```bash
cd frontend
npm install
npm run dev            # → http://localhost:5173
```

Open **http://localhost:5173**. The Vite dev server proxies `/api` to the backend, so both must be running.

### Retrain from scratch (optional)

```bash
cd ML-Backend/src
python3 preprocessing.py   # extract MFCCs from dataset/, save to data/processed/
python3 train.py           # train, save best_model.pth + scaler.pkl
python3 evaluate.py        # regenerate results/ plots
cd .. && python gen_metrics.py   # refresh metrics.json for the UI
```

---

## 🗣️ Talking points

- Why CNN+LSTM instead of either alone
- Why 13 MFCCs beat 40 at this dataset size — a real bias-variance tradeoff
- The siren/scream spurious-correlation finding from the dataset audit
- Why F1, not raw recall, drives early stopping — and the failure mode it prevents
- Why recall is weighted over precision, and what that costs
- **How the browser reaches a PyTorch model** — the React → Express → Python bridge, and why recordings are encoded to WAV client-side
- **Why the confusion matrix is per-model but the spectrogram is per-clip**

---

## 📝 Notes

Built as a solo learning project — the model and training loop were written and debugged personally (chasing non-reproducible results, a `state_dict()` reference bug, and an overfitting spiral), and the Extended edition adds a real full-stack layer on top: an MVC API, a recording pipeline, and honest visualizations. The gaps above are named on purpose — an honest account of what's simple/early is worth more than an oversold claim.

---

_Built with PyTorch, Librosa, React, Express — and a lot of confusion matrices._ 🧧
