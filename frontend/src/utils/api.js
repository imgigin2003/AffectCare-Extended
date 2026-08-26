// Fully client-side inference — no backend. The model (ONNX) and preprocessing
// (librosa-faithful MFCC) run in the browser, so this app is a static site.
import * as ort from 'onnxruntime-web/wasm';
import { clipToMFCC } from './audio';
import { renderSpectrogram } from './spectrogram';

// WASM is served from the site root (bundled into /public); single-threaded so
// no cross-origin-isolation headers are required on the host.
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@dev/dist/';
ort.env.wasm.numThreads = 1;

const THRESHOLD = 0.15;
const N_MFCC = 13;
const N_FRAMES = 173;

let _sessionPromise = null;
let _scalerPromise = null;

function getSession() {
  if (!_sessionPromise) {
    _sessionPromise = ort.InferenceSession.create('/best_model.onnx', {
      executionProviders: ['wasm'],
    });
  }
  return _sessionPromise;
}

function getScaler() {
  if (!_scalerPromise) {
    _scalerPromise = fetch('/scaler.json').then((r) => {
      if (!r.ok) throw new Error('Could not load scaler.json');
      return r.json();
    });
  }
  return _scalerPromise;
}

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

// Scale the (13×173) MFCC exactly as the Python StandardScaler did:
// flatten in C order (f*173 + t), (x - mean) / scale.
function scaleFeatures(mfcc, scaler) {
  const input = new Float32Array(N_MFCC * N_FRAMES);
  for (let f = 0; f < N_MFCC; f++) {
    for (let t = 0; t < N_FRAMES; t++) {
      const i = f * N_FRAMES + t;
      input[i] = (mfcc[f][t] - scaler.mean[i]) / scaler.scale[i];
    }
  }
  return input;
}

export async function predictAudio(fileOrBlob) {
  const [mfcc, scaler, session] = await Promise.all([
    clipToMFCC(fileOrBlob),
    getScaler(),
    getSession(),
  ]);

  const input = scaleFeatures(mfcc, scaler);
  const tensor = new ort.Tensor('float32', input, [1, 1, N_MFCC, N_FRAMES]);
  const output = await session.run({ input: tensor });
  const logit = output[session.outputNames[0]].data[0];

  const confidence = sigmoid(logit);
  const label = confidence >= THRESHOLD ? 'distress' : 'normal';
  return { label, confidence: Number(confidence.toFixed(4)), threshold: THRESHOLD };
}

export async function visualizeAudio(fileOrBlob) {
  const mfcc = await clipToMFCC(fileOrBlob);
  let min = Infinity;
  let max = -Infinity;
  for (let f = 0; f < N_MFCC; f++) {
    for (let t = 0; t < N_FRAMES; t++) {
      const v = mfcc[f][t];
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  return {
    mfcc_png: renderSpectrogram(mfcc, min, max),
    stats: {
      n_mfcc: N_MFCC,
      n_frames: N_FRAMES,
      energy_min: Number(min.toFixed(2)),
      energy_max: Number(max.toFixed(2)),
    },
  };
}

export async function getMetrics() {
  const res = await fetch('/metrics.json');
  if (!res.ok) throw new Error('Could not load metrics');
  return res.json();
}

export async function getLossHistory() {
  const res = await fetch('/loss_history.json');
  if (!res.ok) throw new Error('Could not load loss history');
  return res.json(); // number[]
}
