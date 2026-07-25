import { computeMFCC, trimSilence, fixLength } from './mfcc';

const SR = 22050;
const TARGET = 4 * SR;

// Decode any browser-supported audio blob (wav/mp3/…) to mono Float32 @ 22050,
// matching librosa.load(path, sr=22050, mono=True). Web Audio handles decode +
// resample + downmix.
export async function decodeToMono22k(blob) {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const ctx = new AudioCtx();
  let decoded;
  try {
    decoded = await ctx.decodeAudioData(await blob.arrayBuffer());
  } finally {
    ctx.close();
  }

  const frames = Math.max(1, Math.ceil(decoded.duration * SR));
  const offline = new OfflineAudioContext(1, frames, SR);
  const src = offline.createBufferSource();
  src.buffer = decoded;
  src.connect(offline.destination);
  src.start();
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0);
}

// Full preprocessing: decode → trim → fix_length → MFCC (13 × 173), the exact
// pipeline the Python model was trained on (validated to ~1e-6 vs librosa).
export async function clipToMFCC(blob) {
  const audio = await decodeToMono22k(blob);
  const trimmed = trimSilence(audio, 18);
  const fixed = fixLength(trimmed, TARGET);
  return computeMFCC(fixed, SR, 13);
}
