// A faithful port of librosa.feature.mfcc(y, sr=22050, n_mfcc=13) with all
// librosa defaults, so the browser produces the SAME features the model was
// trained on. Validated against librosa's output on real signals (see the
// reference-validation harness). Pure math — no browser APIs — so it runs in
// Node for testing too.
//
// Pipeline (all librosa defaults):
//   STFT  n_fft=2048, hop=512, periodic Hann, center=True, zero-pad
//   power spectrum (|.|^2)
//   mel  n_mels=128, fmin=0, fmax=sr/2, Slaney scale + Slaney norm
//   power_to_db  ref=1.0, amin=1e-10, top_db=80  (global max floor)
//   DCT-II  orthonormal, over the mel axis, keep first n_mfcc

const N_FFT = 2048;
const HOP = 512;
const N_MELS = 128;

// ---- FFT (iterative radix-2, in-place) ----
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1;
      let curIm = 0;
      for (let k = 0; k < len / 2; k++) {
        const uRe = re[i + k];
        const uIm = im[i + k];
        const vRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm;
        const vIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe;
        re[i + k] = uRe + vRe;
        im[i + k] = uIm + vIm;
        re[i + k + len / 2] = uRe - vRe;
        im[i + k + len / 2] = uIm - vIm;
        const nextRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
      }
    }
  }
}

// ---- periodic Hann window (librosa: get_window('hann', N, fftbins=True)) ----
function hann(n) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
  return w;
}

// ---- Slaney mel scale (librosa htk=False) ----
function hzToMel(f) {
  const fSp = 200.0 / 3;
  const minLogHz = 1000.0;
  const minLogMel = minLogHz / fSp; // = 15
  const logstep = Math.log(6.4) / 27.0;
  return f >= minLogHz ? minLogMel + Math.log(f / minLogHz) / logstep : f / fSp;
}
function melToHz(m) {
  const fSp = 200.0 / 3;
  const minLogHz = 1000.0;
  const minLogMel = minLogHz / fSp;
  const logstep = Math.log(6.4) / 27.0;
  return m >= minLogMel ? minLogHz * Math.exp(logstep * (m - minLogMel)) : fSp * m;
}

// ---- mel filterbank (librosa.filters.mel, norm='slaney') ----
function melBasis(sr) {
  const nBins = N_FFT / 2 + 1; // 1025
  const fftFreqs = new Float32Array(nBins);
  for (let i = 0; i < nBins; i++) fftFreqs[i] = (i * sr) / N_FFT;

  const fmin = 0.0;
  const fmax = sr / 2;
  const melMin = hzToMel(fmin);
  const melMax = hzToMel(fmax);
  const melF = new Float64Array(N_MELS + 2);
  for (let i = 0; i < N_MELS + 2; i++) {
    melF[i] = melToHz(melMin + ((melMax - melMin) * i) / (N_MELS + 1));
  }

  const weights = [];
  for (let m = 0; m < N_MELS; m++) {
    const row = new Float64Array(nBins);
    const fLower = melF[m];
    const fCenter = melF[m + 1];
    const fUpper = melF[m + 2];
    const dLow = fCenter - fLower;
    const dUp = fUpper - fCenter;
    const enorm = 2.0 / (fUpper - fLower); // Slaney normalization
    for (let k = 0; k < nBins; k++) {
      const lower = (fftFreqs[k] - fLower) / dLow;
      const upper = (fUpper - fftFreqs[k]) / dUp;
      const w = Math.max(0, Math.min(lower, upper));
      row[k] = w * enorm;
    }
    weights.push(row);
  }
  return weights; // N_MELS x nBins
}

// ---- librosa.effects.trim(y, top_db=18, frame_length=2048, hop_length=512) ----
// Removes leading/trailing frames whose RMS is > top_db below the loudest frame.
export function trimSilence(audio, topDb = 45, frameLength = 2048, hop = 512) {
  const pad = frameLength >> 1; // center=True zero-pad
  const padded = new Float32Array(audio.length + 2 * pad);
  padded.set(audio, pad);
  const nFrames = 1 + Math.floor((padded.length - frameLength) / hop);

  const rms = new Float64Array(nFrames);
  let maxRms = 0;
  for (let t = 0; t < nFrames; t++) {
    const start = t * hop;
    let s = 0;
    for (let i = 0; i < frameLength; i++) {
      const v = padded[start + i];
      s += v * v;
    }
    rms[t] = Math.sqrt(s / frameLength);
    if (rms[t] > maxRms) maxRms = rms[t];
  }

  const amin = 1e-10;
  const refDb = 20 * Math.log10(Math.max(amin, maxRms));
  let first = -1;
  let last = -1;
  for (let t = 0; t < nFrames; t++) {
    const db = 20 * Math.log10(Math.max(amin, rms[t])) - refDb;
    if (db > -topDb) {
      if (first === -1) first = t;
      last = t;
    }
  }
  if (first === -1) return audio.slice(0, 0); // all silent

  const startSample = first * hop;
  const endSample = Math.min(audio.length, (last + 1) * hop);
  return audio.slice(startSample, endSample);
}

// ---- librosa.util.fix_length(y, size) ----
export function fixLength(audio, size) {
  if (audio.length === size) return audio;
  const out = new Float32Array(size);
  out.set(audio.subarray(0, Math.min(audio.length, size)));
  return out; // zero-padded if shorter, truncated if longer
}

let _melCacheSr = null;
let _melCache = null;
let _win = null;

export function computeMFCC(audio, sr = 22050, nMfcc = 13) {
  if (_melCacheSr !== sr || !_melCache) {
    _melCache = melBasis(sr);
    _melCacheSr = sr;
  }
  if (!_win) _win = hann(N_FFT);
  const mel = _melCache;
  const nBins = N_FFT / 2 + 1;

  // center=True: zero-pad by n_fft/2 on both ends
  const pad = N_FFT / 2;
  const padded = new Float32Array(audio.length + 2 * pad);
  padded.set(audio, pad);

  const nFrames = 1 + Math.floor((padded.length - N_FFT) / HOP);

  // mel-power spectrogram → power_to_db, collected per frame
  const dbSpec = []; // nFrames arrays of length N_MELS
  let globalMax = -Infinity;
  const amin = 1e-10;

  const re = new Float64Array(N_FFT);
  const im = new Float64Array(N_FFT);

  for (let t = 0; t < nFrames; t++) {
    const start = t * HOP;
    for (let i = 0; i < N_FFT; i++) {
      re[i] = padded[start + i] * _win[i];
      im[i] = 0;
    }
    fft(re, im);

    // power spectrum (first nBins), then mel projection
    const power = new Float64Array(nBins);
    for (let k = 0; k < nBins; k++) power[k] = re[k] * re[k] + im[k] * im[k];

    const melFrame = new Float64Array(N_MELS);
    for (let m = 0; m < N_MELS; m++) {
      const row = mel[m];
      let s = 0;
      for (let k = 0; k < nBins; k++) s += row[k] * power[k];
      // power_to_db: 10*log10(max(amin, S))  (ref=1.0 → -0)
      const db = 10 * Math.log10(Math.max(amin, s));
      melFrame[m] = db;
      if (db > globalMax) globalMax = db;
    }
    dbSpec.push(melFrame);
  }

  // top_db=80 floor, applied against the global max
  const floor = globalMax - 80.0;
  for (let t = 0; t < nFrames; t++) {
    const f = dbSpec[t];
    for (let m = 0; m < N_MELS; m++) if (f[m] < floor) f[m] = floor;
  }

  // DCT-II orthonormal over the mel axis, keep first nMfcc
  // X[0]   = sqrt(1/N)  * sum(x)
  // X[k>0] = sqrt(2/N)  * sum_n x[n] cos(pi k (2n+1) / (2N))
  const N = N_MELS;
  const scale0 = Math.sqrt(1 / N);
  const scaleK = Math.sqrt(2 / N);
  // precompute cosine table (nMfcc x N)
  const cos = [];
  for (let k = 0; k < nMfcc; k++) {
    const rowk = new Float64Array(N);
    for (let n = 0; n < N; n++) rowk[n] = Math.cos((Math.PI * k * (2 * n + 1)) / (2 * N));
    cos.push(rowk);
  }

  // output mfcc[k][t]  (nMfcc x nFrames)
  const mfcc = [];
  for (let k = 0; k < nMfcc; k++) mfcc.push(new Float32Array(nFrames));
  for (let t = 0; t < nFrames; t++) {
    const f = dbSpec[t];
    for (let k = 0; k < nMfcc; k++) {
      let s = 0;
      const rowk = cos[k];
      for (let n = 0; n < N; n++) s += f[n] * rowk[n];
      mfcc[k][t] = (k === 0 ? scale0 : scaleK) * s;
    }
  }
  return mfcc; // nMfcc x nFrames
}
