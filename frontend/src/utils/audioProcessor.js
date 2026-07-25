import * as Meyda from "meyda";

// Replicate librosa.load(filepath, sr=22050, mono=True)
async function loadAudio(audioBuffer, sr = 22050) {
  // Meyda works with AudioBuffer, so we assume audioBuffer is already loaded
  // We need to resample if the audioBuffer's sample rate is not the target sr
  if (audioBuffer.sampleRate !== sr) {
    const offlineContext = new OfflineAudioContext(audioBuffer.numberOfChannels, audioBuffer.duration * sr, sr);
    const source = offlineContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(offlineContext.destination);
    source.start();
    const resampledBuffer = await offlineContext.startRendering();
    return resampledBuffer.getChannelData(0); // Assuming mono for now
  }
  return audioBuffer.getChannelData(0); // Assuming mono
}

// Replicate librosa.effects.trim(audio, top_db=18)
function trimAudio(audioData, top_db = 18) {
  // This is a simplified trim. librosa's trim is more sophisticated.
  // For a full replication, a more complex algorithm would be needed.
  // This version removes leading/trailing silence below top_db threshold.
  const threshold = Math.pow(10, top_db / 20) * 0.001; // Convert dB to amplitude ratio
  let start = 0;
  let end = audioData.length - 1;

  while (start < audioData.length && Math.abs(audioData[start]) < threshold) {
    start++;
  }
  while (end >= 0 && Math.abs(audioData[end]) < threshold) {
    end--;
  }

  if (start > end) {
    return new Float32Array(0); // Entire audio is silent
  }
  return audioData.slice(start, end + 1);
}

// Replicate librosa.util.fix_length(audio_trimmed, size=target_length)
function fixLength(audioData, targetLength) {
  if (audioData.length === targetLength) {
    return audioData;
  } else if (audioData.length < targetLength) {
    const paddedAudio = new Float32Array(targetLength).fill(0);
    paddedAudio.set(audioData);
    return paddedAudio;
  } else {
    return audioData.slice(0, targetLength);
  }
}

// Replicate librosa.feature.mfcc(y=audio_fixed, sr=sr, n_mfcc=13)
function extractMfccs(audioData, sr, n_mfcc = 13) {
  // Meyda requires an AudioContext and an Analyzer
  // For offline processing, we can create a dummy context and analyzer
  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const bufferSize = 512; // Meyda's default buffer size

  const analyzer = Meyda.createMeydaAnalyzer({
    "audioContext": audioContext,
    "source": audioContext.createBufferSource(), // Dummy source
    "bufferSize": bufferSize,
    "featureExtractors": ["mfcc"],
    "numberOfMFCCCoefficients": n_mfcc,
    "sampleRate": sr,
  });

  // Meyda.extract expects a Float32Array of audio data
  const features = Meyda.extract("mfcc", audioData, bufferSize, sr);

  if (!features || !features.mfcc) {
    throw new Error("MFCC extraction failed.");
  }

  // Meyda returns MFCCs as an array of arrays (frames x coefficients)
  // We need to reshape it to (1, n_mfcc, num_frames) to match PyTorch model input
  const mfccs = features.mfcc;
  const numFrames = mfccs.length;
  const reshapedMfccs = new Float32Array(1 * n_mfcc * numFrames);

  for (let i = 0; i < numFrames; i++) {
    for (let j = 0; j < n_mfcc; j++) {
      reshapedMfccs[i * n_mfcc + j] = mfccs[i][j];
    }
  }

  return reshapedMfccs; // This will be (n_mfcc * num_frames) flat array
}

export async function preprocessAudio(audioBuffer, sr = 22050, targetSeconds = 4, n_mfcc = 13) {
  const audioData = await loadAudio(audioBuffer, sr);
  const trimmedAudio = trimAudio(audioData);
  const fixedAudio = fixLength(trimmedAudio, targetSeconds * sr);
  const mfccs = extractMfccs(fixedAudio, sr, n_mfcc);

  // Reshape to (1, 1, 13, 173) for ONNX model input
  // This assumes num_frames is 173 based on the Python model's input shape
  // If num_frames is different, this will need adjustment.
  const numFrames = Math.floor(fixedAudio.length / (sr / (Meyda.bufferSize / 2))); // Approximate frames
  const finalShape = [1, 1, n_mfcc, numFrames]; // Adjust numFrames if needed

  // For now, return the flat array and let the ONNX runtime handle reshaping if it supports it
  // or we'll need to manually reshape it to a 4D tensor before feeding to ONNX.
  return { mfccs, finalShape };
}
