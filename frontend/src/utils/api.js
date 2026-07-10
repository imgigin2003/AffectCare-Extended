// Sends an audio file/blob to the backend and returns the prediction.
export async function predictAudio(fileOrBlob, filename = 'clip.wav') {
  const formData = new FormData();
  formData.append('audio', fileOrBlob, filename);

  const res = await fetch('/api/predict', {
    method: 'POST',
    body: formData,
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Prediction failed');
  return data; // { label, confidence, threshold }
}

// Requests the MFCC spectrogram (what the CNN sees) for one clip.
export async function visualizeAudio(fileOrBlob, filename = 'clip.wav') {
  const formData = new FormData();
  formData.append('audio', fileOrBlob, filename);

  const res = await fetch('/api/visualize', {
    method: 'POST',
    body: formData,
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Visualization failed');
  return data; // { mfcc_png, stats }
}

export async function getMetrics() {
  const res = await fetch('/api/model/metrics');
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Could not load metrics');
  return data;
}

export async function getLossHistory() {
  const res = await fetch('/api/model/loss-history');
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Could not load loss history');
  return data.loss; // number[]
}
