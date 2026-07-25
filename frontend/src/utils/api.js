import * as ort from "onnxruntime-web";
import { preprocessAudio } from "./audioProcessor";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

let onnxSession = null;

async function loadOnnxModel() {
  if (onnxSession) {
    return onnxSession;
  }
  try {
    // Load the ONNX model from the public directory
    onnxSession = await ort.InferenceSession.create("/best_model.onnx");
    console.log("ONNX model loaded successfully.");
    return onnxSession;
  } catch (e) {
    console.error("Failed to load ONNX model:", e);
    throw new Error("Failed to load ONNX model.");
  }
}

// Sends an audio file/blob to the backend and returns the prediction.
export async function predictAudio(fileOrBlob, filename = "clip.wav") {
  await loadOnnxModel();

  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const arrayBuffer = await fileOrBlob.arrayBuffer();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

  const { mfccs, finalShape } = await preprocessAudio(audioBuffer);

  // Create an ONNX tensor from the preprocessed MFCCs
  const inputTensor = new ort.Tensor("float32", mfccs, finalShape);

  const feeds = { input: inputTensor };
  const results = await onnxSession.run(feeds);

  // Assuming the model outputs a single tensor named 'output'
  const output = results.output.data[0];
  const probability = 1 / (1 + Math.exp(-output)); // Sigmoid activation

  const threshold = 0.15; // From Python model
  const label = probability >= threshold ? "distress" : "normal";

  return { label, confidence: parseFloat(probability.toFixed(4)), threshold };
}

// Requests the MFCC spectrogram (what the CNN sees) for one clip.
// This will still call the backend for visualization, as client-side visualization is more complex.
export async function visualizeAudio(fileOrBlob, filename = "clip.wav") {
  const formData = new FormData();
  formData.append("audio", fileOrBlob, filename);

  const res = await fetch(`${API_BASE_URL}/api/visualize`, {
    method: "POST",
    body: formData,
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Visualization failed");
  return data; // { mfcc_png, stats }
}

export async function getMetrics() {
  const res = await fetch(`${API_BASE_URL}/api/model/metrics`);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Could not load metrics");
  return data;
}

export async function getLossHistory() {
  const res = await fetch(`${API_BASE_URL}/api/model/loss-history`);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Could not load loss history");
  return data.loss; // number[]
}
