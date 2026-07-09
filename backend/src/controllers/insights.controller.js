import fs from "fs/promises";
import { runVisualization } from "../models/visualization.model.js";
import { METRICS_FILE, LOSS_HISTORY_FILE } from "../config/config.js";

// Model-level metrics (cached by gen_metrics.py from the held-out test split)
export async function getMetrics(req, res) {
  try {
    const raw = await fs.readFile(METRICS_FILE, "utf-8");
    res.json(JSON.parse(raw));
  } catch {
    res.status(404).json({ error: "Metrics not generated yet — run gen_metrics.py" });
  }
}

// Per-epoch training loss, recorded by train.py
export async function getLossHistory(req, res) {
  try {
    const raw = await fs.readFile(LOSS_HISTORY_FILE, "utf-8");
    res.json({ loss: JSON.parse(raw) });
  } catch {
    res.status(404).json({ error: "Loss history not found" });
  }
}

// Per-clip MFCC spectrogram — same upload/score/delete lifecycle as predict
export async function visualizeAudio(req, res, next) {
  if (!req.file) {
    return res.status(400).json({ error: "No audio file provided" });
  }

  const filePath = req.file.path;
  try {
    const result = await runVisualization(filePath);
    res.json(result);
  } catch (err) {
    next(err);
  } finally {
    fs.unlink(filePath).catch(() => {});
  }
}
