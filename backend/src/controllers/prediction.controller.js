import fs from "fs/promises";
import { runPrediction } from "../models/prediction.model.js";

export async function predictAudio(req, res, next) {
  if (!req.file) {
    return res.status(400).json({ error: "No audio file provided" });
  }

  const filePath = req.file.path;
  try {
    const result = await runPrediction(filePath);
    res.json(result);
  } catch (err) {
    next(err);
  } finally {
    // No DB, no retention — the clip is deleted as soon as it's scored
    fs.unlink(filePath).catch(() => {});
  }
}

export function healthCheck(req, res) {
  res.json({ status: "ok" });
}
