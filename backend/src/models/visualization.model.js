import { execFile } from "child_process";
import { PYTHON_BIN, VISUALIZE_SCRIPT, ML_DIR } from "../config/config.js";

/**
 * Runs the Python MFCC-spectrogram renderer on an audio file and resolves with
 * { mfcc_png (base64), stats }. Stateless — mirrors runPrediction.
 */
export function runVisualization(audioPath) {
  return new Promise((resolve, reject) => {
    execFile(
      PYTHON_BIN,
      [VISUALIZE_SCRIPT, audioPath],
      { cwd: ML_DIR, timeout: 60_000, maxBuffer: 1024 * 1024 * 16 },
      (error, stdout, stderr) => {
        try {
          const result = JSON.parse(stdout.trim().split("\n").pop());
          if (result.error) {
            const err = new Error(result.error);
            err.status = 422;
            return reject(err);
          }
          return resolve(result);
        } catch {
          return reject(
            new Error(error ? `Visualization failed: ${stderr || error.message}` : "Unparseable output")
          );
        }
      }
    );
  });
}
