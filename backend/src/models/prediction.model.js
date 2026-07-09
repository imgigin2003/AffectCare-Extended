import { execFile } from "child_process";
import { PYTHON_BIN, PREDICT_SCRIPT, ML_DIR } from "../config/config.js";

/**
 * Runs the Python ML pipeline on an audio file and resolves with
 * { label, confidence, threshold }. No persistence — stateless by design.
 */
export function runPrediction(audioPath) {
  return new Promise((resolve, reject) => {
    execFile(
      PYTHON_BIN,
      [PREDICT_SCRIPT, audioPath],
      { cwd: ML_DIR, timeout: 60_000 },
      (error, stdout, stderr) => {
        // predict_api.py prints JSON even on failure, so try parsing first
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
            new Error(error ? `ML pipeline failed: ${stderr || error.message}` : "Unparseable ML output")
          );
        }
      }
    );
  });
}
