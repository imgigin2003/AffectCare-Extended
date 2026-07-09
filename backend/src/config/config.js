import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export const PORT = process.env.PORT || 5001;

// Root of the repo (backend/src/config → three levels up)
export const REPO_ROOT = path.resolve(__dirname, "..", "..", "..");

export const ML_DIR = path.join(REPO_ROOT, "ML-Backend");
export const PYTHON_BIN =
  process.env.PYTHON_BIN || path.join(ML_DIR, "src", "venv", "bin", "python");
export const PREDICT_SCRIPT = path.join(ML_DIR, "predict_api.py");
export const VISUALIZE_SCRIPT = path.join(ML_DIR, "visualize_api.py");

// Model-level artifacts produced by evaluate.py / gen_metrics.py / train.py
export const RESULTS_DIR = path.join(ML_DIR, "results");
export const METRICS_FILE = path.join(RESULTS_DIR, "metrics.json");
export const LOSS_HISTORY_FILE = path.join(ML_DIR, "src", "loss_history.json");

// Temp storage for uploaded/recorded clips — wiped after every prediction
export const UPLOAD_DIR = path.join(REPO_ROOT, "backend", "uploads");
