# gen_metrics.py
# Computes the model's held-out test metrics once and caches them to
# results/metrics.json so the UI can show real, reproducible numbers instead
# of hardcoded ones. Run from anywhere: paths are anchored to this file.

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

import torch  # noqa: E402
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix  # noqa: E402
from model import ElderCareModel  # noqa: E402
from prepare_data import prepare_data  # noqa: E402


def main(threshold=0.15):
    processed_path = os.path.join(BASE_DIR, "data", "processed")
    X_train, X_test, y_train, y_test, _ = prepare_data(processed_path=processed_path)

    X_test = torch.from_numpy(X_test.reshape(-1, 1, 13, 173)).float()
    y_true = y_test

    model = ElderCareModel()
    model.load_state_dict(torch.load(os.path.join(SRC_DIR, "best_model.pth")))
    model.eval()

    with torch.no_grad():
        probs = torch.sigmoid(model(X_test)).squeeze(1)
    preds = (probs >= threshold).int().numpy()

    cm = confusion_matrix(y_true, preds)  # [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "threshold": threshold,
        "test_size": int(len(y_true)),
        "recall": round(float(recall_score(y_true, preds)), 4),
        "precision": round(float(precision_score(y_true, preds)), 4),
        "f1": round(float(f1_score(y_true, preds)), 4),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }

    out_path = os.path.join(BASE_DIR, "results", "metrics.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
