# Thin JSON wrapper around predict.py for the Node/Express backend.
# Prints a single JSON object to stdout so the server can parse it reliably.

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "src"))

from predict import load_trained_model, load_scaler, predict  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print(
            json.dumps({"error": "usage: python3 predict_api.py <path_to_audio_file>"})
        )
        sys.exit(1)

    filepath = sys.argv[1]

    try:
        model = load_trained_model(os.path.join(BASE_DIR, "src", "best_model.pth"))
        scaler = load_scaler(os.path.join(BASE_DIR, "src", "scaler.pkl"))
        label, probability = predict(filepath, model, scaler)
        print(
            json.dumps(
                {
                    "label": label,
                    "confidence": round(probability, 4),
                    "threshold": 0.15,
                }
            )
        )
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
