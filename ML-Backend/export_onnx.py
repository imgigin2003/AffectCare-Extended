# export_onnx.py
# Prepares everything the browser-only build needs:
#   frontend/public/best_model.onnx   — the model, for onnxruntime-web
#   frontend/public/scaler.json       — StandardScaler mean_/scale_ (2249 each)
#   frontend/public/metrics.json      — held-out test metrics (Insights tab)
#   frontend/public/loss_history.json — per-epoch training loss (Insights tab)
#
# Run once (and again whenever the model retrains):  python export_onnx.py

import json
import os
import pickle
import shutil
import sys

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "src"))

from model import ElderCareModel  # noqa: E402

PUBLIC_DIR = os.path.join(BASE_DIR, "..", "frontend", "public")
SRC_DIR = os.path.join(BASE_DIR, "src")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def export_model():
    model = ElderCareModel()
    model.load_state_dict(torch.load(os.path.join(SRC_DIR, "best_model.pth"), map_location="cpu"))
    model.eval()

    dummy = torch.zeros(1, 1, 13, 173)
    out_path = os.path.join(PUBLIC_DIR, "best_model.onnx")
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["logit"],
        opset_version=17,
        dynamic_axes=None,  # fixed (1,1,13,173) — no dynamic dims needed
    )
    print(f"  ✓ {out_path}")


def export_scaler():
    with open(os.path.join(SRC_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    payload = {
        "mean": scaler.mean_.astype(float).tolist(),
        "scale": scaler.scale_.astype(float).tolist(),
        "n_features": int(scaler.mean_.shape[0]),
    }
    out_path = os.path.join(PUBLIC_DIR, "scaler.json")
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"  ✓ {out_path}  ({payload['n_features']} features)")


def copy_static():
    for src in [
        os.path.join(RESULTS_DIR, "metrics.json"),
        os.path.join(SRC_DIR, "loss_history.json"),
    ]:
        if os.path.exists(src):
            dst = os.path.join(PUBLIC_DIR, os.path.basename(src))
            shutil.copyfile(src, dst)
            print(f"  ✓ {dst}")
        else:
            print(f"  ! missing (skipped): {src}")


if __name__ == "__main__":
    os.makedirs(PUBLIC_DIR, exist_ok=True)
    print("Exporting client-side assets → frontend/public/")
    export_model()
    export_scaler()
    copy_static()
    print("Done.")
