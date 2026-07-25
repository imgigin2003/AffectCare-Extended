import os
import sys
import json
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# Add the src directory to the Python path to import model.py and preprocessing.py
# This assumes app.py is in ML-Backend and src is in ML-Backend/src
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "src"))

from predict import load_trained_model, load_scaler, predict # type: ignore
from visualize_api import visualize_audio # type: ignore

app = FastAPI()

# Configure CORS to allow requests from your Cloudflare Pages frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://affectcare-extended.pages.dev", "http://localhost:5173"], # Add your Cloudflare Pages domain and local dev URL
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# --- Global ML Model Loading ---
# Load model and scaler once when the app starts
model = None
scaler = None

@app.on_event("startup")
async def load_ml_artifacts():
    global model, scaler
    try:
        model_path = os.path.join(BASE_DIR, "src", "best_model.pth")
        scaler_path = os.path.join(BASE_DIR, "src", "scaler.pkl")
        model = load_trained_model(model_path)
        scaler = load_scaler(scaler_path)
        print("ML model and scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading ML artifacts: {e}")
        # Depending on criticality, you might want to raise an exception here
        # or handle it gracefully if the app can run without the model.

# --- API Endpoints ---

@app.post("/api/predict")
async def predict_audio_endpoint(audio: UploadFile = File(...)):
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="ML model not loaded.")

    # Save the uploaded audio file temporarily
    temp_audio_path = os.path.join(BASE_DIR, "temp_audio.wav") # Or use a more robust temp file solution
    try:
        with open(temp_audio_path, "wb") as buffer:
            buffer.write(await audio.read())

        label, probability = predict(temp_audio_path, model, scaler)
        return JSONResponse({"label": label, "confidence": round(probability, 4), "threshold": 0.15})
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"ML pipeline failed: {e}")
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

@app.post("/api/visualize")
async def visualize_audio_endpoint(audio: UploadFile = File(...)):
    # Save the uploaded audio file temporarily
    temp_audio_path = os.path.join(BASE_DIR, "temp_audio.wav")
    output_image_path = os.path.join(BASE_DIR, "mfcc_spectrogram.png")
    try:
        with open(temp_audio_path, "wb") as buffer:
            buffer.write(await audio.read())

        # Call the visualize_audio function from visualize_api.py
        visualize_audio(temp_audio_path, output_image_path)

        # Return the generated image
        return FileResponse(output_image_path, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Visualization failed: {e}")
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if os.path.exists(output_image_path):
            os.remove(output_image_path)

@app.get("/api/model/metrics")
async def get_metrics_endpoint():
    metrics_file = os.path.join(BASE_DIR, "results", "metrics.json")
    if not os.path.exists(metrics_file):
        raise HTTPException(status_code=404, detail="Metrics file not found.")
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    return JSONResponse(metrics)

@app.get("/api/model/loss-history")
async def get_loss_history_endpoint():
    loss_history_file = os.path.join(BASE_DIR, "src", "loss_history.json")
    if not os.path.exists(loss_history_file):
        raise HTTPException(status_code=404, detail="Loss history file not found.")
    with open(loss_history_file, "r") as f:
        loss_history = json.load(f)
    return JSONResponse(loss_history)

@app.get("/health")
async def health_check():
    return JSONResponse({"status": "ok"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
