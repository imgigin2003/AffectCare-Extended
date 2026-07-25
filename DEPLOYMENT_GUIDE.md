# AffectCare Free Backend Deployment Guide

This guide explains how to deploy your machine learning backend for **$0** using **Streamlit Community Cloud**. This is the most reliable free option that does not require a credit card and supports PyTorch.

## Phase 1: Create an Account on Streamlit

1.  **Sign Up:** Go to [Streamlit Community Cloud](https://streamlit.io/cloud) and click **"Sign up"**.
2.  **Connect GitHub:** Connect your GitHub account.

## Phase 2: Deploy Your Backend

1.  **New App:** Click **"New app"**.
2.  **Select Repository:** Select your `AffectCare-Extended` repository.
3.  **Main File Path:** Set this to `ML-Backend/streamlit_api.py`.
4.  **Deploy:** Click **"Deploy!"**.

## Phase 3: Prepare Your Code

Streamlit will automatically detect your `requirements.txt` and install the necessary libraries.

1.  **Model Weights:** Ensure `best_model.pth` and `scaler.pkl` are in `ML-Backend/src/` in your GitHub repository.
2.  **Requirements:** I have created a `requirements.txt` in your `ML-Backend` folder. Streamlit will use this.


### The Backend Code: `streamlit_api.py`
This file serves as your backend. It provides a simple UI for testing and can be adapted to serve your frontend.

```python
import streamlit as st
import os
import sys
import json
import numpy as np

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "src"))

from predict import load_trained_model, load_scaler, predict
from visualize_api import visualize_audio

# Load model and scaler
@st.cache_resource
def load_ml_artifacts():
    model_path = os.path.join(BASE_DIR, "src", "best_model.pth")
    scaler_path = os.path.join(BASE_DIR, "src", "scaler.pkl")
    return load_trained_model(model_path), load_scaler(scaler_path)

model, scaler = load_ml_artifacts()

st.title("AffectCare Backend API")
uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])

if uploaded_file:
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    label, prob = predict(temp_path, model, scaler)
    st.write(f"Prediction: {label} ({prob*100:.2f}%)")
    os.remove(temp_path)
```

### 2. `requirements.txt`
These libraries will be installed by Streamlit during the build.

```text
streamlit
librosa
torch --extra-index-url https://download.pytorch.org/whl/cpu
numpy
scikit-learn
matplotlib
soundfile
seaborn
pandas
```

## Phase 4: Update Your Frontend

1.  **Get your API URL:** Once Streamlit shows "Your app is live!", copy the URL from your browser (e.g., `https://affectcare-backend.streamlit.app`).
2.  **Cloudflare Pages:** Go to your Cloudflare Pages dashboard.
3.  **Environment Variables:** Add or update `VITE_API_BASE_URL` with your Streamlit URL.
4.  **Redeploy:** Trigger a new deployment of your frontend.

---
**You're done!** Your backend is now running for free on Streamlit Community Cloud.
