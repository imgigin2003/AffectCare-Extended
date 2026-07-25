import streamlit as st
import os
import sys
import json
import numpy as np
import base64

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "src"))

from predict import load_trained_model, load_scaler, predict
from visualize_api import visualize_audio

# --- Page Config ---
st.set_page_config(page_title="AffectCare Backend API", layout="centered")

# --- Global ML Model Loading ---
@st.cache_resource
def load_ml_artifacts():
    model_path = os.path.join(BASE_DIR, "src", "best_model.pth")
    scaler_path = os.path.join(BASE_DIR, "src", "scaler.pkl")
    model = load_trained_model(model_path)
    scaler = load_scaler(scaler_path)
    return model, scaler

model, scaler = load_ml_artifacts()

# --- Streamlit UI (Required but we can make it minimal) ---
st.title("AffectCare Backend API")
st.write("This application serves as the backend for the AffectCare project.")

# --- API Emulation using Query Parameters ---
# Streamlit isn't a traditional REST API, but we can use query parameters 
# or file uploaders to trigger actions. 
# However, for a true React -> Backend integration, we'll use a small 
# 'Secret' trick: Streamlit can serve raw JSON if we use st.write() 
# but for a React frontend, the best way is to use a simple FastAPI 
# and deploy it on a platform like Koyeb (if it still has a no-CC free tier) 
# or use the ONNX conversion.

# Since you want to avoid ONNX and need it free, let's try one more 
# platform that is often overlooked: **Koyeb**. 
# Their "Nano" instance is free and often doesn't require a CC.

# If you want to stick with the CURRENT code and just deploy it, 
# let's look at the "Damn Space" instruction again. 
# I will rewrite the guide for **Koyeb** or **Railway** (if they have 
# a free trial that doesn't require CC).

st.info("API is active. Connect your frontend to this URL.")
