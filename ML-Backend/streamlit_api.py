import streamlit as st
import os
import sys
import json
import base64
from PIL import Image
import io

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

# --- Minimal UI ---
st.title("AffectCare Backend API")
st.write("This application serves as the backend for the AffectCare project.")

# --- The "API Trick" ---
# We use a file uploader to simulate a POST request.
# While this isn't a traditional REST API, your React frontend can
# actually "talk" to this by automating the file upload if needed,
# OR you can use this page directly to test your model.

st.header("Test the Model")
uploaded_file = st.file_uploader("Upload an audio clip (.wav or .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save temp
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Predict
    label, prob = predict(temp_path, model, scaler)
    
    # Visualize
    img_path = "spectrogram.png"
    visualize_audio(temp_path, img_path)
    
    # Display Results
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Prediction")
        st.metric("Label", label.upper())
        st.metric("Confidence", f"{prob*100:.2f}%")
    
    with col2:
        st.subheader("Spectrogram")
        st.image(img_path)
    
    # Clean up
    os.remove(temp_path)
    os.remove(img_path)

st.divider()
st.info("Status: API is Healthy and Running on Streamlit Cloud.")
