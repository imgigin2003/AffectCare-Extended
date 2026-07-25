---
title: AffectCare ML Backend
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# AffectCare ML Backend

This Space hosts the machine learning backend for the AffectCare project. It provides API endpoints for audio prediction and visualization.

## Endpoints:

*   `/api/predict` (POST): Predicts distress from an audio file.
*   `/api/visualize` (POST): Generates an MFCC spectrogram for an audio file.
*   `/api/model/metrics` (GET): Retrieves model evaluation metrics.
*   `/api/model/loss-history` (GET): Retrieves the model training loss history.
*   `/health` (GET): Health check endpoint.

## Setup:

This application is built with FastAPI and uses a Docker environment. The `app.py` file serves as the entry point.

**Important:** This Space requires the `best_model.pth` and `scaler.pkl` files to be present in the `src/` directory within this repository. These files are typically generated during model training.
