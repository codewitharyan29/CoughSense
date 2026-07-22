"""
CoughSense — FastAPI backend + custom frontend, served together.
Built for HuggingFace Docker Spaces (port 7860), but runs identically anywhere.
"""

import os
import tempfile
import numpy as np
import torch
import joblib
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from features import load_audio, extract_statistical_features, extract_mel_spectrogram
from dl_model import CoughCNN
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = FastAPI(title="CoughSense API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = ["covid", "healthy"]

rf_model = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
xgb_model = XGBClassifier()
xgb_model.load_model(os.path.join(MODELS_DIR, "xgb_model.json"))

cnn_model = CoughCNN(num_classes=len(CLASSES))
cnn_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "cnn_augmented_best.pt"), map_location="cpu"))
cnn_model.eval()


@app.get("/api/health")
def health_check():
    return {"status": "ok", "classes": CLASSES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    tmp_path = os.path.join(tempfile.gettempdir(), "coughsense_temp_audio.wav")
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)
    y = load_audio(tmp_path)

    stat_feats = extract_statistical_features(y)
    feat_vector = np.array([list(stat_feats.values())])
    feat_vector_scaled = scaler.transform(feat_vector)
    rf_pred = rf_model.predict(feat_vector_scaled)[0]
    rf_proba = rf_model.predict_proba(feat_vector_scaled)[0]

    xgb_pred = int(xgb_model.predict(feat_vector)[0])
    xgb_proba = xgb_model.predict_proba(feat_vector)[0]

    mel_spec = extract_mel_spectrogram(y)
    spec_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        cnn_logits = cnn_model(spec_tensor)
        cnn_proba = torch.softmax(cnn_logits, dim=1).numpy()[0]
        cnn_pred = int(np.argmax(cnn_proba))

    return {
        "random_forest": {
            "prediction": CLASSES[rf_pred],
            "confidence": float(np.max(rf_proba)),
            "probabilities": {CLASSES[i]: float(p) for i, p in enumerate(rf_proba)},
        },
        "xgboost": {
            "prediction": CLASSES[xgb_pred],
            "confidence": float(np.max(xgb_proba)),
            "probabilities": {CLASSES[i]: float(p) for i, p in enumerate(xgb_proba)},
        },
        "cnn": {
            "prediction": CLASSES[cnn_pred],
            "confidence": float(np.max(cnn_proba)),
            "probabilities": {CLASSES[i]: float(p) for i, p in enumerate(cnn_proba)},
        },
        "disclaimer": "This is a screening aid, not a medical diagnosis. Consult a doctor for confirmation.",
    }


# Serve the custom frontend (index.html, dashboard.html, favicon, etc.)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
