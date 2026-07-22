"""
FastAPI inference service.
POST an audio file -> get back risk classification from both models.

Run: uvicorn serve:app --reload --port 8000
"""

import io
import numpy as np
import torch
import joblib
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from features import load_audio, extract_statistical_features, extract_mel_spectrogram
from dl_model import CoughCNN

app = FastAPI(title="Cough/Breath Disease Screening API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = ["covid", "healthy"]  # matches alphabetical folder order from build_dataset.py

# Load models at startup
rf_model = joblib.load("../models/rf_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

cnn_model = CoughCNN(num_classes=len(CLASSES))
cnn_model.load_state_dict(torch.load("../models/cnn_augmented_best.pt", map_location="cpu"))
cnn_model.eval()


@app.get("/")
def health_check():
    return {"status": "ok", "classes": CLASSES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    # Save to temp buffer and load with librosa (cross-platform temp path)
    import tempfile, os
    tmp_path = os.path.join(tempfile.gettempdir(), "coughsense_temp_audio.wav")
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)
    y = load_audio(tmp_path)

    # ML baseline prediction
    stat_feats = extract_statistical_features(y)
    feat_vector = np.array([list(stat_feats.values())])
    feat_vector_scaled = scaler.transform(feat_vector)
    rf_pred = rf_model.predict(feat_vector_scaled)[0]
    rf_proba = rf_model.predict_proba(feat_vector_scaled)[0]

    # DL prediction
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
        "cnn": {
            "prediction": CLASSES[cnn_pred],
            "confidence": float(np.max(cnn_proba)),
            "probabilities": {CLASSES[i]: float(p) for i, p in enumerate(cnn_proba)},
        },
        "disclaimer": "This is a screening aid, not a medical diagnosis. Consult a doctor for confirmation.",
    }