"""
Test whether RICHER features improve accuracy on the clinical data.

Adds to the original 37 features:
  - delta-MFCCs (how each MFCC changes over time — captures cough dynamics)
  - delta2-MFCCs (acceleration)
  - spectral contrast (peak-vs-valley energy, good for distinguishing cough textures)
  - more spectral stats

Then runs the same 5-fold CV on RF + XGBoost so the number is comparable.
If it helps, we fold these features into the main pipeline. If not, we keep
the simpler set (honest — no point adding complexity that doesn't help).

Run:  python enhanced_features.py
"""

import numpy as np
import librosa
import os
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

SAMPLE_RATE = 22050
DURATION = 2.0
N_MFCC = 13
HOP = 512
RAW_DIR = "../data/raw"


def load_audio(fp):
    y, _ = librosa.load(fp, sr=SAMPLE_RATE, duration=DURATION)
    target = int(SAMPLE_RATE * DURATION)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]
    return y


def rich_features(y, sr=SAMPLE_RATE):
    feats = []
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    for arr in (mfcc, d1, d2):
        feats.extend(np.mean(arr, axis=1))
        feats.extend(np.std(arr, axis=1))

    # spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=HOP)
    feats.extend(np.mean(contrast, axis=1))
    feats.extend(np.std(contrast, axis=1))

    # other spectral
    for f in [
        librosa.feature.spectral_centroid(y=y, sr=sr),
        librosa.feature.spectral_rolloff(y=y, sr=sr),
        librosa.feature.spectral_bandwidth(y=y, sr=sr),
        librosa.feature.zero_crossing_rate(y),
        librosa.feature.rms(y=y),
    ]:
        feats.append(np.mean(f))
        feats.append(np.std(f))

    return np.array(feats)


def main():
    classes = sorted([d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))])
    X, y = [], []
    for idx, c in enumerate(classes):
        cdir = os.path.join(RAW_DIR, c)
        for f in os.listdir(cdir):
            if f.lower().endswith((".wav", ".mp3", ".flac")):
                try:
                    X.append(rich_features(load_audio(os.path.join(cdir, f))))
                    y.append(idx)
                except Exception:
                    pass
    X, y = np.array(X), np.array(y)
    print(f"Data: {X.shape[0]} clips, {X.shape[1]} features (was 37)\n")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf = make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=400, max_depth=14, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1))
    xgb = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=42)

    rf_scores = cross_val_score(rf, X, y, cv=skf)
    xgb_scores = cross_val_score(xgb, X, y, cv=skf)

    print(f"Random Forest (rich): {rf_scores.mean()*100:.1f}% +/- {rf_scores.std()*100:.1f}%")
    print(f"XGBoost (rich):       {xgb_scores.mean()*100:.1f}% +/- {xgb_scores.std()*100:.1f}%")
    print(f"\nCompare to original 37-feature XGBoost: 85.2%")


if __name__ == "__main__":
    main()
