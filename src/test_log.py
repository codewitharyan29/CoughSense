"""
Testing & Validation log.

Runs a sample of cough clips through all three trained models (Random Forest,
XGBoost, CNN) and records what each predicted vs the true label. Produces:

  - a printed table (easy to screenshot for the report)
  - reports/test_log.csv  (open in Excel, or embed in the report)

This is the "we actually tested it" evidence — shows the system working on
real clips, not just cross-validation numbers.

Run:  python test_log.py --n 20
"""

import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import joblib
import warnings
warnings.filterwarnings("ignore")

from features import load_audio, extract_statistical_features, extract_mel_spectrogram
from dl_model import CoughCNN
from xgboost import XGBClassifier

RAW_DIR = "../data/raw"
CLASSES = ["covid", "healthy"]  # alphabetical = label index 0,1


def load_models():
    rf = joblib.load("../models/rf_model.pkl")
    scaler = joblib.load("../models/scaler.pkl")
    xgb = XGBClassifier()
    xgb.load_model("../models/xgb_model.json")
    cnn = CoughCNN(num_classes=len(CLASSES))
    cnn.load_state_dict(torch.load("../models/cnn_augmented_best.pt", map_location="cpu"))
    cnn.eval()
    return rf, scaler, xgb, cnn


def predict_one(filepath, rf, scaler, xgb, cnn):
    y = load_audio(filepath)

    # tabular models
    feats = np.array([list(extract_statistical_features(y).values())])
    feats_s = scaler.transform(feats)
    rf_pred = CLASSES[rf.predict(feats_s)[0]]
    xgb_pred = CLASSES[int(xgb.predict(feats)[0])]

    # CNN
    mel = extract_mel_spectrogram(y)
    t = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        cnn_pred = CLASSES[int(cnn(t).argmax(1))]

    return rf_pred, xgb_pred, cnn_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="how many clips to test (split evenly across classes)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    rf, scaler, xgb, cnn = load_models()

    # Gather a balanced sample
    per_class = max(1, args.n // len(CLASSES))
    samples = []
    for cls in CLASSES:
        cdir = os.path.join(RAW_DIR, cls)
        files = [f for f in os.listdir(cdir) if f.lower().endswith((".mp3", ".wav", ".flac"))]
        random.shuffle(files)
        for f in files[:per_class]:
            samples.append((os.path.join(cdir, f), cls, f))

    rows = []
    for path, true_label, fname in samples:
        rf_p, xgb_p, cnn_p = predict_one(path, rf, scaler, xgb, cnn)
        # majority vote across the 3 models
        votes = [rf_p, xgb_p, cnn_p]
        consensus = max(set(votes), key=votes.count)
        rows.append({
            "clip": fname[:28],
            "actual": true_label,
            "RF": rf_p,
            "XGBoost": xgb_p,
            "CNN": cnn_p,
            "consensus": consensus,
            "correct": "yes" if consensus == true_label else "no",
        })

    df = pd.DataFrame(rows)

    # Print a clean table
    print("\n" + "=" * 78)
    print("  COUGHSENSE — TESTING & VALIDATION LOG")
    print("=" * 78)
    print(df.to_string(index=False))
    print("=" * 78)

    # Per-model accuracy on this sample
    n = len(df)
    for m in ["RF", "XGBoost", "CNN", "consensus"]:
        acc = (df[m] == df["actual"]).mean() * 100
        print(f"  {m:10s} correct on sample: {acc:.0f}%  ({(df[m]==df['actual']).sum()}/{n})")
    print("=" * 78)

    os.makedirs("../reports", exist_ok=True)
    out = "../reports/test_log.csv"
    df.to_csv(out, index=False)
    print(f"\n  Saved -> {out}")
    print("  (open in Excel, or embed the table in your report)")


if __name__ == "__main__":
    main()
