"""
Builds train-ready datasets from a directory of labeled audio clips.

Expected input layout:
    data/raw/
        healthy/
            clip1.wav
            clip2.wav
        covid/
            clip1.wav
        asthma/
            ...

Outputs:
    data/processed/features_tabular.csv   (for ML baseline)
    data/processed/spectrograms.npz       (for DL model: X, y arrays)
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from features import process_file

RAW_DATA_DIR = "../data/raw"
PROCESSED_DIR = "../data/processed"


def build_datasets(raw_dir=RAW_DATA_DIR, out_dir=PROCESSED_DIR):
    os.makedirs(out_dir, exist_ok=True)

    classes = sorted(
        [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    )
    print(f"Found classes: {classes}")

    tabular_rows = []
    spec_list = []
    label_list = []

    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(raw_dir, class_name)
        files = [f for f in os.listdir(class_dir) if f.lower().endswith((".wav", ".mp3", ".flac"))]

        for fname in tqdm(files, desc=f"Processing {class_name}"):
            filepath = os.path.join(class_dir, fname)
            try:
                stat_feats, mel_spec = process_file(filepath)
            except Exception as e:
                print(f"  Skipping {filepath}: {e}")
                continue

            stat_feats["label"] = class_name
            stat_feats["label_idx"] = label_idx
            tabular_rows.append(stat_feats)

            spec_list.append(mel_spec)
            label_list.append(label_idx)

    # Save tabular features (for ML baseline)
    df = pd.DataFrame(tabular_rows)
    df.to_csv(os.path.join(out_dir, "features_tabular.csv"), index=False)
    print(f"Saved tabular features: {df.shape}")

    # Save spectrograms (for DL model)
    X = np.stack(spec_list)  # (N, n_mels, time_steps)
    y = np.array(label_list)
    np.savez_compressed(
        os.path.join(out_dir, "spectrograms.npz"), X=X, y=y, classes=classes
    )
    print(f"Saved spectrograms: X={X.shape}, y={y.shape}")

    return df, X, y, classes


if __name__ == "__main__":
    build_datasets()
