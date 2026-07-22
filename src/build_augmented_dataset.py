"""
Builds an augmented spectrogram dataset for the CNN.

CRITICAL: split into train/val/test at the FILE level first, then augment
ONLY the training files. This avoids leakage (an augmented copy of a test
clip ending up in training) which would inflate accuracy artificially.
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from features import load_audio, extract_mel_spectrogram
from augment import augment_clip

RAW_DATA_DIR = "../data/raw"
OUT_DIR = "../data/processed"


def build_augmented_dataset(raw_dir=RAW_DATA_DIR, out_dir=OUT_DIR, seed=42):
    os.makedirs(out_dir, exist_ok=True)

    classes = sorted(
        [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    )
    print(f"Classes: {classes}")

    # Collect all filepaths + labels
    all_files, all_labels = [], []
    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(raw_dir, class_name)
        files = [f for f in os.listdir(class_dir) if f.lower().endswith((".wav", ".mp3", ".flac"))]
        for fname in files:
            all_files.append(os.path.join(class_dir, fname))
            all_labels.append(label_idx)

    all_files = np.array(all_files)
    all_labels = np.array(all_labels)

    # Split FIRST, at file level, stratified — before any augmentation happens
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        all_files, all_labels, test_size=0.3, stratify=all_labels, random_state=seed
    )
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=seed
    )

    print(f"Train files: {len(train_files)} | Val files: {len(val_files)} | Test files: {len(test_files)}")

    def process_split(files, labels, augment=False):
        specs, out_labels = [], []
        for filepath, label in tqdm(zip(files, labels), total=len(files)):
            y = load_audio(filepath)
            specs.append(extract_mel_spectrogram(y))
            out_labels.append(label)

            if augment:
                for aug_y in augment_clip(y, sr=22050):
                    specs.append(extract_mel_spectrogram(aug_y))
                    out_labels.append(label)

        return np.stack(specs), np.array(out_labels)

    print("\nProcessing train split (with augmentation)...")
    X_train, y_train = process_split(train_files, train_labels, augment=True)

    print("Processing val split (no augmentation)...")
    X_val, y_val = process_split(val_files, val_labels, augment=False)

    print("Processing test split (no augmentation)...")
    X_test, y_test = process_split(test_files, test_labels, augment=False)

    np.savez_compressed(
        os.path.join(out_dir, "spectrograms_augmented.npz"),
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        classes=classes,
    )

    print(f"\nFinal shapes:")
    print(f"  Train: {X_train.shape} (was {len(train_files)} before augmentation -> {len(train_files)*4} after)")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, classes


if __name__ == "__main__":
    build_augmented_dataset()
