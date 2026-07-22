"""
K-fold cross-validation for reliable performance estimates.

With only 121 samples, a single 80/10/10 split is noisy — a handful of
"hard" samples landing in the test fold can swing accuracy by 10-15
points either way. 5-fold CV trains/tests 5 times on different splits
and reports mean +/- std, which is what you should actually cite in
a report or pitch instead of a single split's number.
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

from features import load_audio, extract_mel_spectrogram
from augment import augment_clip
from dl_model import CoughCNN, SpectrogramDataset, train_model, evaluate_model
from torch.utils.data import DataLoader


def cv_tabular_models(csv_path="../data/processed/features_tabular.csv", n_splits=5):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["label", "label_idx"]).values
    y = df["label_idx"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    rf_accs, rf_f1s = [], []
    xgb_accs, xgb_f1s = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        rf = RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        rf.fit(X_train_s, y_train)
        rf_pred = rf.predict(X_test_s)
        rf_accs.append(accuracy_score(y_test, rf_pred))
        rf_f1s.append(f1_score(y_test, rf_pred, average="weighted"))

        xgb = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            eval_metric="mlogloss", random_state=42
        )
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        xgb_accs.append(accuracy_score(y_test, xgb_pred))
        xgb_f1s.append(f1_score(y_test, xgb_pred, average="weighted"))

        print(f"Fold {fold+1}: RF acc={rf_accs[-1]:.3f}  XGB acc={xgb_accs[-1]:.3f}")

    print(f"\nRandom Forest: {np.mean(rf_accs)*100:.1f}% +/- {np.std(rf_accs)*100:.1f}%  (F1: {np.mean(rf_f1s)*100:.1f}%)")
    print(f"XGBoost:       {np.mean(xgb_accs)*100:.1f}% +/- {np.std(xgb_accs)*100:.1f}%  (F1: {np.mean(xgb_f1s)*100:.1f}%)")

    return rf_accs, xgb_accs


def cv_cnn(raw_dir="../data/raw", n_splits=5, epochs=20):
    classes = sorted([d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])

    all_files, all_labels = [], []
    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(raw_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".wav", ".mp3", ".flac")):
                all_files.append(os.path.join(class_dir, fname))
                all_labels.append(label_idx)

    all_files = np.array(all_files)
    all_labels = np.array(all_labels)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cnn_accs = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(all_files, all_labels)):
        train_files, test_files = all_files[train_idx], all_files[test_idx]
        train_labels, test_labels = all_labels[train_idx], all_labels[test_idx]

        # Augment training fold only
        X_train, y_train = [], []
        for fp, lbl in zip(train_files, train_labels):
            y_audio = load_audio(fp)
            X_train.append(extract_mel_spectrogram(y_audio))
            y_train.append(lbl)
            for aug_y in augment_clip(y_audio, sr=22050):
                X_train.append(extract_mel_spectrogram(aug_y))
                y_train.append(lbl)

        X_test, y_test = [], []
        for fp, lbl in zip(test_files, test_labels):
            y_audio = load_audio(fp)
            X_test.append(extract_mel_spectrogram(y_audio))
            y_test.append(lbl)

        X_train, y_train = np.stack(X_train), np.array(y_train)
        X_test, y_test = np.stack(X_test), np.array(y_test)

        train_loader = DataLoader(SpectrogramDataset(X_train, y_train), batch_size=16, shuffle=True)
        test_loader = DataLoader(SpectrogramDataset(X_test, y_test), batch_size=16)

        model = CoughCNN(num_classes=len(classes))
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            model.train()
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                preds = model(Xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        fold_acc = correct / total
        cnn_accs.append(fold_acc)
        print(f"Fold {fold+1}: CNN acc={fold_acc:.3f}")

    print(f"\nCNN: {np.mean(cnn_accs)*100:.1f}% +/- {np.std(cnn_accs)*100:.1f}%")
    return cnn_accs


if __name__ == "__main__":
    print("=== 5-Fold CV: Tabular Models (RF, XGBoost) ===")
    rf_accs, xgb_accs = cv_tabular_models()

    print("\n=== 5-Fold CV: CNN (with per-fold augmentation) ===")
    cnn_accs = cv_cnn()

    print("\n=== FINAL SUMMARY (5-fold CV, mean +/- std) ===")
    print(f"Random Forest: {np.mean(rf_accs)*100:.1f}% +/- {np.std(rf_accs)*100:.1f}%")
    print(f"XGBoost:       {np.mean(xgb_accs)*100:.1f}% +/- {np.std(xgb_accs)*100:.1f}%")
    print(f"CNN:           {np.mean(cnn_accs)*100:.1f}% +/- {np.std(cnn_accs)*100:.1f}%")
