"""
CNN training on the AUGMENTED dataset (train split augmented 4x, val/test untouched).
Compare this file's results against the original dl_model.py run
to see the improvement from augmentation.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from dl_model import CoughCNN, SpectrogramDataset, train_model, evaluate_model


def main():
    data = np.load("../data/processed/spectrograms_augmented.npz", allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    classes = data["classes"]

    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    train_loader = DataLoader(SpectrogramDataset(X_train, y_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(SpectrogramDataset(X_val, y_val), batch_size=16)
    test_loader = DataLoader(SpectrogramDataset(X_test, y_test), batch_size=16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = CoughCNN(num_classes=len(classes))
    model = train_model(model, train_loader, val_loader, epochs=30, device=device)

    print("\n=== Final Test Set Evaluation (Augmented Training) ===")
    evaluate_model(model, test_loader, device=device)

    torch.save(model.state_dict(), "../models/cnn_augmented_best.pt")
    print("\nSaved to ../models/cnn_augmented_best.pt")


if __name__ == "__main__":
    main()