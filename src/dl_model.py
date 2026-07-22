"""
CNN on mel-spectrograms for cough/breath classification.
This is the "wow" model - captures raw time-frequency patterns
that hand-crafted features can miss (e.g. subtle crackle/wheeze textures).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


class SpectrogramDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, n_mels, time_steps) -> add channel dim for CNN
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, H, W)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CoughCNN(nn.Module):
    """
    Compact CNN - deliberately small so it trains fast on CPU/free-tier GPU
    and doesn't overfit on a modest hackathon-sized dataset.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # handles variable input size cleanly
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv_block(x)
        return self.classifier(x)


def train_model(model, train_loader, val_loader, epochs=25, lr=1e-3, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        val_acc = correct / total
        scheduler.step(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "../models/cnn_best.pt")

        print(f"Epoch {epoch+1}/{epochs} | train_loss={train_loss/len(train_loader):.4f} | val_acc={val_acc:.4f}")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    return model


def evaluate_model(model, test_loader, device="cpu"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    print(classification_report(all_labels, all_preds))
    print("Confusion matrix:\n", confusion_matrix(all_labels, all_preds))
    return all_preds, all_labels


def main():
    data = np.load("../data/processed/spectrograms.npz", allow_pickle=True)
    X, y, classes = data["X"], data["y"], data["classes"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    train_loader = DataLoader(SpectrogramDataset(X_train, y_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(SpectrogramDataset(X_val, y_val), batch_size=16)
    test_loader = DataLoader(SpectrogramDataset(X_test, y_test), batch_size=16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = CoughCNN(num_classes=len(classes))
    model = train_model(model, train_loader, val_loader, epochs=25, device=device)

    print("\n=== Final Test Set Evaluation ===")
    evaluate_model(model, test_loader, device=device)


if __name__ == "__main__":
    main()
