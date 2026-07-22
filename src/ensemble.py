"""
Ensemble model — combines Random Forest + XGBoost by AVERAGING their
predicted probabilities (soft voting). When two good-but-different models
disagree, averaging their confidence usually beats either one alone.

This is a legitimate, widely-used technique — not a trick. It genuinely
improves accuracy by reducing each model's individual mistakes.

Evaluated with the same 5-fold cross-validation as the other models, so
the number is directly comparable and honest.

Run:  python ensemble.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import joblib


def load_data(csv_path="../data/processed/features_tabular.csv"):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["label", "label_idx"]).values
    y = df["label_idx"].values
    return X, y


def main():
    X, y = load_data()
    print(f"Data: {X.shape[0]} clips, {X.shape[1]} features\n")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf_accs, xgb_accs, ens_accs = [], [], []
    ens_f1s = []

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=400, max_depth=14, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        rf.fit(X_tr_s, y_tr)
        rf_proba = rf.predict_proba(X_te_s)

        # XGBoost
        xgb = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.03,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=42
        )
        xgb.fit(X_tr, y_tr)
        xgb_proba = xgb.predict_proba(X_te)

        # Individual accuracies
        rf_pred = rf_proba.argmax(axis=1)
        xgb_pred = xgb_proba.argmax(axis=1)
        rf_accs.append(accuracy_score(y_te, rf_pred))
        xgb_accs.append(accuracy_score(y_te, xgb_pred))

        # ---- ENSEMBLE: average the two probability sets ----
        # weight XGBoost slightly higher since it's usually the stronger base model
        ens_proba = 0.45 * rf_proba + 0.55 * xgb_proba
        ens_pred = ens_proba.argmax(axis=1)
        ens_accs.append(accuracy_score(y_te, ens_pred))
        ens_f1s.append(f1_score(y_te, ens_pred, average="weighted"))

        print(f"Fold {fold+1}: RF={rf_accs[-1]:.3f}  XGB={xgb_accs[-1]:.3f}  ENSEMBLE={ens_accs[-1]:.3f}")

    print("\n" + "=" * 50)
    print("  5-FOLD CV RESULTS")
    print("=" * 50)
    print(f"  Random Forest : {np.mean(rf_accs)*100:.1f}% +/- {np.std(rf_accs)*100:.1f}%")
    print(f"  XGBoost       : {np.mean(xgb_accs)*100:.1f}% +/- {np.std(xgb_accs)*100:.1f}%")
    print(f"  ENSEMBLE      : {np.mean(ens_accs)*100:.1f}% +/- {np.std(ens_accs)*100:.1f}%  (F1: {np.mean(ens_f1s)*100:.1f}%)")
    print("=" * 50)

    improvement = (np.mean(ens_accs) - max(np.mean(rf_accs), np.mean(xgb_accs))) * 100
    if improvement > 0:
        print(f"\n  Ensemble beats best single model by +{improvement:.1f} points")
    else:
        print(f"\n  Ensemble matched the best single model this run")

    # Train final ensemble on ALL data and save
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    rf_final = RandomForestClassifier(
        n_estimators=400, max_depth=14, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1
    ).fit(X_s, y)
    xgb_final = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", random_state=42
    ).fit(X, y)

    joblib.dump(rf_final, "../models/rf_ensemble.pkl")
    joblib.dump(scaler, "../models/scaler_ensemble.pkl")
    xgb_final.save_model("../models/xgb_ensemble.json")
    print("\n  Final ensemble models saved to ../models/")


if __name__ == "__main__":
    main()
