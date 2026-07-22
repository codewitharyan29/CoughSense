"""
Classic ML baseline: Random Forest + XGBoost on statistical audio features.
Interpretable, fast to train, good sanity-check against the DL model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)
from xgboost import XGBClassifier
import joblib


def load_tabular_data(csv_path="../data/processed/features_tabular.csv"):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["label", "label_idx"])
    y = df["label_idx"]
    return X, y, df["label"].unique()


def train_random_forest(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)

    print("=== Random Forest ===")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Feature importance - useful for judges/explainability
    importances = pd.Series(clf.feature_importances_, index=X_train.columns)
    print("\nTop 10 important features:")
    print(importances.sort_values(ascending=False).head(10))

    return clf, scaler, y_pred, y_proba


def train_xgboost(X_train, y_train, X_test, y_test):
    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        eval_metric="mlogloss", random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    print("=== XGBoost ===")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    return clf, y_pred, y_proba


def main():
    X, y, class_names = load_tabular_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    rf_model, scaler, rf_pred, rf_proba = train_random_forest(
        X_train, y_train, X_test, y_test
    )
    xgb_model, xgb_pred, xgb_proba = train_xgboost(
        X_train, y_train, X_test, y_test
    )

    # Save best model
    joblib.dump(rf_model, "../models/rf_model.pkl")
    joblib.dump(scaler, "../models/scaler.pkl")
    xgb_model.save_model("../models/xgb_model.json")
    print("\nModels saved to ../models/")


if __name__ == "__main__":
    main()
