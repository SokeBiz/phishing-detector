"""Model training and evaluation for phishing email detection."""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from detector.features import extract_features


def load_data(csv_path: str = "data/emails.csv") -> pd.DataFrame:
    """Load and prepare the dataset."""
    df = pd.read_csv(csv_path)
    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple:
    """Extract features from all emails in the dataframe."""
    feature_list = []
    for _, row in df.iterrows():
        feats = extract_features(str(row["subject"]), str(row["body"]))
        feature_list.append(feats)

    X = pd.DataFrame(feature_list)
    y = df["label"].values
    return X, y


def train_model(X: pd.DataFrame, y: np.ndarray, model_type: str = "random_forest") -> tuple:
    """Train a classifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=42
        )
    elif model_type == "logistic_regression":
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="f1")
    metrics["cv_f1_mean"] = cv_scores.mean()
    metrics["cv_f1_std"] = cv_scores.std()

    return model, scaler, X_test_scaled, y_test, y_pred, y_proba, metrics


def print_report(metrics: dict, y_test, y_pred, feature_names):
    """Print training report."""
    print("=" * 60)
    print("PHISHING EMAIL DETECTOR - TRAINING REPORT")
    print("=" * 60)
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"CV F1:     {metrics['cv_f1_mean']:.4f} (+/- {metrics['cv_f1_std']:.4f})")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))


def save_model(model, scaler, feature_names, metrics, output_dir="models"):
    """Save trained model, scaler, and metadata."""
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(model, os.path.join(output_dir, "phishing_model.joblib"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    joblib.dump(feature_names, os.path.join(output_dir, "feature_names.joblib"))

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to {output_dir}/")


def main():
    print("Loading data...")
    df = load_data()

    print("Extracting features...")
    X, y = build_feature_matrix(df)

    print(f"Training on {len(X)} samples with {X.shape[1]} features...")
    model, scaler, X_test, y_test, y_pred, y_proba, metrics = train_model(X, y, model_type="random_forest")

    print_report(metrics, y_test, y_pred, list(X.columns))
    save_model(model, scaler, list(X.columns), metrics)

    return model, scaler, metrics


if __name__ == "__main__":
    main()
