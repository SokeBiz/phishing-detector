"""Prediction / inference module for phishing email detection."""
import os

import joblib
import numpy as np
import pandas as pd

from detector.features import extract_features

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")


def load_model(model_dir: str = MODEL_DIR):
    """Load trained model, scaler, and feature names."""
    model_path = os.path.join(model_dir, "phishing_model.joblib")
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    feature_names_path = os.path.join(model_dir, "feature_names.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run model training first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(feature_names_path)

    return model, scaler, feature_names


def predict(subject: str, body: str, model_dir: str = MODEL_DIR) -> dict:
    """Predict whether an email is phishing."""
    model, scaler, feature_names = load_model(model_dir)

    # Extract features
    features_dict = extract_features(subject, body)
    features_df = pd.DataFrame([features_dict])

    # Ensure correct column order
    features_df = features_df.reindex(columns=feature_names, fill_value=0)

    # Scale and predict
    X_scaled = scaler.transform(features_df)
    proba = model.predict_proba(X_scaled)[0]
    prediction = model.predict(X_scaled)[0]

    verdict = "PHISHING" if prediction == 1 else "LEGITIMATE"
    confidence = float(proba[1] if prediction == 1 else proba[0])

    # Top risk factors
    importances = dict(zip(feature_names, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "phishing_probability": round(float(proba[1]), 4),
        "features": features_dict,
        "top_risk_factors": top_features,
    }


def predict_from_file(file_path: str, model_dir: str = MODEL_DIR) -> dict:
    """Read an email from a text file and predict."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Simple parsing: first line as subject, rest as body
    lines = content.strip().split("\n")
    if lines[0].lower().startswith("subject:"):
        subject = lines[0][8:].strip()
        body = "\n".join(lines[1:]).strip()
    else:
        subject = lines[0]
        body = "\n".join(lines[1:]) if len(lines) > 1 else ""

    return predict(subject, body, model_dir)


if __name__ == "__main__":
    # Quick test
    test_email = {
        "subject": "URGENT: Verify your account!!!",
        "body": "Dear user, click here to verify: http://192.168.1.1/login.html. Your account will be suspended!",
    }
    result = predict(test_email["subject"], test_email["body"])
    print(f"Verdict: {result['verdict']} (confidence: {result['confidence']})")
    print(f"Phishing probability: {result['phishing_probability']}")
