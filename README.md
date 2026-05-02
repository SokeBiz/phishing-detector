# Phishing Email Detector

A machine learning-based system to detect phishing emails from legitimate ones. Uses a RandomForest classifier with 33 hand-crafted features extracted from email subject and body.

## Features

- **Text analysis** — suspicious keywords, urgency indicators, greeting patterns
- **URL analysis** — IP-based URLs, suspicious TLDs (.tk, .ml, .ga, .top, .xyz), URL shorteners
- **HTML inspection** — form tags, embedded links, script tags
- **Subject-line heuristics** — all-caps, exclamation marks, trigger words
- **RandomForest classifier** with balanced class weights and cross-validation
- **FastAPI web API** for integration with other tools
- **CLI interface** for quick scanning from terminal

## Quick Start

```bash
# Install
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Download NLTK data
python3 -m nltk.downloader punkt punkt_tab stopwords

# Train model
python3 -m src.detector.model

# Scan an email
python main.py -s "URGENT: Verify your account!!!" -b "Click http://evil.ml to confirm"

# Scan from file
python main.py -f suspicious_email.txt

# JSON output
python main.py -s "Meeting at 3pm" -b "Let's sync up" --json
```

## Usage

### CLI

```bash
# Single email
python main.py --subject "URGENT" --body "Click here now: http://192.168.1.1/login"

# From file (first line = Subject, rest = body)
python main.py --file email.txt

# Machine-readable output
python main.py --subject "Hello" --body "Test" --json
```

Exit code: `0` for legitimate, `1` for phishing.

### Web API

```bash
# Start server
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000

# Check health
curl http://localhost:8000/health

# Analyze an email
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"subject": "URGENT", "body": "Click http://evil.ml"}'

# Open interactive docs
# http://localhost:8000/docs
```

### Python API

```python
from detector.predict import predict

result = predict("URGENT", "Click http://evil.ml/login")
print(result["verdict"])          # "PHISHING" or "LEGITIMATE"
print(result["confidence"])       # 0.9876
print(result["phishing_probability"])  # 0.9876
print(result["top_risk_factors"]) # [(feature, importance), ...]
```

## Project Structure

```
phishing-detector/
├── data/              # Training data (synthetic)
├── models/            # Saved trained model + scaler
├── src/
│   ├── detector/
│   │   ├── __init__.py
│   │   ├── data_generator.py   # Synthetic data generation
│   │   ├── features.py         # Feature extraction (33 features)
│   │   ├── model.py            # Training & evaluation
│   │   └── predict.py          # Inference API
│   └── app.py                  # FastAPI web service
├── tests/
│   ├── test_features.py        # Feature extraction tests
│   └── test_predict.py         # Prediction tests
├── main.py                     # CLI entry point
├── pyproject.toml
└── README.md
```

## Training

```bash
# Train (default: RandomForest with 200 estimators)
python3 -m src.detector.model

# Model artifacts saved to models/:
#   phishing_model.joblib   — trained RandomForest
#   scaler.joblib            — feature scaler
#   feature_names.joblib     — column order
#   metrics.json             — accuracy, precision, recall, F1, ROC-AUC
```

## Testing

```bash
python3 -m pytest tests/ -v
```

## License

MIT
