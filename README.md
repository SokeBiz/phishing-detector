# Phishing Email Detector

A machine learning-based system to detect phishing emails from legitimate ones.

## Features

- Text-based feature extraction (suspicious keywords, urgency indicators)
- URL analysis (IP-based URLs, suspicious TLDs, link count)
- Email header parsing
- RandomForest classifier with high accuracy
- Simple CLI and FastAPI web interface

## Installation

```bash
pip install -e .
python -m nltk.downloader punkt stopwords wordnet
```

## Usage

### CLI
```bash
python main.py --email "path/to/email.txt"
```

### Web API
```bash
uvicorn src.app:app --reload
```

## Project Structure

```
phishing-detector/
├── data/           # Training data
├── models/         # Saved trained models
├── src/detector/   # Core detection logic
├── tests/          # Unit tests
├── main.py         # CLI entry point
└── README.md
```
