"""FastAPI web interface for Phishing Email Detector."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from detector.predict import predict

app = FastAPI(
    title="Phishing Email Detector",
    description="ML-based API for detecting phishing emails",
    version="0.1.0",
)


class EmailInput(BaseModel):
    subject: str
    body: str = ""


class RiskFactor(BaseModel):
    name: str
    importance: float


class PredictionResult(BaseModel):
    verdict: str
    confidence: float
    phishing_probability: float
    top_risk_factors: list[RiskFactor]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check if the service is running and model is loaded."""
    import os

    model_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "phishing_model.joblib"
    )
    return HealthResponse(
        status="ok",
        model_loaded=os.path.exists(model_path),
    )


@app.post("/predict", response_model=PredictionResult)
async def analyze_email(email: EmailInput):
    """Analyze an email and determine if it's phishing."""
    try:
        result = predict(email.subject, email.body)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run model training first.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictionResult(
        verdict=result["verdict"],
        confidence=result["confidence"],
        phishing_probability=result["phishing_probability"],
        top_risk_factors=[
            RiskFactor(name=name, importance=round(imp, 4))
            for name, imp in result.get("top_risk_factors", [])[:5]
        ],
    )


@app.get("/")
async def root():
    return {
        "service": "Phishing Email Detector",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }
