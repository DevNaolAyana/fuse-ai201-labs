import os
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Initialize FastAPI application
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="FastAPI service for real-time credit card fraud detection using trained XGBoost and SMOTE models.",
    version="1.0.0"
)

# Resolve model artifact paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fraud_detector_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "feature_scaler.pkl"

# Fallback path if run from project root
if not MODEL_PATH.exists():
    MODEL_PATH = Path("models/fraud_detector_model.pkl")
    SCALER_PATH = Path("models/feature_scaler.pkl")

# Artifact storage in app state
model = None
scaler = None


def load_artifacts():
    """Load model and feature scaler artifacts."""
    global model, scaler
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        if not SCALER_PATH.exists():
            raise FileNotFoundError(f"Scaler file not found at: {SCALER_PATH}")

        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
        print(f"Scaler loaded successfully from {SCALER_PATH}")
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        raise e


# Pre-load artifacts at import time
try:
    load_artifacts()
except Exception as e:
    print(f"Warning: Artifacts could not be loaded immediately: {e}")


@app.on_event("startup")
def startup_event():
    """Load model artifacts on startup."""
    if model is None or scaler is None:
        load_artifacts()


class TransactionInput(BaseModel):
    """Pydantic schema representing 30 transaction features."""
    Time: float = Field(..., description="Seconds elapsed since the first transaction in the dataset")
    V1: float = Field(..., description="PCA Component 1")
    V2: float = Field(..., description="PCA Component 2")
    V3: float = Field(..., description="PCA Component 3")
    V4: float = Field(..., description="PCA Component 4")
    V5: float = Field(..., description="PCA Component 5")
    V6: float = Field(..., description="PCA Component 6")
    V7: float = Field(..., description="PCA Component 7")
    V8: float = Field(..., description="PCA Component 8")
    V9: float = Field(..., description="PCA Component 9")
    V10: float = Field(..., description="PCA Component 10")
    V11: float = Field(..., description="PCA Component 11")
    V12: float = Field(..., description="PCA Component 12")
    V13: float = Field(..., description="PCA Component 13")
    V14: float = Field(..., description="PCA Component 14")
    V15: float = Field(..., description="PCA Component 15")
    V16: float = Field(..., description="PCA Component 16")
    V17: float = Field(..., description="PCA Component 17")
    V18: float = Field(..., description="PCA Component 18")
    V19: float = Field(..., description="PCA Component 19")
    V20: float = Field(..., description="PCA Component 20")
    V21: float = Field(..., description="PCA Component 21")
    V22: float = Field(..., description="PCA Component 22")
    V23: float = Field(..., description="PCA Component 23")
    V24: float = Field(..., description="PCA Component 24")
    V25: float = Field(..., description="PCA Component 25")
    V26: float = Field(..., description="PCA Component 26")
    V27: float = Field(..., description="PCA Component 27")
    V28: float = Field(..., description="PCA Component 28")
    Amount: float = Field(..., description="Monetary transaction amount in USD")

    model_config = {
        "json_schema_extra": {
            "example": {
                "Time": 406.0,
                "V1": -2.31,
                "V2": 1.95,
                "V3": -1.61,
                "V4": 3.99,
                "V5": -0.52,
                "V6": -1.43,
                "V7": -2.54,
                "V8": 1.39,
                "V9": -2.77,
                "V10": -2.77,
                "V11": 3.20,
                "V12": -2.90,
                "V13": -0.60,
                "V14": -4.29,
                "V15": 0.39,
                "V16": -1.14,
                "V17": -2.83,
                "V18": -0.02,
                "V19": 0.42,
                "V20": 0.13,
                "V21": 0.52,
                "V22": -0.04,
                "V23": -0.47,
                "V24": 0.32,
                "V25": 0.04,
                "V26": 0.18,
                "V27": 0.26,
                "V28": -0.14,
                "Amount": 239.93
            }
        }
    }


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    fraud_probability: float
    decision: str


@app.get("/", tags=["Health Check"])
def root():
    """Root health-check endpoint."""
    return {
        "service": "Credit Card Fraud Detection API",
        "status": "online",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }


@app.get("/health", tags=["Health Check"])
def health():
    """Detailed health status."""
    if model is None or scaler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or scaler artifacts are not loaded"
        )
    return {
        "status": "healthy",
        "model_path": str(MODEL_PATH),
        "scaler_path": str(SCALER_PATH)
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(transaction: TransactionInput):
    """
    Accepts 30 transaction features (Time, V1-V28, Amount),
    standardizes Time and Amount using the saved scaler,
    and returns fraud probability and decision.
    """
    if model is None or scaler is None:
        load_artifacts()

    try:
        # Convert Pydantic model to dictionary
        data = transaction.model_dump()

        # Step 1: Standardize Time and Amount using the pre-fitted scaler
        time_amt_df = pd.DataFrame([[data["Time"], data["Amount"]]], columns=["Time", "Amount"])
        scaled_time_amt = scaler.transform(time_amt_df)
        scaled_time = scaled_time_amt[0, 0]
        scaled_amount = scaled_time_amt[0, 1]

        # Step 2: Construct full 30-feature vector in exact training order
        feature_dict = {"Time": scaled_time}
        for i in range(1, 29):
            feature_dict[f"V{i}"] = data[f"V{i}"]
        feature_dict["Amount"] = scaled_amount

        feature_df = pd.DataFrame([feature_dict])

        # Step 3: Model inference
        prob = float(model.predict_proba(feature_df)[0][1])
        prediction = int(model.predict(feature_df)[0])

        decision = "Fraud" if (prediction == 1 or prob >= 0.5) else "Legitimate"

        return PredictionResponse(
            fraud_probability=round(prob, 4),
            decision=decision
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )
