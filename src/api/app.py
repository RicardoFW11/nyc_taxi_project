"""
FastAPI application for NYC Taxi Fare Prediction
"""

import sys
from datetime import datetime
from pathlib import Path
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Fix imports
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Local imports
from src.api.predictor import FarePredictor
from src.api.schemas import TripRequest, PredictionResponse, HealthResponse
from src.config.settings import MODEL_FEATURES



# --------------------------------------------------------------------
# APP CONFIG
# --------------------------------------------------------------------
APP_MODE = os.getenv("APP_MODE", "prod")

app = FastAPI(
    title="NYC Taxi Fare Prediction API",
    description="Predict taxi fare using ML model trained on NYC dataset",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor: FarePredictor | None = None
# --------------------------------------------------------------------


# ================================================================
# STARTUP EVENT
# ================================================================
@app.on_event("startup")
async def startup_event():
    global predictor
    try:
        predictor = FarePredictor(model_name="linear_fare.pkl")
        print("🚀 ML model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


# ================================================================
# BASE ROUTE
# ================================================================
@app.get("/", tags=["General"])
async def root():
    return {
        "message": "NYC Taxi Fare Prediction API",
        "version": "1.0.0",
        "routes": {
            "/health": "Check API status",
            "/predict": "POST taxi trip to get fare",
            "/model-info": "Metadata of ML model",
        }
    }


# ================================================================
# HEALTH CHECK
# ================================================================
@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    return HealthResponse(
        status="healthy" if predictor and predictor.is_loaded() else "unhealthy",
        model_loaded=predictor.is_loaded() if predictor else False,
        timestamp=datetime.utcnow(),
    )


# ================================================================
# PREDICTION
# ================================================================
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_fare(trip: TripRequest, mode: Optional[str] = None):
    """
    Predict taxi fare based on trip details.
    mode: "prod" | "dev"
    """
    current_mode = mode or APP_MODE

    # ========== DEV MODE MOCK ==========
    if current_mode == "dev":
        return PredictionResponse(
            predicted_fare=123.45,
            model_version="mock_dev",
            prediction_timestamp=datetime.utcnow(),
            input_features=trip.model_dump()
        )

    # ========== REAL MODEL ==========
    if predictor is None or not predictor.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Try again later."
        )

    try:
        prediction = predictor.predict(trip)

        return PredictionResponse(
            predicted_fare=prediction["fare"],
            model_version=predictor.model_version,
            prediction_timestamp=datetime.utcnow(),
            input_features=prediction["features"]
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


# ================================================================
# MODEL METADATA
# ================================================================
@app.get("/model-info", tags=["Model"])
async def model_info():
    if predictor is None or not predictor.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_version": predictor.model_version,
        "model_path": str(predictor.model_path),
        "features_used": MODEL_FEATURES,
        "model_type": predictor.model_type,
    }
