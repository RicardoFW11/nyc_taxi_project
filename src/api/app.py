"""
FastAPI application for NYC Taxi Fare Prediction
"""

import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.predictor import PredictionManager
from src.api.schemas import HealthResponse, PredictionResponse, TripRequest
#from src.config.settings import MODEL_FEATURES

# Initialize FastAPI app
app = FastAPI(
    title="NYC Taxi Fare & Duration Prediction API",
    description="API for predicting taxi fares and trip duration in New York City",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware (allows Streamlit to call the API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor (loaded once when app starts)
predictor = None


@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    global predictor
    try:
        #predictor = FarePredictor(model_name="linear_fare.pkl")
        #predictor = FarePredictor(model_name="xgboost_fare.pkl")
        predictor = PredictionManager(
            fare_model_name="xgboost_fare_amount_advanced.pkl", # Usamos XGBoost optimizado como principal
            duration_model_name="random_forest_trip_duration_minutes_advanced.pkl" # Usamos RF optimizado
        )
        print("🚀 API started successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise


@app.get("/", tags=["General"])
async def root():
    """Welcome endpoint"""
    return {
        "message": "NYC Taxi Fare & Duration Prediction API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy"
        if predictor and predictor.is_loaded()
        else "unhealthy",
        model_loaded=predictor.is_loaded() if predictor else False,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
#async def predict_fare(trip: TripRequest):
async def predict_trip(trip: TripRequest):
    """
    Predict taxi fare for a given trip

    Args:
        trip: TripRequest with trip details

    Returns:
        PredictionResponse with predicted fare and metadata
    """
    if predictor is None or not predictor.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later.",
        )

    try:
        # Make prediction
        result = predictor.predict(trip)

        return PredictionResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}"
        )


@app.get("/model-info", tags=["Model"])
async def model_info():
    """Get information about the loaded model"""
    if predictor is None or not predictor.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "fare_model": { # CAMBIO
            "version": predictor.model_versions['fare'],
            #"path": str(predictor._get_model_path(predictor.model_versions['fare'] + '.pkl')),
            "metrics": predictor.model_metrics.get('fare', {"MAE": "N/A"}),
            "features_used_count": len(predictor.model_features['fare']),
        },
        "duration_model": { # CAMBIO
            "version": predictor.model_versions['duration'],
            #"path": str(predictor._get_model_path(predictor.model_versions['duration'] + '.pkl')),
            "metrics": predictor.model_metrics.get('duration', {"MAE": "N/A"}),
            "features_used_count": len(predictor.model_features['duration']),
        },
        "total_features_engineered": len(predictor.feature_engineer.feature_stats.get('column_types', {}).get('numeric', [])) # CAMBIO
    }

    #return {
     #   "model_version": predictor.model_version,
     #   "model_path": str(predictor.model_path),
     #   "features_used": MODEL_FEATURES,
     #   "model_type": "Linear Regression",
    #}
