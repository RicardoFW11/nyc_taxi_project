from fastapi import FastAPI, HTTPException
from src.api.schemas import TripInput, PredictionOutput
from src.api.predictor import predictor_service

app = FastAPI(title="NYC Taxi Predictor", version="1.0")

@app.on_event("startup")
def load_models():
    # Ensure models are loaded
    if predictor_service.model_fare is None:
        predictor_service.load_models()

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "nyc-taxi-predictor"}

@app.post("/predict", response_model=PredictionOutput)
def predict_trip(payload: TripInput):
    try:
        results = predictor_service.predict(payload)
        return PredictionOutput(
            predicted_fare=round(results['fare'], 2),
            predicted_duration_minutes=round(results['duration'], 1),
            model_version="xgboost_v1"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
