import sys
from pathlib import Path

import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# Import LangGraph workflow
from orchestration.workflows.airops_workflow import air_ops_graph


# Initialize API
app = FastAPI(
    title="AirOps AI",
    description="AI-Powered Airline Operations Intelligence Platform",
    version="2.0"
)


# Load ML model
model = joblib.load(
    PROJECT_ROOT / "ml/models/risk_model.pkl"
)

encoder = joblib.load(
    PROJECT_ROOT / "ml/models/label_encoder.pkl"
)


FEATURES = [
    "distance_km",
    "temperature_c",
    "wind_speed_kmh",
    "visibility_km",
    "altitude_ft",
    "velocity_kmh",
]


class FlightData(BaseModel):
    distance_km: float
    temperature_c: float
    wind_speed_kmh: float
    visibility_km: float
    altitude_ft: float
    velocity_kmh: float


# Health endpoint
@app.get("/")
def health():
    return {
        "status": "AirOps AI API running",
        "version": "2.0"
    }


# Individual ML prediction
@app.post("/predict")
def predict_risk(flight: FlightData):

    data = pd.DataFrame(
        [flight.model_dump()],
        columns=FEATURES
    )

    prediction = model.predict(data)[0]
    probabilities = model.predict_proba(data)[0]

    return {
        "predicted_risk": encoder.inverse_transform(
            [prediction]
        )[0],

        "confidence": round(
            float(probabilities.max()), 4
        ),

        "risk_probabilities": {
            label: round(float(probability), 4)
            for label, probability in zip(
                encoder.classes_,
                probabilities
            )
        },

        "model_type": "Random Forest"
    }


# Integrated AI-powered operations report
@app.get("/operations/report")
def operations_report():

    try:
        result = air_ops_graph.invoke({})

        return {
            "platform": "AirOps AI",
            "version": "2.0",

            "flight_monitoring": result.get(
                "flight_findings", {}
            ),

            "weather_intelligence": result.get(
                "weather_findings", []
            ),

            "overall_assessment": result.get(
                "overall_assessment", ""
            ),

            "operational_recommendations": result.get(
                "operational_recommendations", []
            ),

            "llm_model": result.get(
                "llm_model", None
            ),

            "llm_error": result.get(
                "llm_error", None
            )
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Operations report generation failed: {str(e)}"
        )  