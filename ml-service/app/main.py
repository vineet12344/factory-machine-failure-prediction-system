"""
FastAPI ML Service for Factory Machine Failure Prediction
This service will be called by your Go backend
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Get the directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# Create FastAPI app
app = FastAPI(
    title="Factory Machine Failure Prediction API",
    description="Predicts machine failure based on sensor data",
    version="1.0.0"
)

# ============================================
# LOAD MODEL AND ARTIFACTS (on startup)
# ============================================
print("=" * 50)
print("Loading ML Model and Artifacts...")
print("=" * 50)

# Load the trained model
model = joblib.load(os.path.join(MODEL_DIR, 'failure_prediction_model.pkl'))
print("✓ Model loaded")

# Load the scaler
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
print("✓ Scaler loaded")

# Load label encoder
label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
print("✓ Label encoder loaded")

# Load feature names
feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
print(f"✓ Features loaded: {len(feature_names)} features")

# Load model metadata
metadata = joblib.load(os.path.join(MODEL_DIR, 'model_metadata.pkl'))
print(f"✓ Model metadata loaded (ROC-AUC: {metadata['roc_auc']:.3f})")

print("\n✅ All artifacts loaded successfully!")
print("=" * 50)

# ============================================
# DEFINE REQUEST/RESPONSE MODELS
# ============================================

class SensorData(BaseModel):
    """Sensor data from a single machine"""
    type: str = Field(..., description="Product type: L, M, or H", example="M")
    air_temperature: float = Field(..., description="Air temperature in Kelvin", example=300.0)
    process_temperature: float = Field(..., description="Process temperature in Kelvin", example=310.0)
    rotational_speed: float = Field(..., description="Rotational speed in RPM", example=1500.0)
    torque: float = Field(..., description="Torque in Nm", example=45.0)
    tool_wear: float = Field(..., description="Tool wear in minutes", example=100.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "M",
                "air_temperature": 300.0,
                "process_temperature": 310.0,
                "rotational_speed": 1500.0,
                "torque": 55.0,
                "tool_wear": 200.0
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response from the API"""
    machine_id: Optional[str] = None
    failure_probability: float = Field(..., description="Probability of failure (0 to 1)")
    prediction: str = Field(..., description="Prediction: NORMAL or FAILURE")
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, HIGH, or CRITICAL")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version used")
    threshold_used: float = Field(..., description="Decision threshold used")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request for multiple machines"""
    machines: List[SensorData]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    roc_auc: float
    features: List[str]


# ============================================
# HELPER FUNCTIONS
# ============================================

def encode_type(type_str: str) -> int:
    """Convert L, M, H to numerical values"""
    try:
        return label_encoder.transform([type_str])[0]
    except:
        raise ValueError(f"Invalid type: {type_str}. Must be L, M, or H")


def engineer_features(data: np.ndarray) -> np.ndarray:
    """
    Add engineered features to match training data
    Input: [type, air_temp, process_temp, speed, torque, tool_wear]
    Output: [type, air_temp, process_temp, speed, torque, tool_wear, 
             temp_diff, power, tool_wear_percent]
    """
    type_val = data[0]
    air_temp = data[1]
    process_temp = data[2]
    speed = data[3]
    torque = data[4]
    tool_wear = data[5]
    
    # Engineered features
    temp_difference = process_temp - air_temp
    power = torque * speed
    tool_wear_percent = (tool_wear / 250) * 100  # 250 is max tool wear
    
    # Combine all features
    features = np.array([
        type_val,
        air_temp,
        process_temp,
        speed,
        torque,
        tool_wear,
        temp_difference,
        power,
        tool_wear_percent
    ])
    
    return features.reshape(1, -1)


def get_risk_level(probability: float) -> str:
    """Determine risk level based on probability"""
    if probability >= 0.7:
        return "CRITICAL"
    elif probability >= 0.5:
        return "HIGH"
    elif probability >= 0.3:
        return "MEDIUM"
    else:
        return "LOW"


# ============================================
# API ENDPOINTS
# ============================================


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Factory Machine Failure Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Check service health",
            "/predict": "Predict single machine",
            "/predict/batch": "Predict multiple machines"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for your Go backend"""
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        roc_auc=metadata['roc_auc'],
        features=feature_names
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(sensor_data: SensorData):
    """
    Predict machine failure for a single machine
    
    Your Go backend will send sensor data and receive failure probability
    """
    try:
        # Step 1: Extract data
        type_encoded = encode_type(sensor_data.type)
        
        # Step 2: Create base features array
        base_features = np.array([
            type_encoded,
            sensor_data.air_temperature,
            sensor_data.process_temperature,
            sensor_data.rotational_speed,
            sensor_data.torque,
            sensor_data.tool_wear
        ])
        
        # Step 3: Engineer features
        features = engineer_features(base_features)
        
        # Step 4: Scale features
        features_scaled = scaler.transform(features)
        
        # Step 5: Get prediction probability
        probability = model.predict_proba(features_scaled)[0][1]
        
        # Step 6: Determine prediction based on threshold
        threshold = 0.5  # You can adjust this
        prediction = "FAILURE" if probability >= threshold else "NORMAL"
        risk_level = get_risk_level(probability)
        
        # Step 7: Return response
        return PredictionResponse(
            failure_probability=round(probability, 4),
            prediction=prediction,
            risk_level=risk_level,
            timestamp=datetime.now().isoformat(),
            model_version="1.0.0",
            threshold_used=threshold
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict machine failure for multiple machines
    Useful for batch processing in your Go backend
    """
    results = []
    
    for i, machine in enumerate(request.machines):
        try:
            type_encoded = encode_type(machine.type)
            base_features = np.array([
                type_encoded,
                machine.air_temperature,
                machine.process_temperature,
                machine.rotational_speed,
                machine.torque,
                machine.tool_wear
            ])
            
            features = engineer_features(base_features)
            features_scaled = scaler.transform(features)
            probability = model.predict_proba(features_scaled)[0][1]
            
            threshold = 0.5
            prediction = "FAILURE" if probability >= threshold else "NORMAL"
            risk_level = get_risk_level(probability)
            
            results.append({
                "machine_index": i,
                "failure_probability": round(probability, 4),
                "prediction": prediction,
                "risk_level": risk_level
            })
            
        except Exception as e:
            results.append({
                "machine_index": i,
                "error": str(e)
            })
    
    return {
        "total_machines": len(request.machines),
        "predictions": results,
        "timestamp": datetime.now().isoformat()
    }


# ============================================
# RUN WITH: uvicorn main:app --reload --host 0.0.0.0 --port 8001
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)