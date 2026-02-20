"""
FastAPI application for Iris classification predictions.
"""
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Initialize FastAPI app
app = FastAPI(
    title="Iris Classifier API",
    description="Machine Learning API for classifying Iris flowers",
    version="0.1.0"
)

# Global variables for model and metadata
model = None
feature_names = None
target_names = None


class IrisFeatures(BaseModel):
    """Input features for Iris classification."""
    sepal_length: float = Field(..., ge=0, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=0, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, description="Petal length in cm")
    petal_width: float = Field(..., ge=0, description="Petal width in cm")
    
    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predicted_class: int
    predicted_species: str
    confidence: float
    probabilities: dict


class BatchIrisFeatures(BaseModel):
    """Batch input for multiple predictions."""
    samples: list[IrisFeatures]


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: list[PredictionResponse]


@app.on_event("startup")
async def load_model():
    """Load the trained model and metadata on startup."""
    global model, feature_names, target_names
    
    # Get artifacts directory
    artifacts_dir = Path(__file__).parent.parent.parent / "artifacts"
    
    # Load model
    model_file = artifacts_dir / "iris_classifier.joblib"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    model = joblib.load(model_file)
    
    # Load feature names
    feature_names_file = artifacts_dir / "feature_names.joblib"
    if feature_names_file.exists():
        feature_names = joblib.load(feature_names_file)
    else:
        feature_names = ["sepal length (cm)", "sepal width (cm)", 
                        "petal length (cm)", "petal width (cm)"]
    
    # Load target names
    target_names_file = artifacts_dir / "target_names.joblib"
    if target_names_file.exists():
        target_names = joblib.load(target_names_file)
    else:
        target_names = ["setosa", "versicolor", "virginica"]
    
    print(f"✓ Model loaded successfully from {model_file}")
    print(f"✓ Features: {feature_names}")
    print(f"✓ Classes: {target_names}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Iris Classifier API",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features": feature_names,
        "classes": target_names
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: IrisFeatures):
    """
    Predict the Iris species from input features.
    
    Args:
        features: Iris flower measurements
        
    Returns:
        Prediction with class, species name, confidence, and probabilities
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Prepare input data
    input_data = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    # Get confidence (max probability)
    confidence = float(np.max(probabilities))
    
    # Prepare probabilities dict
    prob_dict = {
        target_names[i]: float(probabilities[i])
        for i in range(len(target_names))
    }
    
    return PredictionResponse(
        predicted_class=int(prediction),
        predicted_species=target_names[prediction],
        confidence=confidence,
        probabilities=prob_dict
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: BatchIrisFeatures):
    """
    Predict Iris species for multiple samples.
    
    Args:
        batch: List of Iris flower measurements
        
    Returns:
        List of predictions
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Prepare input data
    input_data = np.array([
        [s.sepal_length, s.sepal_width, s.petal_length, s.petal_width]
        for s in batch.samples
    ])
    
    # Make predictions
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    
    # Prepare response
    results = []
    for _i, (pred, probs) in enumerate(zip(predictions, probabilities, strict=True)):
        confidence = float(np.max(probs))
        prob_dict = {
            target_names[j]: float(probs[j])
            for j in range(len(target_names))
        }
        
        results.append(PredictionResponse(
            predicted_class=int(pred),
            predicted_species=target_names[pred],
            confidence=confidence,
            probabilities=prob_dict
        ))
    
    return BatchPredictionResponse(predictions=results)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
