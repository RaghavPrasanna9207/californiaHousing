"""
FastAPI application for California Housing model predictions.
"""
import json
import logging
import pathlib
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.utils.model_loader import find_latest_artifact, load_model, preprocess_input, predict
from .schemas import HealthResponse, PredictRequest, PredictResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="California Housing Prediction API",
    description="API for predicting California housing prices using machine learning models",
    version="1.0.0"
)

# Add CORS middleware with restricted origins for security
# For local development, you can use ["http://localhost:3000", "http://127.0.0.1:3000"]
# For production, replace with your actual frontend domain(s)
ALLOWED_ORIGINS = [
    "http://localhost:3000",    # React default dev server
    "http://127.0.0.1:3000",   # Alternative localhost
    "http://localhost:5173",    # Vite default dev server
    "http://127.0.0.1:5173",   # Alternative localhost for Vite
    # Add your production frontend domain here:
    # "https://your-frontend-domain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """
    Startup event to load model and metadata.
    """
    logger.info("Starting up California Housing Prediction API...")
    
    try:
        # Find latest model artifact
        model_path = find_latest_artifact("artifacts")
        if model_path is None:
            logger.error("No model artifacts found in artifacts/ directory")
            raise RuntimeError("No model artifacts found. Please train a model first.")
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        model = load_model(model_path)
        app.state.model = model
        logger.info(f"Successfully loaded model: {type(model)}")
        
        # Try to load metadata
        metadata_path = model_path.with_suffix('.metadata.json')
        metadata = None
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Successfully loaded metadata from {metadata_path}")
            except Exception as e:
                logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
        else:
            logger.warning(f"No metadata file found at {metadata_path}")
            logger.info("Will use identity preprocessing (no schema validation)")
        
        app.state.metadata = metadata
        
        # Log model info
        if metadata:
            model_version = metadata.get("version", "unknown")
            train_metrics = metadata.get("train_metrics", {})
            feature_names = metadata.get("feature_names", [])
            
            logger.info(f"Model version: {model_version}")
            logger.info(f"Training metrics: {train_metrics}")
            logger.info(f"Feature names: {feature_names}")
        else:
            logger.info("No metadata available - using fallback preprocessing")
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise RuntimeError(f"Failed to start API: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event for cleanup.
    """
    logger.info("Shutting down California Housing Prediction API...")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    """
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(request: PredictRequest):
    """
    Prediction endpoint for California housing prices.
    """
    try:
        # Get model and metadata from app state
        model = getattr(app.state, 'model', None)
        metadata = getattr(app.state, 'metadata', None)
        
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Preprocess input
        logger.info(f"Processing prediction request with {len(request.raw) if isinstance(request.raw, list) else 1} record(s)")
        
        try:
            df = preprocess_input(request.raw, metadata)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Input validation failed: {str(e)}")
        
        # Make predictions
        try:
            predictions = predict(model, df)
        except ValueError as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
        # Prepare response
        response_data = {
            "predictions": predictions.tolist(),
            "model_version": None,
            "metadata": None
        }
        
        # Add metadata if available
        if metadata:
            response_data["model_version"] = metadata.get("version")
            response_data["metadata"] = {
                "train_metrics": metadata.get("train_metrics"),
                "feature_names": metadata.get("feature_names")
            }
        
        logger.info(f"Successfully generated {len(predictions)} predictions")
        return PredictResponse(**response_data)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in prediction endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "California Housing Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
