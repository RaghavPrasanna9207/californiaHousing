"""
Pydantic schemas for FastAPI request/response validation.
"""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """
    Request schema for prediction endpoint.
    Accepts either a single record (dict) or multiple records (list of dicts).
    """
    # Using Union to accept either dict or list[dict]
    # If Pydantic version doesn't support this exact syntax, we'll handle validation in preprocess
    raw: Union[Dict[str, Any], List[Dict[str, Any]]] = Field(
        ...,
        description="Either a single record (dict) or list of records for batch prediction",
        examples=[
            # Single record example
            {
                "MedInc": 5.0,
                "HouseAge": 20,
                "AveRooms": 6.0,
                "AveBedrms": 2.2,
                "Population": 1500,
                "AveOccup": 3.0,
                "Latitude": 37.5,
                "Longitude": -121.9
            },
            # Multiple records example
            [
                {
                    "MedInc": 5.0,
                    "HouseAge": 20,
                    "AveRooms": 6.0,
                    "AveBedrms": 2.2,
                    "Population": 1500,
                    "AveOccup": 3.0,
                    "Latitude": 37.5,
                    "Longitude": -121.9
                },
                {
                    "MedInc": 3.2,
                    "HouseAge": 35,
                    "AveRooms": 4.5,
                    "AveBedrms": 1.8,
                    "Population": 2500,
                    "AveOccup": 2.6,
                    "Latitude": 34.2,
                    "Longitude": -118.3
                }
            ]
        ]
    )

    class Config:
        schema_extra = {
            "examples": [
                {
                    "raw": {
                        "MedInc": 5.0,
                        "HouseAge": 20,
                        "AveRooms": 6.0,
                        "AveBedrms": 2.2,
                        "Population": 1500,
                        "AveOccup": 3.0,
                        "Latitude": 37.5,
                        "Longitude": -121.9
                    }
                },
                {
                    "raw": [
                        {
                            "MedInc": 5.0,
                            "HouseAge": 20,
                            "AveRooms": 6.0,
                            "AveBedrms": 2.2,
                            "Population": 1500,
                            "AveOccup": 3.0,
                            "Latitude": 37.5,
                            "Longitude": -121.9
                        }
                    ]
                }
            ]
        }


class PredictResponse(BaseModel):
    """
    Response schema for prediction endpoint.
    """
    predictions: List[float] = Field(
        ...,
        description="List of predicted house values",
        examples=[[2.408, 2.383]]
    )
    model_version: Optional[str] = Field(
        None,
        description="Version of the model used for prediction",
        examples=["20250909_130932"]
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Model metadata including training metrics and feature names",
        examples=[{
            "train_metrics": {"rmse": 49401.57, "r2": 0.8239},
            "feature_names": ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
        }]
    )


class HealthResponse(BaseModel):
    """
    Response schema for health check endpoint.
    """
    status: str = Field(
        ...,
        description="Health status of the application",
        examples=["ok"]
    )
