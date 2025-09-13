"""
Smoke tests for the FastAPI application.
"""
import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

# Import the FastAPI app
from app.main import app

# Create test client
client = TestClient(app)


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    model = MagicMock()
    model.predict.return_value = np.array([2.408, 2.383])
    return model


@pytest.fixture
def mock_metadata():
    """Mock metadata for testing."""
    return {
        "version": "20250909_130932",
        "train_metrics": {"rmse": 49401.57, "r2": 0.8239},
        "feature_names": ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"],
        "input_schema": {
            "MedInc": "float64",
            "HouseAge": "float64",
            "AveRooms": "float64",
            "AveBedrms": "float64",
            "Population": "float64",
            "AveOccup": "float64",
            "Latitude": "float64",
            "Longitude": "float64"
        }
    }


@patch('app.main.find_latest_artifact')
@patch('app.main.load_model')
def test_startup_with_model_and_metadata(mock_load_model, mock_find_artifact, mock_model, mock_metadata):
    """Test that the app starts up correctly with model and metadata."""
    # Mock the artifact finding and loading
    mock_find_artifact.return_value = "artifacts/model_v20250909_130932.joblib"
    mock_load_model.return_value = mock_model
    
    # Mock metadata loading
    with patch('builtins.open', create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_metadata)
        
        # Test startup
        with TestClient(app) as test_client:
            # The startup event should have run
            assert hasattr(test_client.app.state, 'model')
            assert hasattr(test_client.app.state, 'metadata')


def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "ok"


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "endpoints" in data


@patch('app.main.app.state.model')
@patch('app.main.app.state.metadata')
def test_predict_endpoint_single_record(mock_metadata, mock_model, mock_model_obj, mock_metadata_obj):
    """Test prediction endpoint with a single record."""
    # Set up mocks
    mock_model.return_value = mock_model_obj
    mock_metadata.return_value = mock_metadata_obj
    
    # Test data
    test_data = {
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
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "predictions" in data
    assert "model_version" in data
    assert "metadata" in data
    assert len(data["predictions"]) == 1
    assert isinstance(data["predictions"][0], (int, float))


@patch('app.main.app.state.model')
@patch('app.main.app.state.metadata')
def test_predict_endpoint_multiple_records(mock_metadata, mock_model, mock_model_obj, mock_metadata_obj):
    """Test prediction endpoint with multiple records."""
    # Set up mocks
    mock_model.return_value = mock_model_obj
    mock_metadata.return_value = mock_metadata_obj
    
    # Test data with multiple records
    test_data = {
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
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    assert all(isinstance(pred, (int, float)) for pred in data["predictions"])


@patch('app.main.app.state.model')
@patch('app.main.app.state.metadata')
def test_predict_endpoint_missing_features(mock_metadata, mock_model, mock_model_obj, mock_metadata_obj):
    """Test prediction endpoint with missing features."""
    # Set up mocks
    mock_model.return_value = mock_model_obj
    mock_metadata.return_value = mock_metadata_obj
    
    # Test data with missing features
    test_data = {
        "raw": {
            "MedInc": 5.0,
            "HouseAge": 20,
            # Missing other required features
        }
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 400
    
    data = response.json()
    assert "detail" in data
    assert "Missing required features" in data["detail"]


@patch('app.main.app.state.model')
@patch('app.main.app.state.metadata')
def test_predict_endpoint_invalid_data(mock_metadata, mock_model, mock_model_obj, mock_metadata_obj):
    """Test prediction endpoint with invalid data."""
    # Set up mocks
    mock_model.return_value = mock_model_obj
    mock_metadata.return_value = mock_metadata_obj
    
    # Test data with invalid types
    test_data = {
        "raw": {
            "MedInc": "invalid",  # Should be numeric
            "HouseAge": 20,
            "AveRooms": 6.0,
            "AveBedrms": 2.2,
            "Population": 1500,
            "AveOccup": 3.0,
            "Latitude": 37.5,
            "Longitude": -121.9
        }
    }
    
    response = client.post("/predict", json=test_data)
    # Should either succeed with type coercion or fail with validation error
    assert response.status_code in [200, 400]


def test_predict_endpoint_no_model():
    """Test prediction endpoint when no model is loaded."""
    # Temporarily remove model from app state
    original_model = getattr(app.state, 'model', None)
    app.state.model = None
    
    try:
        test_data = {
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
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 500
        
        data = response.json()
        assert "detail" in data
        assert "Model not loaded" in data["detail"]
    
    finally:
        # Restore original model
        app.state.model = original_model


def test_predict_endpoint_invalid_json():
    """Test prediction endpoint with invalid JSON structure."""
    # Test with completely invalid structure
    test_data = {
        "raw": "invalid_data"
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__])
