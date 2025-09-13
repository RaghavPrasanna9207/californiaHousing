"""
Functional helpers for model loading, preprocessing, and prediction.
"""
import json
import logging
import pathlib
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_latest_artifact(artifacts_dir: str = "artifacts") -> Optional[pathlib.Path]:
    """
    Find the latest model artifact in the artifacts directory.
    
    Args:
        artifacts_dir: Path to artifacts directory
        
    Returns:
        Path to latest .joblib or .pkl file, or None if not found
    """
    artifacts_path = pathlib.Path(artifacts_dir)
    if not artifacts_path.exists():
        logger.warning(f"Artifacts directory {artifacts_dir} does not exist")
        return None
    
    # Look for .joblib files first, then .pkl files
    joblib_files = list(artifacts_path.glob("*.joblib"))
    pkl_files = list(artifacts_path.glob("*.pkl"))
    
    all_files = joblib_files + pkl_files
    
    if not all_files:
        logger.warning(f"No model artifacts found in {artifacts_dir}")
        return None
    
    # Sort by modification time and return the latest
    latest_file = max(all_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Found latest model artifact: {latest_file}")
    return latest_file


def load_model(path: Union[str, pathlib.Path]) -> Any:
    """
    Load model from file using joblib, with fallback to pickle.
    
    Args:
        path: Path to model file
        
    Returns:
        Loaded model object
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model loading fails
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    try:
        # Try joblib first (preferred for sklearn models)
        model = joblib.load(path)
        logger.info(f"Successfully loaded model using joblib from {path}")
        return model
    except Exception as joblib_error:
        logger.warning(f"Joblib loading failed: {joblib_error}")
        try:
            # Fallback to pickle
            import pickle
            with open(path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Successfully loaded model using pickle from {path}")
            return model
        except Exception as pickle_error:
            logger.error(f"Both joblib and pickle loading failed. Joblib: {joblib_error}, Pickle: {pickle_error}")
            raise Exception(f"Failed to load model from {path}. Joblib error: {joblib_error}, Pickle error: {pickle_error}")


def infer_schema_from_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
    """
    Infer input schema from model metadata.
    
    Args:
        metadata: Model metadata dictionary
        
    Returns:
        Dictionary mapping feature names to their types
    """
    schema = {}
    
    if "feature_names" in metadata:
        feature_names = metadata["feature_names"]
        input_schema = metadata.get("input_schema", {})
        
        for feature_name in feature_names:
            if feature_name in input_schema:
                schema[feature_name] = input_schema[feature_name]
            else:
                # Default to float64 for numeric features
                schema[feature_name] = "float64"
                logger.warning(f"No type info for feature {feature_name}, defaulting to float64")
    
    return schema


def preprocess_input(raw: Union[Dict[str, Any], List[Dict[str, Any]]], metadata: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Robust preprocessing routine for input data.
    
    Args:
        raw: Either a single record (dict) or list of records
        metadata: Optional model metadata for schema validation
        
    Returns:
        Preprocessed DataFrame ready for prediction
        
    Raises:
        ValueError: If input validation fails
    """
    # Convert single record to list for uniform processing
    if isinstance(raw, dict):
        records = [raw]
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError(f"Input must be dict or list of dicts, got {type(raw)}")
    
    if not records:
        raise ValueError("Input records list is empty")
    
    # Create DataFrame from records
    try:
        df = pd.DataFrame(records)
    except Exception as e:
        raise ValueError(f"Failed to create DataFrame from records: {e}")
    
    if df.empty:
        raise ValueError("DataFrame is empty after conversion")
    
    # If metadata is available, use it for validation and ordering
    if metadata and "feature_names" in metadata:
        expected_features = metadata["feature_names"]
        logger.info(f"Using metadata schema with {len(expected_features)} features")
        
        # Check for missing features
        missing_features = set(expected_features) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {list(missing_features)}")
        
        # Reorder columns to match expected feature order
        df = df[expected_features]
        
        # Try to convert to appropriate types based on metadata
        input_schema = metadata.get("input_schema", {})
        for feature in expected_features:
            if feature in input_schema:
                target_type = input_schema[feature]
                try:
                    if target_type in ["float64", "float32"]:
                        df[feature] = pd.to_numeric(df[feature], errors='coerce')
                    elif target_type in ["int64", "int32"]:
                        df[feature] = pd.to_numeric(df[feature], errors='coerce').astype('Int64')
                    # Add more type conversions as needed
                except Exception as e:
                    logger.warning(f"Failed to convert {feature} to {target_type}: {e}")
        
        # Check for NaN values after conversion
        nan_features = df.columns[df.isnull().any()].tolist()
        if nan_features:
            raise ValueError(f"Features contain NaN values after type conversion: {nan_features}")
    
    else:
        logger.warning("No metadata available, attempting to coerce all features to numeric")
        # Fallback: try to convert all columns to numeric
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                logger.warning(f"Failed to convert {col} to numeric: {e}")
        
        # Check for NaN values
        nan_features = df.columns[df.isnull().any()].tolist()
        if nan_features:
            raise ValueError(f"Features contain NaN values: {nan_features}")
    
    logger.info(f"Successfully preprocessed {len(df)} records with {len(df.columns)} features")
    return df


def predict(model: Any, df: pd.DataFrame) -> np.ndarray:
    """
    Make predictions using the loaded model.
    
    Args:
        model: Loaded model object
        df: Preprocessed DataFrame
        
    Returns:
        Array of predictions
        
    Raises:
        ValueError: If prediction fails
    """
    try:
        # Handle different model types
        if hasattr(model, 'predict'):
            # Standard sklearn-style model
            predictions = model.predict(df)
            logger.info(f"Made predictions using model.predict() method")
        elif isinstance(model, dict) and "pipeline" in model:
            # Model wrapped in dict with pipeline key
            pipeline = model["pipeline"]
            if hasattr(pipeline, 'predict'):
                predictions = pipeline.predict(df)
                logger.info(f"Made predictions using pipeline.predict() method")
            else:
                raise ValueError("Pipeline object does not have predict method")
        else:
            raise ValueError(f"Model type {type(model)} not supported. Expected model with predict method or dict with 'pipeline' key")
        
        # Ensure predictions are numpy array
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
        
        logger.info(f"Generated {len(predictions)} predictions")
        return predictions
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise ValueError(f"Prediction failed: {e}")


def main():
    """
    CLI interface for local prediction testing.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run local prediction test")
    parser.add_argument("--model", type=str, required=True, help="Path to model file")
    parser.add_argument("--input_json", type=str, required=True, help="Path to input JSON file")
    args = parser.parse_args()
    
    try:
        # Load model
        model = load_model(args.model)
        logger.info(f"Loaded model: {type(model)}")
        
        # Load input data
        with open(args.input_json, 'r') as f:
            input_data = json.load(f)
        
        # Handle different JSON formats
        if isinstance(input_data, dict) and "records" in input_data:
            records = input_data["records"]
        elif isinstance(input_data, list):
            records = input_data
        elif isinstance(input_data, dict):
            records = [input_data]
        else:
            raise ValueError(f"Unexpected input format: {type(input_data)}")
        
        # Preprocess and predict
        df = preprocess_input(records)
        predictions = predict(model, df)
        
        # Print results
        print("Predictions:")
        for i, pred in enumerate(predictions):
            print(f"  Record {i+1}: {pred:.6f}")
        
        logger.info("Local prediction test completed successfully")
        
    except Exception as e:
        logger.error(f"Local prediction test failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
