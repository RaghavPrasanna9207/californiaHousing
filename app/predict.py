"""
Functional helpers for model loading, preprocessing, and prediction.
This module now imports centralized utilities from src.utils.model_loader.
"""
# Import centralized utilities
from src.utils.model_loader import (
    find_latest_artifact,
    load_model,
    infer_schema_from_metadata,
    preprocess_input,
    predict
)


def main():
    """
    CLI interface for local prediction testing.
    """
    import json
    import logging
    import argparse
    
    # Configure logging for CLI usage
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
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
