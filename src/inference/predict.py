import argparse
import json
import os
from typing import Dict

import pandas as pd

# Import centralized utilities
from src.utils.model_loader import load_model, predict as predict_array


def main() -> int:
    parser = argparse.ArgumentParser(description="Run inference with a saved model")
    parser.add_argument("--model", type=str, required=True, help="Path to model .joblib file")
    parser.add_argument("--input_json", type=str, required=True, help="Path to JSON file with records list")
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="Where to write predictions")
    args = parser.parse_args()

    model = load_model(args.model)

    with open(args.input_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    if isinstance(records, dict) and "records" in records:
        records = records["records"]
    if not isinstance(records, list):
        raise ValueError("Input JSON must be a list of records or an object with key 'records'.")

    input_df = pd.DataFrame.from_records(records)
    preds_array = predict_array(model, input_df)
    preds = pd.Series(preds_array, name="prediction")
    out_df = input_df.copy()
    out_df["prediction"] = preds
    out_df.to_csv(args.output_csv, index=False)
    print(f"Wrote predictions to {args.output_csv}")
    return 0


if __name__ == "__main__":
    exit(main())
