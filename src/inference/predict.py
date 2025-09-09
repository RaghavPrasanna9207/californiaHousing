import argparse
import json
import os
from typing import Any, Dict

import joblib
import pandas as pd


def load_model(path: str):
    return joblib.load(path)


def predict(model, input_df: pd.DataFrame) -> pd.Series:
    preds = model.predict(input_df)
    return pd.Series(preds, name="prediction")


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
    preds = predict(model, input_df)
    out_df = input_df.copy()
    out_df["prediction"] = preds
    out_df.to_csv(args.output_csv, index=False)
    print(f"Wrote predictions to {args.output_csv}")
    return 0


if __name__ == "__main__":
    exit(main())
