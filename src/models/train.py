import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.load_data import load_data, RANDOM_SEED_DEFAULT
from src.data.preprocessing import build_preprocessing_pipeline


@dataclass
class Config:
    project_name: str
    random_seed: int
    test_size: float
    data: Dict[str, Any]
    artifacts_dir: str
    model: Dict[str, Any]
    notebook_source: str


def set_seeds(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    # sklearn estimators take random_state; nothing global to set there


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Config(
        project_name=cfg.get("project_name", "california_housing_model"),
        random_seed=int(cfg.get("random_seed", RANDOM_SEED_DEFAULT)),
        test_size=float(cfg.get("test_size", 0.2)),
        data=cfg.get("data", {}),
        artifacts_dir=cfg.get("artifacts_dir", "artifacts"),
        model=cfg.get("model", {}),
        notebook_source=cfg.get("notebook_source", "housingCalifornia.ipynb"),
    )


def train_and_evaluate(cfg: Config) -> Tuple[Pipeline, Dict[str, float], pd.DataFrame, pd.Series]:
    set_seeds(cfg.random_seed)

    data_path = cfg.data.get("path")
    target_column = cfg.data.get("target_column", "MedHouseVal")

    X, y = load_data(path=data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_seed
    )

    preprocessor = build_preprocessing_pipeline(sample_df=X_train)

    model_type = (cfg.model or {}).get("type", "RandomForestRegressor")
    model_params = (cfg.model or {}).get("params", {})

    if model_type != "RandomForestRegressor":
        # TODO: If your notebook used a different model, implement selection here.
        # Defaulting to RandomForestRegressor.
        pass

    regressor = RandomForestRegressor(**model_params)

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", regressor)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {"rmse": rmse, "r2": r2}
    return pipeline, metrics, X, y


def ensure_dir(dir_path: str) -> None:
    os.makedirs(dir_path, exist_ok=True)


def save_artifacts(
    pipeline: Pipeline,
    metrics: Dict[str, float],
    X: pd.DataFrame,
    cfg: Config,
    config_file_path: str,
) -> Tuple[str, str]:
    ensure_dir(cfg.artifacts_dir)

    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_basename = f"model_v{version}"
    joblib_path = os.path.join(cfg.artifacts_dir, f"{model_basename}.joblib")
    metadata_path = os.path.join(cfg.artifacts_dir, f"{model_basename}.metadata.json")

    joblib.dump(pipeline, joblib_path)

    feature_names = list(X.columns)
    input_schema = {col: str(dtype) for col, dtype in X.dtypes.items()}

    metadata: Dict[str, Any] = {
        "model_name": cfg.project_name,
        "version": version,
        "datetime": datetime.utcnow().isoformat() + "Z",
        "framework": "scikit-learn",
        "train_metrics": metrics,
        "feature_names": feature_names,
        "input_schema": input_schema,
        "training_params": cfg.model.get("params", {}),
        "random_seed": cfg.random_seed,
        "notebook_source": cfg.notebook_source,
        "todo": [
            "TODO: Confirm dataset path and schema in configs/config.yaml",
            "TODO: If using a different model than RandomForest, adjust config and code",
        ],
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save a copy of the config used into artifacts dir for reproducibility
    try:
        shutil.copy2(config_file_path, os.path.join(cfg.artifacts_dir, f"config_v{version}.yaml"))
    except Exception:
        pass

    return joblib_path, metadata_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train model and export artifacts")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("configs", "config.yaml"),
        help="Path to YAML config",
    )
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        pipeline, metrics, X, _ = train_and_evaluate(cfg)
        joblib_path, metadata_path = save_artifacts(pipeline, metrics, X, cfg, args.config)
        print(f"Saved model to: {joblib_path}")
        print(f"Saved metadata to: {metadata_path}")
        print(f"Training metrics: {metrics}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    exit(main())
