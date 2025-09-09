import json
import os

import pandas as pd

from src.data.load_data import make_synthetic_data
from src.data.preprocessing import build_preprocessing_pipeline
from src.models.train import Config, train_and_evaluate, save_artifacts


def test_training_and_artifacts(tmp_path):
    # Build config targeting tmp artifacts dir
    cfg = Config(
        project_name="test_model",
        random_seed=42,
        test_size=0.2,
        data={"path": None, "target_column": "MedHouseVal"},
        artifacts_dir=str(tmp_path / "artifacts"),
        model={"type": "RandomForestRegressor", "params": {"n_estimators": 50, "random_state": 42, "n_jobs": -1}},
        notebook_source="housingCalifornia.ipynb",
    )

    pipeline, metrics, X, y = train_and_evaluate(cfg)
    assert "rmse" in metrics and "r2" in metrics

    cfg_path = str(tmp_path / "config.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write("dummy: true\n")

    joblib_path, metadata_path = save_artifacts(pipeline, metrics, X, cfg, cfg_path)
    assert os.path.exists(joblib_path)
    assert os.path.exists(metadata_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["framework"] == "scikit-learn"
    assert isinstance(meta["feature_names"], list)


def test_preprocessing_predict_shape():
    X, y = make_synthetic_data(n=20)
    pre = build_preprocessing_pipeline(sample_df=X)
    import numpy as np
    transformed = pre.fit_transform(X, y)
    # Should have shape (n_samples, n_features)
    assert transformed.shape[0] == X.shape[0]
