import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd


RANDOM_SEED_DEFAULT = 42


def make_synthetic_data(n: int = 500, random_seed: int = RANDOM_SEED_DEFAULT) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create a small synthetic dataset that resembles the California Housing dataset
    schema used in many tutorials. This matches the typical columns from
    sklearn.datasets.fetch_california_housing(as_frame=True).

    Columns:
      - MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
    Target:
      - MedHouseVal

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with 8 numeric features.
    y : pd.Series
        Target series (float).

    Notes
    -----
    - The synthetic generation is simple but deterministic and provides signal so that
      models can learn and tests are meaningful.
    - TODO: Replace with real dataset by setting `configs/config.yaml:data.path` and
      ensuring the CSV has the required columns.
    """
    rng = np.random.default_rng(random_seed)

    med_inc = rng.gamma(shape=2.0, scale=2.0, size=n)  # ~ income proxy
    house_age = rng.integers(low=1, high=52, size=n).astype(float)
    ave_rooms = rng.normal(loc=5.0, scale=1.5, size=n).clip(1, None)
    ave_bedrms = (ave_rooms / rng.normal(loc=2.5, scale=0.3, size=n)).clip(0.5, None)
    population = rng.integers(low=200, high=8000, size=n).astype(float)
    ave_occup = rng.normal(loc=3.0, scale=1.0, size=n).clip(0.5, None)
    latitude = rng.uniform(32.5, 42.0, size=n)
    longitude = rng.uniform(-124.5, -114.0, size=n)

    # Construct a target with some relationship to features + noise
    noise = rng.normal(loc=0.0, scale=0.5, size=n)
    med_house_val = (
        0.8 * med_inc
        + 0.02 * house_age
        + 0.1 * ave_rooms
        - 0.15 * ave_bedrms
        - 0.00005 * population
        + 0.05 * ave_occup
        + 0.03 * (latitude - 37.0)
        - 0.03 * (longitude + 120.0)
        + noise
    )

    X = pd.DataFrame(
        {
            "MedInc": med_inc,
            "HouseAge": house_age,
            "AveRooms": ave_rooms,
            "AveBedrms": ave_bedrms,
            "Population": population,
            "AveOccup": ave_occup,
            "Latitude": latitude,
            "Longitude": longitude,
        }
    )
    y = pd.Series(med_house_val, name="MedHouseVal")
    return X, y


essential_columns = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


def load_data(path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load dataset from CSV path if available, otherwise try sklearn's built-in
    California Housing dataset, and finally fall back to synthetic data.

    The returned data matches columns documented in `make_synthetic_data`.

    Parameters
    ----------
    path : Optional[str]
        CSV path with columns including the target `MedHouseVal`.

    Returns
    -------
    (X, y) : Tuple[pd.DataFrame, pd.Series]

    Behavior
    --------
    - If `path` exists, we read it with pandas.
    - Else we attempt to import and use sklearn.datasets.fetch_california_housing(as_frame=True).
    - Else we fall back to `make_synthetic_data()`.

    TODO
    ----
    - If you have a custom dataset with different columns, update
      `src/data/preprocessing.py` and `configs/config.yaml` accordingly.
    """
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        if "MedHouseVal" not in df.columns:
            raise ValueError(
                "Expected target column 'MedHouseVal' in provided CSV. "
                "TODO: If your target differs, set configs/config.yaml:data.target_column and update code."
            )
        missing = [c for c in essential_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing expected columns in provided CSV: {missing}. TODO: Update preprocessing and config to match your schema."
            )
        X = df.drop(columns=["MedHouseVal"])  # TODO: support custom target name
        y = df["MedHouseVal"]
        return X, y

    # Try sklearn dataset
    try:
        from sklearn.datasets import fetch_california_housing

        ds = fetch_california_housing(as_frame=True)
        df = ds.frame
        X = df.drop(columns=["MedHouseVal"])  # target column is MedHouseVal in this dataset
        y = df["MedHouseVal"]
        return X, y
    except Exception:
        # Fall back to synthetic
        return make_synthetic_data(n=500, random_seed=RANDOM_SEED_DEFAULT)

