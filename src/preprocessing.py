from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# TODO: If you add categorical features to your real dataset, extend this pipeline
# with OneHotEncoder for categoricals and adjust `numeric_features` and
# `categorical_features` lists accordingly.


def infer_numeric_features(sample_df: pd.DataFrame) -> List[str]:
    """Infer numeric columns from a sample DataFrame."""
    return [c for c in sample_df.columns if pd.api.types.is_numeric_dtype(sample_df[c])]


def build_preprocessing_pipeline(
    feature_names: List[str] | None = None,
    sample_df: pd.DataFrame | None = None,
) -> ColumnTransformer:
    """
    Build a preprocessing pipeline similar to many tutorials:
      - Numeric: impute missing values with median, then StandardScaler

    Parameters
    ----------
    feature_names : Optional[List[str]]
        If provided, restrict preprocessing to these columns.
    sample_df : Optional[pd.DataFrame]
        If provided and feature_names is None, infer numeric features from it.

    Returns
    -------
    ColumnTransformer
    """
    if feature_names is None:
        if sample_df is None:
            raise ValueError("Either feature_names or sample_df must be provided")
        numeric_features = infer_numeric_features(sample_df)
    else:
        numeric_features = feature_names

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            # ("cat", categorical_transformer, categorical_features),  # TODO: extend if needed
        ]
    )
    return preprocessor
