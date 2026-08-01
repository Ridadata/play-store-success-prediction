"""Feature engineering: custom transformers and the preprocessing pipeline.

Every transformation lives inside a single scikit-learn ``Pipeline`` /
``ColumnTransformer`` so that ``fit`` only ever sees the training split and
``transform`` is the only thing ever applied to validation/test data --
there is no path through this module that lets test-set statistics leak
into the fitted imputers, scalers or encoders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from playstore.config import BOOLEAN_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_RAW_FEATURES
from playstore.data import parse_app_size_mb, parse_min_android_version


class SizeParser(BaseEstimator, TransformerMixin):
    """Replace the free-text ``Size`` column with a numeric ``size_mb``."""

    def fit(self, X: pd.DataFrame, y=None) -> SizeParser:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["size_mb"] = X["Size"].apply(parse_app_size_mb)
        return X.drop(columns=["Size"])

    def get_feature_names_out(self, input_features=None):
        return np.array([f for f in input_features if f != "Size"] + ["size_mb"])


class AndroidVersionExtractor(BaseEstimator, TransformerMixin):
    """Replace ``Minimum Android`` free text with a numeric ``android_version``."""

    def fit(self, X: pd.DataFrame, y=None) -> AndroidVersionExtractor:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["android_version"] = X["Minimum Android"].apply(parse_min_android_version)
        return X.drop(columns=["Minimum Android"])

    def get_feature_names_out(self, input_features=None):
        return np.array(
            [f for f in input_features if f != "Minimum Android"] + ["android_version"]
        )


class BooleanInteractionFeatures(BaseEstimator, TransformerMixin):
    """Add interaction flags between the monetization booleans.

    Created before one-hot encoding on purpose: encoding first would turn
    ``Free``/``Ad Supported``/``In App Purchases`` into separate dummy
    columns, making a simple product interaction between them awkward to
    express and easy to get wrong.
    """

    def fit(self, X: pd.DataFrame, y=None) -> BooleanInteractionFeatures:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        free = X["Free"].astype(int)
        X["free_with_ads"] = free * X["Ad Supported"].astype(int)
        X["paid_with_iap"] = (1 - free) * X["In App Purchases"].astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(list(input_features) + ["free_with_ads", "paid_with_iap"])


def _bool_to_int(X: pd.DataFrame) -> pd.DataFrame:
    return X.astype(int)


def build_preprocessing_pipeline(
    categorical_features: list[str] = CATEGORICAL_FEATURES,
    boolean_features: list[str] = BOOLEAN_FEATURES,
    numerical_features: list[str] = NUMERICAL_RAW_FEATURES,
) -> Pipeline:
    """Build the full raw-features -> model-ready-matrix pipeline.

    Structure:
        raw columns
          -> SizeParser            (``Size`` -> ``size_mb``)
          -> AndroidVersionExtractor (``Minimum Android`` -> ``android_version``)
          -> BooleanInteractionFeatures (adds 2 interaction flags)
          -> ColumnTransformer:
               numeric:     median impute -> standard scale
               categorical: constant-impute "Unknown" -> one-hot
               boolean:     cast to int -> most-frequent impute
    """
    engineered_numerical = numerical_features[:]
    if "Size" in engineered_numerical:
        engineered_numerical = [f for f in engineered_numerical if f != "Size"] + ["size_mb"]
    if "Minimum Android" in engineered_numerical:
        engineered_numerical = [
            f for f in engineered_numerical if f != "Minimum Android"
        ] + ["android_version"]

    engineered_boolean = boolean_features + ["free_with_ads", "paid_with_iap"]

    numerical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    boolean_pipeline = Pipeline(
        [
            (
                "to_int",
                FunctionTransformer(_bool_to_int, validate=False, feature_names_out="one-to-one"),
            ),
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    column_transformer = ColumnTransformer(
        [
            ("num", numerical_pipeline, engineered_numerical),
            ("cat", categorical_pipeline, categorical_features),
            ("bool", boolean_pipeline, engineered_boolean),
        ],
        sparse_threshold=0.3,
    )

    return Pipeline(
        [
            ("size_parser", SizeParser()),
            ("android_extractor", AndroidVersionExtractor()),
            ("interactions", BooleanInteractionFeatures()),
            ("columns", column_transformer),
        ]
    )


def get_output_feature_names(pipeline_or_column_transformer) -> list[str]:
    """Human-readable names for every column the preprocessing step produces.

    Accepts either the ``ColumnTransformer`` itself or any fitted pipeline
    that contains it under a step named ``"columns"`` (as produced by
    ``build_preprocessing_pipeline`` or ``modeling.build_model_pipeline``).
    """
    column_transformer = pipeline_or_column_transformer
    if hasattr(pipeline_or_column_transformer, "named_steps"):
        column_transformer = pipeline_or_column_transformer.named_steps["columns"]
    return list(column_transformer.get_feature_names_out())
