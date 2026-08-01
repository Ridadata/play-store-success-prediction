import pandas as pd

from playstore.config import FEATURE_COLUMNS
from playstore.features import (
    AndroidVersionExtractor,
    BooleanInteractionFeatures,
    SizeParser,
    build_preprocessing_pipeline,
    get_output_feature_names,
)


def test_size_parser_replaces_size_with_size_mb():
    df = pd.DataFrame({"Size": ["10M", "1G", None], "other": [1, 2, 3]})
    out = SizeParser().fit_transform(df)
    assert "Size" not in out.columns
    assert out["size_mb"].iloc[0] == 10.0
    assert out["size_mb"].iloc[1] == 1024.0
    assert out["size_mb"].isna().iloc[2]


def test_android_version_extractor_replaces_column():
    df = pd.DataFrame({"Minimum Android": ["4.1 and up", "5.0 and up"]})
    out = AndroidVersionExtractor().fit_transform(df)
    assert "Minimum Android" not in out.columns
    assert list(out["android_version"]) == [4.1, 5.0]


def test_boolean_interaction_features_logic():
    df = pd.DataFrame(
        {
            "Free": [True, False, True, False],
            "Ad Supported": [True, True, False, False],
            "In App Purchases": [True, True, False, False],
        }
    )
    out = BooleanInteractionFeatures().fit_transform(df)
    assert list(out["free_with_ads"]) == [1, 0, 0, 0]
    assert list(out["paid_with_iap"]) == [0, 1, 0, 0]


def test_preprocessing_pipeline_fits_on_train_and_transforms_test(synthetic_raw_df):
    X = synthetic_raw_df[FEATURE_COLUMNS]
    X_train, X_test = X.iloc[:200], X.iloc[200:]

    pipeline = build_preprocessing_pipeline()
    X_train_prepared = pipeline.fit_transform(X_train)
    X_test_prepared = pipeline.transform(X_test)

    assert X_train_prepared.shape[0] == len(X_train)
    assert X_test_prepared.shape[0] == len(X_test)
    # Same number of engineered columns on both splits.
    assert X_train_prepared.shape[1] == X_test_prepared.shape[1]

    feature_names = get_output_feature_names(pipeline)
    assert len(feature_names) == X_train_prepared.shape[1]


def test_preprocessing_pipeline_imputer_stats_come_only_from_train(synthetic_raw_df):
    X = synthetic_raw_df[FEATURE_COLUMNS]
    X_train, X_test = X.iloc[:200], X.iloc[200:]

    pipeline = build_preprocessing_pipeline()
    pipeline.fit(X_train)
    learned_stats = pipeline.named_steps["columns"].named_transformers_["num"].named_steps[
        "imputer"
    ].statistics_.copy()

    # Transforming unseen data must not refit (and therefore not change) the imputer.
    pipeline.transform(X_test)
    refit_stats = pipeline.named_steps["columns"].named_transformers_["num"].named_steps[
        "imputer"
    ].statistics_

    assert (learned_stats == refit_stats).all()
