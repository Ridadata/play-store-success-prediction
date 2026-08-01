"""End-to-end smoke test on a tiny synthetic dataset.

Not a claim about model quality -- just proof that raw rows can flow all
the way from `load_raw_data`-shaped input through labeling, preprocessing,
SMOTE and a classifier without raising, and that the fitted pipeline can
score new data. Runs in well under a second, so it belongs in CI.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from playstore.config import FEATURE_COLUMNS
from playstore.features import build_preprocessing_pipeline
from playstore.modeling import build_model_pipeline, get_candidate_models
from playstore.target import fit_success_thresholds, make_success_label

LABEL_TO_INT = {"Failure": 0, "Success": 1}


def test_full_pipeline_smoke(synthetic_raw_df):
    train_df, test_df = train_test_split(synthetic_raw_df, test_size=0.25, random_state=0)

    thresholds = fit_success_thresholds(train_df)
    y_train = make_success_label(train_df, thresholds).map(LABEL_TO_INT).to_numpy()
    y_test = make_success_label(test_df, thresholds).map(LABEL_TO_INT).to_numpy()

    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]

    preprocessing = build_preprocessing_pipeline()
    classifier = get_candidate_models(random_state=0)["logistic_regression"]
    pipeline = build_model_pipeline(preprocessing, classifier, use_smote=True, random_state=0)

    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]

    assert proba.shape == (len(X_test),)
    assert np.all((proba >= 0) & (proba <= 1))

    preds = pipeline.predict(X_test)
    assert set(np.unique(preds)).issubset({0, 1})
    assert y_test.shape == preds.shape
