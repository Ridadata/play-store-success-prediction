"""Cross-validated model comparison, threshold selection and test-set scoring.

Threshold selection deliberately runs on out-of-fold predictions from the
*training* split (via ``cross_val_predict``), never on the test split. The
test set is only ever touched once, at the very end, to report the metrics
that go in the README. Picking a decision threshold by scanning the test
set (as the original notebook did) quietly fits that threshold to the same
data used to report "final" performance, which overstates how the model
will behave on truly unseen apps.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict, cross_validate

from playstore.config import PRIMARY_METRIC, RANDOM_STATE, SCORING
from playstore.modeling import build_model_pipeline, get_param_distributions

logger = logging.getLogger(__name__)


def run_cross_validation(
    models: dict,
    preprocessing,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv,
    scoring: dict = SCORING,
    use_smote: bool = True,
    random_state: int = RANDOM_STATE,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Compare every candidate model with multi-metric cross-validation.

    Returns one row per model with the mean and std of every metric in
    ``scoring`` -- fixing the original notebook, which defined a six-metric
    ``scoring`` dict but only ever passed ``scoring='accuracy'`` to
    ``cross_val_score``, so precision/recall/F1/ROC-AUC were never actually
    compared across models despite being reported as if they had been.

    ``n_jobs`` defaults to 1, not -1: with SMOTE in the pipeline, each
    parallel fold fit does its own O(minority_size^2) nearest-neighbour
    search and holds its own copy of the training data, so naively firing
    off one process per fold can multiply peak memory use by ``n_jobs`` --
    exactly what caused an out-of-memory crash on a several-hundred-
    thousand-row training set during development of this pipeline. Raise it
    only if you know the machine running this has the RAM to spare.
    """
    rows = []
    for name, estimator in models.items():
        pipeline = build_model_pipeline(preprocessing, estimator, use_smote, random_state)
        logger.info("Cross-validating %s", name)
        cv_result = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=n_jobs,
            return_train_score=False,
        )
        row = {"model": name}
        for metric in scoring:
            scores = cv_result[f"test_{metric}"]
            row[f"test_{metric}"] = scores.mean()
            row[f"test_{metric}_std"] = scores.std()
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(f"test_{PRIMARY_METRIC}", ascending=False)
    return result.reset_index(drop=True)


def tune_model(
    model_name: str,
    estimator,
    preprocessing,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv,
    n_iter: int = 25,
    scoring: str = PRIMARY_METRIC,
    use_smote: bool = True,
    random_state: int = RANDOM_STATE,
    n_jobs: int = 1,
) -> RandomizedSearchCV:
    """Randomized hyperparameter search over the full (preprocessing + model) pipeline.

    See ``run_cross_validation`` for why ``n_jobs`` defaults to 1 rather
    than -1 when SMOTE is in the pipeline.
    """
    pipeline = build_model_pipeline(preprocessing, estimator, use_smote, random_state)
    param_distributions = get_param_distributions(model_name)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=random_state,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def out_of_fold_probabilities(
    pipeline, X_train: pd.DataFrame, y_train: np.ndarray, cv, n_jobs: int = 1
) -> np.ndarray:
    """Predicted P(success) for every training row, from folds that never saw it during fit."""
    proba = cross_val_predict(
        pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=n_jobs
    )
    return proba[:, 1]


def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray, metric: str = "f1") -> float:
    """Scan thresholds and return the one maximizing ``metric`` (default F1)."""

    def score(t: float) -> float:
        y_pred = (y_proba >= t).astype(int)
        if metric == "accuracy":
            return accuracy_score(y_true, y_pred)
        scorers = {"f1": f1_score, "precision": precision_score, "recall": recall_score}
        return scorers[metric](y_true, y_pred, zero_division=0)

    thresholds = np.arange(0.05, 0.96, 0.01)
    scores = [score(float(t)) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def evaluate_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }
