"""End-to-end training entry point.

    python -m playstore.train --sample-frac 0.25 --tuning-iterations 15

Pipeline:
    1. Load raw data (optionally subsampled for speed).
    2. Split into train/test BEFORE any label or feature statistic is computed.
    3. Fit the success-label thresholds on the train split only, apply to both.
    4. Cross-validate several candidate models (multi-metric) on the train split.
    5. Hyperparameter-tune the best one with RandomizedSearchCV.
    6. Pick a decision threshold from out-of-fold train predictions.
    7. Score once on the held-out test split, at both 0.5 and the chosen threshold.
    8. Persist the fitted pipeline, metrics, and evaluation plots.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from playstore import visualization as viz
from playstore.config import FEATURE_COLUMNS, TrainConfig
from playstore.data import load_raw_data
from playstore.evaluate import (
    evaluate_at_threshold,
    find_best_threshold,
    out_of_fold_probabilities,
    run_cross_validation,
    tune_model,
)
from playstore.features import build_preprocessing_pipeline, get_output_feature_names
from playstore.modeling import get_candidate_models
from playstore.target import fit_success_thresholds, make_success_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LABEL_TO_INT = {"Failure": 0, "Success": 1}


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TrainConfig()
    parser.add_argument("--data-path", type=Path, default=defaults.raw_data_path)
    parser.add_argument("--artifacts-dir", type=Path, default=defaults.artifacts_dir)
    parser.add_argument("--figures-dir", type=Path, default=defaults.figures_dir)
    parser.add_argument("--sample-frac", type=float, default=defaults.sample_frac)
    parser.add_argument("--test-size", type=float, default=defaults.test_size)
    parser.add_argument("--cv-folds", type=int, default=defaults.cv_folds)
    parser.add_argument("--tuning-iterations", type=int, default=defaults.tuning_iterations)
    parser.add_argument("--no-smote", dest="use_smote", action="store_false")
    parser.add_argument("--random-state", type=int, default=defaults.random_state)
    parser.add_argument(
        "--n-jobs", type=int, default=defaults.n_jobs,
        help="Parallel workers for CV/tuning. Keep at 1 with SMOTE unless you have RAM to spare.",
    )
    args = parser.parse_args()
    return TrainConfig(
        raw_data_path=args.data_path,
        artifacts_dir=args.artifacts_dir,
        figures_dir=args.figures_dir,
        sample_frac=args.sample_frac,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        tuning_iterations=args.tuning_iterations,
        use_smote=args.use_smote,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )


def main(config: TrainConfig | None = None) -> dict:
    config = config or parse_args()
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    viz.set_plot_style()

    run_meta: dict = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": {k: str(v) for k, v in vars(config).items()},
    }

    # 1-2. Load and split RAW rows before any label/feature statistic exists.
    raw_df = load_raw_data(
        config.raw_data_path, sample_frac=config.sample_frac, random_state=config.random_state
    )
    train_df, test_df = train_test_split(
        raw_df, test_size=config.test_size, random_state=config.random_state
    )
    logger.info("Train rows: %d | Test rows: %d", len(train_df), len(test_df))

    # 3. Label thresholds fit on TRAIN ONLY, then applied to both splits.
    thresholds = fit_success_thresholds(train_df)
    logger.info("Success thresholds (fit on train): %s", thresholds)
    y_train_labels = make_success_label(train_df, thresholds)
    y_test_labels = make_success_label(test_df, thresholds)
    y_train = y_train_labels.map(LABEL_TO_INT).to_numpy()
    y_test = y_test_labels.map(LABEL_TO_INT).to_numpy()
    logger.info(
        "Train label balance: %s", dict(y_train_labels.value_counts(normalize=True).round(3))
    )

    X_train = train_df[FEATURE_COLUMNS].reset_index(drop=True)
    X_test = test_df[FEATURE_COLUMNS].reset_index(drop=True)

    # 4. Cross-validated model comparison.
    preprocessing = build_preprocessing_pipeline()
    models = get_candidate_models(config.random_state)
    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)

    cv_results = run_cross_validation(
        models, preprocessing, X_train, y_train, cv,
        config.scoring, config.use_smote, config.random_state, config.n_jobs,
    )
    cv_results.to_csv(config.artifacts_dir / "cv_results.csv", index=False)
    viz.plot_cv_model_comparison(
        cv_results,
        metric=config.primary_metric,
        save_path=config.figures_dir / "cv_model_comparison.png",
    )
    logger.info("Cross-validation results:\n%s", cv_results.to_string(index=False))

    best_model_name = cv_results.iloc[0]["model"]
    logger.info("Best model from cross-validation: %s", best_model_name)

    # 5. Hyperparameter tuning of the best model.
    search = tune_model(
        best_model_name,
        models[best_model_name],
        preprocessing,
        X_train,
        y_train,
        cv,
        n_iter=config.tuning_iterations,
        scoring=config.primary_metric,
        use_smote=config.use_smote,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )
    best_pipeline = search.best_estimator_
    logger.info("Best params: %s", search.best_params_)
    logger.info("Best CV %s after tuning: %.4f", config.primary_metric, search.best_score_)

    # 6. Threshold selection on out-of-fold TRAIN predictions (never the test set).
    oof_proba = out_of_fold_probabilities(best_pipeline, X_train, y_train, cv, config.n_jobs)
    chosen_threshold = find_best_threshold(y_train, oof_proba, metric=config.primary_metric)
    logger.info("Chosen decision threshold (from train OOF): %.2f", chosen_threshold)

    # 7. Final, single evaluation on the held-out test split.
    test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    metrics_default = evaluate_at_threshold(y_test, test_proba, 0.5)
    metrics_chosen = evaluate_at_threshold(y_test, test_proba, chosen_threshold)
    logger.info("Test metrics @0.50: %s", metrics_default)
    logger.info("Test metrics @%.2f: %s", chosen_threshold, metrics_chosen)

    labels = ["Failure", "Success"]
    y_test_pred_chosen = (test_proba >= chosen_threshold).astype(int)
    viz.plot_confusion_matrix(
        y_test, y_test_pred_chosen, labels,
        save_path=config.figures_dir / "confusion_matrix.png",
        title=f"Confusion Matrix - {best_model_name} (threshold={chosen_threshold:.2f})",
    )
    viz.plot_roc_curve(
        y_test, test_proba,
        save_path=config.figures_dir / "roc_curve.png",
        title=f"ROC Curve - {best_model_name}",
    )
    viz.plot_pr_curve(
        y_test, test_proba,
        save_path=config.figures_dir / "pr_curve.png",
        title=f"Precision-Recall Curve - {best_model_name}",
    )
    viz.plot_threshold_curve(
        y_train, oof_proba,
        save_path=config.figures_dir / "threshold_curve.png",
        chosen_threshold=chosen_threshold,
    )

    classifier = best_pipeline.named_steps["classifier"]
    feature_names = get_output_feature_names(best_pipeline)
    if hasattr(classifier, "feature_importances_"):
        viz.plot_feature_importance(
            feature_names, np.asarray(classifier.feature_importances_),
            save_path=config.figures_dir / "feature_importance.png",
        )

    # 8. Persist everything needed to reproduce or serve this model.
    joblib.dump(best_pipeline, config.artifacts_dir / "model_pipeline.joblib")

    run_meta.update(
        {
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "n_train_rows": len(X_train),
            "n_test_rows": len(X_test),
            "success_thresholds": {
                "installs": thresholds.installs,
                "rating_count": thresholds.rating_count,
            },
            "best_model": best_model_name,
            "best_params": search.best_params_,
            "chosen_threshold": chosen_threshold,
            "cv_results": cv_results.to_dict(orient="records"),
            "test_metrics_default_threshold": metrics_default,
            "test_metrics_chosen_threshold": metrics_chosen,
        }
    )
    with open(config.artifacts_dir / "metrics.json", "w") as f:
        json.dump(run_meta, f, indent=2, default=str)

    logger.info("Artifacts written to %s", config.artifacts_dir)
    return run_meta


if __name__ == "__main__":
    main()
