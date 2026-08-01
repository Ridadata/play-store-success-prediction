"""Candidate models and the full (preprocessing + resampling + classifier) pipeline.

SMOTE is wired in through ``imblearn.pipeline.Pipeline`` rather than
``sklearn.pipeline.Pipeline``. That is not a cosmetic choice: imblearn's
pipeline only resamples the data flowing through ``fit``/``fit_resample``
and always calls plain ``transform`` (no resampling) on ``predict``/
``predict_proba``. A plain sklearn ``Pipeline`` has no concept of
resampling at all, so bolting SMOTE onto it the naive way -- oversampling
once up front and then cross-validating -- leaks synthetic neighbours of
validation-fold points into the training folds.
"""

from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from playstore.config import RANDOM_STATE


def get_candidate_models(random_state: int = RANDOM_STATE) -> dict:
    """Models compared via cross-validation before hyperparameter tuning.

    None of these set ``class_weight="balanced"`` / ``is_unbalance=True``.
    ``build_model_pipeline`` already rebalances the training folds with
    SMOTE by default, and doing both at once double-corrects for the same
    imbalance: the classifier would see an (already oversampled) near-1:1
    training distribution and *also* upweight the minority class again,
    which skews predicted probabilities away from being well calibrated.
    Pick one mechanism -- this project uses SMOTE -- not both. Running with
    ``--no-smote`` trains on the raw class distribution with no rebalancing
    at all; that is an intentional, documented trade-off of that flag, not
    an oversight.

    Kernel SVM and Gaussian Naive Bayes were dropped from the original
    exploration: RBF-kernel ``SVC`` scales roughly quadratically-to-cubically
    with the number of rows, which is not tractable on a dataset with
    hundreds of thousands to millions of apps, and Naive Bayes underfits
    this feature set badly enough (independence assumption violated by the
    one-hot category columns) that it never competed for "best model" --
    it added runtime without ever changing the outcome.

    Each classifier uses all cores itself (``n_jobs=-1``), while
    ``TrainConfig.n_jobs`` (passed to ``cross_validate`` /
    ``RandomizedSearchCV`` / ``cross_val_predict``) defaults to 1, i.e. folds
    run one at a time. That split -- parallel *inside* a fit, sequential
    *across* folds -- avoids two different failure modes seen while building
    this pipeline: nesting ``n_jobs=-1`` on both levels oversubscribes the
    CPU (e.g. 4 parallel folds x an all-core model each on an 8-core
    machine) and, worse, running several folds' SMOTE resampling
    (memory-hungry nearest-neighbour search) at once multiplied peak memory
    use until the process was OOM-killed.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=random_state, n_jobs=-1
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5, subsample=0.8,
            random_state=random_state,
        ),
        "lightgbm": LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        ),
    }


def get_param_distributions(model_name: str) -> dict:
    """RandomizedSearchCV search spaces, keyed by the names in ``get_candidate_models``."""
    distributions = {
        "logistic_regression": {
            "classifier__C": uniform(0.01, 10),
            "classifier__penalty": ["l2"],
        },
        "random_forest": {
            "classifier__n_estimators": randint(100, 400),
            "classifier__max_depth": randint(8, 30),
            "classifier__min_samples_split": randint(2, 20),
            "classifier__min_samples_leaf": randint(1, 10),
        },
        "gradient_boosting": {
            "classifier__n_estimators": randint(100, 400),
            "classifier__learning_rate": uniform(0.01, 0.2),
            "classifier__max_depth": randint(3, 8),
            "classifier__subsample": uniform(0.6, 0.4),
        },
        "lightgbm": {
            "classifier__n_estimators": randint(100, 500),
            "classifier__learning_rate": uniform(0.01, 0.2),
            "classifier__num_leaves": randint(20, 150),
            "classifier__max_depth": randint(3, 15),
            "classifier__min_child_samples": randint(10, 100),
            "classifier__subsample": uniform(0.6, 0.4),
            "classifier__colsample_bytree": uniform(0.6, 0.4),
        },
    }
    if model_name not in distributions:
        raise KeyError(f"No parameter distribution defined for '{model_name}'")
    return distributions[model_name]


def build_model_pipeline(
    preprocessing: Pipeline,
    classifier,
    use_smote: bool = True,
    random_state: int = RANDOM_STATE,
) -> ImbPipeline:
    """Chain preprocessing -> (optional SMOTE) -> classifier as one estimator.

    ``imblearn``'s ``Pipeline`` refuses a nested ``sklearn.pipeline.Pipeline``
    as an intermediate step, so ``preprocessing``'s own steps are unpacked
    (not wrapped) into the combined pipeline instead.
    """
    steps = list(preprocessing.steps)
    if use_smote:
        # "auto" balances the minority class up to the majority count (1:1)
        # regardless of which class happens to be smaller in a given fold or
        # subsample. A fixed ratio like 0.7 looks fine against the full
        # dataset's ~35% success rate, but silently breaks (raises) on any
        # fold/subsample where the "minority" class turns out to already
        # exceed that ratio -- exactly the kind of edge case a small
        # stratified CV fold or a quick subsampled run can hit.
        smote = SMOTE(sampling_strategy="auto", k_neighbors=5, random_state=random_state)
        steps.append(("smote", smote))
    steps.append(("classifier", classifier))
    return ImbPipeline(steps)
