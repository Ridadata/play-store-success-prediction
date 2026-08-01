"""Central configuration for the Playstore success-prediction pipeline.

Keeping every path, column name, and hyperparameter default in one place
means a single source of truth instead of magic strings scattered across
notebooks and scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Google-Playstore.csv"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RANDOM_STATE = 42

# Columns that define the target label. Kept separate from FEATURE_COLUMNS
# so it is obvious none of them ever reach the model as inputs.
LABEL_INSTALLS_COL = "Minimum Installs"
LABEL_RATING_COUNT_COL = "Rating Count"
LABEL_EDITORS_CHOICE_COL = "Editors Choice"

# Quantile (computed on the TRAIN split only, see target.py) above which an
# app is considered to have reached meaningful adoption / engagement.
SUCCESS_INSTALLS_QUANTILE = 0.80
SUCCESS_RATING_COUNT_QUANTILE = 0.80

# Features available at (or very near) release time. Deliberately excludes
# Rating, Rating Count, Installs, Maximum Installs and Editors Choice: those
# either define the label directly or only exist after an app has already
# accumulated post-release engagement, so using them as inputs would leak
# the answer into the features.
CATEGORICAL_FEATURES = ["Category", "Content Rating"]
BOOLEAN_FEATURES = ["Free", "Ad Supported", "In App Purchases"]
NUMERICAL_RAW_FEATURES = ["Price", "Size", "Minimum Android"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + BOOLEAN_FEATURES + NUMERICAL_RAW_FEATURES

# Columns actually read from the (very large) source CSV: features + the
# columns needed to build the label. Restricting usecols avoids loading
# ~15 unused text columns (App Name, Developer Email, ...) into memory.
RAW_USECOLS = sorted(
    set(FEATURE_COLUMNS)
    | {LABEL_INSTALLS_COL, LABEL_RATING_COUNT_COL, LABEL_EDITORS_CHOICE_COL}
)

TEST_SIZE = 0.2
CV_FOLDS = 5

PRIMARY_METRIC = "f1"
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}


@dataclass
class TrainConfig:
    """Runtime knobs for a single training run of ``playstore.train``."""

    raw_data_path: Path = DEFAULT_RAW_DATA_PATH
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR
    figures_dir: Path = DEFAULT_FIGURES_DIR
    random_state: int = RANDOM_STATE
    test_size: float = TEST_SIZE
    cv_folds: int = CV_FOLDS
    # Uniform random subsample of the *raw* rows, applied before the
    # train/test split, purely for tractability on an 8-core laptop.
    # 1.0 uses the full ~2.3M-row dataset.
    sample_frac: float = 1.0
    use_smote: bool = True
    tuning_iterations: int = 25
    primary_metric: str = PRIMARY_METRIC
    scoring: dict = field(default_factory=lambda: dict(SCORING))
    # Parallel workers for cross_validate/RandomizedSearchCV/cross_val_predict.
    # Kept at 1 by default: see evaluate.run_cross_validation for why running
    # SMOTE inside several parallel worker processes at once is a memory
    # hazard, not just a speed knob.
    n_jobs: int = 1
