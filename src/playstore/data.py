"""Loading and low-level parsing of the raw Google Play Store export."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from playstore.config import DEFAULT_RAW_DATA_PATH, RAW_USECOLS

logger = logging.getLogger(__name__)

_SIZE_UNIT_TO_MB = {"k": 1 / 1024, "M": 1.0, "G": 1024.0}


def load_raw_data(
    path: str | Path = DEFAULT_RAW_DATA_PATH,
    usecols: list[str] | None = RAW_USECOLS,
    sample_frac: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Read the Play Store CSV export.

    Parameters
    ----------
    path:
        Location of ``Google-Playstore.csv``.
    usecols:
        Restrict which columns are parsed. Pass ``None`` to load everything
        (useful for exploratory notebook work).
    sample_frac:
        Uniform random subsample taken *after* loading, purely to keep
        iteration fast on the ~2.3M-row dataset. ``1.0`` keeps every row.
    random_state:
        Seed for the subsample so runs are reproducible.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. See README.md for download "
            "instructions -- the CSV is not committed to the repository."
        )

    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    logger.info("Loaded %d rows, %d columns", *df.shape)

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
        logger.info("Subsampled to %d rows (frac=%.3f)", len(df), sample_frac)

    return df


def parse_app_size_mb(size_value: object) -> float:
    """Convert a Play Store ``Size`` string (e.g. ``"14M"``, ``"512k"``) to MB.

    Returns ``NaN`` for missing values and the ``"Varies with device"``
    sentinel, since neither carries an actual numeric size.
    """
    if pd.isna(size_value) or size_value == "Varies with device":
        return np.nan

    text = str(size_value).strip().replace(",", "")
    if not text:
        return np.nan

    unit = text[-1]
    multiplier = _SIZE_UNIT_TO_MB.get(unit)
    if multiplier is None:
        return np.nan

    try:
        return float(text[:-1]) * multiplier
    except ValueError:
        return np.nan


def parse_min_android_version(version_value: object) -> float:
    """Extract the leading numeric Android version (e.g. ``"5.0 and up"`` -> 5.0)."""
    if pd.isna(version_value):
        return np.nan

    text = str(version_value).strip()
    try:
        return float(text.split()[0])
    except (ValueError, IndexError):
        return np.nan
