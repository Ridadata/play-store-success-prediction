"""Definition of the binary `success` target.

An app is labelled ``Success`` if, at the time the dataset was scraped, it
had reached *any* of:

* meaningful adoption -- ``Minimum Installs`` at or above a high quantile,
* meaningful engagement -- ``Rating Count`` at or above a high quantile,
* official recognition -- ``Editors Choice``.

Everything else is ``Failure``.

Leakage note
------------
The two quantile thresholds are statistics computed from data, so they must
be derived from the **training split only** and then applied unchanged to
the test split -- exactly like an imputer or scaler. Computing them on the
full dataset before splitting (as an earlier version of this project did)
lets information about the test set influence how every label is defined,
which is a subtle form of target leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from playstore.config import (
    LABEL_EDITORS_CHOICE_COL,
    LABEL_INSTALLS_COL,
    LABEL_RATING_COUNT_COL,
    SUCCESS_INSTALLS_QUANTILE,
    SUCCESS_RATING_COUNT_QUANTILE,
)

SUCCESS = "Success"
FAILURE = "Failure"


@dataclass(frozen=True)
class SuccessThresholds:
    installs: float
    rating_count: float


def fit_success_thresholds(
    df: pd.DataFrame,
    installs_quantile: float = SUCCESS_INSTALLS_QUANTILE,
    rating_count_quantile: float = SUCCESS_RATING_COUNT_QUANTILE,
) -> SuccessThresholds:
    """Compute the label thresholds from a (training) dataframe."""
    return SuccessThresholds(
        installs=df[LABEL_INSTALLS_COL].quantile(installs_quantile),
        rating_count=df[LABEL_RATING_COUNT_COL].quantile(rating_count_quantile),
    )


def make_success_label(df: pd.DataFrame, thresholds: SuccessThresholds) -> pd.Series:
    """Apply fixed thresholds to label every row ``Success``/``Failure``."""
    is_success = (
        (df[LABEL_INSTALLS_COL] >= thresholds.installs)
        | (df[LABEL_RATING_COUNT_COL] >= thresholds.rating_count)
        | (df[LABEL_EDITORS_CHOICE_COL].astype(bool))
    )
    return pd.Series(
        pd.Categorical(is_success.map({True: SUCCESS, False: FAILURE})),
        index=df.index,
        name="success_label",
    )
