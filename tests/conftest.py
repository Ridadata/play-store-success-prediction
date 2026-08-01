from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from playstore.config import FEATURE_COLUMNS


@pytest.fixture
def synthetic_raw_df() -> pd.DataFrame:
    """A small, deterministic stand-in for the real (646MB) Play Store export.

    Only large enough to exercise every branch of the pipeline; never used
    to make claims about real-world model performance.
    """
    rng = np.random.default_rng(42)
    n = 300

    categories = rng.choice(["GAME", "EDUCATION", "TOOLS", "SOCIAL"], size=n)
    content_ratings = rng.choice(["Everyone", "Teen", "Mature 17+"], size=n)
    sizes = rng.choice(["10M", "512k", "1.2G", "Varies with device", None], size=n)
    android_versions = rng.choice(["4.1 and up", "5.0 and up", None], size=n)
    free = rng.choice([True, False], size=n, p=[0.8, 0.2])
    price = np.where(free, 0.0, rng.uniform(0.99, 9.99, size=n))

    df = pd.DataFrame(
        {
            "Category": categories,
            "Content Rating": content_ratings,
            "Free": free,
            "Ad Supported": rng.choice([True, False], size=n),
            "In App Purchases": rng.choice([True, False], size=n),
            "Price": price,
            "Size": sizes,
            "Minimum Android": android_versions,
            "Minimum Installs": rng.choice(
                [10, 1_000, 50_000, 500_000, 5_000_000], size=n
            ),
            "Rating Count": rng.integers(0, 20_000, size=n),
            "Editors Choice": rng.choice([True, False], size=n, p=[0.02, 0.98]),
        }
    )
    assert set(FEATURE_COLUMNS).issubset(df.columns)
    return df
