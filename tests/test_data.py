import math

import numpy as np
import pytest

from playstore.data import parse_app_size_mb, parse_min_android_version


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("14M", 14.0),
        ("512k", 0.5),
        ("1G", 1024.0),
        ("1,024M", 1024.0),
        ("Varies with device", None),
        (None, None),
        (np.nan, None),
        ("garbage", None),
    ],
)
def test_parse_app_size_mb(raw, expected):
    result = parse_app_size_mb(raw)
    if expected is None:
        assert math.isnan(result)
    else:
        assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("5.0 and up", 5.0),
        ("4.1 - 7.1.1", 4.1),
        (None, None),
        ("Varies with device", None),
    ],
)
def test_parse_min_android_version(raw, expected):
    result = parse_min_android_version(raw)
    if expected is None:
        assert math.isnan(result)
    else:
        assert result == pytest.approx(expected)
