import pandas as pd

from playstore.target import (
    FAILURE,
    SUCCESS,
    SuccessThresholds,
    fit_success_thresholds,
    make_success_label,
)


def _row(installs, rating_count, editors_choice=False):
    return {
        "Minimum Installs": installs,
        "Rating Count": rating_count,
        "Editors Choice": editors_choice,
    }


def test_thresholds_are_computed_from_given_frame_only():
    df = pd.DataFrame([_row(i * 10, i * 5) for i in range(1, 11)])
    thresholds = fit_success_thresholds(df, installs_quantile=0.8, rating_count_quantile=0.8)
    assert thresholds.installs == df["Minimum Installs"].quantile(0.8)
    assert thresholds.rating_count == df["Rating Count"].quantile(0.8)


def test_label_success_via_installs_threshold():
    df = pd.DataFrame([_row(installs=1000, rating_count=0)])
    thresholds = SuccessThresholds(installs=500, rating_count=999_999)
    label = make_success_label(df, thresholds)
    assert label.iloc[0] == SUCCESS


def test_label_success_via_rating_count_threshold():
    df = pd.DataFrame([_row(installs=0, rating_count=1000)])
    thresholds = SuccessThresholds(installs=999_999, rating_count=500)
    label = make_success_label(df, thresholds)
    assert label.iloc[0] == SUCCESS


def test_label_success_via_editors_choice_regardless_of_other_metrics():
    df = pd.DataFrame([_row(installs=0, rating_count=0, editors_choice=True)])
    thresholds = SuccessThresholds(installs=999_999, rating_count=999_999)
    label = make_success_label(df, thresholds)
    assert label.iloc[0] == SUCCESS


def test_label_failure_when_below_every_threshold():
    df = pd.DataFrame([_row(installs=1, rating_count=1, editors_choice=False)])
    thresholds = SuccessThresholds(installs=999_999, rating_count=999_999)
    label = make_success_label(df, thresholds)
    assert label.iloc[0] == FAILURE


def test_thresholds_fit_on_train_can_be_reused_on_unseen_test_rows():
    train_df = pd.DataFrame([_row(i * 100, i * 10) for i in range(1, 21)])
    test_df = pd.DataFrame([_row(installs=5000, rating_count=0)])

    thresholds = fit_success_thresholds(train_df)
    # Applying train-derived thresholds to a row the thresholds never saw
    # must not raise and must depend only on the fixed threshold values.
    label = make_success_label(test_df, thresholds)
    assert label.iloc[0] in (SUCCESS, FAILURE)
