"""Reusable, consistently-styled plots for model evaluation and reporting.

Centralising these (instead of re-writing the same ``plt.subplots`` / color
choices in every notebook cell, as the original exploration notebook did)
means every figure in the README, the reports and the notebook shares one
visual language, and each plot is unit-testable in isolation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)

PALETTE = {"Failure": "#e15759", "Success": "#4e79a7"}


def set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("colorblind")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.frameon": False,
        }
    )


def _save(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")


def plot_confusion_matrix(
    y_true, y_pred, labels: list[str], save_path: str | Path | None = None, title: str = ""
) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title or "Confusion Matrix")
    _save(fig, save_path)
    return fig


def plot_roc_curve(
    y_true, y_proba, save_path: str | Path | None = None, title: str = ""
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name="Model")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_title(title or "ROC Curve")
    ax.legend(loc="lower right")
    _save(fig, save_path)
    return fig


def plot_pr_curve(
    y_true, y_proba, save_path: str | Path | None = None, title: str = ""
) -> plt.Figure:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    baseline = float(np.mean(y_true))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"Model (AP = {ap:.3f})")
    ax.axhline(baseline, color="k", linestyle="--", lw=1, label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title or "Precision-Recall Curve")
    ax.legend(loc="lower left")
    _save(fig, save_path)
    return fig


def plot_threshold_curve(
    y_true, y_proba, save_path: str | Path | None = None, chosen_threshold: float | None = None
) -> plt.Figure:
    from sklearn.metrics import f1_score, precision_score, recall_score

    thresholds = np.arange(0.05, 0.96, 0.01)
    rows = []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        rows.append(
            {
                "threshold": t,
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0),
            }
        )
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    for col, marker in [("precision", "s"), ("recall", "^"), ("f1", "o")]:
        ax.plot(df["threshold"], df[col], marker=marker, markersize=3, label=col.capitalize())
    ax.axvline(0.5, color="gray", linestyle=":", label="Default (0.5)")
    if chosen_threshold is not None:
        ax.axvline(
            chosen_threshold, color="red", linestyle="--",
            label=f"Chosen ({chosen_threshold:.2f})",
        )
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Metrics vs. Decision Threshold (selected on train OOF predictions)")
    ax.legend()
    _save(fig, save_path)
    return fig


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    top_n: int = 15,
    save_path: str | Path | None = None,
) -> plt.Figure:
    order = np.argsort(importances)[::-1][:top_n]
    top_features = np.array(feature_names)[order]
    top_importances = importances[order]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(top_features)), top_importances[::-1], color="#4e79a7")
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    _save(fig, save_path)
    return fig


def plot_cv_model_comparison(
    cv_results: pd.DataFrame, metric: str = "f1", save_path: str | Path | None = None
) -> plt.Figure:
    col = f"test_{metric}"
    ordered = cv_results.sort_values(col, ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4 + 0.3 * len(ordered)))
    ax.barh(ordered["model"], ordered[col], xerr=ordered[f"{col}_std"], color="#59a14f")
    ax.set_xlabel(f"Cross-validated {metric}")
    ax.set_title("Model comparison (cross-validation on training set)")
    _save(fig, save_path)
    return fig
