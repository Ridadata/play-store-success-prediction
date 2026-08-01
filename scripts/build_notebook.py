"""One-off generator for notebooks/01_eda.ipynb.

Kept in the repo (rather than thrown away) so the notebook's structure is
reproducible from a plain-text diff instead of hand-edited JSON.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells.append(md(
"""# Google Play Store: Exploratory Data Analysis

This notebook explores the Play Store export and demonstrates the labeling
and preprocessing logic used by the pipeline. **Training, tuning and final
evaluation live in `src/playstore` and run via:**

```bash
python -m playstore.train --sample-frac 0.3 --tuning-iterations 20
```

so that the exact same, tested code path produces the numbers in the README
-- nothing here is re-implemented ad hoc. See `../README.md` for the full
methodology write-up and the latest results.
"""
))

cells.append(code(
"""import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from playstore import visualization as viz
from playstore.config import RAW_USECOLS
from playstore.data import load_raw_data
from playstore.target import fit_success_thresholds, make_success_label

viz.set_plot_style()
pd.set_option("display.max_columns", None)
%matplotlib inline"""
))

cells.append(md("## 1. Load data\n\nA few extra descriptive columns (`Rating`, `Installs`) are pulled in beyond what the model uses, purely to make the EDA plots below more informative."))

cells.append(code(
"""eda_usecols = sorted(set(RAW_USECOLS) | {"Rating", "Installs"})
df = load_raw_data(usecols=eda_usecols)
print(f"Rows: {len(df):,}  Columns: {len(df.columns)}")
df.head()"""
))

cells.append(code(
"""df.info()
missing = df.isna().mean().sort_values(ascending=False)
missing[missing > 0].to_frame("missing_fraction")"""
))

cells.append(md(
"""## 2. Target variable

An app is `Success` if it has reached a high quantile of installs, a high
quantile of rating count, or has been given Editors' Choice.

**Leakage note:** the thresholds below are fit on the *entire* dataframe,
which is fine for exploring what the label looks like, but is **not** how
`playstore.train` builds it for modeling -- there, the same
`fit_success_thresholds` function is called on the training split only, and
the resulting fixed numbers are applied to the test split. Fitting them
here on all the data would leak test-set information into the label
definition if reused for training."""
))

cells.append(code(
"""thresholds = fit_success_thresholds(df)
df["success_label"] = make_success_label(df, thresholds)
print(thresholds)

counts = df["success_label"].value_counts()
ax = counts.reindex(["Failure", "Success"]).plot(
    kind="bar", color=[viz.PALETTE["Failure"], viz.PALETTE["Success"]], rot=0
)
ax.set_title("Label distribution (whole dataset, illustrative only)")
ax.set_ylabel("Number of apps")
for i, v in enumerate(counts.reindex(["Failure", "Success"])):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom")"""
))

cells.append(md("## 3. Category and monetization vs. success rate"))

cells.append(code(
"""top_categories = df["Category"].value_counts().head(15).index
success_by_category = (
    pd.crosstab(df["Category"], df["success_label"], normalize="index")["Success"]
    .loc[top_categories]
    .sort_values()
)

fig, ax = plt.subplots(figsize=(8, 6))
success_by_category.plot(kind="barh", ax=ax, color="#4e79a7")
ax.set_xlabel("Success rate")
ax.set_title("Success rate by category (top 15 by app count)")
plt.tight_layout()"""
))

cells.append(code(
"""fig, axes = plt.subplots(1, 2, figsize=(12, 5))

pd.crosstab(df["Free"], df["success_label"], normalize="index").plot(
    kind="bar", ax=axes[0], color=[viz.PALETTE["Failure"], viz.PALETTE["Success"]], rot=0
)
axes[0].set_title("Success rate: Free vs. Paid")

pd.crosstab(df["Ad Supported"], df["success_label"], normalize="index").plot(
    kind="bar", ax=axes[1], color=[viz.PALETTE["Failure"], viz.PALETTE["Success"]], rot=0
)
axes[1].set_title("Success rate: Ad supported")
plt.tight_layout()"""
))

cells.append(md(
"""## 4. Modeling demo

This is a **small, fast** run (a subsample and a short hyperparameter
search) purely so the notebook executes in well under a minute. The
headline numbers reported in the README come from a full run with more
data and a larger search budget -- rerun the CLI command at the top of this
notebook to reproduce those."""
))

cells.append(code(
"""from pathlib import Path

from playstore.config import DEFAULT_ARTIFACTS_DIR, DEFAULT_FIGURES_DIR, TrainConfig
from playstore.train import main as run_training

# Writes to artifacts/demo/ and reports/figures/demo/ -- NOT the paths the
# README's figures/metrics point at -- so re-running this notebook can never
# silently overwrite the full run's reported results with this tiny demo's.
demo_config = TrainConfig(
    sample_frac=0.01,
    tuning_iterations=3,
    cv_folds=2,
    artifacts_dir=Path(DEFAULT_ARTIFACTS_DIR) / "demo",
    figures_dir=Path(DEFAULT_FIGURES_DIR) / "demo",
)
demo_results = run_training(demo_config)

print("Best model:", demo_results["best_model"])
print("Chosen threshold:", demo_results["chosen_threshold"])
print("Test metrics @ chosen threshold:", demo_results["test_metrics_chosen_threshold"])"""
))

cells.append(md(
"""## 5. Where to go next

* `src/playstore/train.py` -- the full, tested pipeline (cross-validation,
  tuning, leakage-safe threshold selection, artifact saving).
* `tests/` -- unit tests for the label logic and feature transformers, plus
  an end-to-end smoke test.
* `README.md` -- methodology, latest results, and known limitations.
"""
))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}

with open("notebooks/01_eda.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("wrote notebooks/01_eda.ipynb")
