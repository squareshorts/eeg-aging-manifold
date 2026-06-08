"""Leakage-safe 10-fold EEG brain-age benchmark.

This script expects the full baseline EEG feature table produced by the
notebook. It fits log transformation for absolute-power columns, median
imputation, winsorization, scaling, and RidgeCV only within each outer
training fold.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


METADATA_COLS = {
    "participant_id",
    "session",
    "task",
    "acquisition",
    "eeg_file",
    "filename",
    "sex",
    "handedness",
    "session1",
    "late_ses1",
    "session2",
    "late_ses2",
    "age",
}
SEED = 20260605
BOOTSTRAP_N = 5000


class EEGFoldLocalTransformer(BaseEstimator, TransformerMixin):
    """Fold-local EEG feature transform: log abs power, impute, winsorize."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def _as_frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X, columns=self.feature_names_in_)

    def _basic_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in X.columns:
            if "_abs_" in col:
                X[col] = np.log10(X[col] + 1e-30)
        return X.replace([np.inf, -np.inf], np.nan)

    def fit(self, X, y=None):
        X = self._as_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        X = self._basic_transform(X)
        self.medians_ = X.median(axis=0)
        X = X.fillna(self.medians_)
        self.lower_bounds_ = X.quantile(self.lower_quantile, axis=0)
        self.upper_bounds_ = X.quantile(self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        X = self._as_frame(X)
        X = self._basic_transform(X)
        X = X.fillna(self.medians_)
        X = X.clip(self.lower_bounds_, self.upper_bounds_, axis=1)
        return X


def feature_columns(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include="number").columns
    return [
        col
        for col in numeric
        if col not in METADATA_COLS
        and not any(term in col.lower() for term in ["session", "ses", "late"])
    ]


def bootstrap_ci(out: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    y = out["true_age"].to_numpy(float)
    pred = out["predicted_age"].to_numpy(float)
    n = len(out)
    rows = []
    metric_specs = [
        ("MAE", lambda yt, yp: mean_absolute_error(yt, yp)),
        ("Pearson r", lambda yt, yp: np.corrcoef(yt, yp)[0, 1]),
        ("R2", lambda yt, yp: r2_score(yt, yp)),
    ]

    for name, func in metric_specs:
        point = float(func(y, pred))
        vals = []
        for _ in range(BOOTSTRAP_N):
            idx = rng.integers(0, n, size=n)
            val = func(y[idx], pred[idx])
            if np.isfinite(val):
                vals.append(float(val))
        arr = np.asarray(vals)
        rows.append(
            {
                "metric": name,
                "estimate": point,
                "ci_lower": float(np.percentile(arr, 2.5)),
                "ci_upper": float(np.percentile(arr, 97.5)),
                "bootstrap_n": len(arr),
            }
        )
    return pd.DataFrame(rows)


def make_figure(out: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(out["true_age"], out["predicted_age"], alpha=0.7)
    lims = [
        min(out["true_age"].min(), out["predicted_age"].min()),
        max(out["true_age"].max(), out["predicted_age"].max()),
    ]
    ax.plot(lims, lims, linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Chronological age")
    ax.set_ylabel("Predicted EEG age")
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    if figure_path.suffix.lower() == ".pdf":
        fig.savefig(figure_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_benchmark(
    input_path: Path,
    output_path: Path,
    metrics_path: Path,
    figure_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    df = pd.read_csv(input_path)
    cols = feature_columns(df)
    if "age" not in df.columns:
        raise ValueError("Input feature table must contain an age column.")
    if not cols:
        raise ValueError("No numeric EEG feature columns were found.")

    x = df[cols].copy()
    y = df["age"].to_numpy(float)
    cv = KFold(n_splits=10, shuffle=True, random_state=SEED)
    model = Pipeline(
        [
            ("features", EEGFoldLocalTransformer()),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-2, 6, 80))),
        ]
    )

    pred = cross_val_predict(model, x, y, cv=cv)
    out = pd.DataFrame(
        {
            "participant_id": df["participant_id"] if "participant_id" in df.columns else np.arange(len(df)),
            "sex": df["sex"] if "sex" in df.columns else "",
            "true_age": y,
            "predicted_age": pred,
        }
    )
    out["brain_age_gap"] = out["predicted_age"] - out["true_age"]

    bias_model = LinearRegression()
    bias_model.fit(out[["true_age"]], out["predicted_age"])
    predicted_from_age = bias_model.predict(out[["true_age"]])
    out["predicted_age_corrected"] = out["predicted_age"] - predicted_from_age + out["true_age"]
    out["brain_age_gap_corrected"] = out["predicted_age_corrected"] - out["true_age"]

    metrics = {
        "n": float(len(out)),
        "n_features": float(len(cols)),
        "bootstrap_n": float(BOOTSTRAP_N),
        "mae": float(mean_absolute_error(out["true_age"], out["predicted_age"])),
        "pearson_r": float(np.corrcoef(out["true_age"], out["predicted_age"])[0, 1]),
        "r2": float(r2_score(out["true_age"], out["predicted_age"])),
        "raw_gap_age_r": float(np.corrcoef(out["true_age"], out["brain_age_gap"])[0, 1]),
        "corrected_gap_age_r": float(
            np.corrcoef(out["true_age"], out["brain_age_gap_corrected"])[0, 1]
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    ci_df = bootstrap_ci(out)
    metrics_df = pd.DataFrame(
        [
            {
                **metrics,
                "mae_ci_lower": float(
                    ci_df.loc[ci_df["metric"] == "MAE", "ci_lower"].iloc[0]
                ),
                "mae_ci_upper": float(
                    ci_df.loc[ci_df["metric"] == "MAE", "ci_upper"].iloc[0]
                ),
                "pearson_r_ci_lower": float(
                    ci_df.loc[ci_df["metric"] == "Pearson r", "ci_lower"].iloc[0]
                ),
                "pearson_r_ci_upper": float(
                    ci_df.loc[ci_df["metric"] == "Pearson r", "ci_upper"].iloc[0]
                ),
                "r2_ci_lower": float(
                    ci_df.loc[ci_df["metric"] == "R2", "ci_lower"].iloc[0]
                ),
                "r2_ci_upper": float(
                    ci_df.loc[ci_df["metric"] == "R2", "ci_upper"].iloc[0]
                ),
                "input_file": str(input_path),
                "preprocessing_scope": (
                    "Fold-local: log absolute-power transform, median imputation, "
                    "winsorization, standardization, and RidgeCV fitted inside each "
                    "outer training fold."
                ),
            }
        ]
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)
    ci_df.to_csv(metrics_path.with_name(metrics_path.stem + "_long.csv"), index=False)
    make_figure(out, figure_path)
    return out, metrics_df, metrics


def main() -> None:
    np.random.seed(SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-features", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/brain_age_predictions_fold_local.csv"),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("outputs/brain_age_predictions_fold_local_metrics.csv"),
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("manuscript/figures/fig1_brain_age_prediction_fold_local.pdf"),
    )
    args = parser.parse_args()
    _, metrics_df, metrics = run_benchmark(
        args.baseline_features,
        args.output,
        args.metrics_output,
        args.figure_output,
    )
    print("Leakage-safe fold-local brain-age benchmark")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Predictions: {args.output}")
    print(f"Metrics: {args.metrics_output}")
    print(f"Long-form CIs: {args.metrics_output.with_name(args.metrics_output.stem + '_long.csv')}")
    print(f"Figure: {args.figure_output}")
    print(f"Figure PNG: {args.figure_output.with_suffix('.png')}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
