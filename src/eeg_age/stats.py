"""Statistical summaries and bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

from .config import BOOTSTRAP_N, SEED


def finite_xy(x, y) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    return x_arr[valid], y_arr[valid]


def safe_pearson(x, y) -> tuple[float, float, int]:
    x_arr, y_arr = finite_xy(x, y)
    if len(x_arr) < 3 or np.unique(x_arr).size < 2 or np.unique(y_arr).size < 2:
        return np.nan, np.nan, len(x_arr)
    res = pearsonr(x_arr, y_arr)
    return float(res.statistic), float(res.pvalue), len(x_arr)


def safe_spearman(x, y) -> tuple[float, float, int]:
    x_arr, y_arr = finite_xy(x, y)
    if len(x_arr) < 3 or np.unique(x_arr).size < 2 or np.unique(y_arr).size < 2:
        return np.nan, np.nan, len(x_arr)
    res = spearmanr(x_arr, y_arr)
    return float(res.statistic), float(res.pvalue), len(x_arr)


def prediction_metrics(y_true, y_pred) -> dict[str, float]:
    y_arr, pred_arr = finite_xy(y_true, y_pred)
    pearson, pearson_p, n = safe_pearson(y_arr, pred_arr)
    return {
        "n": float(n),
        "mae": float(mean_absolute_error(y_arr, pred_arr)) if n else np.nan,
        "pearson_r": pearson,
        "pearson_p": pearson_p,
        "r2": float(r2_score(y_arr, pred_arr)) if n else np.nan,
    }


def bootstrap_metric(
    df: pd.DataFrame,
    func: Callable[[pd.DataFrame], float],
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = SEED,
) -> tuple[float, float, float, int]:
    clean = df.reset_index(drop=True)
    point = float(func(clean))
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    n = len(clean)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        val = func(clean.iloc[idx])
        if np.isfinite(val):
            vals.append(float(val))
    if not vals:
        return point, np.nan, np.nan, 0
    arr = np.asarray(vals)
    return point, float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), len(arr)


def metric_table_for_predictions(
    label: str,
    y_true,
    y_pred,
    *,
    seed: int = SEED,
    n_boot: int = BOOTSTRAP_N,
) -> pd.DataFrame:
    df = pd.DataFrame({"age": y_true, "predicted_age": y_pred}).replace([np.inf, -np.inf], np.nan).dropna()
    specs = {
        "MAE": lambda d: mean_absolute_error(d["age"], d["predicted_age"]),
        "Pearson r": lambda d: safe_pearson(d["age"], d["predicted_age"])[0],
        "R2": lambda d: r2_score(d["age"], d["predicted_age"]),
    }
    rows = []
    for offset, (metric, func) in enumerate(specs.items()):
        point, lo, hi, used = bootstrap_metric(df, func, n_boot=n_boot, seed=seed + offset)
        rows.append(
            {
                "analysis": label,
                "metric": metric,
                "estimate": point,
                "ci_lower": lo,
                "ci_upper": hi,
                "bootstrap_n": used,
                "n": len(df),
            }
        )
    return pd.DataFrame(rows)


def metric_table_for_association(
    label: str,
    x,
    y,
    *,
    seed: int = SEED,
    n_boot: int = BOOTSTRAP_N,
) -> pd.DataFrame:
    df = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    specs = {
        "Pearson r": lambda d: safe_pearson(d["x"], d["y"])[0],
        "Spearman rho": lambda d: safe_spearman(d["x"], d["y"])[0],
    }
    rows = []
    for offset, (metric, func) in enumerate(specs.items()):
        point, lo, hi, used = bootstrap_metric(df, func, n_boot=n_boot, seed=seed + 100 + offset)
        if metric == "Pearson r":
            _, pvalue, _ = safe_pearson(df["x"], df["y"])
        else:
            _, pvalue, _ = safe_spearman(df["x"], df["y"])
        rows.append(
            {
                "analysis": label,
                "metric": metric,
                "estimate": point,
                "pvalue": pvalue,
                "ci_lower": lo,
                "ci_upper": hi,
                "bootstrap_n": used,
                "n": len(df),
            }
        )
    return pd.DataFrame(rows)
