"""Generate robustness outputs for the EEG aging manifold project.

The repository currently contains derived summary tables, but not the full
baseline/follow-up EEG feature tables. This runner computes every requested
robustness analysis that is possible from the available derived tables and
writes explicit "not run" rows for feature-dependent analyses that cannot be
recomputed as specified.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import wrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import rankdata, spearmanr, t, wilcoxon
from sklearn.metrics import mean_absolute_error, r2_score


SEED = 20260605
BOOTSTRAP_N = 5000
PERMUTATION_N = 5000


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "outputs"
OUT_DIR = ROOT / "robustness_outputs"


def ensure_outputs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return pd.read_csv(path)


def safe_spearman(x, y) -> tuple[float, float]:
    df = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 3 or df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return np.nan, np.nan
    res = spearmanr(df["x"], df["y"])
    return float(res.statistic), float(res.pvalue)


def safe_pearson(x, y) -> float:
    df = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 3 or df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return np.nan
    return float(np.corrcoef(df["x"], df["y"])[0, 1])


def safe_wilcoxon_p(values) -> float:
    x = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    x = x[x != 0]
    if len(x) == 0:
        return np.nan
    return float(wilcoxon(x).pvalue)


def wilcoxon_rank_biserial(values) -> float:
    """Matched-pairs rank-biserial effect size for paired differences."""
    x = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    x = x[x != 0]
    if len(x) == 0:
        return np.nan
    ranks = rankdata(np.abs(x))
    w_pos = float(ranks[x > 0].sum())
    w_neg = float(ranks[x < 0].sum())
    total = float(ranks.sum())
    return (w_pos - w_neg) / total if total else np.nan


def bootstrap_ci(df: pd.DataFrame, func, n_boot: int, seed_offset: int = 0) -> tuple[float, float, float, int]:
    rng = np.random.default_rng(SEED + seed_offset)
    point = float(func(df))
    n = len(df)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        val = func(df.iloc[idx])
        if np.isfinite(val):
            vals.append(float(val))
    if not vals:
        return point, np.nan, np.nan, 0
    arr = np.asarray(vals)
    return point, float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), len(arr)


def ols_with_pvalues(y, predictors: pd.DataFrame) -> pd.DataFrame:
    y_arr = np.asarray(y, dtype=float)
    x_df = predictors.copy()
    x_df.insert(0, "intercept", 1.0)
    valid = np.isfinite(y_arr)
    for col in x_df.columns:
        valid &= np.isfinite(x_df[col].to_numpy(dtype=float))
    y_arr = y_arr[valid]
    x_arr = x_df.loc[valid].to_numpy(dtype=float)
    names = x_df.columns.tolist()
    n, p = x_arr.shape
    beta = np.linalg.lstsq(x_arr, y_arr, rcond=None)[0]
    resid = y_arr - x_arr @ beta
    df_resid = n - p
    if df_resid <= 0:
        se = np.full_like(beta, np.nan, dtype=float)
        pvals = np.full_like(beta, np.nan, dtype=float)
    else:
        sigma2 = float((resid @ resid) / df_resid)
        cov = sigma2 * np.linalg.pinv(x_arr.T @ x_arr)
        se = np.sqrt(np.diag(cov))
        tvals = beta / se
        pvals = 2 * t.sf(np.abs(tvals), df_resid)
    return pd.DataFrame(
        {
            "term": names,
            "estimate": beta,
            "std_error": se,
            "pvalue": pvals,
            "n": n,
        }
    )


def get_summary_value(summary: pd.DataFrame, analysis: str, metric: str, column: str = "value") -> float:
    hit = summary[(summary["analysis"] == analysis) & (summary["metric"] == metric)]
    if hit.empty:
        return np.nan
    return float(hit.iloc[0][column])


def projection_policy_ablation(long_df: pd.DataFrame) -> pd.DataFrame:
    rho_base, p_base = safe_spearman(
        long_df["base_trajectory_position"],
        long_df["delta_trajectory_position"],
    )
    rho_age, p_age = safe_spearman(
        long_df["age"],
        long_df["delta_trajectory_position"],
    )

    rows = [
        {
            "policy": "Policy 1: baseline-fitted projection",
            "status": "computed",
            "n": len(long_df),
            "mean_delta_T": long_df["delta_trajectory_position"].mean(),
            "median_delta_T": long_df["delta_trajectory_position"].median(),
            "wilcoxon_p_delta_T": safe_wilcoxon_p(long_df["delta_trajectory_position"]),
            "mean_delta_D": long_df["delta_trajectory_deviation"].mean(),
            "median_delta_D": long_df["delta_trajectory_deviation"].median(),
            "wilcoxon_p_delta_D": safe_wilcoxon_p(long_df["delta_trajectory_deviation"]),
            "baseline_T_delta_T_spearman_rho": rho_base,
            "baseline_T_delta_T_spearman_p": p_base,
            "age_delta_T_spearman_rho": rho_age,
            "age_delta_T_spearman_p": p_age,
            "notes": "Computed from outputs/longitudinal_cadence_shift.csv using saved baseline-fitted projections.",
        },
        {
            "policy": "Policy 2: pooled-session projection",
            "status": "not run",
            "n": np.nan,
            "mean_delta_T": np.nan,
            "median_delta_T": np.nan,
            "wilcoxon_p_delta_T": np.nan,
            "mean_delta_D": np.nan,
            "median_delta_D": np.nan,
            "wilcoxon_p_delta_D": np.nan,
            "baseline_T_delta_T_spearman_rho": np.nan,
            "baseline_T_delta_T_spearman_p": np.nan,
            "age_delta_T_spearman_rho": np.nan,
            "age_delta_T_spearman_p": np.nan,
            "notes": "Requires baseline and follow-up 93-feature tables to refit imputation, winsorization, scaling, PCA, and age-axis parameters on pooled sessions; those feature tables are not present in this repo.",
        },
        {
            "policy": "Policy 3: session-wise refit projection",
            "status": "not run",
            "n": np.nan,
            "mean_delta_T": np.nan,
            "median_delta_T": np.nan,
            "wilcoxon_p_delta_T": np.nan,
            "mean_delta_D": np.nan,
            "median_delta_D": np.nan,
            "wilcoxon_p_delta_D": np.nan,
            "baseline_T_delta_T_spearman_rho": np.nan,
            "baseline_T_delta_T_spearman_p": np.nan,
            "age_delta_T_spearman_rho": np.nan,
            "age_delta_T_spearman_p": np.nan,
            "notes": "Requires baseline and follow-up 93-feature tables. Session-wise refits also create non-identical PCA/axis coordinate systems, so Delta T comparability would be limited even when data are available.",
        },
    ]
    return pd.DataFrame(rows)


def bootstrap_table(brain_df: pd.DataFrame, long_df: pd.DataFrame, cadence_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    metrics = [
        (
            "ridge-regression MAE",
            brain_df,
            lambda d: mean_absolute_error(d["true_age"], d["predicted_age"]),
            "computed",
            "Cross-validated baseline predictions from outputs/brain_age_predictions_full_baseline_corrected.csv.",
        ),
        (
            "prediction Pearson r",
            brain_df,
            lambda d: safe_pearson(d["true_age"], d["predicted_age"]),
            "computed",
            "Cross-validated baseline predictions from outputs/brain_age_predictions_full_baseline_corrected.csv.",
        ),
        (
            "R2",
            brain_df,
            lambda d: r2_score(d["true_age"], d["predicted_age"]),
            "computed",
            "Cross-validated baseline predictions from outputs/brain_age_predictions_full_baseline_corrected.csv.",
        ),
        (
            "trajectory position vs age Spearman rho (longitudinal subset fallback)",
            long_df,
            lambda d: safe_spearman(d["base_trajectory_position"], d["age"])[0],
            "computed_fallback",
            "Full 608-subject cadence_position_deviation.csv is absent; computed on the 208 available baseline projections in outputs/longitudinal_cadence_shift.csv.",
        ),
        (
            "trajectory deviation vs age Spearman rho (longitudinal subset fallback)",
            long_df,
            lambda d: safe_spearman(d["base_trajectory_deviation"], d["age"])[0],
            "computed_fallback",
            "Full 608-subject cadence_position_deviation.csv is absent; computed on the 208 available baseline projections in outputs/longitudinal_cadence_shift.csv.",
        ),
        (
            "mean Delta T",
            long_df,
            lambda d: d["delta_trajectory_position"].mean(),
            "computed",
            "Longitudinal Delta T from outputs/longitudinal_cadence_shift.csv.",
        ),
        (
            "median Delta T",
            long_df,
            lambda d: d["delta_trajectory_position"].median(),
            "computed",
            "Longitudinal Delta T from outputs/longitudinal_cadence_shift.csv.",
        ),
        (
            "Wilcoxon rank-biserial effect size for Delta T",
            long_df,
            lambda d: wilcoxon_rank_biserial(d["delta_trajectory_position"]),
            "computed",
            "Matched-pairs rank-biserial correlation computed from nonzero paired Delta T values.",
        ),
        (
            "baseline T vs future Delta T Spearman rho",
            long_df,
            lambda d: safe_spearman(d["base_trajectory_position"], d["delta_trajectory_position"])[0],
            "computed",
            "Longitudinal table with baseline T and future Delta T.",
        ),
        (
            "chronological age vs future Delta T Spearman rho",
            long_df,
            lambda d: safe_spearman(d["age"], d["delta_trajectory_position"])[0],
            "computed",
            "Longitudinal table with baseline chronological age and future Delta T.",
        ),
    ]

    for i, (name, df, func, status, notes) in enumerate(metrics):
        point, lo, hi, used_n = bootstrap_ci(df, func, BOOTSTRAP_N, seed_offset=i)
        rows.append(
            {
                "Metric": name,
                "Point estimate": point,
                "CI lower": lo,
                "CI upper": hi,
                "Bootstrap n": used_n,
                "Status": status,
                "Notes": notes,
            }
        )

    full_pos = get_summary_value(cadence_summary, "PCA trajectory position", "Spearman rho with age")
    full_pos_p = get_summary_value(cadence_summary, "PCA trajectory position", "Spearman rho with age", "pvalue")
    full_dev = get_summary_value(cadence_summary, "PCA trajectory deviation", "Spearman rho with age")
    full_dev_p = get_summary_value(cadence_summary, "PCA trajectory deviation", "Spearman rho with age", "pvalue")

    rows.insert(
        3,
        {
            "Metric": "trajectory position vs age Spearman rho (full baseline main estimate)",
            "Point estimate": full_pos,
            "CI lower": np.nan,
            "CI upper": np.nan,
            "Bootstrap n": 0,
            "Status": "not run",
            "Notes": f"Point estimate preserved from outputs/cadence_summary_results.csv (p={full_pos_p}); bootstrap requires missing full baseline trajectory table cadence_position_deviation.csv.",
        },
    )
    rows.insert(
        4,
        {
            "Metric": "trajectory deviation vs age Spearman rho (full baseline main estimate)",
            "Point estimate": full_dev,
            "CI lower": np.nan,
            "CI upper": np.nan,
            "Bootstrap n": 0,
            "Status": "not run",
            "Notes": f"Point estimate preserved from outputs/cadence_summary_results.csv (p={full_dev_p}); bootstrap requires missing full baseline trajectory table cadence_position_deviation.csv.",
        },
    )

    return pd.DataFrame(rows)


def derive_axis(base: np.ndarray, fu: np.ndarray, age: np.ndarray) -> dict[str, np.ndarray | float]:
    q20 = float(np.nanquantile(age, 0.20))
    q80 = float(np.nanquantile(age, 0.80))
    young = base[age <= q20].mean(axis=0)
    old = base[age >= q80].mean(axis=0)
    direction = old - young
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Age-axis direction is undefined.")
    direction = direction / norm
    base_t = (base - young) @ direction
    fu_t = (fu - young) @ direction
    base_proj = young + np.outer(base_t, direction)
    fu_proj = young + np.outer(fu_t, direction)
    base_d = np.linalg.norm(base - base_proj, axis=1)
    fu_d = np.linalg.norm(fu - fu_proj, axis=1)
    return {
        "young_center": young,
        "old_center": old,
        "direction": direction,
        "base_t": base_t,
        "fu_t": fu_t,
        "delta_t": fu_t - base_t,
        "base_d": base_d,
        "fu_d": fu_d,
        "delta_d": fu_d - base_d,
    }


def pc_space_row(name: str, status: str, components: list[str], base_t, fu_t, base_d, fu_d, age, notes: str, full_traj=None) -> dict:
    delta_t = np.asarray(fu_t) - np.asarray(base_t)
    delta_d = np.asarray(fu_d) - np.asarray(base_d) if base_d is not None and fu_d is not None else None
    if full_traj is None:
        traj_rho, traj_p = safe_spearman(base_t, age)
    else:
        traj_rho, traj_p = full_traj
    rho_base, p_base = safe_spearman(base_t, delta_t)
    rho_age, p_age = safe_spearman(age, delta_t)
    return {
        "version": name,
        "status": status,
        "components_or_latent_dimensions_used": ", ".join(components),
        "n": len(delta_t),
        "trajectory_age_spearman_rho": traj_rho,
        "trajectory_age_spearman_p": traj_p,
        "delta_T_mean": np.nanmean(delta_t),
        "delta_T_median": np.nanmedian(delta_t),
        "delta_T_wilcoxon_p": safe_wilcoxon_p(delta_t),
        "baseline_T_delta_T_spearman_rho": rho_base,
        "baseline_T_delta_T_spearman_p": p_base,
        "age_delta_T_spearman_rho": rho_age,
        "age_delta_T_spearman_p": p_age,
        "delta_D_mean": np.nanmean(delta_d) if delta_d is not None else np.nan,
        "delta_D_median": np.nanmedian(delta_d) if delta_d is not None else np.nan,
        "delta_D_wilcoxon_p": safe_wilcoxon_p(delta_d) if delta_d is not None else np.nan,
        "notes": notes,
    }


def pc_space_robustness(long_df: pd.DataFrame, cadence_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    full_pos = get_summary_value(cadence_summary, "PCA trajectory position", "Spearman rho with age")
    full_pos_p = get_summary_value(cadence_summary, "PCA trajectory position", "Spearman rho with age", "pvalue")
    rows.append(
        pc_space_row(
            name="Original PC1-PC4 plane",
            status="computed",
            components=["PC1", "PC4"],
            base_t=long_df["base_trajectory_position"],
            fu_t=long_df["fu_trajectory_position"],
            base_d=long_df["base_trajectory_deviation"],
            fu_d=long_df["fu_trajectory_deviation"],
            age=long_df["age"],
            notes="Trajectory-age rho/p preserved from full 608-subject main summary; longitudinal metrics computed from saved baseline-fitted projections.",
            full_traj=(full_pos, full_pos_p),
        )
    )

    pc_cols = [f"PC{i}" for i in range(1, 11)]
    pc_corrs = []
    for pc in pc_cols:
        rho, _ = safe_spearman(long_df[f"base_{pc}"], long_df["age"])
        pc_corrs.append((pc, abs(rho)))
    top_two = [pc for pc, _ in sorted(pc_corrs, key=lambda item: item[1], reverse=True)[:2]]

    variants = [
        ("Top two age-associated PCs among first ten", top_two),
        ("Age-axis in first five PCs", [f"PC{i}" for i in range(1, 6)]),
        ("Age-axis in first ten PCs", [f"PC{i}" for i in range(1, 11)]),
    ]

    age = long_df["age"].to_numpy(float)
    for name, comps in variants:
        base = long_df[[f"base_{pc}" for pc in comps]].to_numpy(float)
        fu = long_df[[f"fu_{pc}" for pc in comps]].to_numpy(float)
        axis = derive_axis(base, fu, age)
        rows.append(
            pc_space_row(
                name=name,
                status="computed_fallback",
                components=comps,
                base_t=axis["base_t"],
                fu_t=axis["fu_t"],
                base_d=axis["base_d"],
                fu_d=axis["fu_d"],
                age=age,
                notes="Recomputed from the 208 saved longitudinal baseline/follow-up PC scores only; full 608-subject PCA coordinate table and raw features are absent.",
            )
        )

    rows.append(
        {
            "version": "PLS-derived aging axis",
            "status": "not run",
            "components_or_latent_dimensions_used": "PLS latent scores unavailable",
            "n": np.nan,
            "trajectory_age_spearman_rho": np.nan,
            "trajectory_age_spearman_p": np.nan,
            "delta_T_mean": np.nan,
            "delta_T_median": np.nan,
            "delta_T_wilcoxon_p": np.nan,
            "baseline_T_delta_T_spearman_rho": np.nan,
            "baseline_T_delta_T_spearman_p": np.nan,
            "age_delta_T_spearman_rho": np.nan,
            "age_delta_T_spearman_p": np.nan,
            "delta_D_mean": np.nan,
            "delta_D_median": np.nan,
            "delta_D_wilcoxon_p": np.nan,
            "notes": "Requires baseline and follow-up PLS latent scores or raw feature tables to fit and project a PLS-derived aging axis.",
        }
    )
    return pd.DataFrame(rows)


def permutation_test(long_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    age = long_df["age"].to_numpy(float)
    base = long_df[["base_PC1", "base_PC4"]].to_numpy(float)
    fu = long_df[["fu_PC1", "fu_PC4"]].to_numpy(float)
    observed_main = float(long_df["delta_trajectory_position"].mean())
    observed_axis = derive_axis(base, fu, age)
    observed_fallback = float(np.mean(observed_axis["delta_t"]))

    rng = np.random.default_rng(SEED)
    null_vals = np.empty(PERMUTATION_N, dtype=float)
    for i in range(PERMUTATION_N):
        perm_age = rng.permutation(age)
        axis = derive_axis(base, fu, perm_age)
        null_vals[i] = float(np.mean(axis["delta_t"]))

    p_two = (1 + np.sum(np.abs(null_vals) >= abs(observed_fallback))) / (PERMUTATION_N + 1)
    p_greater = (1 + np.sum(null_vals >= observed_fallback)) / (PERMUTATION_N + 1)
    p_less = (1 + np.sum(null_vals <= observed_fallback)) / (PERMUTATION_N + 1)

    rows = [
        {
            "analysis": "Requested full-baseline PCA age-axis permutation",
            "status": "not run",
            "permutation_n": 0,
            "observed_mean_delta_T": observed_main,
            "null_mean": np.nan,
            "null_sd": np.nan,
            "empirical_p_two_sided": np.nan,
            "empirical_p_one_sided_greater": np.nan,
            "empirical_p_one_sided_less": np.nan,
            "notes": "Requires full baseline PCA feature space or cadence_position_deviation/PCA coordinates for all 608 baseline participants. Only the 208-subject longitudinal PC table is present.",
        },
        {
            "analysis": "Fallback longitudinal-subset PC1-PC4 permutation",
            "status": "computed_fallback",
            "permutation_n": PERMUTATION_N,
            "observed_mean_delta_T": observed_fallback,
            "null_mean": float(np.mean(null_vals)),
            "null_sd": float(np.std(null_vals, ddof=1)),
            "empirical_p_two_sided": float(p_two),
            "empirical_p_one_sided_greater": float(p_greater),
            "empirical_p_one_sided_less": float(p_less),
            "notes": "Age-axis and null axes recomputed from available 208 baseline PC1/PC4 coordinates; this is not the requested full-baseline permutation.",
        },
    ]
    return pd.DataFrame(rows), null_vals


def feature_sensitivity(feature_names: list[str]) -> pd.DataFrame:
    features = pd.Series(feature_names, dtype=str)

    def count_retained(mask) -> int:
        if features.empty:
            return 0
        return int(mask.sum())

    lower = features.str.lower()
    cases = [
        (
            "Exclude gamma-band features",
            count_retained(~lower.str.contains("gamma")),
            "Feature names are available, so retained-feature count is reported; metrics require missing baseline/follow-up feature tables.",
        ),
        (
            "Exclude beta and gamma features",
            count_retained(~lower.str.contains("beta|gamma")),
            "Feature names are available, so retained-feature count is reported; metrics require missing baseline/follow-up feature tables.",
        ),
        (
            "Exclude frontal beta and gamma features",
            count_retained(~(lower.str.contains("frontal") & lower.str.contains("beta|gamma"))),
            "Feature names are available, so retained-feature count is reported; metrics require missing baseline/follow-up feature tables.",
        ),
        (
            "Relative-power features only",
            count_retained(lower.str.contains("_rel_")),
            "Feature names are available, so retained-feature count is reported; metrics require missing baseline/follow-up feature tables.",
        ),
        (
            "Exclude top 5 percent by high-frequency/spectral-noise QC metric",
            len(feature_names),
            "QC metric requires subject-level high-frequency feature values; feature tables are absent.",
        ),
        (
            "Optional exclude top 10 percent by same QC metric",
            len(feature_names),
            "QC metric requires subject-level high-frequency feature values; feature tables are absent.",
        ),
    ]
    rows = []
    for name, n_features, note in cases:
        rows.append(
            {
                "sensitivity_model": name,
                "status": "not run",
                "n_features_retained": n_features if n_features else np.nan,
                "n_participants_retained": np.nan,
                "trajectory_age_spearman_rho": np.nan,
                "trajectory_age_spearman_p": np.nan,
                "delta_T_mean": np.nan,
                "delta_T_median": np.nan,
                "delta_T_wilcoxon_p": np.nan,
                "baseline_T_delta_T_spearman_rho": np.nan,
                "baseline_T_delta_T_spearman_p": np.nan,
                "age_delta_T_spearman_rho": np.nan,
                "age_delta_T_spearman_p": np.nan,
                "notes": note,
            }
        )
    return pd.DataFrame(rows)


def demographic_robustness(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    df = long_df.copy()
    df["male"] = (df["sex"].astype(str).str.upper() == "M").astype(float)

    ols_t = ols_with_pvalues(
        df["base_trajectory_position"],
        df[["age", "male"]],
    )
    age_row = ols_t[ols_t["term"] == "age"].iloc[0]
    sex_row = ols_t[ols_t["term"] == "male"].iloc[0]
    rows.append(
        {
            "analysis": "Baseline T vs age adjusted for sex",
            "subgroup": "longitudinal subset",
            "status": "computed_fallback",
            "n": int(age_row["n"]),
            "estimate_1_label": "OLS beta for age",
            "estimate_1": age_row["estimate"],
            "pvalue_1": age_row["pvalue"],
            "estimate_2_label": "OLS beta for male sex",
            "estimate_2": sex_row["estimate"],
            "pvalue_2": sex_row["pvalue"],
            "mean_delta_T": np.nan,
            "median_delta_T": np.nan,
            "wilcoxon_p_delta_T": np.nan,
            "direction": "",
            "notes": "Full 608-subject baseline trajectory table is absent; computed on available 208 baseline projections.",
        }
    )

    ols_delta = ols_with_pvalues(
        df["delta_trajectory_position"],
        df[["male"]],
    )
    intercept = ols_delta[ols_delta["term"] == "intercept"].iloc[0]
    male = ols_delta[ols_delta["term"] == "male"].iloc[0]
    rows.append(
        {
            "analysis": "Delta T adjusted for sex",
            "subgroup": "all longitudinal participants",
            "status": "computed",
            "n": int(intercept["n"]),
            "estimate_1_label": "OLS intercept (female mean Delta T)",
            "estimate_1": intercept["estimate"],
            "pvalue_1": intercept["pvalue"],
            "estimate_2_label": "OLS beta for male sex",
            "estimate_2": male["estimate"],
            "pvalue_2": male["pvalue"],
            "mean_delta_T": df["delta_trajectory_position"].mean(),
            "median_delta_T": df["delta_trajectory_position"].median(),
            "wilcoxon_p_delta_T": safe_wilcoxon_p(df["delta_trajectory_position"]),
            "direction": "positive" if df["delta_trajectory_position"].mean() > 0 else "negative",
            "notes": "Linear model Delta T ~ sex; male coefficient tests sex difference.",
        }
    )

    for sex_value, group in df.groupby("sex"):
        if len(group) < 10:
            status = "not run"
            note = "Group sample size below 10."
        else:
            status = "computed"
            note = "Sex-specific longitudinal Delta T summary."
        mean_delta = group["delta_trajectory_position"].mean()
        rows.append(
            {
                "analysis": "Sex-specific Delta T",
                "subgroup": str(sex_value),
                "status": status,
                "n": len(group),
                "estimate_1_label": "",
                "estimate_1": np.nan,
                "pvalue_1": np.nan,
                "estimate_2_label": "",
                "estimate_2": np.nan,
                "pvalue_2": np.nan,
                "mean_delta_T": mean_delta if status == "computed" else np.nan,
                "median_delta_T": group["delta_trajectory_position"].median() if status == "computed" else np.nan,
                "wilcoxon_p_delta_T": safe_wilcoxon_p(group["delta_trajectory_position"]) if status == "computed" else np.nan,
                "direction": "positive" if mean_delta > 0 else "negative" if mean_delta < 0 else "null",
                "notes": note,
            }
        )

    bins = [20, 35, 50, 71]
    labels = ["20-34", "35-49", "50-70"]
    df["age_bin"] = pd.cut(df["age"], bins=bins, labels=labels, right=False, include_lowest=True)
    for label in labels:
        group = df[df["age_bin"] == label]
        mean_delta = group["delta_trajectory_position"].mean()
        rows.append(
            {
                "analysis": "Age-bin Delta T",
                "subgroup": label,
                "status": "computed" if len(group) > 0 else "not run",
                "n": len(group),
                "estimate_1_label": "",
                "estimate_1": np.nan,
                "pvalue_1": np.nan,
                "estimate_2_label": "",
                "estimate_2": np.nan,
                "pvalue_2": np.nan,
                "mean_delta_T": mean_delta,
                "median_delta_T": group["delta_trajectory_position"].median(),
                "wilcoxon_p_delta_T": safe_wilcoxon_p(group["delta_trajectory_position"]),
                "direction": "positive" if mean_delta > 0 else "negative" if mean_delta < 0 else "null",
                "notes": "Age-bin direction is based on the sign of mean Delta T.",
            }
        )

    return pd.DataFrame(rows)


def make_permutation_histogram(null_vals: np.ndarray, perm_results: pd.DataFrame) -> None:
    observed = float(
        perm_results.loc[
            perm_results["analysis"] == "Fallback longitudinal-subset PC1-PC4 permutation",
            "observed_mean_delta_T",
        ].iloc[0]
    )
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.hist(null_vals, bins=45, color="#d9d9d9", edgecolor="#4d4d4d", linewidth=0.4)
    ax.axvline(observed, color="#111111", linewidth=2, label=f"Observed = {observed:.3f}")
    ax.set_xlabel("Mean Delta T under permuted age-axis labels")
    ax.set_ylabel("Permutation count")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "permutation_histogram.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "permutation_histogram.pdf", bbox_inches="tight")
    plt.close(fig)


def make_pipeline_schematic() -> None:
    steps = [
        "EDF selection",
        "Preprocessing",
        "Spectral feature extraction",
        "Baseline transformation fitting",
        "Brain-age benchmark",
        "PCA latent space",
        "Aging-axis definition",
        "Trajectory position/deviation",
        "Fixed follow-up projection",
        "Longitudinal statistics",
    ]
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    ax.axis("off")
    positions = []
    for row in range(2):
        xs = np.linspace(0.11, 0.89, 5)
        y = 0.68 if row == 0 else 0.30
        for x in xs:
            positions.append((x, y))

    for i, (label, (x, y)) in enumerate(zip(steps, positions)):
        box_w = 0.145
        box_h = 0.16
        rect = plt.Rectangle(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            facecolor="#f7f7f7",
            edgecolor="#222222",
            linewidth=1.0,
        )
        ax.add_patch(rect)
        wrapped = "\n".join(wrap(label, width=20))
        ax.text(x, y, wrapped, ha="center", va="center", fontsize=9.5)
        if i < len(steps) - 1:
            x2, y2 = positions[i + 1]
            if i == 4:
                ax.annotate(
                    "",
                    xy=(x2, y2 + box_h / 2),
                    xytext=(x, y - box_h / 2),
                    arrowprops=dict(arrowstyle="->", color="#222222", lw=1.0),
                )
            else:
                ax.annotate(
                    "",
                    xy=(x2 - box_w / 2, y2),
                    xytext=(x + box_w / 2, y),
                    arrowprops=dict(arrowstyle="->", color="#222222", lw=1.0),
                )

    ax.set_xlim(0, 1)
    ax.set_ylim(0.08, 0.88)
    fig.tight_layout(pad=0.5)
    fig.savefig(OUT_DIR / "pipeline_schematic.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "pipeline_schematic.pdf", bbox_inches="tight")
    plt.close(fig)


def compact_summary(
    projection_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
    pc_df: pd.DataFrame,
    demographic_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    p1 = projection_df[projection_df["policy"].str.contains("baseline-fitted")].iloc[0]
    rows.extend(
        [
            {
                "Analysis": "Fixed-projection ablation",
                "Metric": "Baseline-fitted mean Delta T",
                "Estimate": p1["mean_delta_T"],
                "P value": p1["wilcoxon_p_delta_T"],
                "Status": p1["status"],
                "Notes": "Main policy computed from saved longitudinal projections.",
            },
            {
                "Analysis": "Fixed-projection ablation",
                "Metric": "Baseline-fitted mean Delta D",
                "Estimate": p1["mean_delta_D"],
                "P value": p1["wilcoxon_p_delta_D"],
                "Status": p1["status"],
                "Notes": "Main policy computed from saved longitudinal projections.",
            },
        ]
    )
    for metric in [
        "ridge-regression MAE",
        "prediction Pearson r",
        "R2",
        "mean Delta T",
        "baseline T vs future Delta T Spearman rho",
        "chronological age vs future Delta T Spearman rho",
    ]:
        row = bootstrap_df[bootstrap_df["Metric"] == metric].iloc[0]
        rows.append(
            {
                "Analysis": "Bootstrap CI",
                "Metric": metric,
                "Estimate": row["Point estimate"],
                "P value": np.nan,
                "Status": row["Status"],
                "Notes": f"95% CI [{row['CI lower']:.6g}, {row['CI upper']:.6g}], n={int(row['Bootstrap n'])}.",
            }
        )
    perm = permutation_df[permutation_df["status"] == "computed_fallback"].iloc[0]
    rows.append(
        {
            "Analysis": "Permutation",
            "Metric": "Fallback observed mean Delta T vs permuted age-axis null",
            "Estimate": perm["observed_mean_delta_T"],
            "P value": perm["empirical_p_one_sided_greater"],
            "Status": perm["status"],
            "Notes": "Fallback uses 208-subject PC1-PC4 table; full-baseline permutation not run.",
        }
    )
    for _, row in pc_df.iterrows():
        rows.append(
            {
                "Analysis": "PC-space robustness",
                "Metric": row["version"],
                "Estimate": row["delta_T_mean"],
                "P value": row["delta_T_wilcoxon_p"],
                "Status": row["status"],
                "Notes": row["components_or_latent_dimensions_used"],
            }
        )
    for _, row in demographic_df[demographic_df["analysis"].isin(["Sex-specific Delta T", "Age-bin Delta T"])].iterrows():
        rows.append(
            {
                "Analysis": row["analysis"],
                "Metric": row["subgroup"],
                "Estimate": row["mean_delta_T"],
                "P value": row["wilcoxon_p_delta_T"],
                "Status": row["status"],
                "Notes": f"n={row['n']}, direction={row['direction']}.",
            }
        )
    return pd.DataFrame(rows)


def flatten_metrics(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for table_name, df in tables.items():
        for idx, row in df.iterrows():
            status = row.get("status", row.get("Status", "computed"))
            notes = row.get("notes", row.get("Notes", ""))
            label = (
                row.get("policy")
                or row.get("version")
                or row.get("sensitivity_model")
                or row.get("analysis")
                or row.get("Metric")
                or f"row_{idx}"
            )
            for col, value in row.items():
                if col.lower() in {"notes", "status"}:
                    continue
                if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                    rows.append(
                        {
                            "source_table": table_name,
                            "analysis": label,
                            "metric": col,
                            "value": value,
                            "status": status,
                            "notes": notes,
                        }
                    )
    return pd.DataFrame(rows)


def write_notes(
    projection_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
    pc_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    demo_df: pd.DataFrame,
    excel_written: bool,
) -> None:
    methods = f"""Robustness methods note

Inputs used:
- outputs/brain_age_predictions_full_baseline_corrected.csv
- outputs/longitudinal_cadence_shift.csv
- outputs/cadence_summary_results.csv
- outputs/longitudinal_cadence_summary_clean.csv
- outputs/test_retest_stability.csv for feature-name counts only

Unavailable inputs:
- baseline_s3_corrected_features.csv with the 93 EEG features for all baseline participants
- ses2_eyesclosed_pre_features.csv with the 93 EEG features for follow-up participants
- full cadence_position_deviation.csv or pca_latent_trajectory.csv for all 608 baseline participants
- PLS latent scores for baseline and follow-up participants

Analysis A, fixed-projection ablation:
Policy 1 was recomputed from saved baseline-fitted longitudinal projections. Policies 2 and 3 were not run because the feature tables needed to refit imputation, winsorization, scaling, PCA, and age-axis parameters are absent. Policy 3 would also have a coordinate-comparability limitation because separate session refits do not define the same latent coordinate system.

Analysis B, bootstrap uncertainty:
Bootstrap confidence intervals used {BOOTSTRAP_N} participant-level resamples with replacement and fixed seed {SEED}. Ridge MAE, Pearson r, R2, longitudinal Delta T, Wilcoxon rank-biserial effect size, baseline T vs future Delta T, and age vs future Delta T were bootstrapped from available derived tables. Full-baseline trajectory-age and deviation-age CIs were not run because the full baseline trajectory table is absent; the original point estimates were preserved from outputs/cadence_summary_results.csv. Fallback 208-subject trajectory-age and deviation-age CIs were also computed and labeled.

Analysis C, permutation:
The requested full-baseline PCA age-axis permutation was not run because the full baseline PCA feature space is absent. A labeled fallback permutation used the available 208-subject baseline/follow-up PC1-PC4 scores, recomputed the young-to-old axis after permuting baseline age labels, and repeated this {PERMUTATION_N} times with seed {SEED}.

Analysis D, PC-space robustness:
The original PC1-PC4 longitudinal result was recomputed from saved baseline-fitted projections, with trajectory-age rho/p preserved from the full-baseline summary. Alternative top-two-PC, first-five-PC, and first-ten-PC axes were recomputed only from the available 208-subject longitudinal PC scores and are labeled computed_fallback. The PLS-derived axis was not run because PLS latent scores or raw features are absent.

Analysis E, feature-set sensitivity:
Feature restriction metrics were not run because the baseline/follow-up 93-feature tables are absent. Where feature names were available from outputs/test_retest_stability.csv, retained-feature counts were reported.

Analysis F, demographic robustness:
Sex and age-bin checks were computed from the 208-subject longitudinal table. Baseline T vs age adjusted for sex is labeled as a fallback because the full 608-subject trajectory table is absent. Linear models used ordinary least squares. Age bins were 20-34, 35-49, and 50-70 years.

Analysis G, pipeline schematic:
A monochrome schematic was generated with matplotlib and exported to PNG and PDF.
"""

    results = f"""Robustness results note

Fixed projection:
Baseline-fitted projection had n={int(projection_df.loc[0, 'n'])}, mean Delta T={projection_df.loc[0, 'mean_delta_T']:.6f}, median Delta T={projection_df.loc[0, 'median_delta_T']:.6f}, Wilcoxon p={projection_df.loc[0, 'wilcoxon_p_delta_T']:.6g}; mean Delta D={projection_df.loc[0, 'mean_delta_D']:.6f}, median Delta D={projection_df.loc[0, 'median_delta_D']:.6f}, Wilcoxon p={projection_df.loc[0, 'wilcoxon_p_delta_D']:.6g}. Baseline T vs future Delta T rho={projection_df.loc[0, 'baseline_T_delta_T_spearman_rho']:.6f}, p={projection_df.loc[0, 'baseline_T_delta_T_spearman_p']:.6g}; age vs future Delta T rho={projection_df.loc[0, 'age_delta_T_spearman_rho']:.6f}, p={projection_df.loc[0, 'age_delta_T_spearman_p']:.6g}.

Bootstrap:
See bootstrap_confidence_intervals.csv for point estimates and 95% CIs. Computed rows used {BOOTSTRAP_N} bootstrap resamples unless otherwise noted in the table.

Permutation:
The full-baseline permutation was not run. The fallback longitudinal-subset PC1-PC4 permutation had observed mean Delta T={permutation_df.loc[1, 'observed_mean_delta_T']:.6f}, null mean={permutation_df.loc[1, 'null_mean']:.6f}, null SD={permutation_df.loc[1, 'null_sd']:.6f}, two-sided p={permutation_df.loc[1, 'empirical_p_two_sided']:.6g}, one-sided greater p={permutation_df.loc[1, 'empirical_p_one_sided_greater']:.6g}.

PC-space:
Original PC1-PC4 mean Delta T={pc_df.loc[0, 'delta_T_mean']:.6f}, Wilcoxon p={pc_df.loc[0, 'delta_T_wilcoxon_p']:.6g}. Fallback alternatives are reported in pc_space_robustness.csv.

Feature sensitivity:
Feature-set sensitivity models were not run because subject-level EEG feature tables are absent. Retained-feature counts are reported where feature names were available.

Demographics:
Sex-specific and age-bin numerical results are reported in demographic_robustness.csv. Baseline T vs age adjusted for sex is a 208-subject fallback analysis.
"""

    changelog = f"""Change log

1. Created robustness_outputs/.
2. Read existing derived outputs from outputs/.
3. Computed fixed baseline-projection longitudinal metrics from outputs/longitudinal_cadence_shift.csv.
4. Computed {BOOTSTRAP_N} bootstrap confidence intervals for available prediction and longitudinal metrics.
5. Preserved full-baseline trajectory-age point estimates from outputs/cadence_summary_results.csv and marked missing full-baseline bootstrap CIs as not run.
6. Computed a labeled fallback {PERMUTATION_N}-iteration permutation test using available 208-subject PC1-PC4 longitudinal coordinates.
7. Computed original PC1-PC4 longitudinal robustness metrics and labeled fallback PC-space alternatives from available PC1-PC10 longitudinal scores.
8. Marked pooled projection, session-wise projection, PLS-axis, and feature-set sensitivity metrics as not run where required feature or latent tables were unavailable.
9. Computed demographic robustness checks from the available longitudinal table.
10. Generated permutation histogram and pipeline schematic as PNG/PDF.
11. Wrote CSV tables and {'robustness_metrics.xlsx' if excel_written else 'skipped XLSX because no Excel writer engine was available'}.
"""

    (OUT_DIR / "methods_note_robustness.txt").write_text(methods, encoding="utf-8")
    (OUT_DIR / "results_note_robustness.txt").write_text(results, encoding="utf-8")
    (OUT_DIR / "changelog.txt").write_text(changelog, encoding="utf-8")


def write_excel(tables: dict[str, pd.DataFrame]) -> bool:
    path = OUT_DIR / "robustness_metrics.xlsx"
    engines = ["openpyxl", "xlsxwriter"]
    for engine in engines:
        try:
            with pd.ExcelWriter(path, engine=engine) as writer:
                for name, df in tables.items():
                    sheet = name[:31]
                    df.to_excel(writer, sheet_name=sheet, index=False)
            return True
        except ModuleNotFoundError:
            continue
        except ImportError:
            continue
    return False


def main() -> None:
    ensure_outputs()

    brain_df = read_csv_required(INPUT_DIR / "brain_age_predictions_full_baseline_corrected.csv")
    long_df = read_csv_required(INPUT_DIR / "longitudinal_cadence_shift.csv")
    cadence_summary = read_csv_required(INPUT_DIR / "cadence_summary_results.csv")
    read_csv_required(INPUT_DIR / "longitudinal_cadence_summary_clean.csv")

    feature_path = INPUT_DIR / "test_retest_stability.csv"
    feature_names = []
    if feature_path.exists():
        feature_names = pd.read_csv(feature_path)["feature"].astype(str).tolist()

    projection_df = projection_policy_ablation(long_df)
    bootstrap_df = bootstrap_table(brain_df, long_df, cadence_summary)
    permutation_df, null_vals = permutation_test(long_df)
    pc_df = pc_space_robustness(long_df, cadence_summary)
    feature_df = feature_sensitivity(feature_names)
    demo_df = demographic_robustness(long_df)
    summary_df = compact_summary(projection_df, bootstrap_df, permutation_df, pc_df, demo_df)

    tables = {
        "compact_summary_table": summary_df,
        "projection_policy_ablation": projection_df,
        "bootstrap_confidence_intervals": bootstrap_df,
        "permutation_test_results": permutation_df,
        "pc_space_robustness": pc_df,
        "feature_sensitivity": feature_df,
        "demographic_robustness": demo_df,
    }
    metrics_df = flatten_metrics(tables)

    metrics_df.to_csv(OUT_DIR / "robustness_metrics.csv", index=False)
    summary_df.to_csv(OUT_DIR / "compact_summary_table.csv", index=False)
    bootstrap_df.to_csv(OUT_DIR / "bootstrap_confidence_intervals.csv", index=False)
    projection_df.to_csv(OUT_DIR / "projection_policy_ablation.csv", index=False)
    pc_df.to_csv(OUT_DIR / "pc_space_robustness.csv", index=False)
    feature_df.to_csv(OUT_DIR / "feature_sensitivity.csv", index=False)
    demo_df.to_csv(OUT_DIR / "demographic_robustness.csv", index=False)
    permutation_df.to_csv(OUT_DIR / "permutation_test_results.csv", index=False)

    make_permutation_histogram(null_vals, permutation_df)
    make_pipeline_schematic()

    excel_written = write_excel({"robustness_metrics": metrics_df, **tables})
    write_notes(projection_df, bootstrap_df, permutation_df, pc_df, feature_df, demo_df, excel_written)

    print(f"Wrote robustness outputs to {OUT_DIR}")
    print(f"Excel written: {excel_written}")


if __name__ == "__main__":
    main()
