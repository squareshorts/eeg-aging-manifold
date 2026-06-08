"""Reviewer-requested controls from saved longitudinal trajectory coordinates.

This script uses the available derived longitudinal PC table to compute:
- signed orthogonal deviation in the PC1-PC4 plane;
- Oldham-style baseline/change checks;
- a reliability-implied null for baseline-vs-change coupling.

It does not require the missing raw EEG feature tables.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr, rankdata, spearmanr, wilcoxon


SEED = 20260605
SIMULATION_N = 5000

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "longitudinal_cadence_shift.csv"
OUT_DIR = ROOT / "robustness_outputs"


def corr_rows(label: str, x, y) -> list[dict[str, float | str]]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    pearson = pearsonr(x_arr, y_arr)
    spearman = spearmanr(x_arr, y_arr)
    return [
        {
            "analysis": label,
            "metric": "Pearson r",
            "estimate": float(pearson.statistic),
            "pvalue": float(pearson.pvalue),
            "n": len(x_arr),
        },
        {
            "analysis": label,
            "metric": "Spearman rho",
            "estimate": float(spearman.statistic),
            "pvalue": float(spearman.pvalue),
            "n": len(x_arr),
        },
    ]


def wilcoxon_rank_biserial(values) -> float:
    x = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    x = x[x != 0]
    if len(x) == 0:
        return np.nan
    ranks = rankdata(np.abs(x))
    w_pos = float(ranks[x > 0].sum())
    w_neg = float(ranks[x < 0].sum())
    total = float(ranks.sum())
    return (w_pos - w_neg) / total if total else np.nan


def infer_pc14_axis(long_df: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Recover the saved aging-axis direction from PC1/PC4 and T."""
    x = long_df[["base_PC1", "base_PC4"]].to_numpy(float)
    t = long_df["base_trajectory_position"].to_numpy(float)
    design = np.column_stack([np.ones(len(x)), x])
    coef = np.linalg.lstsq(design, t, rcond=None)[0]
    direction = coef[1:]
    direction = direction / np.linalg.norm(direction)
    intercept = float(coef[0])
    return direction, intercept


def signed_deviation(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    direction, intercept = infer_pc14_axis(long_df)
    signed_axis = np.array([-direction[1], direction[0]])

    base_x = long_df[["base_PC1", "base_PC4"]].to_numpy(float)
    fu_x = long_df[["fu_PC1", "fu_PC4"]].to_numpy(float)
    base_unsigned = long_df["base_trajectory_deviation"].to_numpy(float)
    fu_unsigned = long_df["fu_trajectory_deviation"].to_numpy(float)

    signed_raw = np.concatenate([base_x @ signed_axis, fu_x @ signed_axis])
    unsigned = np.concatenate([base_unsigned, fu_unsigned])
    bounds = (signed_raw.min() - unsigned.max(), signed_raw.max() + unsigned.max())
    fit = minimize_scalar(
        lambda offset: float(np.mean((np.abs(signed_raw - offset) - unsigned) ** 2)),
        bounds=bounds,
        method="bounded",
    )
    offset = float(fit.x)

    base_signed = base_x @ signed_axis - offset
    fu_signed = fu_x @ signed_axis - offset
    delta_signed = fu_signed - base_signed
    abs_reconstruction_error = np.abs(
        np.concatenate([np.abs(base_signed), np.abs(fu_signed)]) - unsigned
    )

    subjects = long_df[["participant_id", "age", "sex"]].copy()
    subjects["base_D_signed"] = base_signed
    subjects["fu_D_signed"] = fu_signed
    subjects["delta_D_signed"] = delta_signed
    subjects["base_D_abs"] = base_unsigned
    subjects["fu_D_abs"] = fu_unsigned
    subjects["delta_D_abs"] = fu_unsigned - base_unsigned

    w_signed = wilcoxon(delta_signed)
    w_abs = wilcoxon(subjects["delta_D_abs"])
    rb_signed = wilcoxon_rank_biserial(delta_signed)
    rb_abs = wilcoxon_rank_biserial(subjects["delta_D_abs"])
    rho_age_base = spearmanr(subjects["age"], subjects["base_D_signed"])
    rho_age_delta = spearmanr(subjects["age"], subjects["delta_D_signed"])
    rho_base_delta = spearmanr(subjects["base_D_signed"], subjects["delta_D_signed"])

    summary = pd.DataFrame(
        [
            {
                "metric": "n",
                "estimate": len(subjects),
                "pvalue": np.nan,
                "notes": "Longitudinal participants with saved PC1-PC4 projections.",
            },
            {
                "metric": "aging_axis_PC1_weight",
                "estimate": direction[0],
                "pvalue": np.nan,
                "notes": "Recovered from base_PC1/base_PC4 and saved trajectory position.",
            },
            {
                "metric": "aging_axis_PC4_weight",
                "estimate": direction[1],
                "pvalue": np.nan,
                "notes": "Recovered from base_PC1/base_PC4 and saved trajectory position.",
            },
            {
                "metric": "signed_axis_PC1_weight",
                "estimate": signed_axis[0],
                "pvalue": np.nan,
                "notes": "Counter-clockwise 90-degree rotation of the aging axis.",
            },
            {
                "metric": "signed_axis_PC4_weight",
                "estimate": signed_axis[1],
                "pvalue": np.nan,
                "notes": "Counter-clockwise 90-degree rotation of the aging axis.",
            },
            {
                "metric": "signed_axis_offset",
                "estimate": offset,
                "pvalue": np.nan,
                "notes": "Offset chosen so abs(D_signed) reconstructs saved unsigned D.",
            },
            {
                "metric": "max_abs_reconstruction_error",
                "estimate": float(abs_reconstruction_error.max()),
                "pvalue": np.nan,
                "notes": "Numerical agreement between abs(D_signed) and saved D.",
            },
            {
                "metric": "mean_delta_D_signed",
                "estimate": float(np.mean(delta_signed)),
                "pvalue": float(w_signed.pvalue),
                "effect_size": rb_signed,
                "notes": "Wilcoxon p tests paired signed orthogonal change; effect size is matched-pairs rank-biserial correlation.",
            },
            {
                "metric": "median_delta_D_signed",
                "estimate": float(np.median(delta_signed)),
                "pvalue": float(w_signed.pvalue),
                "effect_size": rb_signed,
                "notes": "Wilcoxon p tests paired signed orthogonal change; effect size is matched-pairs rank-biserial correlation.",
            },
            {
                "metric": "wilcoxon_rank_biserial_delta_D_signed",
                "estimate": rb_signed,
                "pvalue": float(w_signed.pvalue),
                "effect_size": rb_signed,
                "notes": "Matched-pairs rank-biserial correlation for signed orthogonal change.",
            },
            {
                "metric": "mean_delta_D_abs",
                "estimate": float(subjects["delta_D_abs"].mean()),
                "pvalue": float(w_abs.pvalue),
                "effect_size": rb_abs,
                "notes": "Original unsigned dispersion metric; effect size is matched-pairs rank-biserial correlation.",
            },
            {
                "metric": "median_delta_D_abs",
                "estimate": float(subjects["delta_D_abs"].median()),
                "pvalue": float(w_abs.pvalue),
                "effect_size": rb_abs,
                "notes": "Original unsigned dispersion metric; effect size is matched-pairs rank-biserial correlation.",
            },
            {
                "metric": "baseline_D_signed_vs_age_spearman",
                "estimate": float(rho_age_base.statistic),
                "pvalue": float(rho_age_base.pvalue),
                "effect_size": np.nan,
                "notes": "Longitudinal subset only.",
            },
            {
                "metric": "delta_D_signed_vs_age_spearman",
                "estimate": float(rho_age_delta.statistic),
                "pvalue": float(rho_age_delta.pvalue),
                "effect_size": np.nan,
                "notes": "Longitudinal subset only.",
            },
            {
                "metric": "baseline_D_signed_vs_delta_D_signed_spearman",
                "estimate": float(rho_base_delta.statistic),
                "pvalue": float(rho_base_delta.pvalue),
                "effect_size": np.nan,
                "notes": "Baseline signed off-axis position vs signed off-axis change.",
            },
        ]
    )
    return subjects, summary


def baseline_change_control(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_t = long_df["base_trajectory_position"].to_numpy(float)
    fu_t = long_df["fu_trajectory_position"].to_numpy(float)
    delta_t = fu_t - base_t
    oldham_mean = (base_t + fu_t) / 2

    rows: list[dict[str, float | str]] = []
    rows.extend(corr_rows("Baseline T vs Delta T", base_t, delta_t))
    rows.extend(corr_rows("Oldham mean T vs Delta T", oldham_mean, delta_t))
    rows.extend(corr_rows("Baseline T vs follow-up T reliability", base_t, fu_t))

    pearson_reliability = pearsonr(base_t, fu_t).statistic
    expected_equal_variance = -np.sqrt((1.0 - pearson_reliability) / 2.0)
    var_base = np.var(base_t, ddof=1)
    sd_base = np.std(base_t, ddof=1)
    sd_delta = np.std(delta_t, ddof=1)
    cov_base_fu = np.cov(base_t, fu_t, ddof=1)[0, 1]
    reliability_implied_observed_margins = (cov_base_fu - var_base) / (sd_base * sd_delta)

    rng = np.random.default_rng(SEED)
    null = np.empty(SIMULATION_N, dtype=float)
    for i in range(SIMULATION_N):
        x = rng.normal(size=len(base_t))
        e = rng.normal(size=len(base_t))
        y = pearson_reliability * x + np.sqrt(1.0 - pearson_reliability**2) * e
        null[i] = np.corrcoef(x, y - x)[0, 1]
    observed_pearson = pearsonr(base_t, delta_t).statistic
    observed_percentile = (np.sum(null <= observed_pearson) + 1) / (SIMULATION_N + 1)
    empirical_two_sided = 2 * min(observed_percentile, 1 - observed_percentile)
    empirical_two_sided = min(empirical_two_sided, 1.0)

    rows.extend(
        [
            {
                "analysis": "Reliability-implied null",
                "metric": "Expected Pearson r, equal variances",
                "estimate": float(expected_equal_variance),
                "pvalue": np.nan,
                "n": len(base_t),
            },
            {
                "analysis": "Reliability-implied null",
                "metric": "Expected Pearson r, observed margins",
                "estimate": float(reliability_implied_observed_margins),
                "pvalue": np.nan,
                "n": len(base_t),
            },
            {
                "analysis": "Reliability-implied null simulation",
                "metric": "Null mean Pearson r",
                "estimate": float(null.mean()),
                "pvalue": np.nan,
                "n": SIMULATION_N,
            },
            {
                "analysis": "Reliability-implied null simulation",
                "metric": "Null SD Pearson r",
                "estimate": float(null.std(ddof=1)),
                "pvalue": np.nan,
                "n": SIMULATION_N,
            },
            {
                "analysis": "Reliability-implied null simulation",
                "metric": "Null 2.5 percentile",
                "estimate": float(np.percentile(null, 2.5)),
                "pvalue": np.nan,
                "n": SIMULATION_N,
            },
            {
                "analysis": "Reliability-implied null simulation",
                "metric": "Null 97.5 percentile",
                "estimate": float(np.percentile(null, 97.5)),
                "pvalue": np.nan,
                "n": SIMULATION_N,
            },
            {
                "analysis": "Reliability-implied null simulation",
                "metric": "Observed percentile",
                "estimate": float(observed_percentile),
                "pvalue": np.nan,
                "n": SIMULATION_N,
            },
            {
                "analysis": "Reliability-implied null simulation",
                "metric": "Empirical two-sided p",
                "estimate": float(empirical_two_sided),
                "pvalue": float(empirical_two_sided),
                "n": SIMULATION_N,
            },
        ]
    )

    null_df = pd.DataFrame(
        {
            "simulation": np.arange(1, SIMULATION_N + 1),
            "baseline_delta_pearson_r": null,
        }
    )
    return pd.DataFrame(rows), null_df


def make_null_plot(null_df: pd.DataFrame, observed: float) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.hist(
        null_df["baseline_delta_pearson_r"],
        bins=45,
        color="#d9d9d9",
        edgecolor="#4d4d4d",
        linewidth=0.4,
    )
    ax.axvline(observed, color="#111111", linewidth=2, label=f"Observed = {observed:.3f}")
    ax.set_xlabel("Pearson r: baseline T vs Delta T")
    ax.set_ylabel("Simulation count")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "baseline_change_reliability_null.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "baseline_change_reliability_null.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long_df = pd.read_csv(INPUT)

    signed_subjects, signed_summary = signed_deviation(long_df)
    baseline_control, reliability_null = baseline_change_control(long_df)

    signed_subjects.to_csv(OUT_DIR / "signed_orthogonal_deviation_subjects.csv", index=False)
    signed_summary.to_csv(OUT_DIR / "signed_orthogonal_deviation_summary.csv", index=False)
    baseline_control.to_csv(OUT_DIR / "baseline_change_control.csv", index=False)
    reliability_null.to_csv(OUT_DIR / "baseline_change_reliability_null.csv", index=False)

    observed = float(
        baseline_control.loc[
            (baseline_control["analysis"] == "Baseline T vs Delta T")
            & (baseline_control["metric"] == "Pearson r"),
            "estimate",
        ].iloc[0]
    )
    make_null_plot(reliability_null, observed)

    note = f"""Reviewer-requested control results

Input:
- {INPUT}

Signed orthogonal deviation:
- Mean Delta D_signed = {signed_summary.loc[signed_summary['metric'] == 'mean_delta_D_signed', 'estimate'].iloc[0]:.6f}
- Median Delta D_signed = {signed_summary.loc[signed_summary['metric'] == 'median_delta_D_signed', 'estimate'].iloc[0]:.6f}
- Wilcoxon p = {signed_summary.loc[signed_summary['metric'] == 'mean_delta_D_signed', 'pvalue'].iloc[0]:.6g}
- Wilcoxon rank-biserial effect size = {signed_summary.loc[signed_summary['metric'] == 'wilcoxon_rank_biserial_delta_D_signed', 'estimate'].iloc[0]:.6f}
- Original unsigned mean Delta D_abs = {signed_summary.loc[signed_summary['metric'] == 'mean_delta_D_abs', 'estimate'].iloc[0]:.6f}
- Original unsigned Wilcoxon p = {signed_summary.loc[signed_summary['metric'] == 'mean_delta_D_abs', 'pvalue'].iloc[0]:.6g}

Oldham/reliability control:
- Baseline T vs Delta T Spearman rho = {baseline_control.loc[(baseline_control['analysis'] == 'Baseline T vs Delta T') & (baseline_control['metric'] == 'Spearman rho'), 'estimate'].iloc[0]:.6f}
- Oldham mean T vs Delta T Spearman rho = {baseline_control.loc[(baseline_control['analysis'] == 'Oldham mean T vs Delta T') & (baseline_control['metric'] == 'Spearman rho'), 'estimate'].iloc[0]:.6f}
- Baseline T vs follow-up T Pearson reliability = {baseline_control.loc[(baseline_control['analysis'] == 'Baseline T vs follow-up T reliability') & (baseline_control['metric'] == 'Pearson r'), 'estimate'].iloc[0]:.6f}
- Reliability-implied equal-variance Pearson r = {baseline_control.loc[(baseline_control['analysis'] == 'Reliability-implied null') & (baseline_control['metric'] == 'Expected Pearson r, equal variances'), 'estimate'].iloc[0]:.6f}
- Reliability-null 95% range = [{baseline_control.loc[(baseline_control['analysis'] == 'Reliability-implied null simulation') & (baseline_control['metric'] == 'Null 2.5 percentile'), 'estimate'].iloc[0]:.6f}, {baseline_control.loc[(baseline_control['analysis'] == 'Reliability-implied null simulation') & (baseline_control['metric'] == 'Null 97.5 percentile'), 'estimate'].iloc[0]:.6f}]
- Observed percentile in reliability null = {baseline_control.loc[(baseline_control['analysis'] == 'Reliability-implied null simulation') & (baseline_control['metric'] == 'Observed percentile'), 'estimate'].iloc[0]:.6f}
- Empirical two-sided p in reliability null = {baseline_control.loc[(baseline_control['analysis'] == 'Reliability-implied null simulation') & (baseline_control['metric'] == 'Empirical two-sided p'), 'estimate'].iloc[0]:.6f}

Artifacts:
- signed_orthogonal_deviation_subjects.csv
- signed_orthogonal_deviation_summary.csv
- baseline_change_control.csv
- baseline_change_reliability_null.csv
- baseline_change_reliability_null.png
- baseline_change_reliability_null.pdf
"""
    (OUT_DIR / "reviewer_requested_controls_summary.txt").write_text(note, encoding="utf-8")

    print(note)


if __name__ == "__main__":
    main()
