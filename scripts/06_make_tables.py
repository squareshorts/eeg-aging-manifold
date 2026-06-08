"""Generate manuscript tables from saved analysis outputs."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_age.config import (  # noqa: E402
    EXTERNAL_RESULTS_DIR,
    OUTPUTS_DIR,
    PARTICIPANTS,
    RESULTS_TABLES_DIR,
    SEED,
    TABLES_DIR,
)
from eeg_age.io import write_csv  # noqa: E402


def fmt(value: float, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def metric_fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def range_fmt(lo: float, hi: float, digits: int = 3) -> str:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return "NA"
    sep = " to " if lo < 0 or hi < 0 else "--"
    return f"{metric_fmt(lo, digits)}{sep}{metric_fmt(hi, digits)}"


def ci_fmt(row: pd.Series, digits: int = 3) -> str:
    return f"{metric_fmt(row['estimate'], digits)} ({range_fmt(row['ci_lower'], row['ci_upper'], digits)})"


def cohort_summary() -> pd.DataFrame:
    participants = pd.read_csv(PARTICIPANTS)
    followup_path = OUTPUTS_DIR / "ses2_eyesclosed_pre_features.csv"
    if followup_path.exists():
        followup_ids = pd.read_csv(followup_path, usecols=["participant_id"])["participant_id"].astype(str)
    else:
        followup_ids = pd.read_csv(OUTPUTS_DIR / "longitudinal_cadence_shift.csv", usecols=["participant_id"])["participant_id"].astype(str)
    followup = participants[participants["participant_id"].astype(str).isin(followup_ids)].copy()

    def summarize(label: str, df: pd.DataFrame) -> dict[str, object]:
        age = pd.to_numeric(df["age"], errors="coerce")
        sex = df["sex"].astype(str).str.upper().str[0]
        return {
            "cohort": label,
            "participants": int(len(df)),
            "age_available_n": int(age.notna().sum()),
            "age_range_years": f"{int(age.min())}--{int(age.max())}",
            "mean_age_years": float(age.mean()),
            "age_sd_years": float(age.std(ddof=1)),
            "sex_available_n": int(sex.isin(["F", "M"]).sum()),
            "female": int((sex == "F").sum()),
            "male": int((sex == "M").sum()),
            "eeg_condition": "Eyes closed, pre-task",
            "missingness_note": "No age or sex missing",
        }

    return pd.DataFrame([summarize("Baseline", participants), summarize("Follow-up subset", followup)])


def write_cohort_latex(df: pd.DataFrame) -> None:
    baseline = df[df["cohort"] == "Baseline"].iloc[0]
    follow = df[df["cohort"] == "Follow-up subset"].iloc[0]
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Measure & Baseline & Follow-up subset \\",
        r"\midrule",
        f"Participants & {baseline['participants']} & {follow['participants']} \\\\",
        f"Age available, N & {baseline['age_available_n']} & {follow['age_available_n']} \\\\",
        f"Age range, years & {baseline['age_range_years']} & {follow['age_range_years']} \\\\",
        f"Mean age, years & {fmt(baseline['mean_age_years'])} & {fmt(follow['mean_age_years'])} \\\\",
        f"Age SD, years & {fmt(baseline['age_sd_years'])} & {fmt(follow['age_sd_years'])} \\\\",
        f"Sex available, N & {baseline['sex_available_n']} & {follow['sex_available_n']} \\\\",
        f"Female & {baseline['female']} & {follow['female']} \\\\",
        f"Male & {baseline['male']} & {follow['male']} \\\\",
        rf"EEG condition & {baseline['eeg_condition']} & {follow['eeg_condition']} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / "cohort_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def external_performance_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dev_t_path = RESULTS_TABLES_DIR / "development_fixed_projection_metrics.csv"
    if dev_t_path.exists():
        dev_t = pd.read_csv(dev_t_path)
        assoc = dev_t[dev_t["analysis"] == "Development fixed T vs chronological age"]
        pearson = assoc[assoc["metric"] == "Pearson r"].iloc[0]
        spearman = assoc[assoc["metric"] == "Spearman rho"].iloc[0]
        rows.append(
            {
                "dataset": "Development ds005385 session-1",
                "analysis": "Fixed T vs chronological age",
                "n": int(spearman["n"]),
                "summary": (
                    f"Pearson $r={metric_fmt(pearson['estimate'])}$ "
                    f"({range_fmt(pearson['ci_lower'], pearson['ci_upper'])}); "
                    f"Spearman $\\rho={metric_fmt(spearman['estimate'])}$ "
                    f"({range_fmt(spearman['ci_lower'], spearman['ci_upper'])}), "
                    f"$p={spearman['pvalue']:.2e}$"
                ),
                "protocol": "Frozen ds005385 session-1 PCA trajectory",
            }
        )

    dev_cv_path = OUTPUTS_DIR / "brain_age_predictions_fold_local_metrics.csv"
    if dev_cv_path.exists():
        dev = pd.read_csv(dev_cv_path).iloc[0]
        rows.append(
            {
                "dataset": "Development ds005385 session-1",
                "analysis": "Ridge age prediction, 10-fold CV",
                "n": int(dev["n"]),
                "summary": (
                    f"MAE {metric_fmt(dev['mae'], 2)} years "
                    f"({range_fmt(dev['mae_ci_lower'], dev['mae_ci_upper'], 2)}); "
                    f"Pearson $r={metric_fmt(dev['pearson_r'])}$ "
                    f"({range_fmt(dev['pearson_r_ci_lower'], dev['pearson_r_ci_upper'])}); "
                    f"$R^2={metric_fmt(dev['r2'])}$ "
                    f"({range_fmt(dev['r2_ci_lower'], dev['r2_ci_upper'])})"
                ),
                "protocol": "Fold-local development benchmark",
            }
        )

    ext_path = EXTERNAL_RESULTS_DIR / "ds003775_validation_metrics.csv"
    if ext_path.exists():
        ext = pd.read_csv(ext_path)

        def row_for(analysis: str, dataset: str) -> dict[str, object]:
            subset = ext[ext["analysis"] == analysis]
            out = {
                "dataset": dataset,
                "analysis": analysis,
                "n": int(subset["n"].dropna().iloc[0]) if "n" in subset and subset["n"].notna().any() else math.nan,
                "summary": "",
                "protocol": subset["validation_protocol"].iloc[0] if "validation_protocol" in subset else "",
            }
            pieces = []
            for _, metric in subset.iterrows():
                name = metric["metric"]
                if name == "MAE":
                    pieces.append(
                        f"MAE {metric_fmt(metric['estimate'], 2)} years "
                        f"({range_fmt(metric['ci_lower'], metric['ci_upper'], 2)})"
                    )
                elif name == "Pearson r":
                    p = f", $p={metric['pvalue']:.3g}$" if "pvalue" in metric and np.isfinite(metric["pvalue"]) else ""
                    pieces.append(
                        f"Pearson $r={metric_fmt(metric['estimate'])}$ "
                        f"({range_fmt(metric['ci_lower'], metric['ci_upper'])}){p}"
                    )
                elif name == "R2":
                    pieces.append(
                        f"$R^2={metric_fmt(metric['estimate'])}$ "
                        f"({range_fmt(metric['ci_lower'], metric['ci_upper'])})"
                    )
                elif name == "Spearman rho":
                    p = f", $p={metric['pvalue']:.3g}$" if "pvalue" in metric and np.isfinite(metric["pvalue"]) else ""
                    pieces.append(
                        f"Spearman $\\rho={metric_fmt(metric['estimate'])}$ "
                        f"({range_fmt(metric['ci_lower'], metric['ci_upper'])}){p}"
                    )
            out["summary"] = "; ".join(pieces)
            return out

        rows.append(row_for("External T vs chronological age", "External ds003775 adult subset"))
        rows.append(row_for("External T-calibrated age prediction", "External ds003775 adult subset"))
        rows.append(row_for("External ridge age prediction", "External ds003775 adult subset"))

    for ext_dataset_id, label in [("ds003690", "External ds003690 Healthy Aging"), ("ds004148", "External ds004148 Young Adults")]:
        p = EXTERNAL_RESULTS_DIR / f"{ext_dataset_id}_validation_metrics.csv"
        if p.exists():
            ext2 = pd.read_csv(p)
            def row_for_dataset(analysis: str, dataset_label: str) -> dict[str, object]:
                subset = ext2[ext2["analysis"] == analysis]
                out = {
                    "dataset": dataset_label,
                    "analysis": analysis,
                    "n": int(subset["n"].dropna().iloc[0]) if "n" in subset and subset["n"].notna().any() else math.nan,
                    "summary": "",
                    "protocol": subset["validation_protocol"].iloc[0] if "validation_protocol" in subset else "",
                }
                pieces = []
                for _, metric in subset.iterrows():
                    name = metric["metric"]
                    if name == "MAE":
                        pieces.append(
                            f"MAE {metric_fmt(metric['estimate'], 2)} years "
                            f"({range_fmt(metric['ci_lower'], metric['ci_upper'], 2)})"
                        )
                    elif name == "Pearson r":
                        pp = f", $p={metric['pvalue']:.3g}$" if "pvalue" in metric and np.isfinite(metric["pvalue"]) else ""
                        pieces.append(
                            f"Pearson $r={metric_fmt(metric['estimate'])}$ "
                            f"({range_fmt(metric['ci_lower'], metric['ci_upper'])}){pp}"
                        )
                    elif name == "R2":
                        pieces.append(
                            f"$R^2={metric_fmt(metric['estimate'])}$ "
                            f"({range_fmt(metric['ci_lower'], metric['ci_upper'])})"
                        )
                    elif name == "Spearman rho":
                        pp = f", $p={metric['pvalue']:.3g}$" if "pvalue" in metric and np.isfinite(metric["pvalue"]) else ""
                        pieces.append(
                            f"Spearman $\\rho={metric_fmt(metric['estimate'])}$ "
                            f"({range_fmt(metric['ci_lower'], metric['ci_upper'])}){pp}"
                        )
                out["summary"] = "; ".join(pieces)
                return out

            rows.append(row_for_dataset("External T vs chronological age", label))
            rows.append(row_for_dataset("External T-calibrated age prediction", label))
            rows.append(row_for_dataset("External ridge age prediction", label))

    return pd.DataFrame(rows)


def write_performance_latex(df: pd.DataFrame) -> None:
    lines = [
        r"\begin{tabular}{llp{0.58\textwidth}}",
        r"\toprule",
        r"Dataset and analysis & N & Estimate (95\% bootstrap CI) \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        label = f"{row['dataset']}: {row['analysis']}"
        lines.append(f"{label} & {int(row['n'])} & {row['summary']} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / "development_external_performance.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    np.random.seed(SEED)
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    cohort = cohort_summary()
    write_csv(cohort, RESULTS_TABLES_DIR / "cohort_summary.csv")
    write_cohort_latex(cohort)

    perf = external_performance_table()
    write_csv(perf, RESULTS_TABLES_DIR / "development_external_performance.csv")
    if len(perf):
        write_performance_latex(perf)

    print(f"Saved {TABLES_DIR / 'cohort_summary.tex'}")
    print(f"Saved {RESULTS_TABLES_DIR / 'cohort_summary.csv'}")
    if len(perf):
        print(f"Saved {TABLES_DIR / 'development_external_performance.tex'}")
        print(f"Saved {RESULTS_TABLES_DIR / 'development_external_performance.csv'}")


if __name__ == "__main__":
    main()
