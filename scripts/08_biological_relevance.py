"""Test biological relevance of the trajectory coordinate T beyond chronological age.

This script evaluates whether T explains behavioral/cognitive variance in ds003775
beyond what is explained by chronological age and sex.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_age.config import EXTERNAL_RESULTS_DIR, RESULTS_TABLES_DIR  # noqa: E402
from eeg_age.io import write_csv  # noqa: E402


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    ranks = np.empty(n, dtype=int)
    sorted_indices = np.argsort(p_values)
    ranks[sorted_indices] = np.arange(1, n + 1)
    
    sorted_p = p_values[sorted_indices]
    fdr_p = sorted_p * n / np.arange(1, n + 1)
    
    # Enforce monotonicity
    for i in range(n - 2, -1, -1):
        fdr_p[i] = min(fdr_p[i], fdr_p[i + 1])
        
    fdr_p = np.minimum(fdr_p, 1.0)
    
    unsorted_fdr_p = np.empty(n, dtype=float)
    unsorted_fdr_p[sorted_indices] = fdr_p
    return unsorted_fdr_p


def fit_ols(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Fit OLS and return R2, coefficients, and RSS."""
    n, p = X.shape
    coef, rss, _, _ = np.linalg.lstsq(X, y, rcond=None)
    rss = rss[0] if len(rss) > 0 else np.sum((y - X @ coef) ** 2)
    tss = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - (rss / tss)
    return r2, coef, rss


def main() -> None:
    # 1. Load data
    projection_path = EXTERNAL_RESULTS_DIR / "ds003775_external_projection.csv"
    if not projection_path.exists():
        raise FileNotFoundError(f"Missing projection: {projection_path}")
    
    proj_df = pd.read_csv(projection_path)
    
    participants_path = ROOT / "data" / "external" / "ds003775" / "participants.tsv"
    if not participants_path.exists():
        raise FileNotFoundError(f"Missing participants: {participants_path}")
        
    participants = pd.read_csv(participants_path, sep='\t')
    
    # Merge on participant_id
    df = pd.merge(proj_df, participants, on="participant_id", suffixes=('', '_drop'))
    df = df.loc[:, ~df.columns.str.endswith('_drop')]
    
    # Cognitive variables in ds003775
    cog_vars = [
        "ravlt_1", "ravlt_5", "ravlt_tot", "ravlt_imm", "ravlt_del", "ravlt_rec", "ravlt_fp",
        "ds_forw", "ds_back", "ds_seq", "ds_tot",
        "tmt_2", "tmt_3", "tmt_4",
        "cw_1", "cw_2", "cw_3", "cw_4",
        "vf_1", "vf_2", "vf_3"
    ]
    
    # Prepare basic covariates
    df["age_num"] = pd.to_numeric(df["age"], errors="coerce")
    
    sex_str = df["sex"].astype(str).str.upper()
    df["is_female"] = (sex_str.str[0] == "F").astype(float)
    df["is_male"] = (sex_str.str[0] == "M").astype(float)
    
    # Ensure sex is coded properly
    valid_mask = df["age_num"].notna() & df["T"].notna() & (df["is_female"] + df["is_male"] > 0)
    df = df[valid_mask].copy()
    
    results = []
    
    for var in cog_vars:
        if var not in df.columns:
            continue
            
        # Clean specific variable
        df[f"{var}_num"] = pd.to_numeric(df[var], errors="coerce")
        var_mask = df[f"{var}_num"].notna()
        sub_df = df[var_mask].copy()
        
        n_samples = len(sub_df)
        if n_samples < 20:
            continue
            
        y = sub_df[f"{var}_num"].to_numpy(float)
        age = sub_df["age_num"].to_numpy(float)
        sex_f = sub_df["is_female"].to_numpy(float)
        T = sub_df["T"].to_numpy(float)
        
        # Standardize for beta
        y_std = (y - np.mean(y)) / np.std(y, ddof=1)
        age_std = (age - np.mean(age)) / np.std(age, ddof=1)
        sex_f_std = (sex_f - np.mean(sex_f)) / np.std(sex_f, ddof=1)
        T_std = (T - np.mean(T)) / np.std(T, ddof=1)
        
        # Model 1: y ~ age + sex (with intercept)
        X1 = np.column_stack([np.ones(n_samples), age_std, sex_f_std])
        r2_1, coef1, rss1 = fit_ols(X1, y_std)
        p1 = X1.shape[1]
        
        # Model 2: y ~ age + sex + T
        X2 = np.column_stack([np.ones(n_samples), age_std, sex_f_std, T_std])
        r2_2, coef2, rss2 = fit_ols(X2, y_std)
        p2 = X2.shape[1]
        
        # F-test for nested models
        # F = ((RSS1 - RSS2) / (p2 - p1)) / (RSS2 / (n - p2))
        df_num = p2 - p1
        df_den = n_samples - p2
        f_stat = ((rss1 - rss2) / df_num) / (rss2 / df_den)
        p_val = f.sf(f_stat, df_num, df_den)
        
        beta_T = coef2[3]
        
        # t-statistic for T coefficient
        # SE(beta) = sqrt( MSE * (X'X)^-1[j,j] )
        cov_matrix = (rss2 / df_den) * np.linalg.inv(X2.T @ X2)
        se_beta_T = np.sqrt(cov_matrix[3, 3])
        t_stat = beta_T / se_beta_T
        
        results.append({
            "cognitive_variable": var,
            "n_samples": n_samples,
            "r2_model1": r2_1,
            "r2_model2": r2_2,
            "delta_r2": r2_2 - r2_1,
            "f_stat": f_stat,
            "p_value": p_val,
            "beta_T_std": beta_T,
            "se_beta_T": se_beta_T,
            "t_stat_T": t_stat
        })
        
    res_df = pd.DataFrame(results)
    if len(res_df) > 0:
        res_df["fdr_p_value"] = benjamini_hochberg(res_df["p_value"].to_numpy())
        res_df = res_df.sort_values("delta_r2", ascending=False).reset_index(drop=True)
    
    out_path = RESULTS_TABLES_DIR / "biological_relevance_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_path, index=False)
    
    print(f"Evaluated {len(res_df)} cognitive variables.")
    print(f"Results saved to {out_path}")
    print("\nTop results (ranked by delta R^2):")
    cols = ["cognitive_variable", "delta_r2", "beta_T_std", "p_value", "fdr_p_value"]
    print(res_df[cols].head(10).to_string(index=False))
    
    surviving = res_df[res_df["fdr_p_value"] < 0.05]
    if len(surviving) > 0:
        print(f"\n{len(surviving)} variable(s) survived FDR < 0.05:")
        print(surviving[cols].to_string(index=False))
    else:
        print("\nNo variables survived FDR < 0.05.")


if __name__ == "__main__":
    main()
