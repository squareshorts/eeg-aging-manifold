"""Theory-driven cognitive composites analysis for ds003775.

Builds prespecified composites, runs nested regression, and performs
sensitivity analyses (HC3 robust SEs, Cook's distance exclusion, Spearman partial correlation).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f, spearmanr, pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_age.config import EXTERNAL_RESULTS_DIR, RESULTS_TABLES_DIR  # noqa: E402
from eeg_age.io import write_csv  # noqa: E402


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    n = len(p_values)
    ranks = np.empty(n, dtype=int)
    sorted_indices = np.argsort(p_values)
    ranks[sorted_indices] = np.arange(1, n + 1)
    sorted_p = p_values[sorted_indices]
    fdr_p = sorted_p * n / np.arange(1, n + 1)
    for i in range(n - 2, -1, -1):
        fdr_p[i] = min(fdr_p[i], fdr_p[i + 1])
    fdr_p = np.minimum(fdr_p, 1.0)
    unsorted_fdr_p = np.empty(n, dtype=float)
    unsorted_fdr_p[sorted_indices] = fdr_p
    return unsorted_fdr_p


def calculate_hc3_se(X: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """Calculate HC3 robust standard errors."""
    n, p = X.shape
    inv_xtx = np.linalg.inv(X.T @ X)
    H = X @ inv_xtx @ X.T
    h = np.diag(H)
    
    # HC3: e_i / (1 - h_i)
    # Clip h_i to prevent division by zero, though h_i < 1 normally
    h = np.clip(h, 0, 0.9999)
    adj_residuals = residuals / (1.0 - h)
    
    omega = np.diag(adj_residuals ** 2)
    cov_hc3 = inv_xtx @ (X.T @ omega @ X) @ inv_xtx
    return np.sqrt(np.diag(cov_hc3))


def calculate_cooks_distance(X: np.ndarray, residuals: np.ndarray, mse: float) -> np.ndarray:
    n, p = X.shape
    inv_xtx = np.linalg.inv(X.T @ X)
    H = X @ inv_xtx @ X.T
    h = np.diag(H)
    h = np.clip(h, 0, 0.9999)
    # D_i = (e_i^2 / (p * mse)) * (h_i / (1 - h_i)^2)
    cooks_d = (residuals ** 2 / (p * mse)) * (h / (1.0 - h) ** 2)
    return cooks_d


def fit_ols_details(X: np.ndarray, y: np.ndarray) -> dict:
    n, p = X.shape
    coef, rss_arr, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ coef
    rss = float(np.sum(residuals ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (rss / tss) if tss > 0 else 0.0
    mse = rss / (n - p)
    
    hc3_se = calculate_hc3_se(X, residuals)
    cooks_d = calculate_cooks_distance(X, residuals, mse)
    
    # Standard SE
    inv_xtx = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(mse * inv_xtx))
    
    return {
        "coef": coef,
        "rss": rss,
        "r2": r2,
        "mse": mse,
        "se": se,
        "hc3_se": hc3_se,
        "cooks_d": cooks_d,
        "residuals": residuals
    }


def spearman_partial(x: np.ndarray, y: np.ndarray, cov: np.ndarray) -> tuple[float, float]:
    """Partial spearman correlation of x and y controlling for cov."""
    # Rank inputs
    from scipy.stats import rankdata
    rx = rankdata(x)
    ry = rankdata(y)
    rcov = np.apply_along_axis(rankdata, 0, cov)
    
    # Add intercept to cov
    n = len(x)
    rcov_int = np.column_stack([np.ones(n), rcov])
    
    # Regress rx on rcov
    coef_x, _, _, _ = np.linalg.lstsq(rcov_int, rx, rcond=None)
    resid_x = rx - rcov_int @ coef_x
    
    # Regress ry on rcov
    coef_y, _, _, _ = np.linalg.lstsq(rcov_int, ry, rcond=None)
    resid_y = ry - rcov_int @ coef_y
    
    return spearmanr(resid_x, resid_y)


def main() -> None:
    # 1. Define Variable Mapping
    DOMAIN_MAPPING = [
        {"variable": "ravlt_1", "domain": "verbal_memory", "reverse_code": False},
        {"variable": "ravlt_5", "domain": "verbal_memory", "reverse_code": False},
        {"variable": "ravlt_tot", "domain": "verbal_memory", "reverse_code": False},
        {"variable": "ravlt_imm", "domain": "verbal_memory", "reverse_code": False},
        {"variable": "ravlt_del", "domain": "verbal_memory", "reverse_code": False},
        {"variable": "ravlt_rec", "domain": "verbal_memory", "reverse_code": False},
        {"variable": "ravlt_fp", "domain": "verbal_memory", "reverse_code": True},
        
        {"variable": "ds_forw", "domain": "working_memory", "reverse_code": False},
        {"variable": "ds_back", "domain": "working_memory", "reverse_code": False},
        {"variable": "ds_seq", "domain": "working_memory", "reverse_code": False},
        {"variable": "ds_tot", "domain": "working_memory", "reverse_code": False},
        
        {"variable": "tmt_2", "domain": "executive_function", "reverse_code": True},
        {"variable": "tmt_3", "domain": "executive_function", "reverse_code": True},
        {"variable": "tmt_4", "domain": "executive_function", "reverse_code": True},
        {"variable": "cw_1", "domain": "executive_function", "reverse_code": True},
        {"variable": "cw_2", "domain": "executive_function", "reverse_code": True},
        {"variable": "cw_3", "domain": "executive_function", "reverse_code": True},
        {"variable": "cw_4", "domain": "executive_function", "reverse_code": True},
        
        {"variable": "vf_1", "domain": "verbal_fluency", "reverse_code": False},
        {"variable": "vf_2", "domain": "verbal_fluency", "reverse_code": False},
        {"variable": "vf_3", "domain": "verbal_fluency", "reverse_code": False},
    ]
    
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    mapping_df = pd.DataFrame(DOMAIN_MAPPING)
    mapping_path = RESULTS_TABLES_DIR / "cognitive_composite_variable_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False)
    
    # 2. Load data
    projection_path = EXTERNAL_RESULTS_DIR / "ds003775_external_projection.csv"
    proj_df = pd.read_csv(projection_path)
    
    participants_path = ROOT / "data" / "external" / "ds003775" / "participants.tsv"
    participants = pd.read_csv(participants_path, sep='\t')
    
    df = pd.merge(proj_df, participants, on="participant_id", suffixes=('', '_drop'))
    df = df.loc[:, ~df.columns.str.endswith('_drop')]
    
    # Prepare basic covariates
    df["age_num"] = pd.to_numeric(df["age"], errors="coerce")
    sex_str = df["sex"].astype(str).str.upper()
    df["is_female"] = (sex_str.str[0] == "F").astype(float)
    df["is_male"] = (sex_str.str[0] == "M").astype(float)
    valid_mask = df["age_num"].notna() & df["T"].notna() & ((df["is_female"] + df["is_male"]) > 0)
    df = df[valid_mask].copy().reset_index(drop=True)
    
    # 3. Build composites
    for mapping in DOMAIN_MAPPING:
        var = mapping["variable"]
        if var in df.columns:
            df[f"{var}_num"] = pd.to_numeric(df[var], errors="coerce")
    
    composites = mapping_df["domain"].unique()
    for domain in composites:
        domain_vars = mapping_df[mapping_df["domain"] == domain]
        z_scores = []
        for _, row in domain_vars.iterrows():
            var = f"{row['variable']}_num"
            if var not in df.columns:
                continue
            
            val = df[var].copy()
            # z-score
            valid = val.notna()
            if valid.sum() > 0:
                mean_val = val[valid].mean()
                sd_val = val[valid].std(ddof=1)
                z = (val - mean_val) / sd_val if sd_val > 0 else (val - mean_val)
                if row["reverse_code"]:
                    z = -z
                z_scores.append(z)
                
        if not z_scores:
            df[f"comp_{domain}"] = np.nan
            continue
            
        z_matrix = pd.concat(z_scores, axis=1)
        valid_count = z_matrix.notna().sum(axis=1)
        required_count = len(z_scores) / 2.0
        
        comp_scores = z_matrix.mean(axis=1)
        comp_scores[valid_count < required_count] = np.nan
        df[f"comp_{domain}"] = comp_scores

    # 4. Nested Regressions
    results = []
    
    for domain in composites:
        comp_var = f"comp_{domain}"
        if comp_var not in df.columns:
            continue
            
        sub_df = df[df[comp_var].notna()].copy()
        n_samples = len(sub_df)
        if n_samples < 20:
            continue
            
        y = sub_df[comp_var].to_numpy(float)
        age = sub_df["age_num"].to_numpy(float)
        sex_f = sub_df["is_female"].to_numpy(float)
        T = sub_df["T"].to_numpy(float)
        
        # Standardize
        y_std = (y - np.mean(y)) / np.std(y, ddof=1)
        age_std = (age - np.mean(age)) / np.std(age, ddof=1)
        sex_f_std = (sex_f - np.mean(sex_f)) / np.std(sex_f, ddof=1)
        T_std = (T - np.mean(T)) / np.std(T, ddof=1)
        
        # Base Model
        X1 = np.column_stack([np.ones(n_samples), age_std, sex_f_std])
        m1 = fit_ols_details(X1, y_std)
        p1 = X1.shape[1]
        
        # Extended Model
        X2 = np.column_stack([np.ones(n_samples), age_std, sex_f_std, T_std])
        m2 = fit_ols_details(X2, y_std)
        p2 = X2.shape[1]
        
        # Nested F-test
        df_num = p2 - p1
        df_den = n_samples - p2
        f_stat = ((m1["rss"] - m2["rss"]) / df_num) / (m2["rss"] / df_den)
        p_val = f.sf(f_stat, df_num, df_den)
        
        beta_T = m2["coef"][3]
        t_stat = beta_T / m2["se"][3]
        
        # Sensitivity: HC3
        t_stat_hc3 = beta_T / m2["hc3_se"][3]
        # p-value for HC3
        from scipy.stats import t as t_dist
        p_val_hc3 = 2 * t_dist.sf(np.abs(t_stat_hc3), df_den)
        
        # Sensitivity: Cook's distance exclusion
        cooks_mask = m2["cooks_d"] <= (4.0 / n_samples)
        n_retained = cooks_mask.sum()
        if n_retained > p2:
            X2_clean = X2[cooks_mask]
            y_clean = y_std[cooks_mask]
            # Refit to check if direction/significance holds
            m2_clean = fit_ols_details(X2_clean, y_clean)
            beta_T_clean = m2_clean["coef"][3]
            t_stat_clean = beta_T_clean / m2_clean["se"][3]
            p_val_clean = 2 * t_dist.sf(np.abs(t_stat_clean), n_retained - p2)
        else:
            beta_T_clean = np.nan
            p_val_clean = np.nan
            
        # Sensitivity: Spearman partial
        cov_matrix = np.column_stack([age_std, sex_f_std])
        spearman_rho, spearman_p = spearman_partial(y_std, T_std, cov_matrix)
        
        results.append({
            "domain": domain,
            "n_samples": n_samples,
            "r2_base": m1["r2"],
            "r2_extended": m2["r2"],
            "delta_r2": m2["r2"] - m1["r2"],
            "beta_T_std": beta_T,
            "se_T": m2["se"][3],
            "p_value_T": p_val,  # Using nested F-test p-value which is identical to t-test squared
            "f_stat_nested": f_stat,
            "nested_p_value": p_val,
            # Sensitivity
            "hc3_p_value": p_val_hc3,
            "cooks_retained_n": n_retained,
            "cooks_beta_T": beta_T_clean,
            "cooks_p_value": p_val_clean,
            "spearman_partial_rho": spearman_rho,
            "spearman_partial_p": spearman_p
        })
        
    res_df = pd.DataFrame(results)
    if len(res_df) > 0:
        res_df["fdr_p_value"] = benjamini_hochberg(res_df["nested_p_value"].to_numpy())
        res_df = res_df.sort_values("nested_p_value").reset_index(drop=True)
        
    out_path = RESULTS_TABLES_DIR / "cognitive_composite_results.csv"
    res_df.to_csv(out_path, index=False)
    
    print("\n--- Cognitive Composites Results ---")
    print(f"Evaluated {len(res_df)} domains: {', '.join(res_df['domain'].astype(str))}")
    print("\nMain Results:")
    cols = ["domain", "n_samples", "delta_r2", "beta_T_std", "nested_p_value", "fdr_p_value"]
    print(res_df[cols].to_string(index=False))
    
    print("\nSensitivity Analysis (Direction/Significance Check):")
    sens_cols = ["domain", "hc3_p_value", "cooks_beta_T", "cooks_p_value", "spearman_partial_rho", "spearman_partial_p"]
    print(res_df[sens_cols].to_string(index=False))

    surviving = res_df[res_df["fdr_p_value"] < 0.05]
    if len(surviving) > 0:
        print(f"\n{len(surviving)} domain(s) survived FDR < 0.05.")
    else:
        print("\nNo domains survived FDR < 0.05.")


if __name__ == "__main__":
    main()
