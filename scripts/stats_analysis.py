import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import traceback


current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

def _derive_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ensure boolean indicator columns exist
    for col in ["has_nvar", "has_rls", "has_lms", "two_stage_readout", "sr_subcritical", "high_order_nvar"]:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].astype(int)
    # Spectral radius bins/supercritical flag
    sr = pd.to_numeric(df.get("spectral_radius_max", np.nan), errors="coerce")
    bins = [-np.inf, 0.9, 1.1, np.inf]
    labels = ["sub", "near", "super"]
    df["sr_bin"] = pd.cut(sr, bins=bins, labels=labels)
    df["sr_supercritical"] = (df["sr_bin"] == "super").astype(int)
    # Numeric cleanup
    numeric_cols = [
        "nrmse", "r2", "final_node_in_dim", "reservoir_units_max", "spectral_radius_max",
        "leak_rate_mean", "nvar_order_max", "nvar_outputs_max", "num_reservoirs",
        "series_length", "num_variables"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _spearman_corr(df: pd.DataFrame, features: list, targets: list) -> pd.DataFrame:
    rows = []
    for t in targets:
        if t not in df.columns:
            continue
        for f in features:
            if f not in df.columns:
                continue
            sub = df[[t, f]].dropna()
            if sub.shape[0] < 8:
                rho, p = np.nan, np.nan
            else:
                rho, p = stats.spearmanr(sub[t], sub[f])
            rows.append({"target": t, "feature": f, "rho": rho, "p": p, "n": int(sub.shape[0])})
    corr_df = pd.DataFrame(rows)
    if not corr_df.empty:
        corr_df["p_fdr"] = multipletests(corr_df["p"].fillna(1.0), method="fdr_bh")[1]
    return corr_df


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size == 0 or b.size == 0:
        return np.nan
    m = np.sum(a[:, None] > b[None, :])
    n = np.sum(a[:, None] < b[None, :])
    denom = float(a.size * b.size)
    return float((m - n) / denom)


def _group_tests(df: pd.DataFrame, binspecs: list, targets: list) -> pd.DataFrame:
    rows = []
    for name, mask in binspecs:
        for t in targets:
            if t not in df.columns:
                continue
            a = df.loc[mask(df), t].dropna().values
            b = df.loc[~mask(df), t].dropna().values
            if a.size < 5 or b.size < 5:
                u, p, delta = np.nan, np.nan, np.nan
            else:
                try:
                    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                except Exception:
                    u, p = np.nan, np.nan
                delta = _cliffs_delta(a, b)
            rows.append({
                "group": name, "target": t, "u": u, "p": p, "p_fdr": np.nan,
                "cliffs_delta": delta, "n_a": int(a.size), "n_b": int(b.size)
            })
    res = pd.DataFrame(rows)
    if not res.empty:
        res["p_fdr"] = multipletests(res["p"].fillna(1.0), method="fdr_bh")[1]
    return res

def run_per_dataset_analysis(analysis_dir: str) -> None:
    df = pd.read_csv(os.path.join(analysis_dir, "models.csv"))

    motif_numeric = [
        "final_node_in_dim", "reservoir_units_max", "spectral_radius_max",
        "leak_rate_mean", "nvar_order_max", "nvar_outputs_max", "num_reservoirs"
    ]
    motif_binary = [
        "has_nvar", "has_rls", "has_lms", "two_stage_readout",
        "sr_subcritical", "sr_supercritical", "high_order_nvar"
    ]

    per_dataset_dir = os.path.join(analysis_dir, "per_dataset")
    os.makedirs(per_dataset_dir, exist_ok=True)

    for ds_name, g in df.groupby("dataset"):
        g = g.copy()
        # Correlations within dataset (FDR per dataset)
        corr_nrmse = _spearman_corr(g, motif_numeric + motif_binary, ["nrmse"])
        corr_nrmse.to_csv(os.path.join(per_dataset_dir, f"motif_correlations_{ds_name}.csv"), index=False)

        # Group tests within dataset
        binspecs = [
            ("sr_subcritical", lambda d: d["sr_subcritical"] == 1),
            ("sr_supercritical", lambda d: d["sr_supercritical"] == 1),
            ("has_nvar", lambda d: d["has_nvar"] == 1),
            ("has_rls", lambda d: d["has_rls"] == 1),
            ("has_lms", lambda d: d["has_lms"] == 1),
            ("two_stage_readout", lambda d: d["two_stage_readout"] == 1),
            ("high_order_nvar", lambda d: d["high_order_nvar"] == 1),
        ]
        group_res = _group_tests(g, binspecs, ["nrmse"])
        group_res.to_csv(os.path.join(per_dataset_dir, f"motif_group_tests_{ds_name}.csv"), index=False)


if __name__ == "__main__":
    analysis_dir = os.path.join(parent_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    try:
        run_per_dataset_analysis(analysis_dir)
        print(f"Saved per-dataset analysis outputs under: {analysis_dir}")
    except Exception as e:
        print(f"Failed per-dataset analysis: {e}")
        traceback.print_exc()


