"""Build motif-analysis tables from EESNAS search-time validation fitnesses.

This script intentionally uses ``ga.fitnesses`` from the saved search logs.
Those values are validation objectives used by EESNAS during optimization, not
held-out test scores from the final evaluation pipeline.
"""
import sys
import os

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
out_dir = os.path.join(parent_dir, "analysis")
sys.path.insert(0, parent_dir)
import pickle
from src.algorithms.ESN_GA import ESN_GA
import numpy as np
import reservoirpy as rpy
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.utils import constructModel, runModel
rpy.verbosity(0)
import warnings
warnings.filterwarnings("ignore")

binary_motifs = ['has_nvar', 'has_rls', 'has_lms', 'has_ip', 'two_stage_readout', 'sr_subcritical', 'sr_supercritical', 'high_order_nvar']
numeric_motifs = ['final_node_in_dim', 'reservoir_units_max', 'spectral_radius_max', 'leak_rate_max', 'leak_rate_mean', 'nvar_order_max', 'nvar_outputs_max', 'num_reservoirs', 'num_nonnvar_reservoirs', 'num_nvar_reservoirs']

def extract_model_stats(architecture, ga: ESN_GA):
    # Create and run model on tiny input to initialize node input/output dimensions
    model = constructModel(architecture)
    runModel(model, ga.experimentData.trainX[:1])

    reservoir_units_max = 0
    reservoir_units_mean = 0
    spectral_radius_max = 0
    spectral_radius_mean = 0
    leak_rate_max = 0
    leak_rate_mean = 0
    num_nonnvar_reservoirs = 0
    num_nvar_reservoirs = 0
    nvar_order_max = 0
    nvar_outputs_max = 0
    has_rls = False
    has_lms = False
    has_ip = False
    for node in architecture['nodes']:
        if node['type'] == 'Reservoir' or node['type'] == 'IPReservoir':
            reservoir_units_max = max(node['params']['units'], reservoir_units_max)
            spectral_radius_max = max(node['params']['sr'], spectral_radius_max)
            spectral_radius_mean += node['params']['sr']
            reservoir_units_mean += node['params']['units']
            leak_rate_max = max(node['params']['lr'], leak_rate_max)
            leak_rate_mean += node['params']['lr']
            num_nonnvar_reservoirs += 1
        if node['type'] == 'IPReservoir':
            has_ip = True
        if node['type'] == 'NVAR':
            nvar_order_max = max(node['params']['order'], nvar_order_max)
            num_nvar_reservoirs += 1
        if node['type'] == 'RLS':
            has_rls = True
        if node['type'] == 'LMS':
            has_lms = True
    for node in model.nodes:
        if node.name == 'NVAR':
            nvar_outputs_max = max(node.out, nvar_outputs_max)
    
    two_stage_readout = False
    for i in range(len(architecture['nodes']) - 1):
        if architecture['nodes'][i]['type'] in ['RLS', 'LMS', 'Ridge'] and [i, len(architecture['nodes']) - 1] in architecture['edges']:
            two_stage_readout = True
    
    return {
        'final_node_in_dim': model.nodes[-1].input_dim,
        'reservoir_units_max': reservoir_units_max,
        'reservoir_units_mean': reservoir_units_mean / max(1, num_nonnvar_reservoirs),
        'spectral_radius_max': spectral_radius_max,
        'spectral_radius_mean': spectral_radius_mean / max(1, num_nonnvar_reservoirs),
        'leak_rate_max': leak_rate_max,
        'leak_rate_mean': leak_rate_mean / max(1, num_nonnvar_reservoirs),
        'nvar_order_max': nvar_order_max,
        'nvar_outputs_max': nvar_outputs_max,
        'num_reservoirs': num_nonnvar_reservoirs + num_nvar_reservoirs,
        'num_nonnvar_reservoirs': num_nonnvar_reservoirs,
        'has_nvar': num_nvar_reservoirs > 0,
        'has_rls': has_rls,
        'has_lms': has_lms,
        'has_ip': has_ip,
        'two_stage_readout': two_stage_readout,
        'sr_subcritical': spectral_radius_max < 1 and spectral_radius_max > 0,
        'sr_bin': 'super' if spectral_radius_max > 1 else 'sub' if spectral_radius_max < 1 else 'near',
        'sr_supercritical': spectral_radius_max > 1,
        'high_order_nvar': nvar_order_max >= 4,
    }


def build_model_dataset():
    model_dataset = []
    for dataset in ['laser', 'mgs', 'lorenz', 'dde', 'sunspots', 'water']:
        for i in range(5):
            ga: ESN_GA = pickle.load(open("{}/{}/backup_{}.obj".format('results/esnas', dataset, i), "rb"))
            architectures = ga.architectures
            # Search-time validation objectives; do not interpret as held-out test scores.
            fitnesses = [errors[0] for errors in ga.fitnesses]
            sorted_architectures = [x for _, x in sorted(zip(fitnesses, architectures), key=lambda pair: pair[0])]
            sorted_fitnesses = sorted(ga.fitnesses, key=lambda pair: pair[0])
            best_architectures = sorted_architectures[:100]
            for i, architecture in enumerate(best_architectures):
                nrmse = sorted_fitnesses[i][0]
                r2 = sorted_fitnesses[i][1]

                model_info = extract_model_stats(architecture, ga)
                model_info['nrmse'] = nrmse
                model_info['r2'] = r2
                model_info['dataset'] = dataset
                model_dataset.append(model_info)
    
    
    model_df = pd.DataFrame(model_dataset)
    for col in binary_motifs:
        model_df[col] = model_df[col].astype(int)

    model_df.to_csv(os.path.join(out_dir, "models.csv"), index=False)

    return model_df

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

def run_per_dataset_analysis(model_df: pd.DataFrame) -> None:
    per_dataset_dir = os.path.join(out_dir, "per_dataset")
    os.makedirs(per_dataset_dir, exist_ok=True)

    for ds_name, g in model_df.groupby("dataset"):
        g = g.copy()
        # Correlations within dataset (FDR per dataset)
        corr_nrmse = _spearman_corr(g, numeric_motifs, ["nrmse"])
        corr_nrmse.to_csv(os.path.join(per_dataset_dir, f"motif_correlations_{ds_name}.csv"), index=False)

        # Group tests within dataset
        # binspecs = [(motif, lambda d: d[motif] == 1) for motif in binary_motifs]
        binspecs = [
            ("has_nvar", lambda d: d["has_nvar"] == 1),
            ("has_rls", lambda d: d["has_rls"] == 1),
            ("has_lms", lambda d: d["has_lms"] == 1),
            ("has_ip", lambda d: d["has_ip"] == 1),
            ("two_stage_readout", lambda d: d["two_stage_readout"] == 1),
            ("sr_subcritical", lambda d: d["sr_subcritical"] == 1),
            ("sr_supercritical", lambda d: d["sr_supercritical"] == 1),
            ("high_order_nvar", lambda d: d["high_order_nvar"] == 1),
        ]
        group_res = _group_tests(g, binspecs, ["nrmse"])
        group_res.to_csv(os.path.join(per_dataset_dir, f"motif_group_tests_{ds_name}.csv"), index=False)


if __name__ == "__main__":
    # Save both datasets to CSV for downstream analysis
    os.makedirs(out_dir, exist_ok=True)

    # Check if file exists
    models_path = os.path.join(out_dir, "models.csv")
    model_df = pd.read_csv(models_path) if os.path.isfile(models_path) else build_model_dataset()

    run_per_dataset_analysis(model_df)


    
