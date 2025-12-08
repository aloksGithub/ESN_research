import os
import sys
import pickle
from typing import Dict, List, Tuple, Iterable, Union

import numpy as np
import pandas as pd
from scipy import stats

# Silence reservoirpy if imported indirectly
try:
    import reservoirpy as rpy  # type: ignore
    rpy.verbosity(0)
except Exception:
    pass

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.algorithms.ESN_GA import ESN_GA  # noqa: E402
from src.algorithms.ESN_BO import ESN_BO  # noqa: E402
from src.utils import runModel  # noqa: E402


def _read_saved(path: str) -> Union[ESN_GA, ESN_BO]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _compute_metrics(obj: Union[ESN_GA, ESN_BO], is_autoregressive: bool) -> Tuple[float, float]:
    model = obj.bestModel
    if is_autoregressive:
        # Autoregressive multi-step on validation set
        runModel(model, obj.experimentData.trainX)
        prev_out = obj.experimentData.valX[0]
        preds: List[np.ndarray] = []
        for _ in range(len(obj.experimentData.valX)):
            pred = runModel(model, prev_out)
            prev_out = pred
            preds.append(pred[0])
        preds_np = np.array(preds)
        nrmse = obj.evalParams.errorMetrics[0](obj.experimentData.valY, preds_np)
        r2 = obj.evalParams.errorMetrics[1](obj.experimentData.valY, preds_np)
    else:
        # Next-step on held-out test set
        runModel(model, obj.experimentData.valX)
        preds_np = runModel(model, obj.experimentData.testX)
        nrmse = obj.evalParams.errorMetrics[0](obj.experimentData.testY, preds_np)
        r2 = obj.evalParams.errorMetrics[1](obj.experimentData.testY, preds_np)
    return float(nrmse), float(r2)


def _load_method_metrics(results_dir: str, dataset: str, is_autoregressive: bool) -> Tuple[List[float], List[float]]:
    nrmse_vals: List[float] = []
    r2_vals: List[float] = []
    for i in range(5):
        path = os.path.join(results_dir, dataset, f"backup_{i}.obj")
        if not os.path.exists(path):
            continue
        try:
            obj = _read_saved(path)
            nrmse, r2 = _compute_metrics(obj, is_autoregressive)
            nrmse_vals.append(nrmse)
            r2_vals.append(r2)
        except Exception:
            # Skip corrupted/mismatched files gracefully
            continue
    return nrmse_vals, r2_vals


def _has_any_results(results_root: str, datasets: Iterable[str]) -> bool:
    for ds in datasets:
        for i in range(5):
            if os.path.exists(os.path.join(results_root, ds, f"backup_{i}.obj")):
                return True
    return False


def _rank_biserial_from_diffs(diffs: np.ndarray) -> float:
    # Ignore zeros (ties)
    nz = diffs != 0
    diffs = diffs[nz]
    if diffs.size == 0:
        return float("nan")
    ranks = stats.rankdata(np.abs(diffs), method="average")
    r_pos = np.sum(ranks[diffs > 0])
    r_neg = np.sum(ranks[diffs < 0])
    denom = float(ranks.sum())
    if denom == 0:
        return float("nan")
    return float((r_pos - r_neg) / denom)


def _paired_wilcoxon(x: List[float], y: List[float], metric: str) -> Dict[str, float]:
    # Align to min length
    n = min(len(x), len(y))
    if n == 0:
        return {"n": 0, "stat": np.nan, "p_two_sided": np.nan, "p_better": np.nan, "rb": np.nan}
    x_arr = np.asarray(x[:n], dtype=float)
    y_arr = np.asarray(y[:n], dtype=float)
    # Two-sided
    try:
        stat, p_two = stats.wilcoxon(x_arr, y_arr, alternative="two-sided", zero_method="wilcox", correction=False, mode="auto")
    except Exception:
        stat, p_two = np.nan, np.nan
    # Directional: ESNAS better
    if metric.lower() == "nrmse":
        alt = "less"  # ESNAS < baseline
    else:
        alt = "greater"  # ESNAS > baseline (e.g., R2)
    try:
        _, p_dir = stats.wilcoxon(x_arr, y_arr, alternative=alt, zero_method="wilcox", correction=False, mode="auto")
    except Exception:
        p_dir = np.nan
    # Effect size (rank-biserial) using diffs of (ESNAS - baseline) for interpretability
    diffs = x_arr - y_arr
    rb = _rank_biserial_from_diffs(diffs)
    return {
        "n": int(n),
        "stat": float(stat) if stat is not None else np.nan,
        "p_two_sided": float(p_two) if p_two is not None else np.nan,
        "p_better": float(p_dir) if p_dir is not None else np.nan,
        "rb": rb,
    }


def main() -> None:
    # Dataset settings: True => autoregressive validation; False => next-step test
    ds_autoreg: Dict[str, bool] = {
        "mgs": True,
        "lorenz": True,
        "dde": True,
        "laser": True,
        "sunspots": False,
        "water": False,
    }

    roots = {
        "ESNAS": os.path.join(parent_dir, "results", "esnas"),
        "GA-only": os.path.join(parent_dir, "results", "ga"),
        "BO-only": os.path.join(parent_dir, "results", "bo"),
    }

    rows: List[Dict[str, object]] = []
    for ds, is_autoreg in ds_autoreg.items():
        # Load ESNAS
        esnas_nrmse, esnas_r2 = _load_method_metrics(roots["ESNAS"], ds, is_autoreg)
        # Compare with each ablation
        for baseline in ["GA-only", "BO-only"]:
            base_nrmse, base_r2 = _load_method_metrics(roots[baseline], ds, is_autoreg)
            # NRMSE (minimize) — use arrays in (ESNAS, Baseline) order
            res_n = _paired_wilcoxon(esnas_nrmse, base_nrmse, metric="nrmse")
            rows.append({
                "dataset": ds,
                "metric": "NRMSE",
                "baseline": baseline,
                "n_pairs": res_n["n"],
                "median_esnas": float(np.median(esnas_nrmse)) if len(esnas_nrmse) else np.nan,
                "median_baseline": float(np.median(base_nrmse)) if len(base_nrmse) else np.nan,
                "median_delta_esnas_minus_baseline": (float(np.median(esnas_nrmse) - np.median(base_nrmse)) if len(esnas_nrmse) and len(base_nrmse) else np.nan),
                "stat": res_n["stat"],
                "p_two_sided": res_n["p_two_sided"],
                "p_esnas_better": res_n["p_better"],
                "rank_biserial": res_n["rb"],
            })

    res_df = pd.DataFrame(rows)
    if res_df.empty:
        print("No paired results found. Ensure results/esnas, results/ga, and results/bo contain backup_*.obj files.")
        return

    # BH-FDR over all p-values in this analysis
    from statsmodels.stats.multitest import multipletests  # type: ignore

    p = res_df["p_two_sided"].fillna(1.0).to_numpy()
    _, q, _, _ = multipletests(p, method="fdr_bh")
    res_df["q_two_sided"] = q

    p = res_df["p_esnas_better"].fillna(1.0).to_numpy()
    _, q, _, _ = multipletests(p, method="fdr_bh")
    res_df["q_esnas_better"] = q

    res_df.to_csv("analysis/wilcoxon_ablation.csv", index=False)


if __name__ == "__main__":
    main()


