"""
Grid search baseline evaluation on test data.

Uses the best parameters from:
    "Parameterizing echo state networks for multi-step time series prediction"
    https://doi.org/10.1016/j.neucom.2022.11.044

Evaluates seeds on the proper test split (not validation) and reports
median, mean, and best NRMSE/R2 scores.
"""

import numpy as np
import pickle
import sys
import os

current_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "src", "baselines", "grid_search"))

from reproduce_results import network
from src.datasets import getDataMGS, getDataDDE, getDataLaser, getDataLorenz
from src.error_metrics import nrmse, r_squared

SEEDS_PATH = os.path.join(root_dir, "src", "baselines", "grid_search", "seeds.csv")

# Dataset configs
# params: [n_reservoir, leak_rate, spectral_radius, sparsity_W, sparsity_Win,
#          ridge, input_scaling, trainLen, initLen, seed, param_name, sparsity_val, testLen]
S_T = 2000
DATASETS = {
    "dde": {
        "loader": getDataDDE,
        "params": [256, 0.84, 0.995, 0.91, 1, 8.3e-7, 1, S_T, 300, 0, "sparW", 0.91, 500],
        "dim": 6,
    },
    "lorenz": {
        "loader": getDataLorenz,
        "params": [2048, 0.88, 0, 0, 1, 1.1e-6, 1, S_T, 300, 0, "sparW", 0, 444],
        "dim": 3,
    },
    "mgs": {
        "loader": getDataMGS,
        "params": [2048, 0.68, 1.406, 0.44, 1, 6e-7, 1, S_T, 300, 0, "sparW", 0.44, 286],
        "dim": 1,
    },
    "laser": {
        "loader": getDataLaser,
        "params": [1024, 0.41, 0.906, 0.84, 1, 8.1e-7, 1, S_T, 300, 0, "spawW", 0.84, 100],
        "dim": 1,
    },
}


def run_dataset(dataset_name, seedSet):
    config = DATASETS[dataset_name]
    loader = config["loader"]
    base_params = list(config["params"])
    dim = config["dim"]

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} (dim={dim})")
    print(f"{'='*60}")

    train_in, train_out, val_in, val_out, test_in, test_out = loader()

    # Combine train + val for final training (matches ESNAS final evaluation protocol)
    combined_in = np.concatenate([train_in, val_in], axis=0)
    combined_out = np.concatenate([train_out, val_out], axis=0)

    # Initialize global maskW (same as original code: uses default random state)
    np.random.seed(0)
    maskW = np.random.rand(8192 * 2, 8192 * 2)

    results = []
    for i, seed in enumerate(seedSet):
        params = list(base_params)
        params[9] = int(seed)
        try:
            preds = network(params, dim, combined_in, combined_out, test_in, maskW)
        except Exception as e:
            print(f"  Seed {i}: FAILED ({e})")
            preds = None

        if preds is not None:
            nrmse_val = nrmse(test_out, preds)
            r2_val = r_squared(test_out, preds)
            result = {"nrmse": nrmse_val, "r2": r2_val}
            print(f"  Seed {i}: NRMSE={nrmse_val:.6f}, R2={r2_val:.6f}")
        else:
            result = {"nrmse": np.inf, "r2": -np.inf}
            print(f"  Seed {i}: FAILED")
        results.append(result)

    # Compute statistics
    nrmse_vals = np.array([r["nrmse"] for r in results])
    r2_vals = np.array([r["r2"] for r in results])

    valid = np.isfinite(nrmse_vals)
    nrmse_valid = nrmse_vals[valid]
    r2_valid = r2_vals[valid]

    stats = {
        "dataset": dataset_name,
        "num_seeds": len(seedSet),
        "num_valid": int(valid.sum()),
        "per_seed_results": results,
        "nrmse_median": float(np.median(nrmse_valid)) if len(nrmse_valid) > 0 else None,
        "nrmse_mean": float(np.mean(nrmse_valid)) if len(nrmse_valid) > 0 else None,
        "nrmse_best": float(np.min(nrmse_valid)) if len(nrmse_valid) > 0 else None,
        "r2_median": float(np.median(r2_valid)) if len(r2_valid) > 0 else None,
        "r2_mean": float(np.mean(r2_valid)) if len(r2_valid) > 0 else None,
        "r2_best": float(np.max(r2_valid)) if len(r2_valid) > 0 else None,
    }

    print(f"\n--- {dataset_name} Results ({stats['num_valid']}/{len(seedSet)} valid seeds) ---")
    print(f"  NRMSE  - Median: {stats['nrmse_median']:.6f}, Mean: {stats['nrmse_mean']:.6f}, Best: {stats['nrmse_best']:.6f}")
    print(f"  R2     - Median: {stats['r2_median']:.6f}, Mean: {stats['r2_mean']:.6f}, Best: {stats['r2_best']:.6f}")

    return stats


if __name__ == "__main__":
    save_dir = os.path.join(root_dir, "results", "grid_search")
    os.makedirs(save_dir, exist_ok=True)

    seedSet = np.genfromtxt(SEEDS_PATH)

    if len(sys.argv) > 1:
        # Run specific dataset by index: 0=dde, 1=lorenz, 2=mgs, 3=laser
        dataset_names = ["dde", "lorenz", "mgs", "laser"]
        idx = int(sys.argv[1])
        names_to_run = [dataset_names[idx]]
    else:
        names_to_run = ["dde", "lorenz", "mgs", "laser"]

    all_stats = {}
    for name in names_to_run:
        stats = run_dataset(name, seedSet)
        all_stats[name] = stats

        # Save per-dataset results
        with open(os.path.join(save_dir, f"{name}_results.pkl"), "wb") as f:
            pickle.dump(stats, f)

    # Save combined results
    with open(os.path.join(save_dir, "all_results.pkl"), "wb") as f:
        pickle.dump(all_stats, f)

    print(f"\nResults saved to {save_dir}/")
