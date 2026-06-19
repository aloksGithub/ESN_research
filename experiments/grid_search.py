"""
Grid search baseline evaluation on test data.

Uses the best parameters from:
    "Parameterizing echo state networks for multi-step time series prediction"
    https://doi.org/10.1016/j.neucom.2022.11.044

Evaluates seeds on the proper test split (not validation) and reports
median, mean, and best NRMSE/R2 scores.
"""

import argparse
import numpy as np
import pickle
import sys
import os
import torch

current_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "src", "baselines", "grid_search"))

from reproduce_results import pooling, WinPooling
import ESN_Torch as ESN
from scipy.stats import uniform
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


def build_and_train_reservoir(params, dim, train_in, train_out, maskW):
    """Build the reservoir, train it, and return the trained reservoir object."""
    initLen = int(params[8])
    n_reservoir = int(params[0])
    leak_rate = params[1]
    spectral_radius = params[2]
    input_scaling = params[6]
    proba_non_zero_connec_W = params[3]
    regularization_coef = params[5]
    N = n_reservoir
    seed = params[9]

    np.random.seed(int(seed))
    W = np.asarray(uniform.rvs(size=(8192*2, 8192*2)))
    W = pooling(W, int(8192*2/N))
    Win = np.asarray(uniform.rvs(size=(8192*2, dim+1)))
    Win = WinPooling(Win, int(8192*2/N))
    maskWuse = np.asarray(uniform.rvs(size=(N, N)))
    maskWuse = pooling(maskW, int(8192*2/N))

    idx = np.flatnonzero(maskWuse)
    Nso = np.count_nonzero(maskWuse != 0) - int(round(proba_non_zero_connec_W * maskWuse.size))
    if Nso > 0:
        np.put(maskWuse, np.random.choice(idx, size=Nso, replace=False), 0)
    W[maskWuse == 0] = 0
    negMask = np.asarray(uniform.rvs(size=(8192*2, 8192*2)))
    negMask = pooling(negMask, int(8192*2/N))
    W[negMask > .5] *= -1
    neginMask = np.asarray(uniform.rvs(size=(8192*2, dim+1)))
    neginMask = WinPooling(neginMask, int(8192*2/N))
    Win[neginMask > .5] *= -1
    original_spectral_radius = np.max(np.abs(np.linalg.eigvals(W)))

    Win = Win * input_scaling
    if original_spectral_radius != 0:
        W = W * (spectral_radius / original_spectral_radius)

    reservoir = ESN.ESN(lr=leak_rate, W=W, Win=Win, input_bias=True,
                        ridge=regularization_coef, Wfb=None, fbfunc=None)
    if reservoir == 0:
        return None

    reservoir.train(inputs=[train_in], teachers=[train_out],
                    wash_nr_time_step=initLen, verbose=False)
    return reservoir


def warmup_reservoir(reservoir, data):
    """Teacher-forced run through data to update the reservoir's internal state."""
    x = reservoir.x
    di = reservoir.dim_inp
    for t in range(data.shape[0]):
        if reservoir.in_bias:
            u_t = np.concatenate(([1.0], data[t]))
        else:
            u_t = data[t]
        u_t = torch.from_numpy(u_t.astype(reservoir.typefloat)).reshape(di, 1)
        x = (1 - reservoir.lr) * x + reservoir.lr * torch.tanh(
            torch.matmul(reservoir.Win, u_t) + torch.matmul(reservoir.W, x)
        )
    reservoir.x = x


def run_dataset(dataset_name, seedSet):
    config = DATASETS[dataset_name]
    loader = config["loader"]
    base_params = list(config["params"])
    dim = config["dim"]

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} (dim={dim})")
    print(f"{'='*60}")

    train_in, train_out, val_in, val_out, test_in, test_out, warmup_in, _ = loader()
    num_test = len(test_in)

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
            reservoir = build_and_train_reservoir(params, dim, combined_in, combined_out, maskW)
        except Exception as e:
            print(f"  Seed {i}: FAILED ({e})")
            reservoir = None

        if reservoir is not None:
            nrmse_per_set = []
            r2_per_set = []
            for j in range(num_test):
                # Warm up: show val data before first test set, previous test set otherwise
                warmup_reservoir(reservoir, warmup_in[j])

                # Autoregressive prediction on current test set
                output_pred, _ = reservoir.run(inputs=[test_in[j]], reset_state=False)
                preds = output_pred[0]

                nrmse_per_set.append(nrmse(test_out[j], preds))
                r2_per_set.append(r_squared(test_out[j], preds))

            avg_nrmse = float(np.mean(nrmse_per_set))
            avg_r2 = float(np.mean(r2_per_set))
            result = {
                "nrmse": avg_nrmse,
                "r2": avg_r2,
                "nrmse_per_set": nrmse_per_set,
                "r2_per_set": r2_per_set,
            }
            print(f"  Seed {i}: NRMSE={avg_nrmse:.6f}, R2={avg_r2:.6f}")
        else:
            result = {"nrmse": np.inf, "r2": -np.inf, "nrmse_per_set": [], "r2_per_set": []}
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
        "num_test_sets": num_test,
        "per_seed_results": results,
        "nrmse_median": float(np.median(nrmse_valid)) if len(nrmse_valid) > 0 else None,
        "nrmse_mean": float(np.mean(nrmse_valid)) if len(nrmse_valid) > 0 else None,
        "nrmse_best": float(np.min(nrmse_valid)) if len(nrmse_valid) > 0 else None,
        "r2_median": float(np.median(r2_valid)) if len(r2_valid) > 0 else None,
        "r2_mean": float(np.mean(r2_valid)) if len(r2_valid) > 0 else None,
        "r2_best": float(np.max(r2_valid)) if len(r2_valid) > 0 else None,
    }

    print_summary(stats)

    return stats


def print_summary(stats):
    name = stats["dataset"]
    print(f"\n--- {name} Results ({stats['num_valid']}/{stats['num_seeds']} valid seeds) ---")
    print(f"  NRMSE  - Median: {stats['nrmse_median']:.6f}, Mean: {stats['nrmse_mean']:.6f}, Best: {stats['nrmse_best']:.6f}")
    print(f"  R2     - Median: {stats['r2_median']:.6f}, Mean: {stats['r2_mean']:.6f}, Best: {stats['r2_best']:.6f}")


if __name__ == "__main__":
    dataset_names = ["dde", "lorenz", "mgs", "laser"]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_idx", nargs="?", type=int, default=None,
                        help=f"Dataset index: {dict(enumerate(dataset_names))}. If omitted, runs all.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; load saved per-dataset pickles and print summary stats.")
    args = parser.parse_args()

    save_dir = os.path.join(root_dir, "results", "grid_search")

    if args.dataset_idx is not None:
        names_to_run = [dataset_names[args.dataset_idx]]
    else:
        names_to_run = list(dataset_names)

    if args.eval_only:
        all_stats = {}
        for name in names_to_run:
            results_path = os.path.join(save_dir, f"{name}_results.pkl")
            if not os.path.exists(results_path):
                print(f"  {name}: no saved results at {results_path}, skipping")
                continue
            with open(results_path, "rb") as f:
                stats = pickle.load(f)
            all_stats[name] = stats
            print_summary(stats)
        sys.exit(0)

    os.makedirs(save_dir, exist_ok=True)
    seedSet = np.genfromtxt(SEEDS_PATH)

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
