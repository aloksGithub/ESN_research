"""
GE-DESN baseline experiment.

Evaluates the Growing Evolutional Deep ESN on the same datasets and metrics
used by the ESNAS experiments for a fair comparison.

Usage (from project root, using the ge_desn venv):
    envs/ge_desn/Scripts/python experiments/ge_desn.py
"""
import numpy as np
import os
import pickle
import sys
import time

# Add project root to path so we can import src modules
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from src.baselines.ge_desn.esn import run_ge_desn
from src.datasets import (
    getDataMGS, getDataLaser, getDataDDE, getDataLorenz, getDataSunspots,
    getDataWater
)
from src.error_metrics import nrmse, nrmse_sunspots, r_squared


def prepare_data(data_loader):
    """Convert framework data format (samples, features) to GE-DESN format (features, samples).

    Framework datasets return arrays of shape (num_samples, num_features).
    GE-DESN expects shape (num_features, num_samples) — columns are timesteps.
    """
    train_in, train_out, val_in, val_out, test_in, test_out = data_loader()

    # Combine train + val for the training set, test stays separate
    # GE-DESN uses its own washout from the beginning of the training data
    combined_in = np.concatenate([train_in, val_in], axis=0)
    combined_out = np.concatenate([train_out, val_out], axis=0)

    input_dim = combined_in.shape[1]
    output_dim = combined_out.shape[1]

    # Transpose to (features, timesteps)
    all_in = combined_in.T    # (input_dim, train+val_len)
    all_out = combined_out.T  # (output_dim, train+val_len)
    t_in = test_in.T
    t_out = test_out.T

    return all_in, all_out, t_in, t_out, input_dim, output_dim


def run_experiment(dataset_name, data_loader, num_repeats=5,
                   max_layers=4, neurons_per_layer=40, neurons_add=40,
                   washout=100, save_dir='results/ge_desn',
                   nrmse_func=None,
                   min_improvement=0.01, patience=1):
    """Run GE-DESN on one dataset for num_repeats trials."""
    if nrmse_func is None:
        nrmse_func = nrmse
    print(f"\n{'=' * 60}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'=' * 60}")

    all_in, all_out, test_in, test_out, input_dim, output_dim = prepare_data(data_loader)
    total_train_len = all_in.shape[1]

    # Split off washout from training data
    U_init = all_in[:, :washout]
    U_train = all_in[:, washout:]
    Y_train = all_out[:, washout:]

    print(f"  Input dim: {input_dim}, Output dim: {output_dim}")
    print(f"  Washout: {washout}, Train: {U_train.shape[1]}, Test: {test_in.shape[1]}")
    print(f"  Layers: {max_layers}, Neurons/layer: {neurons_per_layer}, "
          f"Extra neurons: {neurons_add}")
    print(f"  Repeats: {num_repeats}")

    pram = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'leaky_rate': 0.92,
        'max_layers': max_layers,
        'neurons_per_layer': [neurons_per_layer] * 16,
        'neurons_add': neurons_add,
    }

    oram = {
        'spare_rate': 1.0,
        'ampWi': 1.2853033,
        'ampWp': 0.53432484,
        'ampWr': 0.8,
        'reg_fac': 1e-10,
        'similarity_method': 0,
        'Q2': 0,
    }

    print(f"  Early stopping: min_improvement={min_improvement}, patience={patience}")

    dataset_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    nrmse_scores = []
    r2_scores = []
    y_true_T = test_out.T
    total_elapsed = 0.0

    for rep in range(num_repeats):
        start = time.time()
        result = run_ge_desn(
            U_init, U_train, Y_train, test_in, test_out,
            pram, oram,
            min_improvement=min_improvement, patience=patience,
        )
        elapsed = time.time() - start
        total_elapsed += elapsed

        Y_pred = result['Y_pred']
        y_pred_T = Y_pred.T

        nrmse_val = nrmse_func(y_true_T, y_pred_T)
        r2_val = r_squared(y_true_T, y_pred_T)

        nrmse_scores.append(nrmse_val)
        r2_scores.append(r2_val)

        print(f"\n  --- Repeat {rep + 1}/{num_repeats} ({elapsed:.1f}s) ---")
        print(f"  NRMSE: {nrmse_val:.6f}, R²: {r2_val:.6f}")
        for i, (nl, ms) in enumerate(zip(result['nrmse_per_layer'],
                                          result['max_similarity_per_layer'])):
            print(f"    Layer {i}: NRMSE={nl:.6f}, MaxSimilarity={ms:.6f}")

        # Save this repeat's result to its own pickle
        repeat_data = {
            'dataset': dataset_name,
            'pram': pram,
            'oram': oram,
            'min_improvement': min_improvement,
            'patience': patience,
            'washout': washout,
            'nrmse': nrmse_val,
            'r2': r2_val,
            'elapsed': elapsed,
            'result': result,
        }
        with open(os.path.join(dataset_dir, f'repeat_{rep}.pkl'), 'wb') as f:
            pickle.dump(repeat_data, f)
        np.save(os.path.join(dataset_dir, 'nrmse_scores.npy'), np.array(nrmse_scores))
        np.save(os.path.join(dataset_dir, 'r2_scores.npy'), np.array(r2_scores))
        print(f"  Checkpoint saved to {dataset_dir}/ ({rep + 1}/{num_repeats} repeats)")

    print(f"\n  Total time: {total_elapsed:.1f}s")

    # Summary
    print(f"\n  --- {dataset_name} Summary ---")
    print(f"  NRMSE: {np.mean(nrmse_scores):.6f} +/- {np.std(nrmse_scores):.6f}")
    print(f"  R²:    {np.mean(r2_scores):.6f} +/- {np.std(r2_scores):.6f}")

    return nrmse_scores, r2_scores


if __name__ == '__main__':
    DATASETS = {
        'mgs': getDataMGS,
        'laser': getDataLaser,
        'dde': getDataDDE,
        'lorenz': getDataLorenz,
        'sunspots': getDataSunspots,
        'water': getDataWater,
    }

    NRMSE_OVERRIDES = {
        'sunspots': nrmse_sunspots,
    }

    all_results = {}
    for name, loader in DATASETS.items():
        nrmse_fn = NRMSE_OVERRIDES.get(name)
        nrmses, r2s = run_experiment(name, loader, num_repeats=5,
                                     nrmse_func=nrmse_fn)
        all_results[name] = {'nrmse': nrmses, 'r2': r2s}

    # Final summary table
    print(f"\n{'=' * 60}")
    print("  FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"  {'Dataset':<12} {'NRMSE':>20} {'R²':>20}")
    print(f"  {'-' * 52}")
    for name, res in all_results.items():
        nm = f"{np.mean(res['nrmse']):.4f} +/- {np.std(res['nrmse']):.4f}"
        r2 = f"{np.mean(res['r2']):.4f} +/- {np.std(res['r2']):.4f}"
        print(f"  {name:<12} {nm:>20} {r2:>20}")
