"""
LCNN baseline experiment.

Evaluates the Locally Connected Neural Network on the same datasets and
metrics used by the ESNAS experiments for a fair comparison.

Usage (from project root, using the lcnn venv):
    envs/lcnn/Scripts/python experiments/lcnn.py
"""
import numpy as np
import os
import pickle
import sys
import time

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from src.baselines.lcnn.optimize import optimize_lcnn
from src.baselines.lcnn.lcnn import LCNN
from src.datasets import (
    getDataMGS, getDataLaser, getDataDDE, getDataLorenz, getDataSunspots,
    getDataWater
)
from src.error_metrics import nrmse, nrmse_sunspots, r_squared


def run_experiment(dataset_name, data_loader, num_repeats=5,
                   washout=100, save_dir='results/lcnn',
                   nrmse_func=None, autoregressive=False,
                   popsize=20, max_evals=300):
    """Run LCNN optimization + evaluation on one dataset."""
    if nrmse_func is None:
        nrmse_func = nrmse
    print(f"\n{'=' * 60}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'=' * 60}")

    train_in, train_out, val_in, val_out, test_in, test_out = data_loader()

    print(f"  Input dim: {train_in.shape[1]}, Output dim: {train_out.shape[1]}")
    if autoregressive:
        test_len_str = f"{len(test_in)}x{len(test_in[0])}"
    else:
        test_len_str = str(len(test_in))
    print(f"  Washout: {washout}, Train: {len(train_in)}, "
          f"Val: {len(val_in)}, Test: {test_len_str}")
    print(f"  Autoregressive: {autoregressive}")
    print(f"  CMA-ES: popsize={popsize}, max_evals={max_evals}")
    print(f"  Repeats: {num_repeats}")

    dataset_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    nrmse_scores = []
    r2_scores = []
    total_elapsed = 0.0

    for rep in range(num_repeats):
        print(f"\n  --- Repeat {rep + 1}/{num_repeats} ---")
        start = time.time()

        # Optimize on validation set
        best_model, best_params, best_val_error = optimize_lcnn(
            train_in, train_out, val_in, val_out,
            washout=washout,
            autoregressive=autoregressive,
            val_error_func=nrmse_func,
            popsize=popsize,
            max_evals=max_evals,
            seed=rep * 1000,
            verbose=True,
        )

        # Evaluate on test set
        # Re-run train+val with teacher forcing to build up reservoir state,
        # then predict on test.
        best_model.state_ = np.zeros_like(best_model.state_)
        best_model.last_output_ = None
        best_model.run(train_in, train_out, washout=0, teacher_forcing=True)
        rep_nrmse = []
        rep_r2 = []
        if autoregressive:
            for i in range(len(test_in)):
                prev_in = val_in if i == 0 else test_in[i - 1]
                prev_out = val_out if i == 0 else test_out[i - 1]
                best_model.run(prev_in, prev_out, washout=0, teacher_forcing=True)
                preds = best_model.predict_autoregressive(test_in[i][0], steps=len(test_in[i]))
                rep_nrmse.append(nrmse_func(test_out[i], preds))
                rep_r2.append(r_squared(test_out[i], preds))
        else:
            best_model.run(val_in, val_out, washout=0, teacher_forcing=True)
            preds = best_model.predict(test_in)
            rep_nrmse.append(nrmse_func(test_out, preds))
            rep_r2.append(r_squared(test_out, preds))

        nrmse_scores.extend(rep_nrmse)
        r2_scores.extend(rep_r2)

        elapsed = time.time() - start
        total_elapsed += elapsed

        print(f"  Repeat {rep + 1}: ({elapsed:.1f}s)")
        for j, (n, r) in enumerate(zip(rep_nrmse, rep_r2)):
            print(f"    Test {j+1}: NRMSE={n:.6f}, R²={r:.6f}")
        print(f"  Best params: {best_params}")

        # Save checkpoint
        repeat_data = {
            'dataset': dataset_name,
            'params': best_params,
            'washout': washout,
            'nrmse': rep_nrmse,
            'r2': rep_r2,
            'val_error': best_val_error,
            'elapsed': elapsed,
        }
        with open(os.path.join(dataset_dir, f'repeat_{rep}.pkl'), 'wb') as f:
            pickle.dump(repeat_data, f)
        np.save(os.path.join(dataset_dir, 'nrmse_scores.npy'), np.array(nrmse_scores))
        np.save(os.path.join(dataset_dir, 'r2_scores.npy'), np.array(r2_scores))
        print(f"  Checkpoint saved to {dataset_dir}/ ({rep + 1}/{num_repeats})")

    print(f"\n  Total time: {total_elapsed:.1f}s")

    print(f"\n  --- {dataset_name} Summary ---")
    print(f"  NRMSE: {np.mean(nrmse_scores):.6f} +/- {np.std(nrmse_scores):.6f}")
    print(f"  R²:    {np.mean(r2_scores):.6f} +/- {np.std(r2_scores):.6f}")

    return nrmse_scores, r2_scores


if __name__ == '__main__':
    DATASETS = {
        'mgs': getDataMGS,
    }

    NRMSE_OVERRIDES = {
        'sunspots': nrmse_sunspots,
    }

    AUTOREGRESSIVE = {'mgs', 'laser', 'dde', 'lorenz'}

    all_results = {}
    for name, loader in DATASETS.items():
        nrmse_fn = NRMSE_OVERRIDES.get(name)
        nrmses, r2s = run_experiment(
            name, loader, num_repeats=5,
            nrmse_func=nrmse_fn,
            autoregressive=name in AUTOREGRESSIVE,
        )
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
