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

from experiments._seed import set_global_seed
from src.baselines.lcnn.optimize import optimize_lcnn
from src.baselines.lcnn.lcnn import LCNN
from src.datasets import (
    getDataMGS, getDataLaser, getDataDDE, getDataLorenz, getDataSunspots,
    getDataWater
)
from src.error_metrics import nrmse, nrmse_sunspots, r_squared


def _evaluate_lcnn_on_test(model, val_in, val_out,
                           test_in, test_out, warmup_in, warmup_out,
                           autoregressive, nrmse_func):
    """Run trained LCNN on test data and return (nrmses, r2s)."""
    model.state_ = np.zeros_like(model.state_)
    model.last_output_ = None
    rep_nrmse, rep_r2 = [], []
    if autoregressive:
        for i in range(len(test_in)):
            model.run(warmup_in[i], warmup_out[i], washout=0, teacher_forcing=True)
            preds = model.predict_autoregressive(test_in[i][0], steps=len(test_in[i]))
            rep_nrmse.append(nrmse_func(test_out[i], preds))
            rep_r2.append(r_squared(test_out[i], preds))
    else:
        model.run(val_in, val_out, washout=0, teacher_forcing=True)
        preds = model.predict(test_in)
        rep_nrmse.append(nrmse_func(test_out, preds))
        rep_r2.append(r_squared(test_out, preds))
    return rep_nrmse, rep_r2


def run_experiment(dataset_name, data_loader, num_repeats=5,
                   washout=100, base_seed=0, save_dir='results/lcnn',
                   nrmse_func=None, autoregressive=False,
                   popsize=20, max_evals=300):
    """Run LCNN optimization + evaluation on one dataset."""
    if nrmse_func is None:
        nrmse_func = nrmse
    print(f"\n{'=' * 60}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'=' * 60}")

    result = data_loader()
    if len(result) == 8:
        train_in, train_out, val_in, val_out, test_in, test_out, warmup_in, warmup_out = result
    else:
        train_in, train_out, val_in, val_out, test_in, test_out = result
        warmup_in = warmup_out = None

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
        seed = base_seed + rep
        set_global_seed(seed)
        print(f"\n  --- Repeat {rep + 1}/{num_repeats} (seed={seed}) ---")
        start = time.time()

        best_model, best_params, best_val_error = optimize_lcnn(
            train_in, train_out, val_in, val_out,
            washout=washout,
            autoregressive=autoregressive,
            val_error_func=nrmse_func,
            popsize=popsize,
            max_evals=max_evals,
            seed=seed,
            verbose=True,
        )

        rep_nrmse, rep_r2 = _evaluate_lcnn_on_test(
            best_model, val_in, val_out,
            test_in, test_out, warmup_in, warmup_out,
            autoregressive, nrmse_func)

        nrmse_scores.extend(rep_nrmse)
        r2_scores.extend(rep_r2)

        elapsed = time.time() - start
        total_elapsed += elapsed

        print(f"  Repeat {rep + 1}: ({elapsed:.1f}s)")
        for j, (n, r) in enumerate(zip(rep_nrmse, rep_r2)):
            print(f"    Test {j+1}: NRMSE={n:.6f}, R²={r:.6f}")
        print(f"  Best params: {best_params}")

        # Save model separately for later reload/eval
        with open(os.path.join(dataset_dir, f'repeat_{rep}_model.pkl'), 'wb') as f:
            pickle.dump(best_model, f)

        repeat_data = {
            'dataset': dataset_name,
            'seed': seed,
            'params': best_params,
            'washout': washout,
            'autoregressive': autoregressive,
            'data': {
                'train_in': train_in, 'train_out': train_out,
                'val_in': val_in, 'val_out': val_out,
                'test_in': test_in, 'test_out': test_out,
                'warmup_in': warmup_in, 'warmup_out': warmup_out,
            },
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


def evaluate_saved(dataset_name, data_loader, save_dir='results/lcnn',
                   nrmse_func=None, autoregressive=False):
    """Load each saved repeat model and re-evaluate on freshly loaded test data."""
    if nrmse_func is None:
        nrmse_func = nrmse

    result = data_loader()
    if len(result) == 8:
        train_in, train_out, val_in, val_out, test_in, test_out, warmup_in, warmup_out = result
    else:
        train_in, train_out, val_in, val_out, test_in, test_out = result
        warmup_in = warmup_out = None

    dataset_dir = os.path.join(save_dir, dataset_name)
    rep = 0
    all_nrmse, all_r2 = [], []
    while True:
        model_path = os.path.join(dataset_dir, f'repeat_{rep}_model.pkl')
        if not os.path.exists(model_path):
            break
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        rep_nrmse, rep_r2 = _evaluate_lcnn_on_test(
            model, val_in, val_out,
            test_in, test_out, warmup_in, warmup_out,
            autoregressive, nrmse_func)
        print(f"  Repeat {rep}: NRMSE={np.mean(rep_nrmse):.6f}, R2={np.mean(rep_r2):.6f}")
        all_nrmse.extend(rep_nrmse)
        all_r2.extend(rep_r2)
        rep += 1
    return all_nrmse, all_r2


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip training; load saved models and evaluate only')
    args = parser.parse_args()

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

    AUTOREGRESSIVE = {'mgs', 'laser', 'dde', 'lorenz'}

    all_results = {}
    for name, loader in DATASETS.items():
        nrmse_fn = NRMSE_OVERRIDES.get(name)
        if args.eval_only:
            nrmses, r2s = evaluate_saved(
                name, loader, nrmse_func=nrmse_fn,
                autoregressive=name in AUTOREGRESSIVE)
        else:
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
