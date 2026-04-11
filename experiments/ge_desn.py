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
from src.baselines.ge_desn.optimize import optimize_oram
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
    train_in, train_out, val_in, val_out, test_in, test_out, warmup_in, warmup_out = data_loader()

    input_dim = train_in.shape[1]
    output_dim = train_out.shape[1]

    # Transpose to (features, timesteps)
    tr_in = train_in.T
    tr_out = train_out.T
    v_in = val_in.T
    v_out = val_out.T
    if isinstance(test_in, list):
        t_in = [t.T for t in test_in]
        t_out = [t.T for t in test_out]
        w_in = [w.T for w in warmup_in]
    else:
        t_in = test_in.T
        t_out = test_out.T
        w_in = None

    return tr_in, tr_out, v_in, v_out, t_in, t_out, w_in, input_dim, output_dim


def run_experiment(dataset_name, data_loader, num_repeats=5,
                   max_layers=8, neurons_per_layer=40, neurons_add=40,
                   washout=100, save_dir='results/ge_desn',
                   nrmse_func=None, autoregressive=False,
                   optimize=True, opt_maxiter=20, opt_popsize=15,
                   opt_seed=None):
    """Run GE-DESN on one dataset for num_repeats trials.

    Args:
        optimize: If True, run differential-evolution optimizer on the
            validation set to tune oram parameters before the main trials.
        opt_maxiter: Max iterations for the optimizer.
        opt_popsize: Population size multiplier for differential evolution.
        opt_seed: Random seed for optimizer reproducibility.
    """
    if nrmse_func is None:
        nrmse_func = nrmse
    print(f"\n{'=' * 60}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'=' * 60}")

    train_in, train_out, val_in, val_out, test_in, test_out, warmup_in, input_dim, output_dim = prepare_data(data_loader)

    # Split off washout from training data
    U_init = train_in[:, :washout]
    U_train = train_in[:, washout:]
    Y_train = train_out[:, washout:]

    print(f"  Input dim: {input_dim}, Output dim: {output_dim}")
    if autoregressive:
        test_len_str = f"{len(test_in)}x{test_in[0].shape[1]}"
    else:
        test_len_str = str(test_in.shape[1])
    print(f"  Washout: {washout}, Train: {U_train.shape[1]}, "
          f"Val: {val_in.shape[1]}, Test: {test_len_str}")
    print(f"  Layers: {max_layers}, Neurons/layer: {neurons_per_layer}, "
          f"Extra neurons: {neurons_add}")
    print(f"  Repeats: {num_repeats}, Optimize: {optimize}")

    pram = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'leaky_rate': 0.7,
        'max_layers': max_layers,
        'neurons_per_layer': [neurons_per_layer] * 16,
        'neurons_add': neurons_add,
    }

    oram = {
        'spare_rate': 1.0,
        'ampWi': 0.5,
        'ampWp': 0.5,
        'ampWr': 0.99,
        'reg_fac': 1e-6,
        'similarity_method': 0,
        'Q2': 0,
    }

    dataset_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    nrmse_scores = []
    r2_scores = []
    total_elapsed = 0.0

    for rep in range(num_repeats):
        start = time.time()

        # Re-optimize per repeat so each trial gets its own tuned params
        rep_pram = pram.copy()
        rep_oram = oram.copy()
        if optimize:
            print(f"\n  --- Optimizing repeat {rep + 1}/{num_repeats} ---")
            opt_oram, opt_leaky = optimize_oram(
                U_init, U_train, Y_train, val_in, val_out,
                rep_pram, rep_oram, autoregressive=autoregressive,
                repeats=3, maxiter=opt_maxiter, popsize=opt_popsize,
                seed=opt_seed)
            rep_oram.update(opt_oram)
            rep_pram['leaky_rate'] = opt_leaky

        # Pass first test set (or single test set) to run_ge_desn
        t_in_arg = test_in[0] if autoregressive else test_in
        t_out_arg = test_out[0] if autoregressive else test_out
        result = run_ge_desn(
            U_init, U_train, Y_train, val_in, val_out,
            t_in_arg, t_out_arg,
            rep_pram, rep_oram, autoregressive=autoregressive,
        )
        elapsed = time.time() - start
        total_elapsed += elapsed

        rep_nrmse = []
        rep_r2 = []
        if autoregressive:
            esn = result['esn']
            esn.Init_reservior(esn.U_init)
            esn.Train_reservoir(esn.U_train, esn.Y_train)
            esn.Reinit_reservoir()
            for j in range(esn.U_train.shape[1]):
                esn.UspanX(esn.U_train[:, j:j + 1], esn.galaph)
            for i in range(len(test_in)):
                esn.Validate_test_data_constant(warmup_in[i])
                Y_pred = esn.Validate_test_data_autoregressive(
                    test_in[i][:, 0:1], test_in[i].shape[1])
                rep_nrmse.append(nrmse_func(test_out[i].T, Y_pred.T))
                rep_r2.append(r_squared(test_out[i].T, Y_pred.T))
        else:
            Y_pred = result['Y_pred']
            y_true_T = test_out.T
            rep_nrmse.append(nrmse_func(y_true_T, Y_pred.T))
            rep_r2.append(r_squared(y_true_T, Y_pred.T))

        nrmse_scores.extend(rep_nrmse)
        r2_scores.extend(rep_r2)

        print(f"\n  --- Repeat {rep + 1}/{num_repeats} ({elapsed:.1f}s) ---")
        for j, (n, r) in enumerate(zip(rep_nrmse, rep_r2)):
            print(f"    Test {j+1}: NRMSE={n:.6f}, R²={r:.6f}")
        for i, (nl, ms) in enumerate(zip(result['nrmse_per_layer'],
                                          result['max_similarity_per_layer'])):
            print(f"    Layer {i}: NRMSE={nl:.6f}, MaxSimilarity={ms:.6f}")

        # Save this repeat's result to its own pickle
        repeat_data = {
            'dataset': dataset_name,
            'pram': rep_pram,
            'oram': rep_oram,
            'washout': washout,
            'nrmse': rep_nrmse,
            'r2': rep_r2,
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
    import argparse

    parser = argparse.ArgumentParser(description='GE-DESN baseline experiment')
    parser.add_argument('--no-optimize', action='store_true',
                        help='Skip hyperparameter optimization (optimization '
                             'is on by default, tuning ampWi, ampWp, ampWr, '
                             'leaky_rate, reg_fac on the validation set)')
    parser.add_argument('--opt-maxiter', type=int, default=20,
                        help='Max iterations for optimizer (default: 20)')
    parser.add_argument('--opt-popsize', type=int, default=15,
                        help='Population size multiplier (default: 15)')
    parser.add_argument('--opt-seed', type=int, default=None,
                        help='Random seed for optimizer')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Specific datasets to run (default: all)')
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

    if args.datasets:
        datasets_to_run = {k: v for k, v in DATASETS.items()
                           if k in args.datasets}
    else:
        datasets_to_run = DATASETS

    all_results = {}
    for name, loader in datasets_to_run.items():
        nrmse_fn = NRMSE_OVERRIDES.get(name)
        nrmses, r2s = run_experiment(name, loader, num_repeats=5,
                                     nrmse_func=nrmse_fn,
                                     autoregressive=name in AUTOREGRESSIVE,
                                     optimize=not args.no_optimize,
                                     opt_maxiter=args.opt_maxiter,
                                     opt_popsize=args.opt_popsize,
                                     opt_seed=args.opt_seed)
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
