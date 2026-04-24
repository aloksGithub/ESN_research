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

from experiments._seed import set_global_seed
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
    result = data_loader()
    if len(result) == 8:
        train_in, train_out, val_in, val_out, test_in, test_out, warmup_in, warmup_out = result
    else:
        train_in, train_out, val_in, val_out, test_in, test_out = result
        warmup_in = warmup_out = None

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


def _evaluate_gedesn_on_test(esn, val_in, test_in, test_out, warmup_in,
                             autoregressive, nrmse_func):
    """Run trained GE-DESN on test data and return (nrmses, r2s)."""
    rep_nrmse, rep_r2 = [], []
    if autoregressive:
        if hasattr(esn, 'PreTestX'):
            for i in range(esn.Stack):
                esn.GroupX[i] = esn.PreTestX[i].copy()
        else:
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
        esn.Validate_test_data_constant(val_in)
        Y_pred, _ = esn.Validate_test_data_constant(test_in)
        rep_nrmse.append(nrmse_func(test_out.T, Y_pred.T))
        rep_r2.append(r_squared(test_out.T, Y_pred.T))
    return rep_nrmse, rep_r2


def run_experiment(dataset_name, data_loader, num_repeats=5,
                   max_layers=8, neurons_per_layer=40, neurons_add=40,
                   washout=100, base_seed=0, save_dir='results/ge_desn',
                   nrmse_func=None, autoregressive=False,
                   optimize=True, opt_maxiter=20, opt_popsize=15):
    """Run GE-DESN on one dataset for num_repeats trials.

    Args:
        optimize: If True, run differential-evolution optimizer on the
            validation set to tune oram parameters before the main trials.
        opt_maxiter: Max iterations for the optimizer.
        opt_popsize: Population size multiplier for differential evolution.
        base_seed: Per-repeat seed = base_seed + rep.
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
        seed = base_seed + rep
        set_global_seed(seed)
        start = time.time()

        rep_pram = pram.copy()
        rep_oram = oram.copy()
        if optimize:
            print(f"\n  --- Optimizing repeat {rep + 1}/{num_repeats} (seed={seed}) ---")
            opt_oram, opt_leaky = optimize_oram(
                U_init, U_train, Y_train, val_in, val_out,
                rep_pram, rep_oram, autoregressive=autoregressive,
                repeats=3, maxiter=opt_maxiter, popsize=opt_popsize,
                seed=seed)
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

        esn = result['esn']
        # Save ESN BEFORE eval so reloaded state matches training-time eval
        with open(os.path.join(dataset_dir, f'repeat_{rep}_model.pkl'), 'wb') as f:
            pickle.dump(esn, f)

        rep_nrmse, rep_r2 = _evaluate_gedesn_on_test(
            esn, val_in, test_in, test_out, warmup_in, autoregressive, nrmse_func)

        nrmse_scores.extend(rep_nrmse)
        r2_scores.extend(rep_r2)

        print(f"\n  --- Repeat {rep + 1}/{num_repeats} ({elapsed:.1f}s) ---")
        for j, (n, r) in enumerate(zip(rep_nrmse, rep_r2)):
            print(f"    Test {j+1}: NRMSE={n:.6f}, R²={r:.6f}")
        for i, (nl, ms) in enumerate(zip(result['nrmse_per_layer'],
                                          result['max_similarity_per_layer'])):
            print(f"    Layer {i}: NRMSE={nl:.6f}, MaxSimilarity={ms:.6f}")

        result_no_esn = {k: v for k, v in result.items() if k != 'esn'}
        repeat_data = {
            'dataset': dataset_name,
            'seed': seed,
            'pram': rep_pram,
            'oram': rep_oram,
            'washout': washout,
            'autoregressive': autoregressive,
            'data': {
                'train_in': train_in, 'train_out': train_out,
                'val_in': val_in, 'val_out': val_out,
                'test_in': test_in, 'test_out': test_out,
                'warmup_in': warmup_in,
            },
            'nrmse': rep_nrmse,
            'r2': rep_r2,
            'elapsed': elapsed,
            'result': result_no_esn,
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


def evaluate_saved(dataset_name, data_loader, save_dir='results/ge_desn',
                   nrmse_func=None, autoregressive=False):
    """Load each saved repeat model and re-evaluate on freshly loaded test data."""
    if nrmse_func is None:
        nrmse_func = nrmse

    _, _, val_in, _, test_in, test_out, warmup_in, _, _ = prepare_data(data_loader)

    dataset_dir = os.path.join(save_dir, dataset_name)
    rep = 0
    all_nrmse, all_r2 = [], []
    while True:
        model_path = os.path.join(dataset_dir, f'repeat_{rep}_model.pkl')
        if not os.path.exists(model_path):
            break
        with open(model_path, 'rb') as f:
            esn = pickle.load(f)
        rep_nrmse, rep_r2 = _evaluate_gedesn_on_test(
            esn, val_in, test_in, test_out, warmup_in, autoregressive, nrmse_func)
        print(f"  Repeat {rep}: NRMSE={np.mean(rep_nrmse):.6f}, R2={np.mean(rep_r2):.6f}")
        all_nrmse.extend(rep_nrmse)
        all_r2.extend(rep_r2)
        rep += 1
    return all_nrmse, all_r2


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='GE-DESN baseline experiment')
    parser.add_argument('--no-optimize', action='store_true',
                        help='Skip hyperparameter optimization (optimization '
                             'is on by default, tuning ampWi, ampWp, ampWr, '
                             'leaky_rate, reg_fac on the validation set)')
    parser.add_argument('--opt-maxiter', type=int, default=40,
                        help='Max iterations for optimizer (default: 40)')
    parser.add_argument('--opt-popsize', type=int, default=15,
                        help='Population size multiplier (default: 15)')
    parser.add_argument('--base-seed', type=int, default=0,
                        help='Base seed (rep seed = base_seed + rep)')
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip training; load saved models and evaluate only')
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
        if args.eval_only:
            nrmses, r2s = evaluate_saved(
                name, loader, nrmse_func=nrmse_fn,
                autoregressive=name in AUTOREGRESSIVE)
        else:
            nrmses, r2s = run_experiment(name, loader, num_repeats=5,
                                         nrmse_func=nrmse_fn,
                                         autoregressive=name in AUTOREGRESSIVE,
                                         optimize=not args.no_optimize,
                                         opt_maxiter=args.opt_maxiter,
                                         opt_popsize=args.opt_popsize,
                                         base_seed=args.base_seed)
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
