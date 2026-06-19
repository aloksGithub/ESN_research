"""
LSTM baseline experiment.

Runs independent BO searches to find the best LSTM model, then evaluates
directly on the test set — mirroring how EESNAS runs are conducted.

Usage (from project root, using the lstm venv):
    envs/lstm/Scripts/python experiments/lstm.py
"""
import numpy as np
import os
import pickle
import sys
import time
import torch

# Add project root to path so we can import src modules
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from experiments._seed import set_global_seed
from src.baselines.lstm import (
    predict_lstm, predict_lstm_autoregressive,
    optimize_lstm,
)
from src.baselines.lstm.model import LSTMForecaster
from src.datasets import (
    getDataMGS, getDataLaser, getDataDDE, getDataLorenz, getDataSunspots,
    getDataWater
)
from src.error_metrics import nrmse, nrmse_sunspots, r_squared


# ---------------------------------------------------------------------------
# Autoregressive vs next-step dataset config
# ---------------------------------------------------------------------------

# Datasets that use autoregressive multi-step prediction at test time
AUTOREGRESSIVE_DATASETS = {'mgs', 'laser', 'dde', 'lorenz'}

# Datasets that use next-step (teacher-forced) prediction at test time
NEXTSTEP_DATASETS = {'sunspots', 'water'}


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def _mse(y_true, y_pred):
    """MSE for use as AR validation objective in BO (smoother than NRMSE)."""
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def _evaluate_lstm_on_test(model, val_in, test_in, test_out, warmup_in,
                           is_autoregressive, nrmse_func, device):
    """Run trained LSTM on test data and return (nrmses, r2s)."""
    rep_nrmse = []
    rep_r2 = []
    model.eval()
    model = model.to(device)
    if is_autoregressive:
        for i in range(len(test_in)):
            x = torch.tensor(warmup_in[i], dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                _, hidden = model(x, None)

            preds = []
            current = torch.tensor(test_in[i][0], dtype=torch.float32).reshape(1, 1, -1).to(device)
            with torch.no_grad():
                for _ in range(len(test_in[i])):
                    out, hidden = model(current, hidden)
                    pred = out[:, -1, :]
                    preds.append(pred.cpu().numpy().flatten())
                    current = pred.unsqueeze(1)

            y_pred = np.array(preds)
            rep_nrmse.append(nrmse_func(test_out[i], y_pred))
            rep_r2.append(r_squared(test_out[i], y_pred))
    else:
        # Warm hidden state through val, then next-step predict on test
        x_val = torch.tensor(val_in, dtype=torch.float32).unsqueeze(0).to(device)
        x_test = torch.tensor(test_in, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            _, hidden = model(x_val, None)
            pred, _ = model(x_test, hidden)
        y_pred = pred.squeeze(0).cpu().numpy()
        rep_nrmse.append(nrmse_func(test_out, y_pred))
        rep_r2.append(r_squared(test_out, y_pred))
    return rep_nrmse, rep_r2


def run_experiment(dataset_name, data_loader, num_repeats=5,
                   n_init=30, n_iter=2000, bo_patience=100,
                   epochs=200, patience=20, base_seed=0,
                   save_dir='results/lstm', nrmse_func=None, device='cpu',
                   val_noise_sigma=0.0, start_rep=0):
    """Run LSTM BO search + evaluation on one dataset.

    Each repeat runs an independent full BO search and takes the best
    model directly for test evaluation — matching EESNAS methodology.

    Args:
        val_noise_sigma: Std of Gaussian noise injected into warmup data
            during AR val scoring. >0 enables noise-robust val (see
            optimize_lstm). Use ~0.05 for memorization-prone datasets
            (laser/mgs). Keep 0 for chaotic systems (lorenz) where
            warmup noise amplifies and penalizes good models.
    """
    if nrmse_func is None:
        nrmse_func = nrmse

    print(f"\n{'=' * 60}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'=' * 60}")

    result = data_loader()
    if len(result) == 8:
        train_in, train_out, val_in, val_out, test_in, test_out, warmup_in, _ = result
    else:
        train_in, train_out, val_in, val_out, test_in, test_out = result
        warmup_in = None
    input_dim = train_in.shape[1]
    output_dim = train_out.shape[1]
    is_autoregressive = dataset_name in AUTOREGRESSIVE_DATASETS

    print(f"  Input dim: {input_dim}, Output dim: {output_dim}")
    if is_autoregressive:
        test_len_str = f"{len(test_in)}x{len(test_in[0])}"
    else:
        test_len_str = str(len(test_in))
    print(f"  Train: {len(train_in)}, Val: {len(val_in)}, Test: {test_len_str}")
    print(f"  Mode: {'autoregressive' if is_autoregressive else 'next-step'}")
    print(f"  BO: {n_init} init + up to {n_iter} iter (patience={bo_patience}), "
          f"Repeats: {num_repeats}")

    dataset_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    nrmse_path = os.path.join(dataset_dir, 'nrmse_scores.npy')
    r2_path = os.path.join(dataset_dir, 'r2_scores.npy')

    if start_rep > 0:
        if not (os.path.exists(nrmse_path) and os.path.exists(r2_path)):
            raise FileNotFoundError(
                f"--start-rep {start_rep} requires existing {nrmse_path} "
                f"and {r2_path}")
        per_rep = len(test_in) if is_autoregressive else 1
        expected = start_rep * per_rep
        nrmse_scores = np.load(nrmse_path).tolist()
        r2_scores = np.load(r2_path).tolist()
        if len(nrmse_scores) != expected or len(r2_scores) != expected:
            raise ValueError(
                f"{dataset_name}: existing scores have "
                f"{len(nrmse_scores)} entries, expected "
                f"{expected} ({start_rep} reps x {per_rep} per rep). "
                f"Refusing to resume from inconsistent state.")
        print(f"  Resuming from rep {start_rep} "
              f"({len(nrmse_scores)} prior scores loaded)")
    else:
        nrmse_scores = []
        r2_scores = []

    for rep in range(start_rep, num_repeats):
        seed = base_seed + rep
        set_global_seed(seed)
        start = time.time()
        print(f"\n  --- Run {rep + 1}/{num_repeats} (seed={seed}) ---")

        best_model, best_params, best_val_loss, optim_log = optimize_lstm(
            train_in, train_out, val_in, val_out,
            input_dim, output_dim,
            n_init=n_init, n_iter=n_iter, bo_patience=bo_patience,
            epochs=epochs, patience=patience,
            device=device, seed=seed,
            autoregressive=is_autoregressive,
            val_error_func=_mse if is_autoregressive else None,
            val_noise_sigma=val_noise_sigma,
        )

        print(f"  Best params: {best_params}")
        print(f"  Best val MSE: {best_val_loss:.6f}")

        rep_nrmse, rep_r2 = _evaluate_lstm_on_test(
            best_model, val_in, test_in, test_out, warmup_in,
            is_autoregressive, nrmse_func, device)

        nrmse_scores.extend(rep_nrmse)
        r2_scores.extend(rep_r2)

        elapsed = time.time() - start
        print(f"  Repeat {rep + 1}: ({elapsed:.1f}s)")
        for j, (n, r) in enumerate(zip(rep_nrmse, rep_r2)):
            print(f"    Test {j+1}: NRMSE={n:.6f}, R2={r:.6f}")

        # Save model separately so it can be reloaded for test eval
        model_path = os.path.join(dataset_dir, f'repeat_{rep}_model.pt')
        torch.save({
            'state_dict': best_model.state_dict(),
            'input_dim': input_dim,
            'output_dim': output_dim,
            'hidden_size': int(best_params['hidden_size']),
            'num_layers': int(best_params['num_layers']),
            'dropout': float(best_params.get('dropout', 0.0)),
        }, model_path)

        repeat_data = {
            'dataset': dataset_name,
            'seed': seed,
            'input_dim': input_dim,
            'output_dim': output_dim,
            'is_autoregressive': is_autoregressive,
            'bo_settings': {
                'n_init': n_init,
                'n_iter': n_iter,
                'bo_patience': bo_patience,
            },
            'training_settings': {
                'epochs': epochs,
                'patience': patience,
            },
            'data': {
                'train_in': train_in, 'train_out': train_out,
                'val_in': val_in, 'val_out': val_out,
                'test_in': test_in, 'test_out': test_out,
                'warmup_in': warmup_in,
            },
            'params': best_params,
            'val_loss': best_val_loss,
            'nrmse': rep_nrmse,
            'r2': rep_r2,
            'elapsed': elapsed,
            'optim_log': optim_log,
        }
        with open(os.path.join(dataset_dir, f'repeat_{rep}.pkl'), 'wb') as f:
            pickle.dump(repeat_data, f)
        np.save(nrmse_path, np.array(nrmse_scores))
        np.save(r2_path, np.array(r2_scores))
        print(f"  Checkpoint saved to {dataset_dir}/ ({rep + 1}/{num_repeats} repeats)")

    # --- Summary ---
    print(f"\n  --- {dataset_name} Summary ---")
    print(f"  NRMSE: {np.mean(nrmse_scores):.6f} +/- {np.std(nrmse_scores):.6f}")
    print(f"  R2:    {np.mean(r2_scores):.6f} +/- {np.std(r2_scores):.6f}")

    return nrmse_scores, r2_scores


def evaluate_saved(dataset_name, data_loader, save_dir='results/lstm',
                   nrmse_func=None, device='cpu'):
    """Load each saved repeat model and re-evaluate on freshly loaded test data."""
    if nrmse_func is None:
        nrmse_func = nrmse
    is_autoregressive = dataset_name in AUTOREGRESSIVE_DATASETS

    result = data_loader()
    if len(result) == 8:
        _, _, val_in, _, test_in, test_out, warmup_in, _ = result
    else:
        _, _, val_in, _, test_in, test_out = result
        warmup_in = None

    dataset_dir = os.path.join(save_dir, dataset_name)
    rep = 0
    all_nrmse, all_r2 = [], []
    while True:
        model_path = os.path.join(dataset_dir, f'repeat_{rep}_model.pt')
        if not os.path.exists(model_path):
            break
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model = LSTMForecaster(
            input_dim=ckpt['input_dim'], output_dim=ckpt['output_dim'],
            hidden_size=ckpt['hidden_size'], num_layers=ckpt['num_layers'],
            dropout=ckpt['dropout'],
        ).to(device)
        model.load_state_dict(ckpt['state_dict'])

        rep_nrmse, rep_r2 = _evaluate_lstm_on_test(
            model, val_in, test_in, test_out, warmup_in,
            is_autoregressive, nrmse_func, device)
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
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Specific datasets to run (default: all)')
    parser.add_argument('--start-rep', type=int, default=0,
                        help='Skip reps [0, N); preload existing npy '
                             'and resume from rep N. Applies to every '
                             'dataset selected.')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Order: laser and mgs first (most diagnostic / currently worst), then
    # lorenz, then the well-behaved ones.
    DATASETS = {
        'laser': getDataLaser,
        'mgs': getDataMGS,
        'lorenz': getDataLorenz,
        'dde': getDataDDE,
        'sunspots': getDataSunspots,
        'water': getDataWater,
    }

    NRMSE_OVERRIDES = {
        'sunspots': nrmse_sunspots,
    }

    # Noise-robust val sigma per dataset. Laser/mgs are prone to
    # memorization (val is a natural continuation of train), so noise on
    # the warmup breaks overfit configs. Lorenz is chaotic — warmup noise
    # amplifies through 2300 steps and wrongly penalizes good models.
    VAL_NOISE_SIGMA = {
        'laser': 0.05,
        'mgs': 0.05,
    }

    # n_init bumped for lorenz: 4/5 prior BO seeds missed the good basin,
    # more diverse random init improves the odds of hitting it.
    N_INIT = {
        'lorenz': 60,
    }
    DEFAULT_N_INIT = 30

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
                name, loader, nrmse_func=nrmse_fn, device=device)
        else:
            nrmses, r2s = run_experiment(
                name, loader, num_repeats=5,
                nrmse_func=nrmse_fn, device=device,
                val_noise_sigma=VAL_NOISE_SIGMA.get(name, 0.0),
                n_init=N_INIT.get(name, DEFAULT_N_INIT),
                start_rep=args.start_rep,
            )
        all_results[name] = {'nrmse': nrmses, 'r2': r2s}

    # Final summary table
    print(f"\n{'=' * 60}")
    print("  FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"  {'Dataset':<12} {'NRMSE':>20} {'R2':>20}")
    print(f"  {'-' * 52}")
    for name, res in all_results.items():
        nm = f"{np.mean(res['nrmse']):.4f} +/- {np.std(res['nrmse']):.4f}"
        r2 = f"{np.mean(res['r2']):.4f} +/- {np.std(res['r2']):.4f}"
        print(f"  {name:<12} {nm:>20} {r2:>20}")
