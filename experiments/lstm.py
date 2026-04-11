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

from src.baselines.lstm import (
    predict_lstm, predict_lstm_autoregressive,
    optimize_lstm,
)
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


def run_experiment(dataset_name, data_loader, num_repeats=5,
                   n_init=30, n_iter=800, bo_patience=50,
                   epochs=200, patience=20,
                   save_dir='results/lstm', nrmse_func=None, device='cpu'):
    """Run LSTM BO search + evaluation on one dataset.

    Each repeat runs an independent full BO search and takes the best
    model directly for test evaluation — matching EESNAS methodology.
    """
    if nrmse_func is None:
        nrmse_func = nrmse

    print(f"\n{'=' * 60}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'=' * 60}")

    train_in, train_out, val_in, val_out, test_in, test_out, warmup_in, _ = data_loader()
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

    nrmse_scores = []
    r2_scores = []

    dataset_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    for rep in range(num_repeats):
        start = time.time()
        print(f"\n  --- Run {rep + 1}/{num_repeats} ---")

        best_model, best_params, best_val_loss = optimize_lstm(
            train_in, train_out, val_in, val_out,
            input_dim, output_dim,
            n_init=n_init, n_iter=n_iter, bo_patience=bo_patience,
            epochs=epochs, patience=patience,
            device=device, seed=rep,
            autoregressive=is_autoregressive,
            val_error_func=_mse if is_autoregressive else None,
        )

        print(f"  Best params: {best_params}")
        print(f"  Best val MSE: {best_val_loss:.6f}")

        # Evaluate best model directly on test set
        rep_nrmse = []
        rep_r2 = []
        if is_autoregressive:
            best_model.eval()
            best_model = best_model.to(device)
            hidden = None

            # Teacher-force through training data
            x = torch.tensor(train_in, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                _, hidden = best_model(x, hidden)

            for i in range(len(test_in)):
                # Teacher-force through warmup segment for this test set
                x = torch.tensor(warmup_in[i], dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    _, hidden = best_model(x, hidden)

                # Autoregressive prediction on current test set
                preds = []
                current = torch.tensor(test_in[i][0], dtype=torch.float32).reshape(1, 1, -1).to(device)
                ar_hidden = hidden
                with torch.no_grad():
                    for _ in range(len(test_in[i])):
                        out, ar_hidden = best_model(current, ar_hidden)
                        pred = out[:, -1, :]
                        preds.append(pred.cpu().numpy().flatten())
                        current = pred.unsqueeze(1)
                hidden = ar_hidden

                y_pred = np.array(preds)
                rep_nrmse.append(nrmse_func(test_out[i], y_pred))
                rep_r2.append(r_squared(test_out[i], y_pred))
        else:
            y_pred = predict_lstm(best_model, test_in, device=device)
            rep_nrmse.append(nrmse_func(test_out, y_pred))
            rep_r2.append(r_squared(test_out, y_pred))

        nrmse_scores.extend(rep_nrmse)
        r2_scores.extend(rep_r2)

        elapsed = time.time() - start
        print(f"  Repeat {rep + 1}: ({elapsed:.1f}s)")
        for j, (n, r) in enumerate(zip(rep_nrmse, rep_r2)):
            print(f"    Test {j+1}: NRMSE={n:.6f}, R2={r:.6f}")

        # Save this repeat's result to its own pickle
        repeat_data = {
            'dataset': dataset_name,
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
            'model_state_dict': best_model.state_dict(),
            'params': best_params,
            'val_loss': best_val_loss,
            'nrmse': rep_nrmse,
            'r2': rep_r2,
            'elapsed': elapsed,
        }
        with open(os.path.join(dataset_dir, f'repeat_{rep}.pkl'), 'wb') as f:
            pickle.dump(repeat_data, f)
        np.save(os.path.join(dataset_dir, 'nrmse_scores.npy'), np.array(nrmse_scores))
        np.save(os.path.join(dataset_dir, 'r2_scores.npy'), np.array(r2_scores))
        print(f"  Checkpoint saved to {dataset_dir}/ ({rep + 1}/{num_repeats} repeats)")

    # --- Summary ---
    print(f"\n  --- {dataset_name} Summary ---")
    print(f"  NRMSE: {np.mean(nrmse_scores):.6f} +/- {np.std(nrmse_scores):.6f}")
    print(f"  R2:    {np.mean(r2_scores):.6f} +/- {np.std(r2_scores):.6f}")

    return nrmse_scores, r2_scores


if __name__ == '__main__':
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

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
        nrmses, r2s = run_experiment(
            name, loader, num_repeats=5,
            nrmse_func=nrmse_fn, device=device,
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
