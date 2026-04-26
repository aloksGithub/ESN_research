"""
Bayesian Optimization of LSTM hyperparameters using bayes_opt.

Searches over: hidden_size, num_layers, dropout, learning_rate,
weight_decay, and sequence_length.

Uses convergence-based early stopping: optimization halts when no
improvement is observed for `bo_patience` consecutive iterations.

Returns the best trained model directly (no retraining), consistent
with how EESNAS returns its best model from the search.
"""
import copy
import time
import numpy as np
from bayes_opt import BayesianOptimization
from .model import LSTMForecaster, train_lstm, predict_lstm_autoregressive


def optimize_lstm(train_in, train_out, val_in, val_out,
                  input_dim, output_dim,
                  n_init=20, n_iter=2000, bo_patience=100,
                  rtol=1e-4,
                  epochs=200, patience=20,
                  device='cpu', seed=None, verbose=2,
                  autoregressive=False, val_error_func=None,
                  val_noise_sigma=0.0, val_noise_samples=3):
    """Run Bayesian Optimization to find the best LSTM model.

    Uses the same bayes_opt library as the rest of the EESNAS codebase.
    BO maximizes, so we negate the validation loss. Stops early if the
    best val score has not improved by a relative factor of `rtol` for
    `bo_patience` consecutive BO iterations.

    Returns the best trained model directly, mirroring how EESNAS
    returns the best model from its search without retraining.

    Args:
        train_in: Training inputs, shape (num_samples, input_dim).
        train_out: Training targets, shape (num_samples, output_dim).
        val_in: Validation inputs, shape (num_samples, input_dim).
        val_out: Validation targets, shape (num_samples, output_dim).
        input_dim: Number of input features.
        output_dim: Number of output features.
        n_init: Number of random initialization points.
        n_iter: Maximum number of BO iterations after initialization.
        bo_patience: Stop if no relative improvement for this many
            consecutive BO iterations (after the init phase).
        rtol: Relative-improvement threshold. A new best counts only if
            it beats the prior best by at least rtol * |prior best|.
        epochs: Max epochs per LSTM training trial.
        patience: Early stopping patience per LSTM training trial.
        device: 'cpu' or 'cuda'.
        seed: Random seed.
        verbose: Verbosity level for bayes_opt (0=silent, 2=all).
        autoregressive: If True, evaluate validation using autoregressive
            prediction (warm up on train data, then AR predict on val).
        val_error_func: Error function for AR validation (e.g. nrmse).
            Required when autoregressive=True. Should return a scalar
            where lower is better.

    Returns:
        best_model: The best trained LSTMForecaster instance.
        best_params: Dict of best hyperparameters found.
        best_val_loss: Best validation loss achieved.
        optim_log: Dict with full optimization trajectory (every eval's
            params + val_loss, best-so-far trace, stop reason, etc.).
    """
    best_model_state = [None]  # mutable container for closure
    best_model_config = [None]
    best_val = [float('inf')]

    trajectory = []
    phase = ['init']

    def black_box(hidden_size, num_layers, dropout, log_lr,
                  log_weight_decay, seq_len, input_noise):
        hidden_size = int(round(hidden_size))
        num_layers = int(round(num_layers))
        seq_len = int(round(seq_len / 10.0)) * 10  # snap to multiples of 10
        seq_len = max(seq_len, 10)

        if num_layers <= 1:
            dropout = 0.0

        lr = 10 ** log_lr
        weight_decay = 10 ** log_weight_decay

        eval_start = time.time()

        model = LSTMForecaster(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        val_loss = train_lstm(
            model, train_in, train_out, val_in, val_out,
            seq_len=seq_len, lr=lr, weight_decay=weight_decay,
            epochs=epochs, patience=patience, device=device,
            input_noise=input_noise,
        )

        # For AR datasets, re-evaluate using autoregressive prediction.
        # When val_noise_sigma > 0, also score with noisy-warmup rollouts
        # and combine as max(clean, mean(noisy)) — needed for laser/mgs
        # where AR from val[0] is a natural continuation of training so
        # overfit models nail the clean rollout. Noisy warmups break
        # memorization (memorizers spike under perturbation while
        # generalizers don't). For chaotic systems (lorenz), noise
        # amplifies exponentially over a long warmup and penalizes good
        # models too, so leave val_noise_sigma=0 there.
        if autoregressive and val_error_func is not None:
            y_pred = predict_lstm_autoregressive(
                model, val_in[0], num_steps=len(val_out),
                device=device, warmup_data=train_in,
            )
            clean_val = val_error_func(val_out, y_pred)

            if val_noise_sigma > 0.0:
                rng = np.random.RandomState(0)
                noisy_errs = []
                for _ in range(val_noise_samples):
                    noise = (rng.randn(*train_in.shape).astype(np.float32)
                             * val_noise_sigma)
                    y_pred_n = predict_lstm_autoregressive(
                        model, val_in[0], num_steps=len(val_out),
                        device=device, warmup_data=train_in + noise,
                    )
                    noisy_errs.append(val_error_func(val_out, y_pred_n))
                val_loss = max(clean_val, float(np.mean(noisy_errs)))
            else:
                val_loss = clean_val

        # Track the best model weights
        is_best = val_loss < best_val[0]
        if is_best:
            best_val[0] = val_loss
            best_model_state[0] = copy.deepcopy(model.state_dict())
            best_model_config[0] = {
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'dropout': dropout,
            }

        trajectory.append({
            'phase': phase[0],
            'params': {
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'dropout': dropout,
                'lr': float(lr),
                'weight_decay': float(weight_decay),
                'seq_len': seq_len,
                'input_noise': float(input_noise),
            },
            'val_loss': float(val_loss),
            'is_best_so_far': bool(is_best),
            'elapsed_s': time.time() - eval_start,
        })

        return -val_loss  # BO maximizes, so negate

    seq_len_max = max(10, (len(val_in) // 10) * 10)
    pbounds = {
        'hidden_size': (16, 512),
        'num_layers': (1, 3),
        'dropout': (0.0, 0.5),
        'log_lr': (-4, -2),             # 1e-4 to 1e-2
        'log_weight_decay': (-6, -2),   # 1e-6 to 1e-2
        'seq_len': (10, seq_len_max),
        'input_noise': (0.0, 0.1),
    }

    random_state = seed if seed is not None else 1
    optimizer = BayesianOptimization(
        f=black_box,
        pbounds=pbounds,
        random_state=random_state,
        allow_duplicate_points=True,
        verbose=verbose,
    )

    # Random initialization phase
    optimizer.maximize(init_points=n_init, n_iter=0)
    phase[0] = 'bo'

    # BO phase with early stopping
    best_target = optimizer.max["target"]
    iters_without_improvement = 0
    total_iters = 0
    stop_reason = 'cap'

    for i in range(n_iter):
        optimizer.maximize(init_points=0, n_iter=1)
        total_iters += 1

        current_best = optimizer.max["target"]
        # best_target is -loss (BO maximizes), so improvement means
        # current_best > best_target by a relative margin.
        improvement_threshold = best_target + rtol * abs(best_target)
        if current_best > improvement_threshold:
            best_target = current_best
            iters_without_improvement = 0
        else:
            iters_without_improvement += 1

        if iters_without_improvement >= bo_patience:
            print(f"  BO early stop after {n_init + total_iters} total evaluations "
                  f"({bo_patience} iters without relative improvement > {rtol})")
            stop_reason = 'patience'
            break

    raw = optimizer.max["params"]
    best_val_loss = -optimizer.max["target"]

    num_layers = int(round(raw['num_layers']))
    best_params = {
        'hidden_size': int(round(raw['hidden_size'])),
        'num_layers': num_layers,
        'dropout': raw['dropout'] if num_layers > 1 else 0.0,
        'lr': 10 ** raw['log_lr'],
        'weight_decay': 10 ** raw['log_weight_decay'],
        'seq_len': max(int(round(raw['seq_len'] / 10.0)) * 10, 10),
        'input_noise': raw['input_noise'],
    }

    # Reconstruct the best model with its saved weights
    cfg = best_model_config[0]
    best_model = LSTMForecaster(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_size=cfg['hidden_size'],
        num_layers=cfg['num_layers'],
        dropout=cfg['dropout'],
    )
    best_model.load_state_dict(best_model_state[0])

    print(f"  BO completed: {n_init + total_iters} total evaluations, "
          f"best val MSE = {best_val_loss:.6f}")

    # Best-so-far trace over the whole trajectory
    best_val_trace = []
    running_best = float('inf')
    for e in trajectory:
        if e['val_loss'] < running_best:
            running_best = e['val_loss']
        best_val_trace.append(running_best)

    optim_log = {
        'optimizer': 'bayesian_optimization',
        'evals_total': len(trajectory),
        'init_evals': n_init,
        'bo_iters_done': total_iters,
        'bo_iters_cap': n_iter,
        'bo_patience': bo_patience,
        'rtol': rtol,
        'patience_at_stop': iters_without_improvement,
        'stop_reason': stop_reason,
        'best_val_trace': best_val_trace,
        'evals': trajectory,
        'pbounds': pbounds,
    }

    return best_model, best_params, best_val_loss, optim_log
