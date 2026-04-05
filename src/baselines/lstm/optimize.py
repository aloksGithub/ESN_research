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
import numpy as np
from bayes_opt import BayesianOptimization
from .model import LSTMForecaster, train_lstm, predict_lstm_autoregressive


def optimize_lstm(train_in, train_out, val_in, val_out,
                  input_dim, output_dim,
                  n_init=20, n_iter=800, bo_patience=30,
                  epochs=200, patience=20,
                  device='cpu', seed=None, verbose=2,
                  autoregressive=False, val_error_func=None):
    """Run Bayesian Optimization to find the best LSTM model.

    Uses the same bayes_opt library as the rest of the EESNAS codebase.
    BO maximizes, so we negate the validation loss. Stops early if no
    improvement is found for `bo_patience` consecutive BO iterations.

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
        bo_patience: Stop if no improvement for this many consecutive
            BO iterations (after the init phase).
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
    """
    best_model_state = [None]  # mutable container for closure
    best_model_config = [None]
    best_val = [float('inf')]

    def black_box(hidden_size, num_layers, dropout, log_lr,
                  log_weight_decay, seq_len):
        hidden_size = int(round(hidden_size))
        num_layers = int(round(num_layers))
        seq_len = int(round(seq_len / 10.0)) * 10  # snap to multiples of 10
        seq_len = max(seq_len, 10)

        if num_layers <= 1:
            dropout = 0.0

        lr = 10 ** log_lr
        weight_decay = 10 ** log_weight_decay

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
        )

        # For AR datasets, re-evaluate using autoregressive prediction
        if autoregressive and val_error_func is not None:
            y_pred = predict_lstm_autoregressive(
                model, val_in[0], num_steps=len(val_out),
                device=device, warmup_data=train_in,
            )
            val_loss = val_error_func(val_out, y_pred)

        # Track the best model weights
        if val_loss < best_val[0]:
            best_val[0] = val_loss
            best_model_state[0] = copy.deepcopy(model.state_dict())
            best_model_config[0] = {
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'dropout': dropout,
            }

        return -val_loss  # BO maximizes, so negate

    pbounds = {
        'hidden_size': (16, 512),
        'num_layers': (1, 3),
        'dropout': (0.0, 0.5),
        'log_lr': (-4, -2),             # 1e-4 to 1e-2
        'log_weight_decay': (-6, -2),   # 1e-6 to 1e-2
        'seq_len': (10, 200),
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

    # BO phase with early stopping
    best_target = optimizer.max["target"]
    iters_without_improvement = 0
    total_iters = 0

    for i in range(n_iter):
        optimizer.maximize(init_points=0, n_iter=1)
        total_iters += 1

        current_best = optimizer.max["target"]
        if current_best > best_target:
            best_target = current_best
            iters_without_improvement = 0
        else:
            iters_without_improvement += 1

        if iters_without_improvement >= bo_patience:
            print(f"  BO early stop after {n_init + total_iters} total evaluations "
                  f"({bo_patience} iters without improvement)")
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

    return best_model, best_params, best_val_loss
