"""
CMA-ES optimization of LCNN hyperparameters.

Searches over reservoir dimensions, kernel size, spectral radius,
leakage, noise, sparsity, input scaling, and ridge regularization.

Evaluates on the validation set (never on test).
"""
import copy
import numpy as np
import cma
from .lcnn import LCNN


# Kernel sizes to search over (must be odd)
KERNEL_OPTIONS = [3, 5, 7]


def _params_from_vector(x):
    """Decode a CMA-ES solution vector into LCNN hyperparameters.

    Bounds are chosen from standard ESN/LCNN tuning guidelines so CMA-ES
    cannot land in pathological regions (e.g. echo-state-property violations
    at SR > 1 or input starvation at sigma_in << 1).

    The vector x has 11 continuous dimensions, mapped as follows:
      0: state_height  (clamped to [3, 30], rounded to int)
      1: state_width   (clamped to [3, 30], rounded to int)
      2: kernel_idx    (index into KERNEL_OPTIONS, same for H and W)
      3: log10(sigma_res)  -> sigma_res, clamped to [~0.316, ~3.16]
      4: sigma_in, clamped to [0.3, 2.5]
      5: leakage       (clamped to [0.01, 1.0])
      6: sparsity      (clamped to [0.0, 0.9])
      7: log10(ridge)
      8: noise         (clamped to [0.0, 0.05])
      9: spectral_radius (clamped to [0.1, 1.1])
     10: noise_augment  (clamped to [0.0, 0.1])
    """
    state_height = int(np.clip(round(x[0]), 3, 30))
    state_width = int(np.clip(round(x[1]), 3, 30))
    kernel_idx = int(np.clip(round(x[2]), 0, len(KERNEL_OPTIONS) - 1))
    kernel_size = KERNEL_OPTIONS[kernel_idx]
    # Kernel can't exceed state dimensions
    kernel_size = min(kernel_size, state_height, state_width)
    if kernel_size % 2 == 0:
        kernel_size -= 1
    kernel_size = max(kernel_size, 3)

    # sigma_res floor of 10^-0.5 (~0.316) avoids near-degenerate random
    # reservoirs being amplified pathologically by the spectral-radius
    # rescaling step.
    sigma_res = 10 ** np.clip(x[3], -0.5, 0.5)
    sigma_in = np.clip(x[4], 0.3, 2.5)
    leakage = np.clip(x[5], 0.01, 1.0)
    sparsity = np.clip(x[6], 0.0, 0.9)
    ridge = 10 ** np.clip(x[7], -6, -1)
    noise = np.clip(x[8], 0.0, 0.05)
    # SR cap of 1.1 keeps the reservoir within/near the echo state regime.
    spectral_radius = np.clip(x[9], 0.1, 1.1)
    noise_augment = np.clip(x[10], 0.0, 0.1)

    return {
        'state_height': state_height,
        'state_width': state_width,
        'kernel_height': kernel_size,
        'kernel_width': kernel_size,
        'topology': 'lcnn',
        'sigma_res': sigma_res,
        'mu_res': 0.0,
        'sigma_in': sigma_in,
        'mu_in': 0.0,
        'sigma_b': 0.0,
        'mu_b': 0.0,
        'sparsity': sparsity,
        'leakage': leakage,
        'noise': noise,
        'ridge': ridge,
        'spectral_radius': spectral_radius,
        'noise_augment': noise_augment,
    }


def _evaluate(params, train_in, train_out, val_in, val_out,
              washout, autoregressive, val_error_func, seed):
    """Train an LCNN with given params and return (error, model)."""
    try:
        noise_augment = params.pop('noise_augment', 0.0)
        model = LCNN(**params, seed=seed)
        model.fit(train_in, train_out, washout=washout,
                  noise_augment=noise_augment)

        if autoregressive:
            preds = model.predict_autoregressive(val_in[0], steps=len(val_out))
        else:
            preds = model.predict(val_in)

        if preds is None or np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
            return 1e6, None

        error = val_error_func(val_out, preds)
        if np.isnan(error) or np.isinf(error):
            return 1e6, None
        return float(error), model
    except Exception:
        return 1e6, None


def optimize_lcnn(train_in, train_out, val_in, val_out,
                  washout=100, autoregressive=False,
                  val_error_func=None,
                  popsize=20, max_evals=600,
                  n_restarts=3, seed=None, verbose=True):
    """Run CMA-ES to find the best LCNN hyperparameters.

    Args:
        train_in: (T_train, n_inputs) training inputs.
        train_out: (T_train, n_outputs) training targets.
        val_in: (T_val, n_inputs) validation inputs.
        val_out: (T_val, n_outputs) validation targets.
        washout: Number of initial timesteps to discard during training.
        autoregressive: If True, evaluate with autoregressive prediction.
        val_error_func: Error function(y_true, y_pred) -> scalar (lower is
            better). Required.
        popsize: CMA-ES population size.
        max_evals: Maximum number of function evaluations.
        n_restarts: Number of independent CMA-ES restarts (best across all).
        seed: Random seed.
        verbose: Print progress.

    Returns:
        best_model: Trained LCNN with best hyperparameters (the actual
            model instance that achieved the best validation score).
        best_params: Dict of best hyperparameters.
        best_error: Best validation error achieved.
    """
    if val_error_func is None:
        raise ValueError("val_error_func is required")

    # Initial guess (center of search space)
    x0 = [
        11,    # state_height
        11,    # state_width
        1,     # kernel_idx (5x5)
        -0.3,  # log10(sigma_res)
        1.0,   # sigma_in
        0.7,   # leakage
        0.1,   # sparsity
        -4,    # log10(ridge)
        0.0,   # noise
        0.9,   # spectral_radius
        0.02,  # noise_augment
    ]
    sigma0 = 1.5

    overall_best_error = 1e6
    overall_best_params = None
    overall_best_model = None

    rng = np.random.RandomState(seed)

    for restart in range(n_restarts):
        run_seed = rng.randint(0, 2**31)
        opts = {
            'popsize': popsize,
            'maxfevals': max_evals,
            'seed': run_seed,
            'verbose': -9 if not verbose else 1,
            'tolfun': 1e-8,
        }

        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        while not es.stop():
            solutions = es.ask()
            fitnesses = []
            models = []
            for s in solutions:
                params = _params_from_vector(s)
                f, m = _evaluate(
                    params, train_in, train_out, val_in, val_out,
                    washout, autoregressive, val_error_func,
                    seed=rng.randint(0, 2**31),
                )
                fitnesses.append(f)
                models.append(m)
            es.tell(solutions, fitnesses)

            # Track the actual best solution and model seen
            best_idx = int(np.argmin(fitnesses))
            if fitnesses[best_idx] < overall_best_error:
                overall_best_error = fitnesses[best_idx]
                overall_best_params = _params_from_vector(solutions[best_idx])
                overall_best_model = models[best_idx]
                if verbose:
                    print(f"  CMA-ES restart {restart+1}: new best = {overall_best_error:.6f}")

        if verbose:
            print(f"  CMA-ES restart {restart+1} done, "
                  f"best = {es.result.fbest:.6f}, "
                  f"evals = {es.result.evaluations}")

    if overall_best_model is None:
        raise RuntimeError("CMA-ES failed to find any valid configuration")

    if verbose:
        print(f"  CMA-ES completed: best val error = {overall_best_error:.6f}")
        print(f"  Best params: {overall_best_params}")

    return overall_best_model, overall_best_params, overall_best_error
