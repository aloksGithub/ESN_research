"""
PSO-style hyperparameter optimizer for GE-DESN.

Adapted from the PSO optimization in the original ESNGIPMAMG.py.
Uses scipy.optimize.differential_evolution (a global optimizer similar to PSO)
to tune the oram weight-scaling parameters before running the full grow/evolve.

The key insight from the original code is that optimization uses a *simple*
(non-growing) ESN for speed — the oram parameters control reservoir dynamics
and are somewhat independent of the evolution process.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution

from .esn import EchoStateNetwork


def _build_and_evaluate(U_init, U_train, Y_train, U_val, Y_val, pram, oram,
                        autoregressive=False, repeats=3):
    """Build a simple (non-growing) deep ESN and return validation NRMSE.

    This mirrors the author's ``fastRepeatOptOram``: construct an ESN with
    ``neurons_add`` neurons per layer (no evolution), train, and evaluate.
    Averaged over *repeats* random seeds to reduce variance.
    """
    neurons = pram['neurons_add']
    n_layers = pram['max_layers']
    scores = np.zeros(repeats)

    for r in range(repeats):
        esn = EchoStateNetwork(
            U_init, U_train, Y_train, U_val, Y_val,
            U_val, Y_val,  # dummy test — not used here
            pram, oram,
        )
        # Build fixed-size network (no evolution)
        esn.Inilize_First_reservoir(neurons)
        for _ in range(1, n_layers):
            esn.Inilize_Stack_a_reservoir(neurons)

        esn.Init_reservior(U_init)
        esn.Train_reservoir(U_train, Y_train)
        esn.Reinit_reservoir()

        if autoregressive:
            for t in range(U_train.shape[1]):
                esn.UspanX(U_train[:, t:t + 1], esn.galaph)
            Yout = esn.Validate_test_data_autoregressive(
                U_val[:, 0:1], U_val.shape[1])
        else:
            Yout, _ = esn.Validate_test_data_constant(U_val)

        err = np.sum((Yout - Y_val) ** 2)
        Ynorm = np.mean(Y_val) * np.ones(Yout.shape)
        nerr = np.sum((Yout - Ynorm) ** 2)
        if nerr == 0:
            scores[r] = 1e6
        else:
            scores[r] = np.sqrt(err / nerr)

    return float(np.mean(scores))


def optimize_oram(U_init, U_train, Y_train, U_val, Y_val, pram,
                  base_oram, autoregressive=False, repeats=3,
                  maxiter=200, popsize=15, gen_patience=20, rtol=1e-4,
                  seed=None, verbose=True):
    """Optimize oram parameters using differential evolution.

    Tunes: ampWi, ampWp (ampWc), ampWr, leaky_rate, reg_fac.
    Uses a simple non-growing ESN evaluated on the validation set.

    Args:
        U_init, U_train, Y_train, U_val, Y_val: data arrays (features, time)
        pram: structural parameters dict (will be modified: leaky_rate updated)
        base_oram: starting oram dict (used for fixed values like spare_rate)
        autoregressive: whether to use autoregressive evaluation
        repeats: number of random restarts per evaluation (reduces variance)
        maxiter: max generations for the optimizer (safety cap; the
            patience callback normally terminates first)
        popsize: population size multiplier for differential evolution
        gen_patience: stop if best score has not improved by `rtol` for
            this many consecutive DE generations
        rtol: relative-improvement threshold for the patience gate
        seed: random seed for reproducibility
        verbose: print progress

    Returns:
        optimized oram dict, optimized leaky_rate, optim_log dict
        (full trajectory of every eval's params + val_loss, generation
        numbers, stop reason, etc.)
    """
    # Parameter bounds — informed by the author's PSO ranges and defaults
    # (ampWi, ampWp, ampWr, leaky_rate, log10_reg_fac)
    bounds = [
        (0.001, 5.0),    # ampWi
        (0.05, 5.0),     # ampWp
        (0.5, 0.99),     # ampWr
        (0.1, 0.99),     # leaky_rate
        (-12, -3),       # log10(reg_fac)
    ]

    best_score = [np.inf]
    eval_count = [0]
    gen_counter = [0]
    trajectory = []

    def objective(x):
        oram = base_oram.copy()
        oram['ampWi'] = x[0]
        oram['ampWp'] = x[1]
        oram['ampWr'] = x[2]
        trial_pram = pram.copy()
        trial_pram['leaky_rate'] = x[3]
        oram['reg_fac'] = 10 ** x[4]

        eval_start = time.time()
        score = _build_and_evaluate(
            U_init, U_train, Y_train, U_val, Y_val,
            trial_pram, oram, autoregressive, repeats)
        eval_elapsed = time.time() - eval_start

        eval_count[0] += 1
        is_best = score < best_score[0]
        if is_best:
            best_score[0] = score
            if verbose:
                print(f"    [eval {eval_count[0]:>4d}] NRMSE={score:.6f}  "
                      f"ampWi={x[0]:.4f} ampWp={x[1]:.4f} ampWr={x[2]:.4f} "
                      f"leak={x[3]:.4f} reg={10**x[4]:.2e}")

        trajectory.append({
            'gen': gen_counter[0],
            'eval_idx': eval_count[0],
            'params': {
                'ampWi': float(x[0]),
                'ampWp': float(x[1]),
                'ampWr': float(x[2]),
                'leaky_rate': float(x[3]),
                'reg_fac': float(10 ** x[4]),
            },
            'val_loss': float(score),
            'is_best_so_far': bool(is_best),
            'elapsed_s': eval_elapsed,
        })
        return score

    if verbose:
        print("  Optimizing oram parameters (differential evolution)...")

    # Generation-level patience tracking. DE's callback fires after each
    # generation; returning True halts the optimizer.
    gen_best = [np.inf]
    gens_without_improvement = [0]
    stop_reason = ['unknown']

    def _patience_callback(xk, convergence):
        gen_counter[0] += 1
        current_best = best_score[0]
        improvement_threshold = gen_best[0] * (1 - rtol)
        if current_best < improvement_threshold:
            gen_best[0] = current_best
            gens_without_improvement[0] = 0
        else:
            gens_without_improvement[0] += 1
        if gens_without_improvement[0] >= gen_patience:
            if verbose:
                print(f"  DE early stop: {gen_patience} generations without "
                      f"relative improvement > {rtol}")
            stop_reason[0] = 'patience'
            return True
        return False

    result = differential_evolution(
        objective, bounds,
        maxiter=maxiter, popsize=popsize,
        seed=seed, tol=1e-4,
        mutation=(0.5, 1.0), recombination=0.7,
        callback=_patience_callback,
    )

    if stop_reason[0] == 'unknown':
        # DE finished without our callback returning True. It either hit
        # maxiter or converged via its internal `tol`.
        if result.nit >= maxiter:
            stop_reason[0] = 'maxiter'
        else:
            stop_reason[0] = 'tol'

    opt_oram = base_oram.copy()
    opt_oram['ampWi'] = result.x[0]
    opt_oram['ampWp'] = result.x[1]
    opt_oram['ampWr'] = result.x[2]
    opt_leaky = result.x[3]
    opt_oram['reg_fac'] = 10 ** result.x[4]

    if verbose:
        print(f"  Optimization complete ({eval_count[0]} evaluations)")
        print(f"    Best NRMSE: {result.fun:.6f}")
        print(f"    ampWi={opt_oram['ampWi']:.4f}, "
              f"ampWp={opt_oram['ampWp']:.4f}, "
              f"ampWr={opt_oram['ampWr']:.4f}, "
              f"leaky_rate={opt_leaky:.4f}, "
              f"reg_fac={opt_oram['reg_fac']:.2e}")

    # Best-so-far trace
    best_val_trace = []
    running_best = float('inf')
    for e in trajectory:
        if e['val_loss'] < running_best:
            running_best = e['val_loss']
        best_val_trace.append(running_best)

    optim_log = {
        'optimizer': 'differential_evolution',
        'evals_total': eval_count[0],
        'gens_total': gen_counter[0],
        'maxiter': maxiter,
        'popsize': popsize,
        'gen_patience': gen_patience,
        'rtol': rtol,
        'stop_reason': stop_reason[0],
        'scipy_nfev': int(result.nfev),
        'scipy_nit': int(result.nit),
        'scipy_message': str(result.message),
        'scipy_success': bool(result.success),
        'best_val_trace': best_val_trace,
        'evals': trajectory,
        'bounds': bounds,
        'eval_repeats': repeats,
    }

    return opt_oram, opt_leaky, optim_log
