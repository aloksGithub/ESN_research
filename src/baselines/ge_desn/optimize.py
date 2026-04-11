"""
PSO-style hyperparameter optimizer for GE-DESN.

Adapted from the PSO optimization in the original ESNGIPMAMG.py.
Uses scipy.optimize.differential_evolution (a global optimizer similar to PSO)
to tune the oram weight-scaling parameters before running the full grow/evolve.

The key insight from the original code is that optimization uses a *simple*
(non-growing) ESN for speed — the oram parameters control reservoir dynamics
and are somewhat independent of the evolution process.
"""
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
                  maxiter=20, popsize=15, seed=None, verbose=True):
    """Optimize oram parameters using differential evolution.

    Tunes: ampWi, ampWp (ampWc), ampWr, leaky_rate, reg_fac.
    Uses a simple non-growing ESN evaluated on the validation set.

    Args:
        U_init, U_train, Y_train, U_val, Y_val: data arrays (features, time)
        pram: structural parameters dict (will be modified: leaky_rate updated)
        base_oram: starting oram dict (used for fixed values like spare_rate)
        autoregressive: whether to use autoregressive evaluation
        repeats: number of random restarts per evaluation (reduces variance)
        maxiter: max iterations for the optimizer
        popsize: population size multiplier for differential evolution
        seed: random seed for reproducibility
        verbose: print progress

    Returns:
        optimized oram dict, optimized leaky_rate
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

    def objective(x):
        oram = base_oram.copy()
        oram['ampWi'] = x[0]
        oram['ampWp'] = x[1]
        oram['ampWr'] = x[2]
        trial_pram = pram.copy()
        trial_pram['leaky_rate'] = x[3]
        oram['reg_fac'] = 10 ** x[4]

        score = _build_and_evaluate(
            U_init, U_train, Y_train, U_val, Y_val,
            trial_pram, oram, autoregressive, repeats)

        eval_count[0] += 1
        if score < best_score[0]:
            best_score[0] = score
            if verbose:
                print(f"    [eval {eval_count[0]:>4d}] NRMSE={score:.6f}  "
                      f"ampWi={x[0]:.4f} ampWp={x[1]:.4f} ampWr={x[2]:.4f} "
                      f"leak={x[3]:.4f} reg={10**x[4]:.2e}")
        return score

    if verbose:
        print("  Optimizing oram parameters (differential evolution)...")

    result = differential_evolution(
        objective, bounds,
        maxiter=maxiter, popsize=popsize,
        seed=seed, tol=1e-4,
        mutation=(0.5, 1.0), recombination=0.7,
    )

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

    return opt_oram, opt_leaky
