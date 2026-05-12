"""Helpers for extracting median-repeat validation convergence traces."""
import math


def _finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def cumulative(values):
    total = 0.0
    out = []
    for value in values:
        value = _finite_float(value)
        total += 0.0 if value is None else value
        out.append(float(total))
    return out


def best_so_far(errors):
    out = []
    best = None
    for error in errors:
        error = _finite_float(error)
        if error is None:
            continue
        if best is None or error < best:
            best = error
        out.append(float(best))
    return out


def median_convergence(convergences):
    valid = [
        conv for conv in convergences
        if conv.get('validation_error')
    ]
    if not valid:
        return None
    ranked = sorted(
        valid,
        key=lambda conv: (
            float(conv.get('final_validation_error', conv['validation_error'][-1])),
            int(conv.get('repeat', 0)),
        ),
    )
    return ranked[len(ranked) // 2]


def convergence_from_optim_log(repeat, optim_log):
    if not optim_log:
        return None

    evals = optim_log.get('trajectory') or optim_log.get('evals') or []
    trace = optim_log.get('best_val_trace')
    if not trace:
        trace = best_so_far(e.get('val_loss') for e in evals)
    else:
        trace = [
            float(value) for value in trace
            if _finite_float(value) is not None
        ]

    if not trace:
        return None

    eval_times = [e.get('elapsed_s', 0.0) for e in evals[:len(trace)]]
    times = cumulative(eval_times)
    if len(times) < len(trace):
        times.extend([times[-1] if times else 0.0] * (len(trace) - len(times)))

    return {
        'repeat': int(repeat),
        'times': times[:len(trace)],
        'validation_error': trace,
        'final_validation_error': float(trace[-1]),
        'n_evals': len(trace),
    }


def convergence_from_bo_experiment(repeat, exp):
    errors = best_so_far(errors[0] for errors in getattr(exp, 'performances', []))
    if not errors:
        return None
    times = cumulative(getattr(exp, 'times', [])[:len(errors)])
    if len(times) < len(errors):
        times.extend([times[-1] if times else 0.0] * (len(errors) - len(times)))
    return {
        'repeat': int(repeat),
        'times': times[:len(errors)],
        'validation_error': errors,
        'final_validation_error': float(errors[-1]),
        'n_evals': len(errors),
    }


def convergence_from_ga_experiment(repeat, exp):
    """Return a generation-level best-so-far validation curve for GA/ESNAS."""
    all_errors = [
        _finite_float(errors[0])
        for errors in getattr(exp, 'fitnesses', [])
    ]
    all_errors = [error for error in all_errors if error is not None]
    generation_times = getattr(exp, 'generationTimes', [])
    if not all_errors or not generation_times:
        return None

    ga_params = getattr(exp, 'gaParams', None)
    population_size = int(getattr(ga_params, 'populationSize', 0) or 0)
    if population_size <= 0:
        population_size = len(all_errors)

    evals_per_individual = 1
    if hasattr(exp, 'bo_iter'):
        # ESNAS probes the default parameters once, then performs bo_init
        # random points and bo_iter BO points for each GA individual.
        evals_per_individual = (
            int(getattr(exp, 'bo_init', 0) or 0)
            + int(getattr(exp, 'bo_iter', 0) or 0)
            + 1
        )

    base_generation_evals = population_size * evals_per_individual
    reset_generation_evals = max(population_size - 1, 0) * evals_per_individual
    reset_generations = set(getattr(exp, 'modelGenerationIndices', []) or [])

    generation_errors = []
    consumed = 0
    for gen_idx in range(len(generation_times)):
        consumed += base_generation_evals
        if gen_idx in reset_generations and gen_idx > 0:
            consumed += reset_generation_evals
        boundary = min(consumed, len(all_errors))
        if boundary <= 0:
            continue
        generation_errors.append(float(min(all_errors[:boundary])))

    if not generation_errors:
        return None

    times = cumulative(generation_times[:len(generation_errors)])
    return {
        'repeat': int(repeat),
        'times': times,
        'validation_error': generation_errors,
        'final_validation_error': float(generation_errors[-1]),
        'n_evals': len(all_errors),
    }
