"""Aggregate per-method results and compute Mann-Whitney U tests.

Reads results/_aggregate/{method}.json, computes per-method summaries, and
computes unpaired Mann-Whitney U tests of EESNAS against each baseline.

Default output:
    thesis/automation/results_summary.json

Run after regenerating aggregate JSONs:
    envs/esnas/Scripts/python scripts/aggregate_results.py
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import mannwhitneyu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGG = os.path.join(ROOT, 'results', '_aggregate')
DEFAULT_OUT = os.path.join(ROOT, 'thesis', 'automation', 'results_summary.json')

DATASETS = ['mgs', 'lorenz', 'dde', 'laser', 'sunspots', 'water']
AR_DATASETS = {'mgs', 'lorenz', 'dde', 'laser'}
NUM_STRETCHES = 5
METHODS = ['esnas', 'ga', 'bo', 'lcnn', 'ge_desn', 'lstm', 'grid_search']
BASELINES = ['ga', 'bo', 'lcnn', 'ge_desn', 'lstm']


def load_method(name):
    path = os.path.join(AGG, f'{name}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def per_run(values, dataset, method=None):
    """Collapse per-stretch AR values only when multiple stretches are present."""
    arr = np.asarray(values, dtype=float)
    if method == 'grid_search':
        return arr
    if dataset in AR_DATASETS and len(arr) == 25:
        arr = arr.reshape(-1, NUM_STRETCHES).mean(axis=1)
    return arr


def summarize(arr, lower_is_better=True):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {'n': 0}
    return {
        'n': int(len(arr)),
        'median': float(np.median(arr)),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=0)),
        'best': float(np.min(arr) if lower_is_better else np.max(arr)),
    }


def cliffs_delta(x, y):
    """Cliff's delta for independent samples, using x - y.

    Negative values mean EESNAS tends to have lower NRMSE than the baseline.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return 0.0
    greater = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])
    total = len(x) * len(y)
    return float((greater - less) / total) if total > 0 else 0.0


def mann_whitney_test(esnas_arr, baseline_arr):
    x = np.asarray(esnas_arr, dtype=float)
    y = np.asarray(baseline_arr, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return None
    try:
        two = mannwhitneyu(x, y, alternative='two-sided', method='auto')
        one = mannwhitneyu(x, y, alternative='less', method='auto')
    except ValueError:
        return None
    return {
        'n_esnas': int(len(x)),
        'n_baseline': int(len(y)),
        'p_two': float(two.pvalue),
        'p_one_less': float(one.pvalue),
        'cliffs_delta': cliffs_delta(x, y),
        'median_esnas': float(np.median(x)),
        'median_baseline': float(np.median(y)),
    }


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([])
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def build_summary():
    raw = {m: load_method(m) for m in METHODS}
    missing = [m for m, v in raw.items() if v is None]
    if missing:
        print(f'Missing aggregates: {missing}')

    summary = {}
    for ds in DATASETS:
        summary[ds] = {}
        for m in METHODS:
            if raw.get(m) is None or ds not in raw[m]:
                continue
            data = raw[m][ds]
            nrmse_arr = per_run(data['nrmse'], ds, method=m)
            r2_arr = per_run(data['r2'], ds, method=m)
            summary[ds][m] = {
                'nrmse': summarize(nrmse_arr, lower_is_better=True),
                'r2': summarize(r2_arr, lower_is_better=False),
                'nrmse_per_run': nrmse_arr.tolist(),
                'r2_per_run': r2_arr.tolist(),
            }

    mann_whitney_tests = {}
    for baseline in BASELINES:
        ds_list = []
        results = {}
        for ds in DATASETS:
            if 'esnas' not in summary[ds] or baseline not in summary[ds]:
                continue
            x = summary[ds]['esnas']['nrmse_per_run']
            y = summary[ds][baseline]['nrmse_per_run']
            res = mann_whitney_test(x, y)
            if res is None:
                continue
            results[ds] = res
            ds_list.append(ds)
        if not ds_list:
            wilcoxon_tests[baseline] = {}
            continue
        p_two = [results[d]['p_two'] for d in ds_list]
        p_one = [results[d]['p_one_less'] for d in ds_list]
        q_two = bh_fdr(p_two).tolist()
        q_one = bh_fdr(p_one).tolist()
        for d, qt, qo in zip(ds_list, q_two, q_one):
            results[d]['q_two'] = qt
            results[d]['q_one_less'] = qo
        mann_whitney_tests[baseline] = results

    return {
        'summary': summary,
        'mann_whitney_vs_esnas': mann_whitney_tests,
        'meta': {
            'datasets': DATASETS,
            'methods': METHODS,
            'baselines': BASELINES,
            'num_stretches_ar': NUM_STRETCHES,
            'ar_datasets': sorted(AR_DATASETS),
        },
    }


def print_summary(out):
    summary = out['summary']
    mann_whitney_tests = out['mann_whitney_vs_esnas']

    print('\n=== NRMSE summary (median / mean / std / best) ===')
    print(f'{"dataset":<10}{"method":<14}{"n":>4}{"median":>14}{"mean":>14}{"std":>14}{"best":>14}')
    for ds in DATASETS:
        for m in METHODS:
            if m not in summary[ds]:
                continue
            s = summary[ds][m]['nrmse']
            if 'median' not in s:
                continue
            print(f'{ds:<10}{m:<14}{s["n"]:>4}{s["median"]:>14.6f}{s["mean"]:>14.6f}{s["std"]:>14.6f}{s["best"]:>14.6f}')

    print('\n=== Mann-Whitney U: EESNAS vs baseline (NRMSE; one-sided less) ===')
    print(f'{"baseline":<10}{"dataset":<10}{"n_e":>4}{"n_b":>4}{"p_two":>12}{"q_two":>12}{"p_one":>12}{"q_one":>12}{"delta":>10}')
    for baseline, ds_results in mann_whitney_tests.items():
        for ds in DATASETS:
            if ds not in ds_results:
                continue
            r = ds_results[ds]
            print(f'{baseline:<10}{ds:<10}{r["n_esnas"]:>4}{r["n_baseline"]:>4}'
                  f'{r["p_two"]:>12.4f}{r["q_two"]:>12.4f}'
                  f'{r["p_one_less"]:>12.4f}{r["q_one_less"]:>12.4f}'
                  f'{r["cliffs_delta"]:>10.3f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=DEFAULT_OUT,
                        help=f'Output JSON path (default: {DEFAULT_OUT})')
    args = parser.parse_args()

    out = build_summary()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Wrote {args.out}')
    print_summary(out)


if __name__ == '__main__':
    main()
