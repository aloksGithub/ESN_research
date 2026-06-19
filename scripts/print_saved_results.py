"""Re-evaluate saved EESNAS / GA / BO experiments and write aggregates.

Running this script evaluates saved best models on fresh test data, extracts
validation convergence traces, and writes:
    results/_aggregate/esnas.json
    results/_aggregate/ga.json
    results/_aggregate/bo.json

The extract_{esnas,ga,bo}_results.py scripts are thin compatibility wrappers
around this module.

Run with the esnas venv:
    envs/esnas/Scripts/python scripts/print_saved_results.py
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import reservoirpy as rpy
rpy.verbosity(0)

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from src.utils import runModel
from src.algorithms.ESN_BO import ESN_BO  # noqa: F401  (needed for unpickle)
from src.algorithms.ESN_GA import ESN_GA  # noqa: F401
from src.datasets import getDataDDE, getDataLaser, getDataLorenz, getDataMGS
from convergence_utils import (
    convergence_from_bo_experiment,
    convergence_from_ga_experiment,
    median_convergence,
)

DATASETS = ['lorenz', 'mgs', 'laser', 'dde', 'sunspots', 'water']
AUTOREGRESSIVE = {'lorenz', 'mgs', 'laser', 'dde'}
AGG_DIR = os.path.join(parent_dir, 'results', '_aggregate')

DATASET_LOADERS = {
    'mgs': getDataMGS,
    'lorenz': getDataLorenz,
    'dde': getDataDDE,
    'laser': getDataLaser,
}

METHODS = {
    'esnas': {
        'src': os.path.join(parent_dir, 'results', 'esnas'),
        'out': os.path.join(AGG_DIR, 'esnas.json'),
        'is_ga': True,
        'label': 'EESNAS',
    },
    'ga': {
        'src': os.path.join(parent_dir, 'results', 'ga'),
        'out': os.path.join(AGG_DIR, 'ga.json'),
        'is_ga': True,
        'label': 'GA-only',
    },
    'bo': {
        'src': os.path.join(parent_dir, 'results', 'bo'),
        'out': os.path.join(AGG_DIR, 'bo.json'),
        'is_ga': False,
        'label': 'BO-only',
    },
}


def readSavedExperiment(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _repeat_time(exp, isGA: bool) -> float:
    """Extract total time spent on this repeat."""
    try:
        if isGA:
            return float(sum(exp.generationTimes))
        return float(sum(exp.times))
    except Exception:
        return 0.0


def collect_saved_experiment(directory: str, dataset: str,
                             isAutoregressive: bool = True,
                             isGA: bool = True,
                             num_repeats: int = 5) -> dict:
    """Re-evaluate one (algorithm, dataset) and return per-repeat metrics."""
    testX = testY = warmupX = None
    if isAutoregressive and dataset in DATASET_LOADERS:
        _, _, _, _, testX, testY, warmupX, _ = DATASET_LOADERS[dataset]()

    nrmse_per_repeat = []
    r2_per_repeat = []
    time_per_repeat = []

    for i in range(num_repeats):
        path = os.path.join(directory, dataset, f'backup_{i}.obj')
        if not os.path.exists(path):
            continue
        exp = readSavedExperiment(path)
        time_per_repeat.append(_repeat_time(exp, isGA))
        best_model = exp.bestModel

        if isAutoregressive:
            ns, rs = [], []
            for j in range(len(testX)):
                runModel(best_model, warmupX[j])
                prev = testX[j][0]
                preds = []
                for _ in range(len(testX[j])):
                    prev = runModel(best_model, prev)
                    preds.append(prev[0])
                preds = np.array(preds)
                ns.append(float(exp.evalParams.errorMetrics[0](testY[j], preds)))
                rs.append(float(exp.evalParams.errorMetrics[1](testY[j], preds)))
            nrmse_per_repeat.append(float(np.mean(ns)))
            r2_per_repeat.append(float(np.mean(rs)))
        else:
            runModel(best_model, exp.experimentData.valX)
            preds = runModel(best_model, exp.experimentData.testX)
            nrmse_per_repeat.append(float(
                exp.evalParams.errorMetrics[0](exp.experimentData.testY, preds)
            ))
            r2_per_repeat.append(float(
                exp.evalParams.errorMetrics[1](exp.experimentData.testY, preds)
            ))

    return {
        'nrmse': nrmse_per_repeat,
        'r2': r2_per_repeat,
        'times': time_per_repeat,
    }


def collect_saved_convergences(directory: str, dataset: str,
                               isGA: bool = True,
                               num_repeats: int = 5) -> dict:
    """Read validation convergence traces from saved GA / BO / EESNAS repeats."""
    convergences = []
    for i in range(num_repeats):
        path = os.path.join(directory, dataset, f'backup_{i}.obj')
        if not os.path.exists(path):
            continue
        exp = readSavedExperiment(path)
        if isGA:
            conv = convergence_from_ga_experiment(i, exp)
        else:
            conv = convergence_from_bo_experiment(i, exp)
        if conv is not None:
            convergences.append(conv)

    return {
        'convergence': convergences,
        'median_convergence': median_convergence(convergences),
    }


def collect_method(method: str, datasets=None, write: bool = True) -> dict:
    """Collect one EESNAS/GA/BO method and optionally write its JSON file."""
    if method not in METHODS:
        raise ValueError(f'Unknown method: {method}')
    cfg = METHODS[method]
    datasets = DATASETS if datasets is None else datasets
    out = {}

    print(f'=============================={cfg["label"]}==============================')
    for ds in datasets:
        if not os.path.isdir(os.path.join(cfg['src'], ds)):
            print(f'  {ds}: missing dir, skipping')
            continue
        print(f'  {ds}...')
        res = collect_saved_experiment(
            cfg['src'], ds,
            isAutoregressive=ds in AUTOREGRESSIVE,
            isGA=cfg['is_ga'],
        )
        res.update(collect_saved_convergences(cfg['src'], ds, isGA=cfg['is_ga']))
        out[ds] = res
        print(
            f'  {ds}: {len(res["nrmse"])} repeats, '
            f'{len(res.get("convergence", []))} convergence traces'
        )

    if write:
        os.makedirs(os.path.dirname(cfg['out']), exist_ok=True)
        with open(cfg['out'], 'w') as f:
            json.dump(out, f, indent=2)
        print(f'Wrote {cfg["out"]}')
    return out


def collect_all(methods=None, datasets=None, write: bool = True) -> dict:
    methods = list(METHODS) if methods is None else methods
    return {method: collect_method(method, datasets=datasets, write=write)
            for method in methods}


def printSavedResults(directory: str, dataset: str,
                      isAutoregressive: bool = True,
                      isGA: bool = True) -> None:
    """Backward-compatible printer for one saved method/dataset pair."""
    res = collect_saved_experiment(directory, dataset, isAutoregressive, isGA)
    print('=' * 62)
    print(f'{dataset} Errors (per-repeat means):')
    print('NRMSE:', res['nrmse'])
    print('R2:', res['r2'])
    if res['nrmse']:
        print(f"NRMSE: {np.average(res['nrmse'])} ({np.std(res['nrmse'])})")
        print(f"R2: {np.average(res['r2'])} ({np.std(res['r2'])})")
    print('Times:', res['times'])
    if res['times']:
        print(f"Average time: {np.average(res['times'])} ({np.std(res['times'])})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods', nargs='+', choices=sorted(METHODS),
                        default=None, help='Subset to collect (default: all)')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS,
                        default=None, help='Subset to collect (default: all)')
    args = parser.parse_args()
    collect_all(methods=args.methods, datasets=args.datasets, write=True)
