"""Re-evaluate saved LSTM models and write aggregate JSON.

This mirrors:
    envs/lstm/Scripts/python experiments/lstm.py --eval-only

The fresh test metrics are produced by loading saved model checkpoints and
running the experiment evaluation path. Times and convergence traces still
come from the per-repeat pickle metadata.
"""
import json
import os
import pickle
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from convergence_utils import convergence_from_optim_log, median_convergence
from experiments.lstm import (
    AUTOREGRESSIVE_DATASETS,
    evaluate_saved,
    getDataDDE,
    getDataLaser,
    getDataLorenz,
    getDataMGS,
    getDataSunspots,
    getDataWater,
    nrmse_sunspots,
)

DATASETS = {
    'laser': getDataLaser,
    'mgs': getDataMGS,
    'lorenz': getDataLorenz,
    'dde': getDataDDE,
    'sunspots': getDataSunspots,
    'water': getDataWater,
}
NRMSE_OVERRIDES = {'sunspots': nrmse_sunspots}
SRC = os.path.join(ROOT, 'results', 'lstm')
OUT = os.path.join(ROOT, 'results', '_aggregate', 'lstm.json')


def _repeat_values(values, dataset):
    arr = np.asarray(values, dtype=float)
    if dataset in AUTOREGRESSIVE_DATASETS and len(arr) > 5 and len(arr) % 5 == 0:
        arr = arr.reshape(-1, 5).mean(axis=1)
    return arr.tolist()


def _load_metadata(ds_dir):
    times, convergences = [], []
    rep = 0
    while True:
        pkl_path = os.path.join(ds_dir, f'repeat_{rep}.pkl')
        if not os.path.exists(pkl_path):
            break
        with open(pkl_path, 'rb') as f:
            d = pickle.load(f)
        times.append(float(d.get('elapsed', 0.0)))
        conv = convergence_from_optim_log(rep, d.get('optim_log'))
        if conv is not None:
            convergences.append(conv)
        rep += 1
    return times, convergences


def main():
    out = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    for ds, loader in DATASETS.items():
        ds_dir = os.path.join(SRC, ds)
        if not os.path.isdir(ds_dir):
            print(f'  {ds}: missing dir, skipping')
            continue
        nrmses, r2s = evaluate_saved(
            ds, loader,
            nrmse_func=NRMSE_OVERRIDES.get(ds),
            device=device,
        )
        nrmse = _repeat_values(nrmses, ds)
        r2 = _repeat_values(r2s, ds)
        times, convergences = _load_metadata(ds_dir)
        if nrmse:
            out[ds] = {
                'nrmse': nrmse,
                'r2': r2,
                'times': times[:len(nrmse)],
                'convergence': convergences,
                'median_convergence': median_convergence(convergences),
            }
            print(f'  {ds}: {len(nrmse)} repeats')
        else:
            print(f'  {ds}: no saved models, skipping')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Wrote {OUT}')


if __name__ == '__main__':
    main()
