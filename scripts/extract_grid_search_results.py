"""Extract grid_search per-seed NRMSE and R² to JSON.

Each seed already stores `nrmse` / `r2` keys with the per-test-set means,
so one value per seed (~100 per dataset). No water/sunspots — those
weren't run for grid_search.

Run with the grid_search venv:
    envs/grid_search/Scripts/python scripts/extract_grid_search_results.py
"""
import json
import os
import pickle

DATASETS = ['lorenz', 'mgs', 'laser', 'dde']
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'results', 'grid_search')
OUT = os.path.join(ROOT, 'results', '_aggregate', 'grid_search.json')


def main():
    out = {}
    for ds in DATASETS:
        path = os.path.join(SRC, f'{ds}_results.pkl')
        if not os.path.exists(path):
            print(f'  {ds}: missing, skipping')
            continue
        with open(path, 'rb') as f:
            stats = pickle.load(f)
        nrmse = [float(s['nrmse']) for s in stats['per_seed_results']]
        r2 = [float(s['r2']) for s in stats['per_seed_results']]
        out[ds] = {'nrmse': nrmse, 'r2': r2}
        print(f'  {ds}: {len(nrmse)} seeds')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Wrote {OUT}')


if __name__ == '__main__':
    main()
