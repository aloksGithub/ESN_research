"""Write the legacy ablation statistics CSV from the aggregate summary.

The Mann-Whitney U implementation now lives in scripts/aggregate_results.py.
This script is kept for compatibility with older analysis notes that expect:
    analysis/wilcoxon_ablation.csv
"""
import csv
import os

from aggregate_results import DATASETS, build_summary

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'analysis', 'wilcoxon_ablation.csv')


def main():
    out = build_summary()
    tests = out['mann_whitney_vs_esnas']
    summary = out['summary']
    rows = []
    for method, label in [('ga', 'GA-only'), ('bo', 'BO-only')]:
        for ds in DATASETS:
            if ds not in tests.get(method, {}):
                continue
            r = tests[method][ds]
            rows.append({
                'dataset': ds,
                'metric': 'NRMSE',
                'baseline': label,
                'test': 'Mann-Whitney U',
                'n_esnas': r['n_esnas'],
                'n_baseline': r['n_baseline'],
                'median_esnas': r['median_esnas'],
                'median_baseline': r['median_baseline'],
                'median_delta_esnas_minus_baseline': (
                    r['median_esnas'] - r['median_baseline']
                ),
                'p_two_sided': r['p_two'],
                'q_two_sided': r['q_two'],
                'p_esnas_better': r['p_one_less'],
                'q_esnas_better': r['q_one_less'],
                'cliffs_delta': r['cliffs_delta'],
                'esnas_nrmse_per_run': summary[ds]['esnas']['nrmse_per_run'],
                f'{method}_nrmse_per_run': summary[ds][method]['nrmse_per_run'],
            })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(OUT, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f'Wrote {OUT}')


if __name__ == '__main__':
    main()
