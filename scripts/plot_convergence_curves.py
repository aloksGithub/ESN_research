"""Plot median-repeat validation convergence curves for all algorithms.

Reads the `convergence` data written by extract_*_results.py. For each
(algorithm, dataset), the displayed run is the repeat with the median final
best validation error.

Run after the extractors:
    envs/esnas/Scripts/python scripts/plot_convergence_curves.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from convergence_utils import median_convergence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGG = os.path.join(ROOT, 'results', '_aggregate')
OUT_DIR = os.path.join(ROOT, 'results', '_aggregate')
THESIS_FIGURES = os.path.join(ROOT, 'thesis', 'figures')

DATASETS = ['lorenz', 'mgs', 'laser', 'dde', 'sunspots', 'water']
MAX_TIME_SECONDS = 45000
X_AXIS_LEFT_BUFFER = 0.02 * MAX_TIME_SECONDS
ALGORITHMS = ['esnas', 'ga', 'bo', 'lstm', 'ge_desn', 'lcnn']
LABELS = {
    'esnas': 'EESNAS',
    'ga': 'GA',
    'bo': 'BO',
    'lstm': 'LSTM',
    'ge_desn': 'GE-DESN',
    'lcnn': 'LCNN',
}
COLORS = {
    'esnas': '#1f77b4',
    'ga': '#d62728',
    'bo': '#2ca02c',
    'lstm': '#9467bd',
    'ge_desn': '#ff7f0e',
    'lcnn': '#17becf',
}


def load_aggregates():
    data = {}
    for algorithm in ALGORITHMS:
        path = os.path.join(AGG, f'{algorithm}.json')
        if not os.path.exists(path):
            print(f'  warning: {path} missing')
            data[algorithm] = {}
            continue
        with open(path) as f:
            data[algorithm] = json.load(f)
    return data


def _selected_convergence(dataset_data):
    conv = dataset_data.get('median_convergence')
    if conv:
        return conv
    return median_convergence(dataset_data.get('convergence', []))


def _finite_curve(conv):
    times = np.asarray(conv.get('times', []), dtype=float)
    errors = np.asarray(conv.get('validation_error', []), dtype=float)
    n = min(len(times), len(errors))
    if n == 0:
        return None, None
    times = times[:n]
    errors = errors[:n]
    mask = np.isfinite(times) & np.isfinite(errors)
    mask &= errors > 0
    mask &= times <= MAX_TIME_SECONDS
    return times[mask], errors[mask]


def draw_dataset(ax, data, dataset):
    plotted = False
    y_values = []
    for algorithm in ALGORITHMS:
        ds_data = data.get(algorithm, {}).get(dataset, {})
        conv = _selected_convergence(ds_data)
        if not conv:
            continue
        times, errors = _finite_curve(conv)
        if times is None or len(times) == 0:
            continue
        ax.plot(
            times,
            errors,
            color=COLORS[algorithm],
            linewidth=1.8,
            label=f'{LABELS[algorithm]} (rep {conv.get("repeat", "?")})',
        )
        y_values.append(errors)
        plotted = True

    ax.set_title(dataset)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Best validation error')
    ax.set_xlim(-X_AXIS_LEFT_BUFFER, MAX_TIME_SECONDS)
    ax.grid(True, which='both', alpha=0.3)
    if plotted:
        all_y = np.concatenate(y_values)
        if np.all(all_y > 0):
            ax.set_yscale('log')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'no convergence data',
                transform=ax.transAxes, ha='center', va='center',
                color='gray')


def render_individual(data):
    os.makedirs(THESIS_FIGURES, exist_ok=True)
    for dataset in DATASETS:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        draw_dataset(ax, data, dataset)
        fig.tight_layout()
        out_path = os.path.join(THESIS_FIGURES, f'{dataset}_convergence.pdf')
        fig.savefig(out_path)
        plt.close(fig)
        print(f'Wrote {out_path}')


def render_combined(data):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), squeeze=False)
    for idx, dataset in enumerate(DATASETS):
        ax = axes[idx // 2][idx % 2]
        draw_dataset(ax, data, dataset)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        out_path = os.path.join(OUT_DIR, f'convergence_curves.{ext}')
        fig.savefig(out_path, dpi=200 if ext == 'png' else None)
        print(f'Wrote {out_path}')
    plt.close(fig)


def main():
    data = load_aggregates()
    render_individual(data)
    render_combined(data)


if __name__ == '__main__':
    main()
