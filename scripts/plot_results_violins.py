"""Plot per-model violins for NRMSE and R² across datasets.

Reads aggregated JSONs in results/_aggregate/ produced by the per-model
extractors. Renders a (datasets x {NRMSE, R²}) grid of violins, one violin
per model. Symlog y-axis on both metrics so catastrophic failures are
visible without squashing the good runs.

Run with the root venv:
    venv/Scripts/python scripts/plot_results_violins.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGG = os.path.join(ROOT, 'results', '_aggregate')
OUT_DIR = os.path.join(ROOT, 'results', '_aggregate')

MODELS = ['grid_search', 'lcnn', 'lstm', 'ge_desn', 'bo', 'ga', 'esnas']
MODEL_LABELS = {
    'grid_search': 'Grid search',
    'lcnn': 'LCNN',
    'lstm': 'LSTM',
    'ge_desn': 'GE-DESN',
    'bo': 'BO',
    'ga': 'GA',
    'esnas': 'ESNAS',
}
MODEL_COLORS = {
    'grid_search': '#888888',
    'lcnn': '#4c72b0',
    'lstm': '#8172b3',
    'ge_desn': '#dd8452',
    'bo': '#c44e52',
    'ga': '#937860',
    'esnas': '#55a868',
}
DATASETS = ['lorenz', 'mgs', 'laser', 'dde', 'sunspots', 'water']
METRICS = ['nrmse', 'r2']
METRIC_LABELS = {'nrmse': 'NRMSE (lower is better)',
                 'r2': 'R² (higher is better)'}


def load_all():
    data = {}
    for m in MODELS:
        path = os.path.join(AGG, f'{m}.json')
        if not os.path.exists(path):
            print(f'  warning: {path} missing')
            data[m] = {}
            continue
        with open(path) as f:
            data[m] = json.load(f)
    return data


def draw_violin(ax, values_per_model, metric):
    # Only assign x positions to models that actually have data here.
    parts_data = []
    for m in MODELS:
        vals = values_per_model.get(m)
        if vals is None or len(vals) == 0:
            continue
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue
        parts_data.append((m, arr))

    if not parts_data:
        ax.set_xticks([])
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                ha='center', va='center', color='gray')
        return

    positions = list(range(len(parts_data)))
    labels = [MODEL_LABELS[m] for m, _ in parts_data]

    rng = np.random.default_rng(0)
    for i, (m, arr) in enumerate(parts_data):
        # KDE on log space for NRMSE so violin shape matches the log y-axis
        if metric == 'nrmse':
            log_arr = np.log10(np.clip(arr, 1e-12, None))
            parts = ax.violinplot([log_arr], positions=[i], widths=0.8,
                                  showextrema=False, showmedians=False)
            for body in parts['bodies']:
                # The default polygon uses the (log) y values; we need to
                # re-map them back to data space because we want a log axis.
                # Easier: don't use violinplot's built-in transform — instead
                # compute KDE manually and draw with fill_betweenx in data
                # coords. But for simplicity, just transform the polygon paths.
                paths = body.get_paths()
                for p in paths:
                    p.vertices[:, 1] = 10 ** p.vertices[:, 1]
                body.set_facecolor(MODEL_COLORS[m])
                body.set_edgecolor(MODEL_COLORS[m])
                body.set_alpha(0.5)
        else:
            parts = ax.violinplot([arr], positions=[i], widths=0.8,
                                  showextrema=False, showmedians=False)
            for body in parts['bodies']:
                body.set_facecolor(MODEL_COLORS[m])
                body.set_edgecolor(MODEL_COLORS[m])
                body.set_alpha(0.5)

        x_jit = i + rng.uniform(-0.12, 0.12, size=len(arr))
        ax.scatter(x_jit, arr, s=8, color=MODEL_COLORS[m],
                   edgecolor='black', linewidth=0.3, alpha=0.8, zorder=3)
        med = np.median(arr)
        ax.hlines(med, i - 0.3, i + 0.3, color='black', linewidth=1.5, zorder=4)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha='right')

    all_arr = np.concatenate([a for _, a in parts_data])
    data_min, data_max = float(np.min(all_arr)), float(np.max(all_arr))

    if metric == 'nrmse':
        ax.set_yscale('log')
        # Pad in log space.
        lo = data_min * 0.5 if data_min > 0 else 1e-6
        hi = data_max * 2.0
        ax.set_ylim(lo, hi)
    else:  # r2
        # asinh compresses catastrophic-failure tails while keeping [0,1]
        # readable. But on narrow good-only clusters (e.g. sunspots) it
        # over-produces ticks, so use linear there.
        if data_min > -0.5:
            ax.set_yscale('linear')
        else:
            ax.set_yscale('asinh', linear_width=0.5)
        span = max(data_max - data_min, 1e-3)
        pad = 0.05 * span
        ax.set_ylim(data_min - pad, min(data_max + pad, 1.05))
        ax.axhline(1.0, color='green', linestyle=':', linewidth=0.8, alpha=0.6)
        if data_min < 0 < data_max:
            ax.axhline(0.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.grid(True, axis='y', which='both', alpha=0.3)


def render_combined(data):
    """Render a 3x4 grid: NRMSE in left two cols, R² in right two cols.

    Row r has datasets [2r, 2r+1]; each dataset gets its NRMSE panel on the
    left half and its R² panel on the right half.
    """
    n_rows, n_cols = 3, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 2.6 * n_rows),
                             squeeze=False)

    for idx, ds in enumerate(DATASETS):
        r = idx // 2
        c_off = idx % 2  # 0 or 1 within the pair
        for m_idx, metric in enumerate(METRICS):
            c = m_idx * 2 + c_off
            ax = axes[r][c]
            values_per_model = {}
            for m in MODELS:
                vals = data.get(m, {}).get(ds, {}).get(metric)
                if vals is not None:
                    values_per_model[m] = vals
            draw_violin(ax, values_per_model, metric)
            label = f'{ds} — {METRIC_LABELS[metric]}'
            ax.set_ylabel(label, fontsize=10, fontweight='bold')

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, 'violins.pdf')
    fig.savefig(out_path)
    plt.close(fig)
    print(f'Wrote {out_path}')


def main():
    data = load_all()
    render_combined(data)


if __name__ == '__main__':
    main()
