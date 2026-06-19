# Evolutionary Echo State Network Architecture Search

This repository contains the code, datasets, saved experiment artifacts, and
analysis scripts for the article **"Evolutionary Echo State Network Architecture
Search for Time-series Prediction"**.

The main method, **EESNAS** (Evolutionary Echo State Network Architecture
Search), combines a genetic algorithm for ESN architecture search with Bayesian
optimization for local hyperparameter tuning. The experiments compare EESNAS
against GA-only and BO-only ablations, reservoir-computing baselines, grid
search, and an LSTM baseline on benchmark time-series prediction tasks.

## Repository layout

```text
data/                  Benchmark datasets used by the experiments
experiments/           Main experiment entry points
src/                   EESNAS, GA, BO, datasets, metrics, and baselines
scripts/               Result extraction, aggregation, plotting, and analysis
results/               Saved models, checkpoints, aggregate JSONs, and figures
analysis/              Derived CSV analyses used in the paper
```

The `thesis/` and `references/` folders are manuscript/reference material and
are not required for reproducing the code results.

## Implemented methods

- `EESNAS`: hybrid GA + BO architecture search (`experiments/esnas.py`,
  `src/algorithms/ESN_GA_BO.py`)
- `GA-only`: architecture search without BO (`experiments/ga.py`)
- `BO-only`: Bayesian optimization on a fixed ESN architecture
  (`experiments/bo.py`)
- `LCNN`: Locally Connected Neural Network baseline
  (`experiments/lcnn.py`, `src/baselines/lcnn/`)
- `GE-DESN`: Growing Evolutional Deep ESN baseline
  (`experiments/ge_desn.py`, `src/baselines/ge_desn/`)
- `Grid search`: published ESN grid-search baseline
  (`experiments/grid_search.py`, `src/baselines/grid_search/`)
- `LSTM`: Bayesian-optimized LSTM baseline
  (`experiments/lstm.py`, `src/baselines/lstm/`)

## Datasets and evaluation

The experiments use six datasets:

- Lorenz
- Mackey-Glass (`mgs`)
- Santa Fe laser (`laser`)
- Neutral delayed differential equation (`dde`)
- Sunspots
- Water

Lorenz, Mackey-Glass, laser, and DDE are evaluated using autoregressive
multi-step prediction. Sunspots and water are evaluated using next-step
prediction. Results are reported with NRMSE and R2.

## Environment setup

The project was developed with Python 3.9. The command examples below assume
Windows PowerShell and the virtual-environment layout already used in this
repository.

Create or refresh the main EESNAS environment:

```powershell
py -3.9 -m venv envs\esnas
envs\esnas\Scripts\python -m pip install --upgrade pip
envs\esnas\Scripts\python -m pip install -r requirements.txt
```

The external baselines have separate dependency sets:

```powershell
py -3.9 -m venv envs\lcnn
envs\lcnn\Scripts\python -m pip install --upgrade pip
envs\lcnn\Scripts\python -m pip install -r src\baselines\lcnn\requirements.txt
envs\lcnn\Scripts\python -m pip install pandas

py -3.9 -m venv envs\ge_desn
envs\ge_desn\Scripts\python -m pip install --upgrade pip
envs\ge_desn\Scripts\python -m pip install -r src\baselines\ge_desn\requirements.txt

py -3.9 -m venv envs\grid_search
envs\grid_search\Scripts\python -m pip install --upgrade pip
envs\grid_search\Scripts\python -m pip install -r src\baselines\grid_search\requirements.txt

py -3.9 -m venv envs\lstm
envs\lstm\Scripts\python -m pip install --upgrade pip
envs\lstm\Scripts\python -m pip install -r src\baselines\lstm\requirements.txt
```

Run all commands from the repository root.

## Reproducing saved paper results

The repository includes saved result artifacts under `results/`. These commands
reload the saved models/checkpoints and re-run the evaluation path on the test
data.

### EESNAS, GA-only, and BO-only

Reproduce the saved EESNAS, GA-only, and BO-only results with:

```powershell
envs\esnas\Scripts\python scripts\print_saved_results.py
```

This writes:

```text
results/_aggregate/esnas.json
results/_aggregate/ga.json
results/_aggregate/bo.json
```

To run only selected methods or datasets:

```powershell
envs\esnas\Scripts\python scripts\print_saved_results.py --methods esnas ga bo
envs\esnas\Scripts\python scripts\print_saved_results.py --datasets lorenz mgs laser dde sunspots water
```

### External baselines

Reproduce external baselines by running the corresponding experiment script with
`--eval-only`:

```powershell
envs\lcnn\Scripts\python experiments\lcnn.py --eval-only
envs\ge_desn\Scripts\python experiments\ge_desn.py --eval-only
envs\grid_search\Scripts\python experiments\grid_search.py --eval-only
envs\lstm\Scripts\python experiments\lstm.py --eval-only
```

For LCNN, GE-DESN, and LSTM, pass `--datasets` to evaluate a subset:

```powershell
envs\lcnn\Scripts\python experiments\lcnn.py --eval-only --datasets laser mgs
```

For grid search, an optional positional dataset index can be supplied:

```text
0 = dde, 1 = lorenz, 2 = mgs, 3 = laser
```

For example:

```powershell
envs\grid_search\Scripts\python experiments\grid_search.py 1 --eval-only
```

## Regenerating aggregate files and figures

After running the evaluation commands, regenerate aggregate JSON files for the
external baselines:

```powershell
envs\lcnn\Scripts\python scripts\extract_lcnn_results.py
envs\ge_desn\Scripts\python scripts\extract_ge_desn_results.py
envs\grid_search\Scripts\python scripts\extract_grid_search_results.py
envs\lstm\Scripts\python scripts\extract_lstm_results.py
```

Build the summary statistics JSON:

```powershell
envs\esnas\Scripts\python scripts\aggregate_results.py --out results\_aggregate\results_summary.json
```

Regenerate the aggregate plots:

```powershell
envs\esnas\Scripts\python scripts\plot_results_violins.py
envs\esnas\Scripts\python scripts\plot_convergence_curves.py
```

Main generated outputs include:

```text
results/_aggregate/*.json
results/_aggregate/violins.pdf
results/_aggregate/convergence_curves.pdf
results/_aggregate/convergence_curves.png
```

## Running experiments from scratch

The saved artifacts are the recommended path for reproducing the paper results.
Full reruns are computationally expensive and can take many hours depending on
hardware.

Run EESNAS and GA-only full experiments:

```powershell
envs\esnas\Scripts\python experiments\esnas.py
envs\esnas\Scripts\python experiments\ga.py
```

Run BO-only for a single repeat index:

```powershell
envs\esnas\Scripts\python experiments\bo.py 0
```

Run external baselines from scratch by omitting `--eval-only`:

```powershell
envs\lcnn\Scripts\python experiments\lcnn.py
envs\ge_desn\Scripts\python experiments\ge_desn.py
envs\grid_search\Scripts\python experiments\grid_search.py
envs\lstm\Scripts\python experiments\lstm.py
```

## Notes

- Commands assume the current working directory is the repository root.
- Saved EESNAS/GA/BO checkpoints are stored as `backup_*.obj` under
  `results/{esnas,ga,bo}/`.
- Saved external-baseline checkpoints are stored under
  `results/{lcnn,ge_desn,grid_search,lstm}/`.
- The precomputed datasets needed by the experiments are included in `data/`.

## License

This project is released under the terms of the license in `LICENSE`.
