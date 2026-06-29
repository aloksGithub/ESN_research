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

This branch is JAX-specific and should be run from WSL with Python 3.12.3.
Open a WSL shell, change to the repository root, and verify the Python version:

```bash
cd /mnt/c/path/to/ESN_research
python3.12 --version
```

The version should be Python 3.12.3. Create or refresh the main JAX environment:

```bash
python3.12 -m venv venv_wsl
source venv_wsl/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

For an existing checkout that already has `venv_wsl`, activate it before running
any project command:

```bash
source venv_wsl/bin/activate
```
```

Run all commands from the repository root in WSL.
