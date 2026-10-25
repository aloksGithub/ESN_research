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

## CeTI-Locomotion SVM and ESNAS experiments

Keep the external dataset at `ceti/CeTI-Locomotion/` (ignored by Git). Copy it
separately when setting up another PC. Shared data formatting, subject splits,
sequence handling, and the paper SVM pipeline live in the `ceti/` package:

```text
ceti/
  CeTI-Locomotion/    External data and supplied scripts; gitignored
  data.py            Subject splits, interpolation, flattened/sequence inputs
  models.py          ESN sequence training, state reset and segment metrics
  search.py          Sequence evaluator for the existing GA + BO ESNAS engine
  svm.py             Supplied paper SVM pipeline and fold selection
```

Use native Windows Python with the main branch dependencies, for example:

```powershell
py -3.11 -m venv envs\main
envs\main\Scripts\python -m pip install -r requirements.txt
```

Run the SVM at either temporal resolution:

```powershell
envs\main\Scripts\python experiments\svm_ceti.py --frames 20
envs\main\Scripts\python experiments\svm_ceti.py --frames 100
```

`--sampling-rate` is an alias for `--frames`: this is the number of interpolated
frames **per complete motion segment, not Hz**. The SVM preserves the supplied
`MaxAbsScaler -> PCA() -> SVC(probability=True)` pipeline, stratified ten-fold
cross-validation by segment, selection by maximum fold balanced accuracy, and
testing of that fitted fold estimator without refitting. It reads TSV values
as float32 before interpolation, like the supplied code. Default training uses
all segments from the 40 non-test subjects. Use `--training-subset small` to
train on the same class/subject-balanced subset used by ESNAS. `--n-jobs`
controls SVM CV workers and defaults to one, like the supplied code.

The shared test subjects come from the explicit manifest at
`results/ceti_timing/subject_split.json`; copy that file to the other PC to
preserve the split. If absent, the loader creates it deterministically. These
are our saved test IDs, not the paper's undocumented original selection.
Neither the SVM pipeline nor the ESNAS search fits preprocessing using tests.

The ESNAS entry point is intended for the experiment PC:

```powershell
envs\main\Scripts\python experiments\esnas_ceti.py
```

Initial ESNAS defaults (all exposed as CLI options):

| Parameter | Default |
|---|---:|
| Frames per segment | 20 |
| Workers (`--n-jobs`) | 20 |
| Training segments per class / validation segments | 40 / all validation-subject segments (964 with the saved split) |
| Training / validation timestep rows | 4,000 / 19,280 with the saved split |
| Input columns / one-hot output columns | 102 / 5 |
| Generations / population | 10 / 20 |
| BO initial random points / iterations | 0 / 3 |
| Initializations per candidate / experiment repeats | 3 / 1 |
| Initial intermediate-node range | 1–2 |
| Warmup / terminal Ridge fitting passes | 3 / 2 |
| Time budget per initialization / memory estimate limit | 60 seconds / 756 MiB |

These parameters are initial guesses. All main-branch node types remain
available, subject to existing validity and memory checks. Fitness minimizes
the mean `1 - segment macro-F1` across three fixed initializations, with mean
output activity after warmup choosing one class per segment. All initializations
must succeed. The saved model uses the first predetermined initialization of
the winning architecture; validation does not select a lucky seed. Reports
distinguish the mean search score from that model's own validation score and
include validation scores per subject. This CeTI policy differs from main's
best-initialization policy for other experiments.

Validation now uses every recording from the same ten validation subjects,
instead of just 50 recordings. Training remains at 200 segments and the test
subject split stays fixed. `--val-per-class 0` selects full validation (the
default); a positive value restores a smaller balanced subset. The larger
validation set costs more inference time for each candidate, without increasing
training rows. It reduces selection noise but cannot guarantee unseen-subject
performance or eliminate overfitting from repeated searches. The default is one
complete search; use `--repeats 5` when measuring variability for a settled setup.

The evaluator resets state and auxiliary history between recordings and supports
native multiprocessing/checkpoint pickling. Time budgets are combined across each BO
worker's `num_evals * (1 + bo_init + bo_iter)` initializations, including the
default BO probe. This is a worker-level deadline, not a separate hard timer
for each individual fit. Test data is loaded after model selection.

Outputs are saved in new directories under `results/svm_ceti/` and
`results/esnas_ceti/`. ESNAS writes per-repeat checkpoints, validation/test
metrics, best architectures and predictions. The SVM writes fold scores,
selected-fold metadata and test predictions; `--save-model` also saves its
selected pipeline.

Use these explicitly bounded checks rather than launching the full search:

```powershell
# Data/configuration only; constructs no ESNAS population or models.
envs\main\Scripts\python experiments\esnas_ceti.py --prepare-only

# Two CV folds on 20 training segments; not a baseline result.
envs\main\Scripts\python experiments\svm_ceti.py --frames 20 --smoke-test
envs\main\Scripts\python experiments\svm_ceti.py --frames 100 --smoke-test

# One generation, four tiny seed architectures, two workers, one repeat,
# with a capped validation subset of 10 segments per class.
envs\main\Scripts\python experiments\esnas_ceti.py --smoke-test
```

## CeTI-Locomotion CPU timing pilot

`experiments/ceti_esn_timing.py` measures a bounded set of ESN profiles without
running GA/BO search or evaluating test subjects. It uses the main branch's
NumPy/SciPy backend and constructors, including Reservoir, IPReservoir, NVAR,
Ridge, LMS, and RLS. Use the native Windows environment `envs/main` (Python
3.11.9, with this branch's pinned `requirements.txt`); the existing WSL `.venv`
belongs to a different backend.

```powershell
envs\main\Scripts\python experiments\ceti_esn_timing.py --repeats 1 --timeout 20 --max-seconds 180
```

Defaults select 40 complete segments per activity for training and 10 per
activity for validation, resampled to 20 frames: **4,000 training and 1,000
validation timesteps**, with five one-hot targets. Sampling spreads segments
across subjects. The full 102 input columns and the six bilateral hip/knee/ankle
angles (`leg6`) use the same segments. Scaling is fitted only on training data.
The first run saves explicit subject IDs in
`results/ceti_timing/subject_split.json`: 30 training, 10 validation, and 10
reserved test subjects. This reproducible split is **not claimed to reproduce
the paper's test IDs**. Subsequent runs reuse it. Test motion contents are not
loaded.

Each trial runs in a separate process with a wall-clock timeout. Reports record
construction, fitting (including lazy initialization), model-copy, validation,
and process times; validation accuracy/F1 are sanity checks on this tiny subset.
State is reset per segment, including the auxiliary NVAR/IP history that
ReservoirPy 0.3's ordinary reset does not clear. The timing adapter preserves
main's two terminal-Ridge fitting passes, with a three-frame warmup appropriate
for these short recordings. Existing training functions are not modified.

Useful variations:

```powershell
# Check only the subset/split, without constructing models.
envs\main\Scripts\python experiments\ceti_esn_timing.py --prepare-only

# Measure worker contention, with two initializations per profile.
envs\main\Scripts\python experiments\ceti_esn_timing.py --workers 4 --repeats 2

# Include heavier IP, RLS, Ridge and nonlinear-expansion profiles.
envs\main\Scripts\python experiments\ceti_esn_timing.py --cases ip512_ridge ip1500_ridge reservoir2000_ridge reservoir400_rls nvar3_ridge --workers 4 --repeats 1

# Keep six channels and increase the data budget to 10,000 / 2,000 timesteps.
envs\main\Scripts\python experiments\ceti_esn_timing.py --feature-sets leg6 --train-per-class 100 --val-per-class 20

# Select exact channels, or retain only particular profiles.
envs\main\Scripts\python experiments\ceti_esn_timing.py --feature-sets LeftHip_flexion,RightHip_flexion --cases nvar_ridge reservoir256_rls
```

Each run creates a new directory under `results/ceti_timing/` containing
`config.json`, `subset.json`, `subset.npz`, `trials.jsonl`, `timings.csv`, and
`report.json`. The subset manifest identifies every selected segment. Reports
retain failed/timed-out cases, and preflight skips allocations exceeding the
existing ESNAS memory/readout-width limits (notably NVAR on all 102 channels).
The nominal search projection includes the initial population, the default BO
probe, all initialization attempts, and five repeats. Its ideal parallel times
exclude overhead, contention, stagnation resets, and unmeasured architectures;
they are illustrative rather than a full-search runtime prediction. Increase
`--timeout` for censored cases before drawing conclusions about their speed.

The supplied SVM code actually selects the best scoring inner-fold estimator,
without refitting. Its subject split uses `list(set(participant_ids))`, so the
NumPy seed alone does not identify a portable paper test split. The pilot does
not execute that cross-validation pipeline or attempt to recover undocumented
test IDs.

Focused checks:

```powershell
envs\main\Scripts\python -m pytest tests\test_ceti_timing.py -q --basetemp .cache\ceti-tests
```

The native environment's resolved dependency versions are recorded in
`envs/main/requirements.freeze.txt`, and each timing run records its core package
versions and Python executable in `config.json`.

## CeTI-Locomotion full-holdout feasibility check

After the timing pilot, train fixed slower profiles on exactly the saved
4,000-row training subset and evaluate every segment of the reserved subjects:

```powershell
envs\main\Scripts\python experiments\ceti_esn_test.py
```

Defaults are five random initializations each of Reservoir 2000 + Ridge and
IPReservoir 1500 + Ridge, all 102 input columns, and four CPU workers. The
training subset, scaling, three-frame warmup, two Ridge passes and 20-frame
segment representation come from `results/ceti_timing/stress_pilot`. No
validation examples are added to training. The same explicit ten-subject
holdout is used; all of its motion segments are loaded, not a small test sample.
Training has a 60-second timeout per model, with a separate 180-second allowance
for full-test inference. A ten-minute overall budget bounds the batch.

Profiles and seeds are saved before test motions are loaded. The report keeps
every model, per-class metrics, per-subject metrics and per-segment predictions;
it does not choose the best test-scoring model. This is a feasibility check,
not an ESNAS search or reproduction of the paper's undocumented test IDs.
Any subsequent optimization should use validation, not these test scores.

Outputs under `results/ceti_test/` include `plan.json`, the exact subset and
holdout manifest, `models.csv`, `report.json`, `summary.txt`, and a `predictions/`
folder. Training-data fractions use complete segments as well as timestep
counts at the same resampling resolution, avoiding confusion between the
pilot's 20 frames and the supplied SVM's 100 frames per segment.

## Notes

- Commands assume the current working directory is the repository root.
- Saved EESNAS/GA/BO checkpoints are stored as `backup_*.obj` under
  `results/{esnas,ga,bo}/`.
- Saved external-baseline checkpoints are stored under
  `results/{lcnn,ge_desn,grid_search,lstm}/`.
- The precomputed datasets needed by the experiments are included in `data/`.
