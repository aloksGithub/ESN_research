# CeTI-Locomotion experiment log

Last updated: 2026-10-29.

This log records the experiments run during the CeTI investigation, their
results, and the changes made in response. Entries follow the order of the
investigation; individual run dates are not assigned where they were not saved.

**Evidence:** local results are linked to saved artifacts. Results run on the
other PC are labeled **user-reported**; their complete reports and configurations
have not been collected in this workspace. Approximate numbers remain approximate.

## Results at a glance

| Experiment | Training data | Evaluation | Macro-F1 | Evidence |
|---|---|---|---:|---|
| Paper SVM reference | Full development cohort, 100 frames/segment | Paper's test subjects | 0.85 | Published Table 2 |
| Fixed Reservoir 2000 + Ridge pilot | 200 segments, 4,000 timestep rows | Our complete test holdout | 0.7000 +/- 0.0360 | Local, five seeds |
| Fixed IPReservoir 1500 + Ridge pilot | 200 segments, 4,000 rows | Our complete test holdout | 0.6712 +/- 0.0152 | Local, five seeds |
| Original ESNAS | 200 segments, 4,000 rows | Our complete test holdout | Approximately 0.716 mean | User-reported, five complete searches |
| Revised ESNAS, larger validation set | 200 segments, 4,000 rows | Our complete test holdout | 0.682 | User-reported, one search |
| SVM, full development cohort | All segments from 40 subjects, 20 frames/segment | Our complete test holdout | Approximately 0.85 | User-reported |
| SVM, small training pool | 200-segment pool, 20 frames/segment | Our complete test holdout | Approximately 0.74 | User-reported |
| Fixed ESN learning curve, smallest set | 200 segments, 4,000 rows | All 964 validation segments | 0.7700 +/- 0.0124 | Local, three seeds |
| Fixed ESN learning curve, middle set | 500 segments, 10,000 rows | All 964 validation segments | 0.8187 +/- 0.0090 | Local, three seeds |
| Fixed ESN learning curve, largest set | 1,000 segments, 20,000 rows | All 964 validation segments | 0.8492 +/- 0.0035 | Local, three seeds |
| ESNAS with 10,000 training rows | 500 segments | Data preparation only | **No search result yet** | Local preparation verified |

Values with `+/-` are means and sample standard deviations across reservoir
initializations, not confidence intervals or uncertainty across subject splits.
**Validation and test scores in this table must not be treated as equivalent.**

## Common data and evaluation protocol

- Dataset: `ceti/CeTI-Locomotion/`, containing 4,672 processed motion segments
  from 50 subjects. The external dataset is gitignored.
- Five classes: `gaitNormal`, `gaitFast`, `gaitNormalBackpack`,
  `gaitNormalBottleCrate`, and `sts`.
- Full input width: 102 channels, representing 17 points with three position
  and three angle values each. The reduced `leg6` option uses bilateral hip
  flexion, knee flexion, and ankle dorsiflexion.
- `--frames 20` means 20 interpolated frames **per complete segment, not 20 Hz**.
  The paper's representation has 100 frames per segment. Interpolation follows
  the supplied code's grid, which excludes the final source endpoint.
- Row counts depend on model format: an ESN receives one row per frame and
  five one-hot targets per row. The SVM receives one row per complete segment,
  with 2,040 flattened features at 20 frames (10,200 at 100 frames) and one
  class label. Thus the small SVM pool has 200 sample rows, while the same
  recordings give an ESN 4,000 timestep rows at 20 frames.
- Shared subject split: 30 training subjects, 10 validation subjects, and
  10 test subjects, saved in
  [subject_split.json](../results/ceti_timing/subject_split.json), seed
  `1234123123`. These groups contain 2,793, 964, and 915 segments respectively.
- Our test IDs are explicit and reproducible, **not the recovered original
  paper test IDs**. Copy the manifest when transferring the experiment to
  another PC; a different subject split changes the comparison.
- Test subjects: `sub-K101`, `sub-K19`, `sub-K27`, `sub-K31`, `sub-K50`,
  `sub-K53`, `sub-K74`, `sub-K82`, `sub-K90`, and `sub-K97`.
- ESN scaling is MaxAbs per input channel, fitted only on the selected training
  timesteps. State is reset between recordings. Training uses one-hot targets
  at each timestep; prediction averages output scores after three warmup frames,
  then takes the highest-scoring class. These scores are not calibrated
  probabilities. There is one final class decision per segment.
- The native main-branch training helper uses two terminal Ridge fitting passes.
  The CeTI adapters also reset NVAR delay history and IPReservoir internal state
  between recordings.
- Completed test evaluations use all 915 test segments at the specified
  representation. Predictions are not upsampled to manufacture additional
  decisions or to claim equivalence to the paper's 100-frame representation.

The initial 200-segment training subset is 7.16% of the training-subject pool,
5.32% of all non-test segments, and 4.28% of the entire dataset. Its 4,000 rows
also reflect using 20 frames per segment; a row comparison with the paper's
100-frame representation mixes temporal resolution with number of recordings.

Local runs used native Windows Python 3.11.9 in `envs/main`, NumPy/SciPy and
ReservoirPy 0.3.13.post1, with the main branch's pinned requirements. The project
was originally developed with Python 3.9; the native environment created for
these checks was 3.11.9. No JAX or WSL backend was used. Runtime measurements
below are local measurements; the other PC can have different timings.

## 1. Initial runtime feasibility pilots

**Question:** can native main-branch ESNs fit this representation within the
roughly 30-60 second model-training budgets used in previous experiments?

Script: [ceti_esn_timing.py](../experiments/ceti_esn_timing.py).

Both pilots used 40 training segments/class (200 total, 4,000 rows) and 10
validation segments/class (50 total, 1,000 rows), at 20 frames. They compared
all 102 inputs with `leg6`. No test motion contents were read.

The [first pilot](../results/ceti_timing/first_pilot/report.json) ran one seed
with one worker and a 20-second complete-trial timeout: 15 successful trials
and one preflight skip. The
[stress pilot](../results/ceti_timing/stress_pilot/report.json) used four workers
and a 30-second timeout: 11 successful trials and one preflight skip.

Selected full-input results:

| Profile | Workers in batch | Fit time (s) | Complete evaluation (s) | Small-validation macro-F1 |
|---|---:|---:|---:|---:|
| Reservoir 128 + Ridge | 1 | 0.70 | 0.77 | 0.7816 |
| Reservoir 512 + Ridge | 1 | 1.12 | 1.20 | 0.7812 |
| Reservoir 1500 + Ridge | 1 | 5.07 | 5.31 | 0.7428 |
| IPReservoir 1500 + Ridge | 4 | 11.19 | 11.51 | 0.7066 |
| Reservoir 2000 + Ridge | 4 | 18.39 | 18.89 | 0.7146 |
| Reservoir 400 + RLS | 4 | 9.98 | 10.09 | 0.8061 |

Complete evaluation includes construction, fitting, a fitted-model copy, and
validation; it excludes some process startup/loading overhead. These are
single-seed sanity checks on a small validation set, not final comparisons.

Reducing inputs from 102 to six gave modest speed savings for these reservoir
profiles and generally poorer validation F1. Full-input NVAR configurations
were rejected by existing preflight memory limits before allocation. Across
the two pilots there were 26 successful trials and two preflight skips.

A separate [timeout check](../results/ceti_timing/timeout_check/report.json)
used a 0.1-second trial budget and terminated as expected. It was a control-flow
check, not a model-performance measurement.

**Decision:** proceed with a small training subset and fixed-model evaluation
on the complete test holdout. Main-branch models were feasible without switching
to JAX or removing all expensive node types.

## 2. Fixed-model feasibility test on the complete holdout

Script: [ceti_esn_test.py](../experiments/ceti_esn_test.py).
Artifacts: [summary](../results/ceti_test/first_full_holdout/summary.txt),
[report](../results/ceti_test/first_full_holdout/report.json), and
[plan](../results/ceti_test/first_full_holdout/plan.json).

Two profiles, five initializations each, were trained on the same 200 segments
with all 102 inputs. All 915 test segments were evaluated at 20 frames.
There was no architecture search; all ten prescribed models completed and
were reported, without choosing the highest test score.

| Profile | Test macro-F1 | Test accuracy | Mean fit time (s) | Mean test inference (s) |
|---|---:|---:|---:|---:|
| Reservoir 2000 + Ridge | 0.7000 +/- 0.0360 | 67.91% +/- 3.63 percentage points | 14.47 +/- 1.77 | 9.47 |
| IPReservoir 1500 + Ridge | 0.6712 +/- 0.0152 | 64.50% +/- 1.60 percentage points | 11.17 +/- 0.73 | 6.85 |

The majority-class baseline had 24.81% accuracy but only 0.0795 macro-F1;
uniform-random expected accuracy and balanced accuracy are 20%. Accuracy
baselines must not be substituted for macro-F1 baselines.

**Finding:** fixed ESNs showed useful competence from a very small training
subset. This justified investigating optimization, but did not show that ESNAS
would improve on a fixed reservoir or match the paper.

## 3. Original ESNAS: five complete searches

**Evidence: user-reported results from the experiment PC.** Complete remote
reports were not supplied locally; exact architectures, runtimes, overrides,
and all validation scores are unavailable.

The script at this stage defaulted to 4,000 training rows, 50 validation
segments, 20 frames, 20 workers, population 20, 10 generations, three BO
iterations after the default probe, and three initializations per candidate.
Candidate evaluation selected its **best initialization** by validation F1.
`--repeats 5` ran five complete independent searches, not five fits of one model.

Reported test macro-F1 scores, rounded by the user:

| Search repeat (zero-based) | Approximate test macro-F1 |
|---:|---:|
| 0 | 0.73 |
| 1 | 0.68 |
| 2 | 0.72 |
| 3 | 0.74 |
| 4 | 0.71 |

The approximate mean is **0.716**, with sample SD approximately 0.023 calculated
from those rounded values. This is not an exact aggregate from raw reports.

The final-repeat screenshot gave more precise values:

- Validation macro-F1: 0.8828.
- Test macro-F1: 0.70550533.
- Test accuracy: 0.68306011.
- Test balanced accuracy: 0.69971379.

The user reported high validation F1 for most repeats. The average test result
was only modestly above the fixed Reservoir + Ridge pilot; the optimization
cost had not produced a convincing improvement.

**Interpretation:** 50 validation segments plus selection of the luckiest
initialization could produce optimistic selection scores. Subject differences
could also explain part of the gap. The validation-test gap alone did not
establish the cause or prove overfitting.

## 4. Revised ESNAS: larger validation set and seed averaging

**Changes:** keep training at 200 segments, use every segment from the ten
validation subjects (964 segments, 19,280 timestep rows), and default to one
complete search. Score a candidate by mean validation loss across three fixed
initializations; all must succeed. Save the first predetermined initialization
of the winning architecture rather than selecting its best-scoring seed.
Reports distinguish mean selection F1 from the saved model's validation F1,
and include validation scores by subject.

**User-reported outcome:** test macro-F1 **0.682**. No raw remote report or
new validation score was provided, so the full effective configuration and
validation-test gap cannot be reconstructed here.

**Finding:** this run did not improve on the original ESNAS results or the
fixed Reservoir + Ridge pilot. Increasing validation data reduced a source
of selection noise, but added no training information and did not guarantee
better test performance. No clear search benefit had been demonstrated.

## 5. SVM controls at 20 frames

Script: [svm_ceti.py](../experiments/svm_ceti.py).
Implementation: [svm.py](svm.py).

The implementation preserves the supplied pipeline:
`MaxAbsScaler -> PCA() -> SVC(probability=True)`, with RBF kernel and default
SVC hyperparameters. It uses stratified ten-fold CV **by segment**, scores
balanced accuracy, and retains the highest-scoring fold's fitted estimator
without a final refit. Scaling and PCA are fitted inside each fold.

### Full development cohort

```powershell
python experiments/svm_ceti.py --frames 20
```

Uses all 3,757 segments from the 40 non-test subjects as the CV pool, including
the ten subjects used as separate validation subjects by ESNAS. Each fold fits
approximately 90% of that pool. The selected fold estimator evaluates all
915 held-out test segments.

**User-reported test macro-F1: approximately 0.85.** Raw report and runtime
were not collected locally.

For reference, the paper reports **0.85 F1 and 84% accuracy** for five-class
action recognition using its 100-frame representation; see
[Table 2, page 7](https://h2t.iar.kit.edu/pdf/Hanisch2024a.pdf#page=7).
Our result uses our own saved test subjects, so it is comparable performance
on the task, not recovery of the paper's exact split or full protocol.

**Finding:** 20 frames can retain enough information for an SVM to reach about
0.85 on our holdout. Temporal downsampling alone cannot explain the ESNAS score.

### Small training pool

```powershell
python experiments/svm_ceti.py --frames 20 --training-subset small --train-per-class 40
```

Uses the same original 200-segment pool sampled from the 30 training subjects.
Ten-fold CV means each individual estimator fits 180 segments and validates
on 20; it is not literally fitted on all 200 segments. Its folds are not
subject-disjoint, unlike ESNAS's separate validation subject group.

**User-reported test macro-F1: approximately 0.74.** Raw report and runtime
were not collected locally.

**Finding:** reducing the SVM development pool substantially reduced its test
score, supporting the hypothesis that training-data volume matters. The
small-data SVM still exceeded the revised ESNAS result by approximately 0.058
and the original ESNAS mean by approximately 0.024. Training amount,
validation procedure, and model representation remain distinct factors.

## 6. Fixed-ESN validation learning curve

**Question:** does adding training data help a fixed ESN, and is its fitting
time practical before paying for another complete ESNAS search?

Script: [ceti_esn_learning_curve.py](../experiments/ceti_esn_learning_curve.py).

```powershell
python experiments/ceti_esn_learning_curve.py
```

Artifacts: [summary](../results/ceti_learning_curve/first_validation_curve/summary.txt),
[report](../results/ceti_learning_curve/first_validation_curve/report.json),
[trial CSV](../results/ceti_learning_curve/first_validation_curve/trials.csv),
and [configuration](../results/ceti_learning_curve/first_validation_curve/config.json).

### Design

- Fixed Reservoir 2000 + Ridge: spectral radius 0.9, leak rate 0.5,
  input connectivity 0.2, recurrent connectivity 0.1, and Ridge parameter 1e-5.
- All 102 inputs, 20 frames/segment, three-frame warmup, two Ridge passes.
- Training sizes of 40, 100, and 200 segments/class: 200, 500, and 1,000
  segments, or 4,000, 10,000, and 20,000 timestep rows.
- Nested subsets: retain earlier recordings and append class-balanced,
  subject-spread additions. The smallest subset exactly matches the original
  pilot's training records. Every size uses the same three reservoir seeds:
  `1234123123`, `1234123124`, and `1234123125`.
- Training-only scaling is refitted at each size. Every model evaluates the
  same full 964-segment validation set. **Test motion contents are not read.**
- One worker, to reduce timing contention. Complete-trial timeout: 180 seconds;
  overall trial-batch budget: 1,200 seconds. Estimated memory allowance: 2,048 MiB.
- No architecture search, hyperparameter optimization, or best-seed selection.

### Aggregate results

| Training segments/class | Training rows | Validation macro-F1 | Fit time (s) | Complete evaluation (s) | Estimated memory (MiB) |
|---:|---:|---:|---:|---:|---:|
| 40 | 4,000 | 0.7700 +/- 0.0124 | 9.37 +/- 1.85 | 15.84 +/- 1.95 | 320.8 |
| 100 | 10,000 | 0.8187 +/- 0.0090 | 21.74 +/- 2.16 | 28.37 +/- 2.38 | 705.7 |
| 200 | 20,000 | 0.8492 +/- 0.0035 | 40.20 +/- 3.37 | 46.76 +/- 3.62 | 1,347.1 |

Validation inference averaged 6.5-6.6 seconds per model. All nine trials
succeeded; every fit completed within 60 seconds. Total wall time, including
preparation and process overhead, was **288.4 seconds**.

### Individual validation results

| Reservoir seed | 4,000 rows | 10,000 rows | 20,000 rows |
|---:|---:|---:|---:|
| 1234123123 | 0.756857 | 0.822599 | 0.845247 |
| 1234123124 | 0.781456 | 0.825140 | 0.851647 |
| 1234123125 | 0.771561 | 0.808422 | 0.850764 |

**Finding:** every paired seed improved at each larger size. Mean validation
macro-F1 increased by 0.0488 at 10,000 rows and by 0.0793 at 20,000 rows,
relative to 4,000 rows. This supports increasing training data for this fixed
profile on this validation split. It does **not** establish 0.85 test F1 or
prove that an architecture search will improve on the fixed model.

**Practical consequence:** 10,000 rows provided a useful gain with roughly
22-second fits and remained under ESNAS's current 756 MiB estimated-memory
limit for this profile. The 20,000-row profile required about 1,347 MiB under
the existing estimator; it would be rejected by the current search limit.
Memory estimates are not measured peak RAM. Single-worker timings do not
predict timings or total memory with 20 concurrent workers.

## 7. Current ESNAS configuration: prepared, search result pending

Following the learning curve, defaults in
[esnas_ceti.py](../experiments/esnas_ceti.py) and its data loader were updated:

| Setting | Current default |
|---|---|
| Training | 100 segments/class; 500 total; 10,000 timestep rows |
| Validation | All 964 segments from the ten validation subjects; 19,280 rows |
| Frames / input channels / output classes | 20 / 102 / 5 |
| Workers | 20, intended for the experiment PC |
| Complete independent searches | 1 |
| Initializations per candidate | 3, mean validation loss; all must succeed |
| Saved initialization | First predetermined seed of winning architecture |
| Generations / population | 10 / 20 |
| BO random initial points / iterations | 0 / 3, plus the default probe |
| Initial intermediate-node count | 1-2 |
| Warmup / terminal Ridge passes | 3 / 2 |
| Time budget / estimated-memory limit | 60 seconds per nominal initialization / 756 MiB |

The timeout is combined across each BO worker's requested initializations
and probes; it is not a separate hard timer around each fit. Architecture
validity checks and memory limits still apply to all native main-branch node types.

The ESNAS loader's ordinary 100-per-class sample is generated directly at that
size. The learning curve uses nested additions from the original 40-per-class
sample. These have the same size and subject pool, but need not contain the
same recordings; the learning-curve F1 is not a prediction for the new search.

Data-only preparation confirmed `(10000, 102)` training inputs and
`(19280, 102)` validation inputs, one repeat, and 20 configured workers.
No models or search populations were constructed by this check, and test
motion contents remained unread. Artifacts:
[configuration](../results/esnas_ceti/ten_thousand_rows_prepare/config.json)
and [data manifest](../results/esnas_ceti/ten_thousand_rows_prepare/data.json).

Run on the experiment PC:

```powershell
python experiments/esnas_ceti.py
```

**Status:** the 10,000-row ESNAS search has been configured and prepared, but
no completed result has been reported in this log. No full 20-worker ESNAS
search was launched on the local PC.

## Implementation checks, separate from scientific results

| Check | Outcome | Artifacts |
|---|---|---|
| SVM 20-frame smoke run: 20 training segments, two folds | Completed; test macro-F1 0.5207 | [Report](../results/svm_ceti/smoke_20/report.json) |
| SVM 100-frame smoke run: 20 training segments, two folds | Completed; test macro-F1 0.5181 | [Report](../results/svm_ceti/smoke_100/report.json) |
| Original ESNAS smoke: four tiny seed architectures, two workers, one generation | Completed; eight candidate evaluations, validation F1 0.5743, test F1 0.6367 | [Report](../results/esnas_ceti/smoke_check/report.json) |
| Revised-evaluator ESNAS smoke, same bounded setup and 50-segment validation cap | Completed; same validation/test F1; mean and saved-model metrics recorded separately | [Report](../results/esnas_ceti/expanded_validation_smoke/report.json) |
| Full-validation preparation | Confirmed 4,000 train / 19,280 validation rows | [Configuration](../results/esnas_ceti/expanded_validation_prepare/config.json) |
| Updated 10,000-row preparation | Confirmed 10,000 train / 19,280 validation rows | [Configuration](../results/esnas_ceti/ten_thousand_rows_prepare/config.json) |
| Automated CeTI tests after learning-curve addition and default-size change | 29 passed | [Experiment tests](../tests/test_ceti_experiments.py), [data/model tests](../tests/test_ceti_timing.py), [learning-curve tests](../tests/test_ceti_learning_curve.py) |

Smoke scores verify execution and reporting. They are not tuned baselines and
must not be confused with the full SVM runs reported from the other PC. No
full 100-frame SVM benchmark has been reported; only its smoke check is recorded.

Tests cover interpolation and SVM parity with supplied code, training-only
scaling, subject separation and test-file access guards, sequence reset and
pickling, mean-initialization selection, nested balanced subsets, and paired
learning-curve summaries. The latest checks passed on Python 3.11.9; this does
not establish Python 3.9 compatibility.

## Conclusions and outstanding evidence

1. Native main-branch ESNs are computationally feasible at the tested sizes;
   the fixed profile's fit cost increased from about 9 to 40 seconds as rows
   increased from 4,000 to 20,000 with one worker.
2. ESNAS has not demonstrated a clear advantage over fixed ESNs or the SVM
   under the original small-training setup. Larger validation alone did not
   produce a better reported test score.
3. More training data helped both the reported SVM test comparison and the
   fixed-ESN validation curve. Different validation procedures, representations,
   and subject cohorts prevent attributing the entire gap to data size alone.
4. The next configured search uses 10,000 training rows. Its validation and
   test outcomes remain to be measured; repeated use of this test holdout for
   development means future scores are exploratory, rather than a fresh
   untouched final evaluation.
5. Collect the remote SVM and ESNAS `config.json`, `report.json`, and prediction
   files to replace approximate numbers and verify exact settings. Local
   artifact links depend on keeping the corresponding `results/` directories;
   those files are not automatically staged with this Markdown log.
