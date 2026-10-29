"""CeTI activity classification with the main-branch GA + BO ESNAS engine."""

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import reservoirpy as rpy

from ceti import DEFAULT_DATA_DIR, DEFAULT_SPLIT
from ceti.data import load_search_inputs, load_search_test
from ceti.models import predict_segments, SegmentMetric
from ceti.search import CetiESNAS
from ceti.svm import classification_metrics
from src.algorithms.types import EvalParams, GAParams, ModelParams


def smoke_models(input_dim):
    """Small fixed initial population for an actual bounded search-path check."""
    models = []
    for hidden, readout in (("Reservoir", "Ridge"), ("IPReservoir", "Ridge"),
                            ("Reservoir", "LMS"), ("Reservoir", "RLS")):
        parameters = {"units": 16, "sr": 0.8, "lr": 0.5, "rc_connectivity": 0.2}
        if hidden == "IPReservoir":
            parameters.update(mu=0.0, sigma=0.5, learning_rate=1e-3)
        output = {"output_dim": 5, **({"ridge": 1e-5} if readout == "Ridge" else {"alpha": 1e-5})}
        models.append({"nodes": [{"type": "Input", "params": {"input_dim": input_dim}},
                                 {"type": hidden, "params": parameters},
                                 {"type": readout, "params": output}], "edges": [(0, 1), (1, 2)]})
    return models


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", "--sampling-rate", type=int, default=20,
                        help="Resampled frames per complete motion, not Hz.")
    parser.add_argument("--n-jobs", type=int, default=20)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=1234123123)
    parser.add_argument("--train-per-class", type=int, default=100,
                        help="Training segments/class; default 100 gives 10,000 rows at 20 frames.")
    parser.add_argument("--val-per-class", type=int, default=0,
                        help="Validation segments/class; 0 (default) uses all validation-subject segments.")
    parser.add_argument("--feature-set", default="all", help="all, leg6 or comma-separated exact column names.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--ridge-passes", type=int, choices=(1, 2), default=2)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--bo-init", type=int, default=0)
    parser.add_argument("--bo-iter", type=int, default=6,
                        help="BO iterations after the default probe and random initial points.")
    parser.add_argument("--num-evals", type=int, default=3,
                        help="Fixed initializations/candidate; architecture fitness averages their metrics.")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Independent complete searches; default 5 to measure variability.")
    parser.add_argument("--timeout", type=float, default=60,
                        help="Seconds/initialization budget, multiplied across each BO worker's evaluations.")
    parser.add_argument("--memory-limit", type=float, default=756)
    parser.add_argument("--min-nodes", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="One generation, four tiny seed models, two workers.")
    args = parser.parse_args(argv)
    if args.smoke_test:
        args.generations, args.population, args.bo_init, args.bo_iter = 1, 4, 0, 0
        args.num_evals, args.repeats, args.n_jobs = 1, 1, 2
        args.train_per_class = min(args.train_per_class, 40)
        args.val_per_class = 10 if args.val_per_class == 0 else min(args.val_per_class, 10)
    if not 0 <= args.warmup < args.frames or args.frames < 2:
        parser.error("Require 0 <= warmup < frames, frames >= 2.")
    if args.val_per_class < 0:
        parser.error("--val-per-class must be nonnegative; 0 means all validation segments.")
    if min(args.n_jobs, args.train_per_class, args.population,
           args.num_evals, args.repeats, args.timeout, args.memory_limit, args.min_nodes) <= 0:
        parser.error("Counts and budgets must be positive.")
    if args.population < 4 or min(args.generations, args.bo_init, args.bo_iter) < 0 or args.max_nodes < args.min_nodes:
        parser.error("Need population >= 4, nonnegative iterations and a valid node-count range.")
    output = args.output_dir or ROOT / "results/esnas_ceti" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(backend="NumPy/SciPy; all main-branch node types available",
                  selection="Mean validation 1 - macro F1 across fixed initializations; all must succeed.",
                  selected_model="First predetermined initialization of the winning architecture; no seed selection.")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    data, metadata = load_search_inputs(args.data_dir, args.split_manifest, args.frames, args.seed,
        args.train_per_class, args.val_per_class, args.feature_set)
    (output / "data.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"ESNAS: train {data.trainX.shape}, validation {data.valX.shape}; "
          f"{args.frames} frames/segment; n_jobs={args.n_jobs}", flush=True)
    if args.prepare_only:
        print(f"Prepared manifest: {output}")
        return 0
    rpy.verbosity(0)
    results = []
    test_x = test_y = test_records = None
    for repeat in range(args.repeats):
        repeat_seed = args.seed + 1000 * repeat
        random.seed(repeat_seed)
        np.random.seed(repeat_seed)
        rpy.set_seed(repeat_seed)
        fold_dir = output / f"repeat_{repeat}"
        fold_dir.mkdir()
        evaluation = EvalParams(numEvals=args.num_evals,
            errorMetrics=[SegmentMetric(args.frames, args.warmup),
                          SegmentMetric(args.frames, args.warmup, "accuracy")],
            defaultErrors=[1.0, 0.0], timeout=args.timeout, memoryLimit=args.memory_limit,
            minimizeFitness=True, isAutoRegressive=False)
        ga_params = GAParams(generations=args.generations, populationSize=args.population,
            crossoverProbability=0.0 if args.smoke_test else 0.7,
            mutationProbability=0.0 if args.smoke_test else 0.2, eliteSize=1, stagnationReset=5)
        seeds = smoke_models(data.trainX.shape[1]) if args.smoke_test else []
        ga = CetiESNAS(data, evaluation, ga_params, ModelParams((args.min_nodes, args.max_nodes)),
            seedModels=seeds, n_jobs=args.n_jobs, saveLocation=str(fold_dir / "backup.obj"),
            bo_init=args.bo_init, bo_iter=args.bo_iter, frames=args.frames, warmup=args.warmup,
            ridge_passes=args.ridge_passes, seed=repeat_seed)
        start = time.perf_counter()
        ga = ga.run()
        elapsed = time.perf_counter() - start
        if ga.bestModel is None:
            raise RuntimeError("Search produced no successful model. Check worker failures/timeouts and data limits.")
        # Search sees neither test features nor labels. Only selected models
        # are evaluated on every segment of the saved ten-subject holdout.
        if test_x is None:
            test_x, test_y, test_records = load_search_test(args.data_dir, metadata)
        validation_x = data.valX.reshape(-1, args.frames, data.valX.shape[-1])
        validation_y = data.valY.reshape(-1, args.frames, 5)[:, 0].argmax(axis=1)
        validation_predicted, _ = predict_segments(ga.bestModel, validation_x, args.warmup)
        validation_metrics = classification_metrics(validation_y, validation_predicted)
        validation_subjects = np.asarray([record["subject"] for record in metadata["validation_records"]])
        validation_by_subject = {
            subject: classification_metrics(validation_y[validation_subjects == subject],
                                            validation_predicted[validation_subjects == subject])
            for subject in sorted(set(validation_subjects.tolist()))}
        predicted, _ = predict_segments(ga.bestModel, test_x, args.warmup)
        best_index = min(range(len(ga.fitnesses)), key=lambda i: ga.fitnesses[i][0])
        result = {"repeat": repeat, "seed": repeat_seed, "search_seconds": elapsed,
                  "selection_mean_validation_macro_f1": 1.0 - ga.bestFitness[0],
                  "selection_mean_validation_accuracy": ga.bestFitness[1],
                  "validation_macro_f1": validation_metrics["macro_f1"],
                  "validation_accuracy": validation_metrics["accuracy"],
                  "validation_segments": len(validation_y),
                  "validation_by_subject": validation_by_subject,
                  "selected_model_seed": repeat_seed,
                  "test_metrics": classification_metrics(test_y, predicted),
                  "test_segments": len(test_y), "evaluated_candidates": len(ga.fitnesses),
                  "best_architecture": ga.architectures[best_index]}
        results.append(result)
        (fold_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        with (fold_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["subject", "segment", "target", "prediction"])
            for record, target, prediction in zip(test_records, test_y, predicted):
                writer.writerow([record["subject"], record["path"], metadata["class_names"][int(target)],
                                 metadata["class_names"][int(prediction)]])
        report = {"repeats": results, "test_ids": metadata["split"]["test"],
                  "mean_test_macro_f1": float(np.mean([r["test_metrics"]["macro_f1"] for r in results])),
                  "mean_test_accuracy": float(np.mean([r["test_metrics"]["accuracy"] for r in results]))}
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Repeat {repeat}: mean-initialization validation F1="
              f"{result['selection_mean_validation_macro_f1']:.4f}; "
              f"selected-model validation F1={result['validation_macro_f1']:.4f}; "
              f"test {result['test_metrics']}", flush=True)
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
