"""Feasibility check: fixed slow ESN profiles on the complete saved holdout.

Uses the timing pilot's frozen training subset and representation. Models and
seeds are fixed before test predictions are made; this does not run ESNAS.
"""

import os

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from collections import Counter
import copy
import csv
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def classification_metrics(targets, predicted):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

    return {
        "accuracy": float(accuracy_score(targets, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predicted)),
        "macro_f1": float(f1_score(targets, predicted, labels=list(range(5)),
                                    average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(targets, predicted, labels=list(range(5))).tolist(),
    }


def prepare_holdout(pilot_dir, data_dir):
    """Load every test segment, while reusing the exact saved pilot training set."""
    import numpy as np
    from ceti.data import inventory, load_segments

    pilot_dir = Path(pilot_dir)
    metadata = json.loads((pilot_dir / "subset.json").read_text(encoding="utf-8"))
    with np.load(pilot_dir / "subset.npz", allow_pickle=False) as cache:
        arrays = {key: cache[key] for key in cache.files}
    root, subjects, records = inventory(data_dir)
    groups = {key: set(metadata["split"][key]) for key in ("train", "validation", "test")}
    if set.union(*groups.values()) != set(subjects) or any(
            groups[a] & groups[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Saved split must cover this dataset without subject overlap.")
    for key in ("train", "validation"):
        selected = metadata[f"{key}_records"]
        if not selected or any(record["subject"] not in groups[key] for record in selected):
            raise ValueError(f"Saved {key} selection does not agree with subject split.")
        prefix = "train" if key == "train" else "val"
        labels = np.asarray([metadata["class_names"].index(r["activity"]) for r in selected])
        expected = (len(selected), metadata["frames_per_segment"], len(metadata["columns"]))
        if arrays[f"{prefix}_x"].shape != expected or not np.array_equal(arrays[f"{prefix}_labels"], labels):
            raise ValueError(f"Saved {key} arrays do not agree with their manifest.")
    test_records = [record for record in records if record["subject"] in groups["test"]]
    if {r["subject"] for r in test_records} != groups["test"]:
        raise ValueError("One or more holdout subjects has no motion segments.")
    arrays["test_x"], arrays["test_labels"], _, test_lengths = load_segments(
        root, test_records, metadata["frames_per_segment"], metadata["columns"])
    group_counts = {key: sum(r["subject"] in group for r in records) for key, group in groups.items()}
    count = len(metadata["train_records"])
    frames = metadata["frames_per_segment"]
    metadata.update(
        test_records=test_records, test_source_lengths=test_lengths,
        test_contents_loaded=True, test_shape=list(arrays["test_x"].shape),
        test_class_counts=dict(Counter(r["activity"] for r in test_records)),
        available_segments_by_subject_partition=group_counts,
        training_fraction={
            "selected_segments": count, "total_dataset_segments": len(records),
            "of_entire_dataset": count / len(records),
            "of_training_subjects": count / group_counts["train"],
            "of_all_non_test_subjects": count / (group_counts["train"] + group_counts["validation"]),
            "selected_training_rows": count * frames,
            "all_dataset_rows_at_same_resolution": len(records) * frames,
            "all_training_subject_rows_at_same_resolution": group_counts["train"] * frames,
            "paper_frames_per_segment": 100,
            "row_ratio_to_entire_dataset_at_paper_resolution": count * frames / (len(records) * 100),
            "interpretation": "Segment fractions compare equal temporal resolution. The ratio to "
                "100-frame rows also includes temporal downsampling; it is not the fraction of recordings used.",
        },
    )
    return arrays, metadata


def prepare_inputs(arrays, metadata, feature_set):
    """Apply the same training-only scale to validation and the full holdout."""
    import numpy as np
    from ceti.data import select_and_scale

    train_x, val_x, columns, scale = select_and_scale(arrays, metadata["columns"], feature_set)
    indices = [metadata["columns"].index(name) for name in columns]
    test_x = arrays["test_x"][:, :, indices].copy() / scale
    if not np.isfinite(test_x).all():
        raise ValueError("Non-finite scaled holdout inputs.")
    return train_x, val_x, test_x, columns


def predict_segments(model, inputs, warmup):
    import numpy as np
    from src.utils import runModel

    scores = []
    for sequence in inputs:
        model.reset()
        outputs = np.asarray(runModel(model, sequence))
        if outputs.shape != (len(sequence), 5) or not np.isfinite(outputs).all():
            raise ValueError("Invalid or non-finite classifier output.")
        scores.append(outputs[warmup:].mean(axis=0))
    scores = np.stack(scores)
    return scores.argmax(axis=1), scores


def evaluation_worker(connection, cache_path, metadata_path, task, options):
    result = dict(task)
    started = time.perf_counter()

    def phase(name):
        connection.send({"phase": name})

    try:
        phase("imports")
        import numpy as np
        import reservoirpy as rpy
        from experiments.ceti_esn_timing import architecture_cases, construct_sequence_model, fit_sequences, preflight
        from sklearn.metrics import classification_report

        if not rpy.__version__.startswith("0.3."):
            raise RuntimeError("Use the native main-branch ReservoirPy 0.3 environment.")
        rpy.verbosity(0)
        rpy.set_seed(task["seed"])
        np.random.seed(task["seed"])
        phase("loading")
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        with np.load(cache_path, allow_pickle=False) as cache:
            arrays = {key: cache[key] for key in cache.files}
        train_x, val_x, test_x, columns = prepare_inputs(arrays, metadata, task["feature_set"])
        targets = [np.repeat(np.eye(5)[label][None, :], train_x.shape[1], axis=0)
                   for label in arrays["train_labels"]]
        architecture = architecture_cases(len(columns))[task["case"]]
        result.update(input_columns=len(columns), train_rows=len(train_x) * train_x.shape[1],
                      validation_rows=len(val_x) * val_x.shape[1], test_rows=len(test_x) * test_x.shape[1],
                      test_segments=len(test_x), architecture=architecture,
                      warmup=options["warmup"], ridge_passes=options["ridge_passes"])
        estimated, reason = preflight(architecture, result["train_rows"], options["memory_limit"])
        result["estimated_memory_mib"] = estimated
        if reason:
            result.update(status="skipped_search_limit", reason=reason)
        else:
            phase("construct")
            start = time.perf_counter()
            model = construct_sequence_model(architecture)
            result["construct_seconds"] = time.perf_counter() - start
            phase("fit")
            start = time.perf_counter()
            fit_sequences(model, list(train_x), targets, options["warmup"], options["ridge_passes"])
            result["fit_seconds"] = time.perf_counter() - start
            phase("snapshot")
            start = time.perf_counter()
            snapshot = copy.deepcopy(model)
            result["snapshot_seconds"] = time.perf_counter() - start
            del snapshot
            phase("validation")
            start = time.perf_counter()
            val_predicted, _ = predict_segments(model, val_x, options["warmup"])
            result["validation_seconds"] = time.perf_counter() - start
            val_metrics = classification_metrics(arrays["val_labels"], val_predicted)
            result.update({f"validation_{key}": value for key, value in val_metrics.items()})
            result["evaluation_seconds"] = sum(result[key] for key in (
                "construct_seconds", "fit_seconds", "snapshot_seconds", "validation_seconds"))
            phase("test")
            start = time.perf_counter()
            test_predicted, scores = predict_segments(model, test_x, options["warmup"])
            result["test_seconds"] = time.perf_counter() - start
            metrics = classification_metrics(arrays["test_labels"], test_predicted)
            result.update({f"test_{key}": value for key, value in metrics.items()})
            result["test_classification_report"] = classification_report(
                arrays["test_labels"], test_predicted, labels=list(range(5)),
                target_names=metadata["class_names"], output_dict=True, zero_division=0)
            result["test_per_subject"] = {}
            for subject in metadata["split"]["test"]:
                mask = np.array([r["subject"] == subject for r in metadata["test_records"]])
                result["test_per_subject"][subject] = dict(
                    segments=int(mask.sum()), **classification_metrics(arrays["test_labels"][mask], test_predicted[mask]))
            path = Path(options["output_dir"]) / "predictions" / f"{task['model_id']}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["subject", "segment", "target", "prediction"] +
                                [f"score_{name}" for name in metadata["class_names"]])
                for record, target, predicted, row in zip(metadata["test_records"], arrays["test_labels"], test_predicted, scores):
                    writer.writerow([record["subject"], record["path"], metadata["class_names"][int(target)],
                                     metadata["class_names"][int(predicted)], *row.tolist()])
            result.update(status="ok", predictions_file=path.name)
    except Exception:
        result.update(status="error", reason=traceback.format_exc())
    result["worker_wall_seconds"] = time.perf_counter() - started
    connection.send({"result": result})
    connection.close()


def aggregate_results(results):
    import numpy as np

    groups = {}
    for result in results:
        if result["status"] == "ok":
            groups.setdefault((result["case"], result["feature_set"]), []).append(result)
    summaries = []
    for (case, feature_set), rows in groups.items():
        summary = {"case": case, "feature_set": feature_set, "successful_models": len(rows)}
        for metric in ("fit_seconds", "evaluation_seconds", "test_seconds", "validation_accuracy",
                       "test_accuracy", "test_balanced_accuracy", "test_macro_f1"):
            values = np.asarray([row[metric] for row in rows])
            summary.update({f"{metric}_mean": float(values.mean()),
                            f"{metric}_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                            f"{metric}_min": float(values.min()), f"{metric}_max": float(values.max())})
        summaries.append(summary)
    return summaries


def write_summary(output, report):
    """Save a readable report without choosing a model using test scores."""
    fraction = report["training_fraction"]
    completed = [row for row in report["models"] if row["status"] == "ok"]
    widths = sorted({row["input_columns"] for row in completed})
    frames = fraction["selected_training_rows"] // fraction["selected_segments"]
    paper_rows = fraction["total_dataset_segments"] * fraction["paper_frames_per_segment"]
    lines = [
        "CeTI-Locomotion: fixed-model feasibility check on the complete holdout",
        "",
        f"Training: {fraction['selected_segments']} whole motion segments, "
        f"{fraction['selected_training_rows']} timesteps; input widths: {widths}.",
        f"Fraction of segments available from training subjects: {100 * fraction['of_training_subjects']:.2f}%.",
        f"Fraction of all dataset segments: {100 * fraction['of_entire_dataset']:.2f}%.",
        f"Fraction of segments from all non-test subjects: {100 * fraction['of_all_non_test_subjects']:.2f}%.",
        f"The subset uses {frames} frames per segment; the paper's classifier uses {fraction['paper_frames_per_segment']}.",
        f"Comparing {fraction['selected_training_rows']:,} rows against the paper's {paper_rows:,} rows "
        f"gives {100 * fraction['row_ratio_to_entire_dataset_at_paper_resolution']:.3f}%, but this",
        "combines sampling fewer recordings with using fewer frames per recording.",
        "",
        f"Test distribution: {report['test_class_counts']}",
        f"Successful models: {len(completed)} / {len(report['models'])}.",
        "All preset seeds/profiles are reported; no selection or tuning by test performance.",
        "",
        "Metrics below are mean +/- sample standard deviation across initializations.",
    ]
    for summary in report["summary"]:
        lines += [
            f"{summary['case']} ({summary['feature_set']}), {summary['successful_models']} initializations:",
            f"  Fit time: {summary['fit_seconds_mean']:.2f} +/- {summary['fit_seconds_std']:.2f} seconds",
            f"  Test accuracy: {100 * summary['test_accuracy_mean']:.2f} +/- {100 * summary['test_accuracy_std']:.2f}%",
            f"  Test balanced accuracy: {100 * summary['test_balanced_accuracy_mean']:.2f} +/- {100 * summary['test_balanced_accuracy_std']:.2f}%",
            f"  Test macro F1: {summary['test_macro_f1_mean']:.4f} +/- {summary['test_macro_f1_std']:.4f}",
            f"  Test inference time: {summary['test_seconds_mean']:.2f} seconds",
        ]
        rows = [r for r in completed if r["case"] == summary["case"] and r["feature_set"] == summary["feature_set"]]
        lines.append("  Mean recall by activity:")
        for activity in report["test_class_counts"]:
            recall = sum(r["test_classification_report"][activity]["recall"] for r in rows) / len(rows)
            lines.append(f"    {activity}: {100 * recall:.2f}%")
    baseline = report["baselines"]
    lines += [
        "", "References:",
        f"  Uniform random expected accuracy/balanced accuracy: {100 * baseline['uniform_random_expected_accuracy']:.1f}%.",
        f"  Training-majority constant prediction accuracy: {100 * baseline['training_majority_class']['accuracy']:.2f}%.",
        f"  Largest test class share: {100 * baseline['largest_test_class_share']:.2f}%.",
        "", report["qualification"],
    ]
    (Path(output) / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    from experiments.ceti_esn_timing import architecture_cases, run_trials
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=ROOT / "results/ceti_timing/stress_pilot")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "ceti/CeTI-Locomotion")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cases", nargs="+", choices=list(architecture_cases(6)),
                        default=["reservoir2000_ridge", "ip1500_ridge"])
    parser.add_argument("--feature-sets", nargs="+", default=["all"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234123123)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fit-timeout", type=float, default=60)
    parser.add_argument("--test-timeout", type=float, default=180)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--memory-limit", type=float, default=756)
    args = parser.parse_args(argv)
    if min(args.repeats, args.workers, args.fit_timeout, args.test_timeout,
           args.max_seconds, args.memory_limit) <= 0 or args.seed < 0:
        parser.error("Counts and budgets must be positive; seed must be nonnegative.")
    output = args.output_dir or ROOT / "results/ceti_test" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    pilot_config = json.loads((args.pilot_dir / "config.json").read_text(encoding="utf-8"))
    # Freeze models/seeds and existing preprocessing before reading test motions.
    options = dict(vars(args), output_dir=str(output), timeout=args.fit_timeout,
                   warmup=pilot_config["warmup"], ridge_passes=pilot_config["ridge_passes"])
    tasks = [{"case": case, "feature_set": feature_set, "seed": args.seed + repeat,
              "model_id": f"{case}_{feature_set}_{repeat}"}
             for repeat in range(args.repeats) for case in args.cases for feature_set in args.feature_sets]
    plan = {key: str(value) if isinstance(value, Path) else value for key, value in options.items()}
    plan.update(models=tasks, python=sys.version, executable=sys.executable, platform=platform.platform(),
                backend="NumPy/SciPy; no JAX", selection="Fixed profiles chosen by prior timing, not test performance.",
                package_versions={name: version(name) for name in ("numpy", "scipy", "scikit-learn", "reservoirpy")})
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    arrays, metadata = prepare_holdout(args.pilot_dir, args.data_dir)
    for feature_set in args.feature_sets:
        prepare_inputs(arrays, metadata, feature_set)
    (output / "subset.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    np.savez(output / "evaluation.npz", **arrays)
    fraction = metadata["training_fraction"]
    print(f"Training: {fraction['selected_segments']} complete segments, {fraction['selected_training_rows']} rows; "
          f"{100 * fraction['of_entire_dataset']:.2f}% of all segments, "
          f"{100 * fraction['of_training_subjects']:.2f}% of training-subject segments.")
    print(f"Full holdout: {len(metadata['test_records'])} segments, "
          f"{len(metadata['split']['test'])} subjects, {metadata['test_shape'][1]} frames each.", flush=True)
    majority_label = int(np.argmax(np.bincount(arrays["train_labels"], minlength=5)))
    baselines = {
        "training_majority_class": dict(label=metadata["class_names"][majority_label],
            **classification_metrics(arrays["test_labels"], np.full(len(arrays["test_labels"]), majority_label))),
        "uniform_random_expected_accuracy": 0.2,
        "uniform_random_expected_balanced_accuracy": 0.2,
        "largest_test_class_share": max(metadata["test_class_counts"].values()) / len(metadata["test_records"]),
    }
    results = run_trials(tasks, output / "evaluation.npz", output / "subset.json", options, output,
                         worker=evaluation_worker,
                         phase_timeouts={"fit": args.fit_timeout, "test": args.test_timeout})
    summaries = aggregate_results(results)
    report = {"training_fraction": fraction, "test_class_counts": metadata["test_class_counts"],
              "baselines": baselines, "summary": summaries, "models": results,
              "qualification": "Feasibility test on our saved holdout, not the paper's recovered test IDs. "
                 "All test recordings are used at the same 20-frame representation as training (or the source pilot's resolution). "
                 "No architecture/hyperparameter search or selection by test results; uncertainty across seeds "
                 "does not measure uncertainty across subject splits. Future optimization must use validation."}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_summary(output, report)
    metrics = ("model_id", "case", "feature_set", "seed", "status", "fit_seconds", "evaluation_seconds",
               "test_seconds", "validation_accuracy", "test_accuracy", "test_balanced_accuracy", "test_macro_f1", "reason")
    with (output / "models.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=metrics, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    for summary in summaries:
        print(f"{summary['case']} ({summary['feature_set']}): "
              f"test accuracy {summary['test_accuracy_mean']:.3f} +/- {summary['test_accuracy_std']:.3f}; "
              f"macro F1 {summary['test_macro_f1_mean']:.3f}; fit {summary['fit_seconds_mean']:.2f}s")
    print(f"Reports and per-segment predictions: {output}", flush=True)
    return 1 if any(row["status"] != "ok" for row in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
