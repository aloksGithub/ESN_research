"""Fixed Reservoir + Ridge learning curve on validation subjects, never tests."""

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ceti import DEFAULT_DATA_DIR, DEFAULT_SPLIT
from ceti.data import inventory, load_segments, sample_segments, select_and_scale, subject_split
from experiments.ceti_esn_timing import architecture_cases, run_trials, trial_worker


def nested_training_records(records, subjects, sizes, seed):
    """Keep earlier recordings and append balanced, subject-spread additions.

    The first size uses exactly the existing pilot's sampling rule and seed.
    All seeds fit the same nested subsets in the same order.
    """
    selected, used, previous, subsets = [], set(), 0, {}
    for size in sorted(set(sizes)):
        remaining = [record for record in records if record["path"] not in used]
        additions = sample_segments(remaining, subjects, size - previous, seed)
        selected.extend(additions)
        used.update(record["path"] for record in additions)
        subsets[size] = list(selected)
        previous = size
    return subsets


def curve_worker(connection, cache_path, metadata_path, task, options):
    directory = Path(cache_path) / f"train_{task['train_per_class']}"
    trial_worker(connection, directory / "subset.npz", directory / "subset.json", task, options)


def summarize(results, sizes):
    summaries = []
    baseline = {row["seed"]: row for row in results
                if row["train_per_class"] == min(sizes) and row["status"] == "ok"}
    for size in sizes:
        rows = [row for row in results if row["train_per_class"] == size and row["status"] == "ok"]
        summary = {"train_per_class": size, "successful_trials": len(rows),
                   "attempted_trials": sum(row["train_per_class"] == size for row in results)}
        if rows:
            for field in ("fit_seconds", "validation_seconds", "evaluation_seconds",
                          "validation_macro_f1", "validation_accuracy"):
                values = [row[field] for row in rows]
                summary[field + "_mean"] = statistics.mean(values)
                summary[field + "_std"] = statistics.stdev(values) if len(values) > 1 else None
            summary.update(train_rows=rows[0]["train_rows"], validation_rows=rows[0]["validation_rows"],
                           estimated_memory_mib=rows[0]["estimated_memory_mib"],
                           fits_within_60_seconds=sum(row["fit_seconds"] <= 60 for row in rows),
                           within_current_search_memory_limit=rows[0]["estimated_memory_mib"] <= 756)
            differences = [row["validation_macro_f1"] - baseline[row["seed"]]["validation_macro_f1"]
                           for row in rows if row["seed"] in baseline]
            summary["paired_f1_change_from_smallest_mean"] = statistics.mean(differences) if differences else None
        summaries.append(summary)
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[40, 100, 200],
                        help="Segments per class, not timestep rows.")
    parser.add_argument("--repeats", type=int, default=3, help="Paired reservoir seeds per size; no search.")
    parser.add_argument("--seed", type=int, default=1234123123)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--ridge-passes", type=int, choices=[1, 2], default=2)
    parser.add_argument("--workers", type=int, default=1, help="Serial by default for comparable fit timings.")
    parser.add_argument("--timeout", type=float, default=180, help="Hard complete-trial limit in seconds.")
    parser.add_argument("--max-seconds", type=float, default=1200, help="Overall trial batch wall-clock limit.")
    parser.add_argument("--memory-limit", type=float, default=2048,
                        help="MiB estimate limit; the 20k-row fixed profile exceeds ESNAS's 756 MiB default.")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if min(args.train_sizes + [args.repeats, args.workers]) < 1:
        parser.error("Training sizes, repeats and workers must be positive.")
    if args.frames < 2 or not 0 <= args.warmup < args.frames:
        parser.error("Need frames >= 2 and 0 <= warmup < frames.")
    if min(args.timeout, args.max_seconds, args.memory_limit) <= 0:
        parser.error("Budgets must be positive.")
    sizes = sorted(set(args.train_sizes))
    import numpy as np

    output = args.output_dir or ROOT / "results/ceti_learning_curve" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    root, subjects, records = inventory(args.data_dir)
    split = subject_split(subjects, args.split_manifest, args.seed)
    subsets = nested_training_records(records, set(split["train"]), sizes, args.seed + 1)
    validation = [record for record in records if record["subject"] in set(split["validation"])]
    # Read the largest training subset once, then take prefixes for smaller sizes.
    train_x, train_labels, columns, lengths = load_segments(root, subsets[sizes[-1]], args.frames)
    val_x, val_labels, _, _ = load_segments(root, validation, args.frames, columns)
    for size in sizes:
        count = len(subsets[size])
        arrays = {"train_x": train_x[:count], "train_labels": train_labels[:count],
                  "val_x": val_x, "val_labels": val_labels}
        _, _, selected_columns, scale = select_and_scale(arrays, columns, "all")
        metadata = {"split": split, "columns": columns, "frames_per_segment": args.frames,
                    "train_records": subsets[size], "validation_records": validation,
                    "train_source_lengths": lengths[:count], "test_contents_loaded": False,
                    "input_columns": selected_columns, "training_max_abs_scale": scale.tolist()}
        directory = output / f"train_{size}"
        directory.mkdir()
        np.savez(directory / "subset.npz", **arrays)
        (directory / "subset.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(python=sys.version, executable=sys.executable, platform=platform.platform(),
                  package_versions={name: version(name) for name in ("numpy", "scipy", "reservoirpy", "scikit-learn")},
                  profile="reservoir2000_ridge", architecture=architecture_cases(len(columns))["reservoir2000_ridge"],
                  subset_policy="Nested class-balanced subsets; smallest matches the existing pilot sampler.",
                  validation_segments=len(validation), test_evaluated=False,
                  data_preparation_seconds=time.perf_counter() - started)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Fixed Reservoir 2000 + Ridge; {len(columns)} inputs; {args.frames} frames/segment.", flush=True)
    print(f"Training rows: {[len(subsets[size]) * args.frames for size in sizes]}; "
          f"validation: all {len(validation)} segments; {args.repeats} paired seeds; workers={args.workers}.", flush=True)
    print(f"Results: {output}", flush=True)
    if args.prepare_only:
        return 0
    tasks = [{"case": "reservoir2000_ridge", "feature_set": "all", "train_per_class": size,
              "seed": args.seed + repeat} for repeat in range(args.repeats) for size in sizes]
    results = run_trials(tasks, output, output / "config.json", vars(args), output, worker=curve_worker)
    summaries = summarize(results, sizes)
    report = {"summary": summaries, "trials": results, "test_evaluated": False,
              "wall_seconds": time.perf_counter() - started,
              "interpretation": "Validation learning curve for one fixed profile and split, not test scores. "
                  "Seed variation does not measure subject-split uncertainty. "
                  "Do not compare these validation numbers directly with prior test F1."}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = ["train_per_class", "train_rows", "seed", "status", "fit_seconds", "validation_seconds",
              "evaluation_seconds", "validation_macro_f1", "validation_accuracy", "estimated_memory_mib", "reason"]
    with (output / "trials.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    lines = ["Fixed Reservoir 2000 + Ridge validation learning curve", "",
             "Training rows | Validation macro-F1 (mean +/- sample SD) | Fit seconds (mean +/- sample SD)"]
    for row in summaries:
        if row["successful_trials"]:
            f1_sd = row["validation_macro_f1_std"]
            fit_sd = row["fit_seconds_std"]
            lines.append(f"{row['train_rows']:>13} | {row['validation_macro_f1_mean']:.4f} +/- "
                         f"{f1_sd if f1_sd is not None else 0:.4f} | {row['fit_seconds_mean']:.2f} +/- "
                         f"{fit_sd if fit_sd is not None else 0:.2f} "
                         f"({row['successful_trials']}/{row['attempted_trials']} successful)")
        else:
            lines.append(f"{row['train_per_class']} segments/class: no successful trials")
    for row in summaries:
        if row["successful_trials"] and not row["within_current_search_memory_limit"]:
            lines.append(f"{row['train_rows']} rows: {row['estimated_memory_mib']:.1f} MiB estimated memory, "
                         "above ESNAS's current 756 MiB default.")
    lines.extend(["", f"Validation: {len(validation)} complete segments from {len(split['validation'])} subjects.",
                  "All sizes use nested training recordings and paired reservoir seeds.",
                  "Training-only MaxAbs scaling, independent-recording state resets, two Ridge passes by default.",
                  report["interpretation"], f"Total wall time: {report['wall_seconds']:.1f} seconds."])
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 1 if any(row["status"] != "ok" for row in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
