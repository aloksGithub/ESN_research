"""Bounded CPU timing pilot, not ESNAS and not a final test evaluation.

Run from the repository root with envs/main/Scripts/python.exe.
"""

import os

# Match src/parallel_processing.py: avoid BLAS oversubscription in search workers.
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
import copy
import csv
import json
from importlib.metadata import version
import math
import multiprocessing as mp
import platform
import statistics
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_CASES = ("reservoir128_ridge", "reservoir512_ridge", "reservoir1500_ridge",
                 "deep256_ridge", "ip128_ridge", "reservoir256_lms",
                 "reservoir256_rls", "nvar_ridge")


def architecture_cases(input_dim, output_dim=5):
    def reservoir(units, kind="Reservoir"):
        params = {"units": units, "lr": 0.5, "sr": 0.9,
                  "input_connectivity": 0.2, "rc_connectivity": 0.1}
        if kind == "IPReservoir":
            params.update(mu=0.0, sigma=0.5, learning_rate=1e-3)
        return {"type": kind, "params": params}

    def readout(kind):
        params = {"output_dim": output_dim}
        params.update({"ridge": 1e-5} if kind == "Ridge" else
                      {"alpha": 1e-5 if kind == "LMS" else 1.0})
        return {"type": kind, "params": params}

    def chain(*nodes):
        nodes = [{"type": "Input", "params": {"input_dim": input_dim}}] + list(nodes)
        return {"nodes": nodes, "edges": [[i, i + 1] for i in range(len(nodes) - 1)]}

    return {
        "reservoir128_ridge": chain(reservoir(128), readout("Ridge")),
        "reservoir512_ridge": chain(reservoir(512), readout("Ridge")),
        "reservoir1500_ridge": chain(reservoir(1500), readout("Ridge")),
        "deep256_ridge": chain(reservoir(256), reservoir(256), readout("Ridge")),
        "ip128_ridge": chain(reservoir(128, "IPReservoir"), readout("Ridge")),
        "reservoir256_lms": chain(reservoir(256), readout("LMS")),
        "reservoir256_rls": chain(reservoir(256), readout("RLS")),
        "nvar_ridge": chain({"type": "NVAR", "params": {"delay": 2, "order": 2, "strides": 1}},
                            readout("Ridge")),
        "ip512_ridge": chain(reservoir(512, "IPReservoir"), readout("Ridge")),
        "ip1500_ridge": chain(reservoir(1500, "IPReservoir"), readout("Ridge")),
        "reservoir2000_ridge": chain(reservoir(2000), readout("Ridge")),
        "reservoir400_rls": chain(reservoir(400), readout("RLS")),
        "nvar3_ridge": chain({"type": "NVAR", "params": {"delay": 3, "order": 3, "strides": 1}},
                             readout("Ridge")),
    }


def preflight(architecture, train_rows, memory_limit):
    from src.memory_estimator import estimateMemory, getInputDimension

    estimate = float(estimateMemory(architecture, train_rows))
    if estimate > memory_limit:
        return estimate, f"estimated {estimate:.1f} MiB exceeds {memory_limit} MiB search limit"
    caps = {"Ridge": 2200, "RLS": 400, "LMS": 3000}
    for i, node in enumerate(architecture["nodes"]):
        width = getInputDimension(architecture, i)
        if node["type"] in caps and width > caps[node["type"]]:
            return estimate, f"{node['type']} input width {width} exceeds existing search cap {caps[node['type']]}"
    return estimate, None


def construct_sequence_model(architecture):
    from ceti.models import construct_sequence_model as construct

    return construct(architecture)


def fit_sequences(model, train_x, train_y, warmup, ridge_passes):
    from ceti.models import fit_sequences as fit

    return fit(model, train_x, train_y, warmup, ridge_passes)


def trial_worker(connection, cache_path, metadata_path, task, options):
    result = dict(task)
    started = time.perf_counter()

    def phase(name):
        connection.send({"phase": name})

    try:
        phase("imports")
        import numpy as np
        import reservoirpy as rpy
        from sklearn.metrics import accuracy_score, f1_score
        from ceti.data import select_and_scale
        from src.utils import runModel

        if not rpy.__version__.startswith("0.3."):
            raise RuntimeError(f"Use main's reservoirpy 0.3 environment; found {rpy.__version__}.")
        rpy.verbosity(0)
        rpy.set_seed(task["seed"])
        np.random.seed(task["seed"])
        phase("loading")
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        with np.load(cache_path, allow_pickle=False) as cache:
            arrays = {key: cache[key] for key in cache.files}
        train_x, val_x, columns, _ = select_and_scale(arrays, metadata["columns"], task["feature_set"])
        train_targets = [np.repeat(np.eye(5)[label][None, :], train_x.shape[1], axis=0)
                         for label in arrays["train_labels"]]
        train_inputs = list(train_x)
        architecture = architecture_cases(len(columns))[task["case"]]
        result.update(input_columns=len(columns), train_rows=len(train_x) * train_x.shape[1],
                      validation_rows=len(val_x) * val_x.shape[1], architecture=architecture,
                      ridge_passes=options["ridge_passes"], warmup=options["warmup"])
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
            fit_sequences(model, train_inputs, train_targets, options["warmup"], options["ridge_passes"])
            result["fit_seconds"] = time.perf_counter() - start
            # evaluateArchitecture copies the fitted model before validation.
            phase("snapshot")
            start = time.perf_counter()
            snapshot = copy.deepcopy(model)
            result["snapshot_seconds"] = time.perf_counter() - start
            del snapshot
            phase("validation")
            start = time.perf_counter()
            predicted = []
            for sequence in val_x:
                model.reset()
                scores = np.asarray(runModel(model, sequence))
                if scores.shape != (len(sequence), 5) or not np.isfinite(scores).all():
                    raise ValueError("Invalid or non-finite validation predictions.")
                predicted.append(int(np.argmax(scores[options["warmup"]:].mean(axis=0))))
            result["validation_seconds"] = time.perf_counter() - start
            result["validation_accuracy"] = float(accuracy_score(arrays["val_labels"], predicted))
            result["validation_macro_f1"] = float(f1_score(arrays["val_labels"], predicted,
                                                          labels=list(range(5)), average="macro", zero_division=0))
            result["evaluation_seconds"] = sum(result[key] for key in
                ("construct_seconds", "fit_seconds", "snapshot_seconds", "validation_seconds"))
            result["status"] = "ok"
    except Exception:
        result.update(status="error", reason=traceback.format_exc())
    result["worker_wall_seconds"] = time.perf_counter() - started
    connection.send({"result": result})
    connection.close()


def run_trials(tasks, cache_path, metadata_path, options, output_dir,
               worker=trial_worker, phase_timeouts=None):
    """Spawn CPU workers with hard per-trial and overall wall-clock budgets."""
    context = mp.get_context("spawn")
    pending, active, results = list(tasks), [], []
    started = time.perf_counter()
    log = Path(output_dir) / "trials.jsonl"
    with log.open("w", encoding="utf-8") as stream:
        def finish(job, result):
            job["process"].join(timeout=0.2)
            if job["process"].is_alive():
                job["process"].terminate()
                job["process"].join(timeout=3)
            job["pipe"].close()
            result["process_wall_seconds"] = time.perf_counter() - job["start"]
            results.append(result)
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            stream.flush()
            duration = result.get("evaluation_seconds", result["process_wall_seconds"])
            print(f"{result['feature_set']:>5} {result['case']:<23} seed={result['seed']} "
                  f"{result['status']:<20} {duration:.3f}s", flush=True)
            if "test_accuracy" in result:
                print(f"      full test: accuracy={result['test_accuracy']:.3f}, "
                      f"balanced accuracy={result['test_balanced_accuracy']:.3f}, "
                      f"macro F1={result['test_macro_f1']:.3f}", flush=True)
            if result["status"] == "error":
                print(result["reason"], flush=True)

        while pending or active:
            elapsed = time.perf_counter() - started
            if elapsed >= options["max_seconds"]:
                for task in pending:
                    result = dict(task, status="not_run_budget", reason="Overall pilot time budget exhausted.")
                    results.append(result)
                    stream.write(json.dumps(result) + "\n")
                pending.clear()
            while pending and len(active) < options["workers"]:
                task = pending.pop(0)
                parent, child = context.Pipe(duplex=False)
                process = context.Process(target=worker,
                    args=(child, str(cache_path), str(metadata_path), task, options))
                process.start()
                child.close()
                active.append({"process": process, "pipe": parent, "task": task,
                               "start": time.perf_counter(), "phase": "starting",
                               "phase_start": time.perf_counter()})
            for job in list(active):
                completed = None
                try:
                    while job["pipe"].poll():
                        message = job["pipe"].recv()
                        if "result" in message:
                            completed = message["result"]
                            break
                        job["phase"] = message["phase"]
                        job["phase_start"] = time.perf_counter()
                except EOFError:
                    pass
                if completed is not None:
                    finish(job, completed)
                    active.remove(job)
                elif (time.perf_counter() - (job["phase_start"] if phase_timeouts else job["start"])
                      >= (phase_timeouts.get(job["phase"], options["timeout"])
                          if phase_timeouts else options["timeout"]) or
                      time.perf_counter() - started >= options["max_seconds"]):
                    finish(job, dict(job["task"], status="timeout", last_phase=job["phase"],
                                     reason="Per-trial or overall wall-clock budget exceeded."))
                    active.remove(job)
                elif not job["process"].is_alive():
                    finish(job, dict(job["task"], status="error", last_phase=job["phase"],
                                     reason=f"Worker exited with code {job['process'].exitcode}"))
                    active.remove(job)
            if active:
                time.sleep(0.05)
    return results


def search_projection(results, args):
    # ESNAS runs an initial population, generations further populations, and
    # probes the default BO point in addition to bo_init + bo_iter evaluations.
    fits = ((args.generations + 1) * args.population * (1 + args.bo_init + args.bo_iter)
            * args.num_evals * args.search_repeats)
    projections = []
    for feature_set in args.feature_sets:
        rows = [row for row in results if row["feature_set"] == feature_set]
        seconds = sorted(row["evaluation_seconds"] for row in rows if row["status"] == "ok")
        if not seconds:
            continue
        median = statistics.median(seconds)
        high = seconds[max(0, math.ceil(0.9 * len(seconds)) - 1)]
        projections.append({
            "feature_set": feature_set, "nominal_model_fits": fits,
            "successful_trials": len(seconds),
            "median_evaluation_seconds": median, "p90_evaluation_seconds": high,
            "serial_hours_at_median": fits * median / 3600,
            "ideal_parallel_hours_at_median": fits * median / args.search_workers / 3600,
            "ideal_parallel_hours_at_p90": fits * high / args.search_workers / 3600,
            "incomplete_trials": sum(row["status"] not in ("ok", "skipped_search_limit") for row in rows),
            "qualification": "Illustrative fixed-case extrapolation, not an ESNAS runtime prediction. "
                "Excludes architecture generation/validity checks, extra stagnation populations, "
                "BO, checkpoint and process overhead, and parallel contention. Timeouts censor slow cases. "
                "Representative profiles are not a random sample of the architecture search space.",
        })
    return projections


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "ceti/CeTI-Locomotion")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--split-manifest", type=Path, default=ROOT / "results/ceti_timing/subject_split.json")
    p.add_argument("--seed", type=int, default=1234123123)
    p.add_argument("--train-per-class", type=int, default=40)
    p.add_argument("--val-per-class", type=int, default=10)
    p.add_argument("--val-subjects", type=int, default=10)
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--feature-sets", nargs="+", default=["all", "leg6"],
                   help="all, leg6, or comma-separated exact input column names")
    p.add_argument("--cases", nargs="+", choices=list(architecture_cases(6)))
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--timeout", type=float, default=30)
    p.add_argument("--max-seconds", type=float, default=600)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--memory-limit", type=float, default=756)
    p.add_argument("--ridge-passes", type=int, choices=(1, 2), default=2)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--population", type=int, default=40)
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--bo-init", type=int, default=0)
    p.add_argument("--bo-iter", type=int, default=5)
    p.add_argument("--num-evals", type=int, default=3)
    p.add_argument("--search-repeats", type=int, default=5)
    p.add_argument("--search-workers", type=int, default=20)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not 0 <= args.warmup < args.frames or args.frames < 2:
        raise SystemExit("Require 0 <= warmup < frames, with at least two frames.")
    if min(args.repeats, args.workers, args.train_per_class, args.val_per_class,
           args.population, args.num_evals, args.search_repeats, args.search_workers) < 1:
        raise SystemExit("Counts must be positive.")
    if min(args.timeout, args.max_seconds, args.memory_limit) <= 0 or min(args.generations, args.bo_init, args.bo_iter) < 0:
        raise SystemExit("Budgets must be positive and search iteration counts nonnegative.")
    import numpy as np
    from ceti.data import prepare_subset, select_and_scale

    output = args.output_dir or ROOT / "results/ceti_timing" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    arrays, metadata = prepare_subset(args.data_dir, args.split_manifest,
        args.train_per_class, args.val_per_class, args.frames, args.seed, args.val_subjects)
    metadata["feature_sets"] = {}
    for feature_set in args.feature_sets:
        _, _, columns, scale = select_and_scale(arrays, metadata["columns"], feature_set)
        metadata["feature_sets"][feature_set] = {"columns": columns, "training_max_abs_scale": scale.tolist()}
    (output / "subset.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    np.savez(output / "subset.npz", **arrays)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(python=sys.version, executable=sys.executable, platform=platform.platform(),
                  cpu_count=os.cpu_count(), processor=platform.processor(),
                  backend="NumPy/SciPy; no JAX", test_evaluated=False,
                  package_versions={name: version(name) for name in
                      ("numpy", "scipy", "scikit-learn", "reservoirpy", "networkx")})
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Training: {metadata['train_rows']} timesteps; validation: {metadata['validation_rows']} timesteps.")
    print(f"Subjects: {len(metadata['split']['train'])} train / {len(metadata['split']['validation'])} validation / "
          f"{len(metadata['split']['test'])} held-out test; test contents are not read.")
    print(f"Output: {output}", flush=True)
    if args.prepare_only:
        return 0
    cases = args.cases or list(DEFAULT_CASES)
    # Alternate widths for each case so a short pilot still compares inputs.
    tasks = [{"case": case, "feature_set": width, "seed": args.seed + repeat}
             for repeat in range(args.repeats) for case in cases for width in args.feature_sets]
    results = run_trials(tasks, output / "subset.npz", output / "subset.json", vars(args), output)
    projections = search_projection(results, args)
    report = {"trials": results, "search_projection": projections}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = ["case", "feature_set", "seed", "status", "input_columns", "train_rows", "validation_rows",
              "construct_seconds", "fit_seconds", "snapshot_seconds", "validation_seconds", "evaluation_seconds",
              "process_wall_seconds", "validation_accuracy", "validation_macro_f1", "estimated_memory_mib", "reason"]
    with (output / "timings.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    for projection in projections:
        print(f"{projection['feature_set']}: median complete evaluation "
              f"{projection['median_evaluation_seconds']:.3f}s; idealized "
              f"{args.search_repeats}-search total at {args.search_workers} workers: "
              f"{projection['ideal_parallel_hours_at_median']:.2f}h. "
              "Excludes overhead/contention and censored slow cases; see report.json.")
    print("No architecture search or test evaluation was run.", flush=True)
    return 1 if any(row["status"] == "error" for row in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
