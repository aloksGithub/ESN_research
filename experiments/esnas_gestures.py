import os

# JAX reads its allocator configuration when the GPU backend is first created.
# Keep this before reservoirpy/JAX imports so search workers grow on demand
# instead of reserving 75% of the GPU in every process.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import reservoirpy as rpy
import numpy as np
import sys
import warnings
import traceback
import sklearn.metrics
import dill
from deap import base, creator
import torch
import copy
import csv
import gc
import json
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from time import perf_counter

# Filter warnings and set reservoirpy verbosity
warnings.filterwarnings("ignore")
rpy.set_seed(0)
np.random.seed(0)
torch.manual_seed(0)
import random as _random
_random.seed(0)

# Add parent directory to sys.path to import from src and gesture_recognition
current_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'gesture_recognition'))
data_dir = os.path.join(root_dir, 'gesture_recognition', 'dataSets')

from src.algorithms.ESN_GA_BO import EvalParams, ExperimentData, GAParams, ModelParams, ESNAS
from src.utils import constructModel, runModel, trainModel
from gesture_recognition.Utils import createData, getData
from gesture_recognition import Evaluation

files = ['s', 'j', 'na', 'l', 'ni']

def testESN(esn, testFiles, fixed_threshold=0.4):
    testF1MaxApps = []
    testAccuracies = []
    
    _, _, trainloader, testloader = createData(
        inputFiles=testFiles,
        testFiles=testFiles,
        dataDir=data_dir,
        nFolds=len(testFiles),
    )

    for test_inputs, test_targets in trainloader:
        inputs = test_inputs[0].numpy()
        targets = test_targets[0].numpy()
        prediction = esn.run(inputs)

        t_target = targets
        threshold = np.ones((prediction.shape[0],1))*fixed_threshold

        t_maxApp_prediction = Evaluation.calcMaxActivityPrediction(prediction,t_target,threshold, 10)

        pred_MaxApp, targ_MaxApp = Evaluation.calcInputSegmentSeries(t_maxApp_prediction, t_target, 0.5)
        testF1MaxApps.append(np.mean(sklearn.metrics.f1_score(targ_MaxApp,pred_MaxApp,average=None)))
        testAccuracies.append(np.mean(sklearn.metrics.accuracy_score(targ_MaxApp,pred_MaxApp)))
    return np.array(testF1MaxApps).mean(), np.array(testAccuracies).mean()

def getSequenceData(loader):
    """Extracts individual sequences from the DataLoader instead of concatenating them."""
    x = []
    y = []
    for inputs, targets in loader:
        x.append(inputs[0].numpy())
        y.append(targets[0].numpy())
    return x, y

def gesture_f1_metric(y_true, y_pred):
    """
    Mimics testESN from Train.py: averages F1 scores across sequences.
    This is used during ESNAS search (validation).
    """
    if not isinstance(y_true, list):
        # Fallback if somehow concatenated arrays are passed
        y_true = [y_true]
        y_pred = [y_pred]
    
    f1_scores = []
    for target, prediction in zip(y_true, y_pred):
        # Following testESN: threshold=0.4, minLength=10
        threshold = np.ones((prediction.shape[0], 1)) * 0.4
        t_maxApp_prediction = Evaluation.calcMaxActivityPrediction(prediction, target, threshold, 10)
        
        # Following testESN: threshold=0.5 for calcInputSegmentSeries
        pred_MaxApp, targ_MaxApp = Evaluation.calcInputSegmentSeries(t_maxApp_prediction, target, 0.5)
        
        # Calculate per-sequence macro F1
        score = np.mean(sklearn.metrics.f1_score(targ_MaxApp, pred_MaxApp, average=None))
        f1_scores.append(score)
    
    # Return 1 - average F1 (to minimize)
    return 1.0 - np.mean(f1_scores)

def gesture_acc_metric(y_true, y_pred):
    """Mimics testESN from Train.py: averages Accuracy across sequences."""
    if not isinstance(y_true, list):
        y_true = [y_true]
        y_pred = [y_pred]
    
    acc_scores = []
    for target, prediction in zip(y_true, y_pred):
        threshold = np.ones((prediction.shape[0], 1)) * 0.4
        t_maxApp_prediction = Evaluation.calcMaxActivityPrediction(prediction, target, threshold, 10)
        pred_MaxApp, targ_MaxApp = Evaluation.calcInputSegmentSeries(t_maxApp_prediction, target, 0.5)
        
        acc = np.mean(sklearn.metrics.accuracy_score(targ_MaxApp, pred_MaxApp))
        acc_scores.append(acc)
    
    return np.mean(acc_scores)

def print_saved_fold_results(idx, ga=None):
    """Evaluate a single saved fold using the jax.export artifact."""
    save_folder = 'results/esnas_gestures/global1'

    print(f"\n======================== Fold {idx+1}/5 ========================")
    testFiles = files[idx:idx+1]
    inputFiles = files[:idx] + files[idx+1:]
    train_dataset, _, train_loader, _ = createData(inputFiles, testFiles, dataDir=data_dir)
    trainX, trainY = getData(train_loader)
    import hashlib
    print(f"trainX hash: {hashlib.md5(trainX.tobytes()).hexdigest()}")
    print(f"trainY hash: {hashlib.md5(trainY.tobytes()).hexdigest()}")

    fold_save_loc = os.path.join(save_folder, f'fold_{idx}')

    # Register deap creator types before loading (they're created dynamically)
    if not hasattr(creator, "Fitness"):
        creator.create("Fitness", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", dict, fitness=creator.Fitness)

    # Reuse a just-completed run when available. Loading its checkpoint would
    # reconstruct a second copy of the same JAX model on the GPU.
    if ga is None:
        with open(os.path.join(fold_save_loc, 'esnas_backup.obj'), "rb") as f:
            ga = dill.load(f)
    
    # model = trainModel(ga.bestModel, trainX, trainY)

    # In-progress checkpoints keep the winning model serialized so loading a
    # checkpoint does not allocate GPU memory until inference is requested.
    best_model = ga.materializeBestModel()
    if best_model is None:
        raise RuntimeError(
            "The saved ESNAS run does not contain a trained best model."
        )

    f1, acc = testESN(best_model, testFiles)
    print(f1, acc)
    return f1, acc

def _as_builtin(value):
    """Convert checkpoint dictionaries and NumPy scalars for JSON output."""
    if isinstance(value, dict):
        return {key: _as_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_builtin(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _select_saved_architecture(ga):
    """Select by search fitness, never by a saved model's test performance."""
    expected = int(ga.gaParams.generations) + 1
    if len(getattr(ga, "generationTimes", [])) < expected:
        raise ValueError("The checkpoint is incomplete; finish the search first.")
    if not ga.architectures or len(ga.architectures) != len(ga.fitnesses):
        raise ValueError("Checkpoint architecture and fitness histories are missing or inconsistent.")
    primary = [float(fitness[0]) for fitness in ga.fitnesses]
    select = min if ga.evalParams.minimizeFitness else max
    best_index = select(range(len(primary)), key=primary.__getitem__)
    return {
        "architecture": _as_builtin(copy.deepcopy(ga.architectures[best_index])),
        "validation_fitness": _as_builtin(ga.fitnesses[best_index]),
        "search_seconds": float(sum(ga.generationTimes)),
    }


def _saved_architecture_worker(checkpoints, connection):
    """Load saved weights on CPU; only lightweight architecture data returns."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    try:
        if not hasattr(creator, "Fitness"):
            creator.create("Fitness", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", dict, fitness=creator.Fitness)
        extracted = []
        for checkpoint in checkpoints:
            with open(checkpoint, "rb") as stream:
                ga = dill.load(stream)
            extracted.append(_select_saved_architecture(ga))
            del ga
            gc.collect()
        connection.send((True, extracted))
    except Exception:
        connection.send((False, traceback.format_exc()))
    finally:
        connection.close()


def _load_saved_architectures(save_folder, folds):
    checkpoints = [
        Path(save_folder) / f"fold_{fold - 1}" / "esnas_backup.obj"
        for fold in folds
    ]
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_saved_architecture_worker, args=(checkpoints, sender),
    )
    process.start()
    sender.close()
    try:
        while not receiver.poll(1):
            if not process.is_alive():
                raise RuntimeError(f"Checkpoint extraction exited with code {process.exitcode}.")
        try:
            ok, payload = receiver.recv()
        except EOFError as exc:
            raise RuntimeError("Checkpoint extraction ended without returning architectures.") from exc
        if not ok:
            raise RuntimeError(f"Unable to extract saved architectures:\n{payload}")
        return payload
    finally:
        receiver.close()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join()


def _architecture_with_seed(architecture, seed):
    seeded = copy.deepcopy(architecture)
    for index, node in enumerate(seeded["nodes"]):
        if node["type"] in {"Reservoir", "IPReservoir"}:
            node["params"]["seed"] = seed * 1000 + index
    return seeded


def _summarize_saved_repeats(records, folds):
    """Keep per-subject SD separate from SD across complete fold averages."""
    fold_summaries = {}
    for fold in folds:
        rows = [row for row in records if row["fold"] == fold]
        fold_summaries[str(fold)] = {
            "test_subject": files[fold - 1],
            "num_repeats": len(rows),
            "f1_mean": float(np.mean([row["test_f1"] for row in rows])),
            "f1_std": float(np.std([row["test_f1"] for row in rows])),
            "accuracy_mean": float(np.mean([row["test_accuracy"] for row in rows])),
            "accuracy_std": float(np.std([row["test_accuracy"] for row in rows])),
        }
    repeat_means = []
    for repeat in sorted({row["repeat"] for row in records}):
        rows = [row for row in records if row["repeat"] == repeat]
        if len(rows) == len(folds) and {row["fold"] for row in rows} == set(folds):
            repeat_means.append({
                "repeat": repeat, "seed": rows[0]["seed"],
                "f1": float(np.mean([row["test_f1"] for row in rows])),
                "accuracy": float(np.mean([row["test_accuracy"] for row in rows])),
            })
    overall = {"num_complete_cv_repeats": len(repeat_means)}
    for metric in ("f1", "accuracy"):
        values = [row[metric] for row in repeat_means]
        overall[f"{metric}_mean"] = float(np.mean(values)) if values else None
        overall[f"{metric}_std"] = float(np.std(values)) if values else None
    return {"folds": fold_summaries, "cv_repeat_means": repeat_means, "overall": overall}


def print_saved_results(repeats=10, base_seed=0, folds=None, save_folder=None, output_dir=None):
    """Retrain each saved winner on all non-test subjects over seeds 0--9.

    Match the former evaluate_esnas_gesture_repeats.py protocol, including
    fresh augmentation per seed and x64 Ridge fitting. Checkpoint weights are
    discarded. Results are saved separately and the summary is returned.
    ``folds`` uses one-based fold numbers; the default evaluates all five.
    """
    import jax

    folds = list(range(1, len(files) + 1)) if folds is None else list(folds)
    if repeats < 1 or base_seed < 0 or base_seed + repeats - 1 > 4294967295:
        raise ValueError("Use positive repeats and seeds within NumPy's uint32 range.")
    if not folds or len(set(folds)) != len(folds) or any(f not in range(1, 6) for f in folds):
        raise ValueError("folds must contain distinct fold numbers from 1 to 5.")
    save_folder = Path(save_folder or Path(root_dir) / "results/esnas_gestures/global1").resolve()
    output_dir = Path(output_dir or save_folder / (
        "repeat_evaluation_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation directory: {output_dir}")

    print("Extracting winning architectures from completed checkpoints...", flush=True)
    extracted = _load_saved_architectures(save_folder, folds)
    jax.config.update("jax_enable_x64", True)
    print("JAX x64 enabled for fresh-model training.", flush=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_path = output_dir / "raw_results.csv"
    fields = [
        "fold", "search_train_subjects", "validation_subject", "final_train_subjects",
        "test_subject", "repeat", "seed", "test_f1", "test_accuracy", "train_and_test_seconds",
    ]
    records = []
    with raw_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        stream.flush()
        for fold, checkpoint_data in zip(folds, extracted):
            index = fold - 1
            test_subject = files[index]
            retrain_subjects = files[:index] + files[index + 1:]
            val_index = index % len(retrain_subjects)
            validation_subject = retrain_subjects[val_index]
            search_subjects = retrain_subjects[:val_index] + retrain_subjects[val_index + 1:]
            print(f"\n======================== Fold {fold}/5 ========================")
            print(f"Final retraining subjects: {retrain_subjects}; test subject: {test_subject}")
            print(f"Saved validation fitness: {checkpoint_data['validation_fitness']}")
            print(f"Architecture: {checkpoint_data['architecture']}")
            for repeat_index in range(repeats):
                seed = base_seed + repeat_index
                print(f"  Repeat {repeat_index + 1}/{repeats} (seed={seed})...", flush=True)
                _random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                rpy.set_seed(seed)
                _, _, train_loader, _ = createData(
                    retrain_subjects, [test_subject], dataDir=data_dir, nFolds=len(retrain_subjects),
                )
                train_x, train_y = getData(train_loader)
                architecture = _architecture_with_seed(checkpoint_data["architecture"], seed)
                started = perf_counter()
                model = constructModel(architecture)
                try:
                    model = trainModel(model, train_x, train_y)
                    f1, accuracy = testESN(model, [test_subject])
                    if not np.isfinite([f1, accuracy]).all():
                        raise ValueError(f"Non-finite evaluation score for fold {fold}, seed {seed}.")
                    record = dict(zip(fields, [
                        fold, "|".join(search_subjects), validation_subject,
                        "|".join(retrain_subjects), test_subject, repeat_index + 1, seed,
                        float(f1), float(accuracy), perf_counter() - started,
                    ]))
                    records.append(record)
                    writer.writerow(record)
                    stream.flush()
                    print(f"    F1={f1:.4f}, accuracy={accuracy:.4f}", flush=True)
                finally:
                    del model
                    gc.collect()

    summary = _summarize_saved_repeats(records, folds)
    summary["configuration"] = {
        "repeats": repeats, "base_seed": base_seed, "folds": folds,
        "checkpoint_root": str(save_folder), "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "training_protocol": (
            "architecture selected with three training subjects and one validation subject; "
            "each final model retrained on all four non-test subjects"
        ),
    }
    summary["architectures"] = {str(fold): data for fold, data in zip(folds, extracted)}
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print("\n======================== Per-fold results ========================")
    for fold, result in summary["folds"].items():
        print(f"Fold {fold}: F1={result['f1_mean']:.4f} (+/- {result['f1_std']:.4f}), "
              f"accuracy={result['accuracy_mean']:.4f} (+/- {result['accuracy_std']:.4f})")
    print(f"\nOverall: mean and SD across {repeats} averages of {len(folds)} fold(s)")
    overall = summary["overall"]
    print(f"Average Test F1: {overall['f1_mean']:.4f} (+/- {overall['f1_std']:.4f})")
    print(f"Average Test Acc: {overall['accuracy_mean']:.4f} (+/- {overall['accuracy_std']:.4f})")
    print(f"Raw results: {raw_path}\nSummary: {summary_path}")
    return summary

def run_esnas_gestures():
    save_folder = 'results/esnas_gestures/global1'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    all_test_f1s = []
    all_test_accs = []

    for idx in range(5):
        print(f"\n======================== Fold {idx+1}/5 ========================")
        inputFiles = files[:idx] + files[idx+1:]
        testFiles = files[idx:idx+1]
        validationFiles = [inputFiles[idx%4]]
        trainFiles = inputFiles[:idx%4] + inputFiles[idx%4+1:]
        
        print(f"Train subjects: {trainFiles}")
        print(f"Validation subject: {validationFiles}")
        print(f"Test subject: {testFiles}")

        # Load data
        data_dir = os.path.join(root_dir, 'gesture_recognition', 'dataSets')
        
        # Training data (Concatenated for fitting)
        train_dataset, _, train_loader, _ = createData(trainFiles, testFiles, dataDir=data_dir)
        trainX, trainY = getData(train_loader)
        
        # Validation data (List of sequences for per-sequence averaging)
        _, val_dataset, _, val_loader = createData(trainFiles, validationFiles, dataDir=data_dir)
        valX, valY = getSequenceData(val_loader)
        valX, valY = valX[0], valY[0]
        
        # Test data (List of sequences)
        _, test_dataset, _, test_loader = createData(trainFiles, testFiles, dataDir=data_dir)
        testX, testY = getSequenceData(test_loader)
        testX, testY = testX[0], testY[0]

        experimentData = ExperimentData(trainX, trainY, valX, valY, testX, testY)

        # Define error metrics following testESN behavior
        errorMetrics = [gesture_f1_metric, gesture_acc_metric]
        
        evalParams = EvalParams(
            numEvals=3,
            errorMetrics=errorMetrics,
            defaultErrors=[1.0, 0.0],
            timeout=40,
            memoryLimit=756,
            minimizeFitness=True,
            isAutoRegressive=False,
        )
        
        gaParams = GAParams(
            generations=10,
            populationSize=20,
            crossoverProbability=0.7,
            mutationProbability=0.2,
            eliteSize=1,
            stagnationReset=5,
        )
        
        modelParams = ModelParams(
            num_nodes_range=(1, 2),
        )

        fold_save_loc = os.path.join(save_folder, f'fold_{idx}')
        if not os.path.exists(fold_save_loc):
            os.makedirs(fold_save_loc)

        # Run ESNAS for this fold
        ga = ESNAS(
            experimentData,
            evalParams,
            gaParams,
            modelParams,
            n_jobs=5,
            saveLocation=os.path.join(fold_save_loc, 'esnas_backup.obj'),
            bo_init=3,
            bo_iter=8
        )
        
        ga.run()

        test_f1, test_acc = print_saved_fold_results(idx, ga=ga)
        all_test_f1s.append(test_f1)
        all_test_accs.append(test_acc)

    print("\n======================== Final Results Across Folds ========================")
    if all_test_f1s:
        print(f"Average Test F1: {np.mean(all_test_f1s):.4f} (+/- {np.std(all_test_f1s):.4f})")
        print(f"Average Test Acc: {np.mean(all_test_accs):.4f} (+/- {np.std(all_test_accs):.4f})")
    else:
        print("No results to report.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Search gesture architectures or evaluate saved winners.")
    parser.add_argument("--print-saved", action="store_true", help="Evaluate saved architectures over fresh seeds.")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--folds", type=int, nargs="+", choices=range(1, 6), default=[1, 2, 3, 4, 5])
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if args.print_saved:
        print_saved_results(args.repeats, args.base_seed, args.folds, output_dir=args.output_dir)
    else:
        run_esnas_gestures()
