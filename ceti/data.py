"""Reproducible, subject-disjoint CeTI-Locomotion experiment inputs.

Training loaders leave test contents unread. Explicit test loaders are called
after model selection. No supplied CeTI scripts are imported or executed.
"""

import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np


ACTIVITIES = (
    "gaitNormal", "gaitFast", "gaitNormalBackpack",
    "gaitNormalBottleCrate", "sts",
)
LEG6 = (
    "LeftHip_flexion", "RightHip_flexion",
    "LeftKnee_flexion", "RightKnee_flexion",
    "LeftAnkle_dorsiflexion", "RightAnkle_dorsiflexion",
)


def inventory(dataset_root):
    root = Path(dataset_root) / "derivatives" / "cut_sequences"
    with (root / "participants.tsv").open(newline="", encoding="utf-8-sig") as stream:
        subjects = sorted(row["participant_id"] for row in csv.DictReader(stream, delimiter="\t"))
    records = []
    for subject in subjects:
        for path in sorted((root / subject / "motion").glob("*_motion.tsv")):
            match = re.search(r"_task-([^_]+)_", path.name)
            if match is None or match.group(1) not in ACTIVITIES:
                raise ValueError(f"Unknown activity in {path}")
            records.append({"subject": subject, "activity": match.group(1),
                            "path": path.relative_to(root).as_posix()})
    if not records:
        raise ValueError(f"No processed motion segments in {root}")
    return root, subjects, records


def subject_split(subjects, manifest_path, seed=1234123123, test_subjects=10, val_subjects=10):
    """Persist explicit IDs; never infer the paper's holdout from its RNG seed."""
    path = Path(manifest_path)
    if path.exists():
        split = json.loads(path.read_text(encoding="utf-8"))
        if split["seed"] != seed:
            raise ValueError("Split seed differs from the existing manifest; use another manifest path.")
    else:
        shuffled = sorted(subjects)
        if test_subjects < 1 or val_subjects < 1 or test_subjects + val_subjects >= len(shuffled):
            raise ValueError("Need nonempty training, validation and test subject groups.")
        random.Random(seed).shuffle(shuffled)
        split = {
            "schema_version": 1, "seed": seed,
            "provenance": "Deterministic timing holdout; NOT the recovered paper holdout.",
            "test": sorted(shuffled[:test_subjects]),
            "validation": sorted(shuffled[test_subjects:test_subjects + val_subjects]),
            "train": sorted(shuffled[test_subjects + val_subjects:]),
        }
    groups = [set(split[key]) for key in ("train", "validation", "test")]
    if any(not group for group in groups) or any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("Subject groups must be nonempty and disjoint.")
    if set.union(*groups) != set(subjects):
        raise ValueError("Subject IDs in the split do not match the dataset.")
    if len(groups[1]) != val_subjects or len(groups[2]) != test_subjects:
        raise ValueError("Requested subject counts differ from the saved split.")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    return split


def sample_segments(records, subjects, per_class, seed):
    """Round-robin across subjects within each class before taking repetitions."""
    if per_class < 1:
        raise ValueError("Segments per class must be positive.")
    rng = random.Random(seed)
    selected = []
    for activity in ACTIVITIES:
        buckets = {}
        for record in records:
            if record["subject"] in subjects and record["activity"] == activity:
                buckets.setdefault(record["subject"], []).append(record)
        if sum(map(len, buckets.values())) < per_class:
            raise ValueError(f"Not enough {activity} segments for the requested subset.")
        for bucket in buckets.values():
            rng.shuffle(bucket)
        count = 0
        while count < per_class:
            order = sorted(subject for subject, bucket in buckets.items() if bucket)
            rng.shuffle(order)
            for subject in order:
                selected.append(buckets[subject].pop())
                count += 1
                if count == per_class:
                    break
    rng.shuffle(selected)
    return selected


def resample(values, frames):
    """Match the author's interpolation grid: arange(frames) * (T-1)/frames."""
    if frames < 2 or len(values) < 2:
        raise ValueError("Need at least two source and resampled frames.")
    old_times = np.arange(len(values))
    new_times = np.arange(frames) * ((len(values) - 1) / frames)
    return np.column_stack([np.interp(new_times, old_times, values[:, i])
                            for i in range(values.shape[1])])


def load_segments(root, records, frames, expected_columns=None, dtype=np.float64):
    inputs, labels, source_lengths = [], [], []
    columns = expected_columns
    for record in records:
        path = root / record["path"]
        with path.open(encoding="utf-8-sig") as stream:
            header = stream.readline().rstrip("\r\n").split("\t")
            values = np.loadtxt(stream, delimiter="\t", ndmin=2, dtype=dtype)
        if columns is None:
            columns = header
        if header != columns or values.shape[1] != len(columns):
            raise ValueError(f"Inconsistent motion columns in {path}")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite motion values in {path}")
        inputs.append(resample(values, frames))
        labels.append(ACTIVITIES.index(record["activity"]))
        source_lengths.append(len(values))
    return np.stack(inputs), np.asarray(labels, dtype=np.int64), columns, source_lengths


def prepare_subset(dataset_root, manifest_path, train_per_class=40, val_per_class=10,
                   frames=20, seed=1234123123, val_subjects=10):
    """Prepare disjoint subjects; val_per_class=0 uses every validation segment."""
    if val_per_class < 0:
        raise ValueError("Validation segments per class must be nonnegative (0 means all).")
    root, subjects, records = inventory(dataset_root)
    split = subject_split(subjects, manifest_path, seed, val_subjects=val_subjects)
    train = sample_segments(records, set(split["train"]), train_per_class, seed + 1)
    if val_per_class == 0:
        val = [record for record in records if record["subject"] in set(split["validation"])]
    else:
        val = sample_segments(records, set(split["validation"]), val_per_class, seed + 2)
    train_x, train_labels, columns, train_lengths = load_segments(root, train, frames)
    val_x, val_labels, _, val_lengths = load_segments(root, val, frames, columns)
    metadata = {
        "split": split, "class_names": list(ACTIVITIES), "columns": columns,
        "frames_per_segment": frames, "train_shape": list(train_x.shape),
        "validation_shape": list(val_x.shape),
        "train_rows": len(train) * frames, "validation_rows": len(val) * frames,
        "inventory_segments": len(records), "inventory_subjects": len(subjects),
        "inventory_class_counts": dict(Counter(r["activity"] for r in records)),
        "train_records": train, "validation_records": val,
        "validation_selection": "all" if val_per_class == 0 else "balanced_subset",
        "validation_class_counts": dict(Counter(r["activity"] for r in val)),
        "train_source_lengths": train_lengths, "validation_source_lengths": val_lengths,
        "test_contents_loaded": False,
    }
    arrays = {"train_x": train_x, "train_labels": train_labels,
              "val_x": val_x, "val_labels": val_labels}
    return arrays, metadata


def select_and_scale(arrays, columns, feature_set):
    if feature_set == "all":
        selected = list(columns)
    elif feature_set == "leg6":
        selected = list(LEG6)
    else:
        selected = [name.strip() for name in feature_set.split(",") if name.strip()]
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("Supply unique input column names.")
    missing = set(selected) - set(columns)
    if missing:
        raise ValueError(f"Unknown input columns: {sorted(missing)}")
    indices = [columns.index(name) for name in selected]
    train_x = arrays["train_x"][:, :, indices].copy()
    val_x = arrays["val_x"][:, :, indices].copy()
    # Match the supplied SVM's MaxAbsScaler, fitted on training timesteps only.
    scale = np.max(np.abs(train_x), axis=(0, 1))
    scale[scale == 0] = 1.0
    return train_x / scale, val_x / scale, selected, scale


def load_svm_inputs(dataset_root, manifest_path, frames=20, seed=1234123123,
                    training_subset="small", train_per_class=100):
    """Paper-format flattened segments, unscaled: scaling belongs inside CV.

    Full mode uses all 40 non-test subjects. Small mode reuses ESNAS's
    class/subject-balanced training subset from 30 training subjects.
    """
    root, subjects, records = inventory(dataset_root)
    split = subject_split(subjects, manifest_path, seed)
    if training_subset == "full":
        selected_subjects = set(split["train"] + split["validation"])
        train = [record for record in records if record["subject"] in selected_subjects]
    elif training_subset == "small":
        train = sample_segments(records, set(split["train"]), train_per_class, seed + 1)
    else:
        raise ValueError("training_subset must be full or small")
    # The supplied script casts TSV data to float32 before interpolation.
    x, labels, columns, lengths = load_segments(root, train, frames, dtype=np.float32)
    test = [record for record in records if record["subject"] in set(split["test"])]
    metadata = {
        "split": split, "class_names": list(ACTIVITIES), "columns": columns,
        "frames_per_segment": frames, "training_subset": training_subset,
        "train_records": train, "test_records": test, "train_source_lengths": lengths,
        "train_shape": list(x.reshape(len(x), -1).shape),
        "test_contents_loaded": False,
    }
    return x.reshape(len(x), -1), labels, metadata


def load_svm_test(dataset_root, metadata):
    root = Path(dataset_root) / "derivatives" / "cut_sequences"
    x, labels, _, _ = load_segments(root, metadata["test_records"],
        metadata["frames_per_segment"], metadata["columns"], dtype=np.float32)
    return x.reshape(len(x), -1), labels


def load_search_inputs(dataset_root, manifest_path, frames=20, seed=1234123123,
                       train_per_class=100, val_per_class=0, feature_set="all"):
    """Flattened arrays for ESNAS bookkeeping, with explicit segment boundaries."""
    from src.algorithms.types import ExperimentData

    arrays, metadata = prepare_subset(dataset_root, manifest_path, train_per_class,
                                      val_per_class, frames, seed)
    train, val, columns, scale = select_and_scale(arrays, metadata["columns"], feature_set)
    train_y = np.repeat(np.eye(5)[arrays["train_labels"]][:, None, :], frames, axis=1)
    val_y = np.repeat(np.eye(5)[arrays["val_labels"]][:, None, :], frames, axis=1)
    metadata.update(input_columns=columns, training_max_abs_scale=scale.tolist(), feature_set=feature_set)
    data = ExperimentData(train.reshape(-1, len(columns)), train_y.reshape(-1, 5),
                          val.reshape(-1, len(columns)), val_y.reshape(-1, 5), None, None)
    return data, metadata


def load_search_test(dataset_root, metadata):
    root, _, records = inventory(dataset_root)
    test = [record for record in records if record["subject"] in set(metadata["split"]["test"])]
    x, labels, _, _ = load_segments(root, test, metadata["frames_per_segment"], metadata["columns"])
    indices = [metadata["columns"].index(name) for name in metadata["input_columns"]]
    return x[:, :, indices] / np.asarray(metadata["training_max_abs_scale"]), labels, test
