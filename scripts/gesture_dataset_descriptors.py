"""Describe gesture sensor inputs with the journal's existing estimators.

Run from the repository root with the research virtual environment.
Three deterministic, evenly spaced 2300-sample windows per input sequence
bound the quadratic RBF change-point cost. Windows never cross shuffled
sequence boundaries, but may contain the dataset's existing gesture joins.
These are descriptive measurements, not a classifier or model-selection rule.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gesture_recognition"))
from gesture_recognition.DataSet import createData as load_subject
from gesture_recognition.Utils import createData
from scripts.dataset_descriptors import compute_dataset_descriptors

SUBJECTS = ["s", "j", "na", "l", "ni"]
WINDOW = 2300
OUT = ROOT / "analysis" / "gesture_descriptors"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data_dir = str(ROOT / "gesture_recognition" / "dataSets")
    sequences = []
    for subject in SUBJECTS:
        x, _ = load_subject(subject, list(range(10)), list(range(10)), dataDir=data_dir)
        # Same group-max normalization as useNormalized=2, without augmentation.
        x = x.copy()
        for start in (0, 3, 6):
            x[:, start:start+3] /= np.max(np.linalg.norm(x[:, start:start+3], axis=1))
        sequences.append(("unaugmented_subject", subject, x))

    for fold, test in enumerate(SUBJECTS):
        others = SUBJECTS[:fold] + SUBJECTS[fold+1:]
        train = others[:fold % 4] + others[fold % 4+1:]
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        dataset, _, _, _ = createData(train, [test], dataDir=data_dir, nFolds=4)
        # Access each sequence once: __getitem__ normalizes in place.
        # Sample one actual shuffled training sequence per fold, not a join.
        x, _ = dataset[0]
        sequences.append(("augmented_search_train", f"fold_{fold+1}", x.copy()))

    rows = []
    for kind, name, x in sequences:
        if len(x) < WINDOW:
            raise ValueError(f"{name} has only {len(x)} samples")
        for window, start in enumerate(np.linspace(0, len(x)-WINDOW, 3, dtype=int)):
            sample = x[start:start+WINDOW]
            print(f"{kind} {name} window {window+1}/3, start={start}, N={len(x)}", flush=True)
            loaders = [(name, lambda sample=sample: (None, sample))]
            row = compute_dataset_descriptors(dataset_funcs=loaders).iloc[0].to_dict()
            row.update(kind=kind, sequence=name, window=window+1, start=int(start), source_length=len(x))
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT / "windows.csv", index=False)
    frame = pd.DataFrame(rows)
    metrics = [c for c in frame.select_dtypes(include="number").columns
               if c not in {"window", "start", "source_length"}]
    summary = frame.groupby(["kind", "sequence"])[metrics].median()
    summary.to_csv(OUT / "by_subject_or_fold.csv")
    overall = frame.groupby("kind")[metrics].median()
    overall.to_csv(OUT / "overall.csv")
    (OUT / "protocol.json").write_text(json.dumps({
        "seed": 0, "window_length": WINDOW, "windows_per_sequence": 3,
        "subjects": SUBJECTS, "channels": "nine input sensors, not class targets",
        "aggregation": "median across channels within each window; median across windows",
        "search_inputs": "first of four shuffled augmented training sequences per fold; three training subjects",
        "clean_inputs": "ten selected gesture files per subject in loader order, group-max normalized",
        "reference": "analysis/datasets.csv; original estimator definitions unchanged",
        "limitations": "Window sample, not full recordings; channel medians omit cross-channel and label relationships. AMI is a lag proxy, not a required memory length. Seasonality is largest non-DC FFT-bin share, not STL seasonal strength."
    }, indent=2), encoding="utf-8")
    print(overall.to_string(), flush=True)


if __name__ == "__main__":
    main()
