import csv
from pathlib import Path

import numpy as np
import pytest

from ceti.data import (
    ACTIVITIES, LEG6, prepare_subset, resample, sample_segments,
    select_and_scale, subject_split,
)


def test_resampling_uses_authors_grid():
    values = np.column_stack([np.arange(11), 2 * np.arange(11)])
    actual = resample(values, 5)
    np.testing.assert_allclose(actual[:, 0], [0, 2, 4, 6, 8])
    np.testing.assert_allclose(actual[:, 1], [0, 4, 8, 12, 16])


def test_saved_subject_split_is_reused_and_validated(tmp_path):
    subjects = [f"sub-K{i}" for i in range(24)]
    manifest = tmp_path / "split.json"
    original = subject_split(subjects, manifest)
    assert subject_split(list(reversed(subjects)), manifest) == original
    assert [len(original[key]) for key in ("train", "validation", "test")] == [4, 10, 10]
    with pytest.raises(ValueError, match="seed"):
        subject_split(subjects, manifest, seed=5)
    with pytest.raises(ValueError, match="IDs"):
        subject_split(subjects[:-1], manifest)


def test_sampling_preserves_classes_and_spreads_subjects():
    records = [{"subject": subject, "activity": activity, "path": f"{subject}/{activity}/{i}"}
               for subject in ("a", "b", "c") for activity in ACTIVITIES for i in range(3)]
    sample = sample_segments(records, {"a", "b", "c"}, per_class=4, seed=10)
    assert len({record["path"] for record in sample}) == 20
    for activity in ACTIVITIES:
        selected = [r for r in sample if r["activity"] == activity]
        assert len(selected) == 4
        assert {r["subject"] for r in selected} == {"a", "b", "c"}


@pytest.mark.parametrize("val_per_class", [1, 0])
def test_preparation_never_reads_test_motion_files(tmp_path, monkeypatch, val_per_class):
    root = tmp_path / "CeTI-Locomotion" / "derivatives" / "cut_sequences"
    root.mkdir(parents=True)
    subjects = [f"sub-K{i}" for i in range(24)]
    with (root / "participants.tsv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["participant_id"])
        writer.writerows([[subject] for subject in subjects])
    split_path = tmp_path / "split.json"
    split = subject_split(subjects, split_path)
    for subject in subjects:
        motion = root / subject / "motion"
        motion.mkdir(parents=True)
        for activity in ACTIVITIES:
            path = motion / f"{subject}_task-{activity}_run-1_motion.tsv"
            if subject in split["test"]:
                path.write_text("TEST DATA MUST NOT BE READ", encoding="utf-8")
            else:
                np.savetxt(path, np.ones((5, 6)), delimiter="\t", header="\t".join(LEG6), comments="")
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if any(subject in path.parts for subject in split["test"]):
            pytest.fail(f"Held-out test file was read: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    arrays, metadata = prepare_subset(tmp_path / "CeTI-Locomotion", split_path,
                                      train_per_class=2, val_per_class=val_per_class, frames=4)
    assert arrays["train_x"].shape == (10, 4, 6)
    assert arrays["val_x"].shape == (5 if val_per_class else 50, 4, 6)
    assert metadata["validation_selection"] == ("balanced_subset" if val_per_class else "all")
    if val_per_class == 0:
        assert {r["subject"] for r in metadata["validation_records"]} == set(split["validation"])
        assert metadata["validation_class_counts"] == dict.fromkeys(ACTIVITIES, 10)
    assert metadata["test_contents_loaded"] is False
    assert set(arrays["train_labels"]) == set(range(5))
    assert not ({r["subject"] for r in metadata["train_records"]} &
                {r["subject"] for r in metadata["validation_records"]})


def test_feature_scaling_uses_only_training_values():
    arrays = {"train_x": np.full((2, 4, 6), 2.0), "val_x": np.full((1, 4, 6), 200.0)}
    train, val, columns, scale = select_and_scale(arrays, list(LEG6), "leg6")
    np.testing.assert_allclose(train, 1.0)
    np.testing.assert_allclose(val, 100.0)
    np.testing.assert_allclose(scale, 2.0)
    assert columns == list(LEG6)
    np.testing.assert_allclose(arrays["train_x"], 2.0)


def test_nvar_width_is_checked_before_model_allocation():
    from experiments.ceti_esn_timing import architecture_cases, preflight

    _, reason = preflight(architecture_cases(102)["nvar_ridge"], 4000, 756)
    assert reason is not None
    _, reason = preflight(architecture_cases(6)["nvar_ridge"], 4000, 756)
    assert reason is None


@pytest.mark.parametrize("case", ["reservoir128_ridge", "ip128_ridge", "reservoir256_lms",
                                  "reservoir256_rls", "nvar_ridge"])
def test_real_node_profiles_train_and_reset_independent_sequences(case):
    import reservoirpy as rpy
    from experiments.ceti_esn_timing import architecture_cases, construct_sequence_model, fit_sequences
    from src.utils import runModel

    rpy.verbosity(0)
    rpy.set_seed(42)
    architecture = architecture_cases(2)[case]
    for node in architecture["nodes"]:
        if "units" in node["params"]:
            node["params"]["units"] = 12
    model = construct_sequence_model(architecture)
    inputs = [np.random.default_rng(seed).normal(size=(10, 2)) for seed in range(3)]
    targets = [np.repeat(np.eye(5)[i][None, :], 10, axis=0) for i in range(3)]
    fit_sequences(model, inputs, targets, warmup=2, ridge_passes=2)
    model.reset()
    expected = runModel(model, inputs[0])
    runModel(model, inputs[1])
    model.reset()
    actual = runModel(model, inputs[0])
    assert actual.shape == (10, 5)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected)


def test_full_holdout_uses_training_scale_without_refitting():
    from experiments.ceti_esn_test import prepare_inputs

    arrays = {"train_x": np.full((2, 4, 6), 2.0), "val_x": np.full((1, 4, 6), 200.0),
              "test_x": np.full((3, 4, 6), 2000.0)}
    train, val, test, columns = prepare_inputs(arrays, {"columns": list(LEG6)}, "leg6")
    np.testing.assert_allclose(train, 1.0)
    np.testing.assert_allclose(val, 100.0)
    np.testing.assert_allclose(test, 1000.0)
    np.testing.assert_allclose(arrays["test_x"], 2000.0)
    assert columns == list(LEG6)


def test_competence_metrics_detect_constant_predictions_on_imbalanced_data():
    from experiments.ceti_esn_test import classification_metrics

    targets = np.array([0] * 80 + [1, 2, 3, 4] * 5)
    metrics = classification_metrics(targets, np.zeros(len(targets), dtype=int))
    assert metrics["accuracy"] == 0.8
    assert metrics["balanced_accuracy"] == 0.2
    assert metrics["macro_f1"] < 0.2


def test_holdout_loading_includes_every_reserved_segment(tmp_path):
    import json
    from experiments.ceti_esn_test import prepare_holdout

    dataset = tmp_path / "dataset"
    root = dataset / "derivatives" / "cut_sequences"
    root.mkdir(parents=True)
    subjects = [f"sub-K{i}" for i in range(24)]
    (root / "participants.tsv").write_text("participant_id\n" + "\n".join(subjects) + "\n")
    for subject in subjects:
        directory = root / subject / "motion"
        directory.mkdir(parents=True)
        for activity in ACTIVITIES:
            np.savetxt(directory / f"{subject}_task-{activity}_run-1_motion.tsv", np.ones((5, 6)),
                       delimiter="\t", header="\t".join(LEG6), comments="")
    arrays, metadata = prepare_subset(dataset, tmp_path / "split.json", train_per_class=2,
                                     val_per_class=1, frames=4)
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    np.savez(pilot / "subset.npz", **arrays)
    (pilot / "subset.json").write_text(json.dumps(metadata))
    loaded, report = prepare_holdout(pilot, dataset)
    assert loaded["test_x"].shape == (50, 4, 6)
    assert len({r["subject"] for r in report["test_records"]}) == 10
    assert set(loaded["test_labels"]) == set(range(5))
    assert report["training_fraction"]["of_training_subjects"] == 0.5
    assert report["training_fraction"]["of_entire_dataset"] == 10 / 120
    np.testing.assert_array_equal(loaded["train_x"], arrays["train_x"])
