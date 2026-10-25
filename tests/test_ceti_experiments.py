import importlib.util
import pickle
import re

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from ceti import DEFAULT_DATA_DIR
from ceti.data import resample
from ceti.models import construct_sequence_model, fit_sequences, predict_segments, SegmentMetric
from ceti.svm import fit_paper_svm


@pytest.fixture
def supplied_svm():
    path = DEFAULT_DATA_DIR / "code/verification_experiments.py"
    if not path.exists():
        pytest.skip("External dataset/code is intentionally gitignored.")
    spec = importlib.util.spec_from_file_location("supplied_ceti_verification", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("frames", [20, 100])
def test_interpolation_matches_supplied_svm(supplied_svm, frames):
    values = np.random.default_rng(10).normal(size=(67, 102)).astype(np.float32)
    np.testing.assert_allclose(resample(values, frames).reshape(-1),
                               supplied_svm.flatten(values, frames), rtol=1e-11, atol=1e-12)


def test_svm_scores_match_supplied_classification_code(supplied_svm, capsys):
    rng = np.random.default_rng(1)
    labels = np.repeat(np.arange(5), 15)
    inputs = rng.normal(size=(75, 8))
    inputs[:, 0] += labels * 0.5
    names = np.array([f"activity{i}" for i in labels])
    np.random.seed(42)
    supplied_svm.classify(inputs.tolist(), names.tolist())
    printed = capsys.readouterr().out
    reference_accuracy = float(re.search(r"Classification accuracy: ([0-9.e+-]+)", printed)[1])
    reference_f1 = float(re.search(r"Classification f1 score: ([0-9.e+-]+)", printed)[1])
    reference_validation = float(re.search(r"Validation score: ([0-9.e+-]+)", printed)[1])
    np.random.seed(42)
    train_x, test_x, train_y, test_y = train_test_split(inputs, labels, test_size=0.2, stratify=labels)
    model, report = fit_paper_svm(train_x, train_y, seed=42)
    predicted = model.predict(test_x)
    assert accuracy_score(test_y, predicted) == reference_accuracy
    assert f1_score(test_y, predicted, average="macro") == reference_f1
    assert report["selected_validation_score"] == reference_validation


@pytest.mark.parametrize("kind", ["Reservoir", "IPReservoir", "NVAR"])
def test_sequence_model_survives_standard_pickle_for_search_workers(kind):
    import reservoirpy as rpy

    rpy.verbosity(0)
    rpy.set_seed(0)
    params = {"units": 12, "sr": 0.8} if kind != "NVAR" else {"delay": 2, "order": 2}
    architecture = {"nodes": [{"type": "Input", "params": {"input_dim": 2}},
                               {"type": kind, "params": params},
                               {"type": "Ridge", "params": {"output_dim": 5, "ridge": 1e-4}}],
                    "edges": [(0, 1), (1, 2)]}
    model = construct_sequence_model(architecture)
    inputs = np.random.default_rng(0).normal(size=(5, 10, 2))
    targets = [np.repeat(np.eye(5)[i][None, :], 10, axis=0) for i in range(5)]
    fit_sequences(model, list(inputs), targets, warmup=2)
    expected_labels, expected_scores = predict_segments(model, inputs, warmup=2)
    restored = pickle.loads(pickle.dumps(model))
    labels, scores = predict_segments(restored, inputs, warmup=2)
    np.testing.assert_array_equal(labels, expected_labels)
    np.testing.assert_allclose(scores, expected_scores)


def test_search_metrics_pool_segments_and_discard_warmup():
    targets = np.repeat(np.eye(5)[:, None, :], 4, axis=1)
    scores = targets.copy()
    scores[:, :2] = np.roll(targets[:, :2], 1, axis=-1)
    assert SegmentMetric(frames=4, warmup=2)(targets.reshape(-1, 5), scores) == 0.0
    assert SegmentMetric(frames=4, warmup=2, metric="accuracy")(targets, scores) == 1.0


def test_initial_node_count_range_survives_invalid_architecture_retry(monkeypatch):
    from src import utils

    attempts = []

    def valid_on_retry(architecture, inputs, memory_limit):
        attempts.append(architecture)
        return len(attempts) > 1

    monkeypatch.setattr(utils, "isValidArchitecture", valid_on_retry)
    architecture = utils.generateRandomArchitectureOld(5, np.zeros((20, 6)), num_nodes_range=(1, 1))
    assert len(attempts) == 2
    assert len(architecture["nodes"]) == 3  # Input, one intermediate node, readout.
