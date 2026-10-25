"""Main-branch ESN adapters for independent, short motion recordings."""

import numpy as np
from reservoirpy.node import Node
from reservoirpy.nodes import IPReservoir, NVAR


class SequenceNVAR(NVAR):
    """Reset delayed inputs as well as the public output state."""

    def reset(self, to_state=None):
        Node.reset(self, to_state=to_state)
        if (to_state is None or not np.any(to_state)) and self.params.get("store") is not None:
            self.set_param("store", np.zeros_like(self.store))
        return self


class SequenceIPReservoir(IPReservoir):
    """Reset internal pre-activation history between recordings."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._backward = fit_ip_independent_sequences

    def reset(self, to_state=None):
        Node.reset(self, to_state=to_state)
        if (to_state is None or not np.any(to_state)) and self.params.get("internal_state") is not None:
            self.set_param("internal_state", np.zeros_like(self.internal_state))
        return self


def fit_ip_independent_sequences(node, X=None, *args, **kwargs):
    from reservoirpy.nodes.reservoirs.intrinsic_plasticity import ip

    for _ in range(node.epochs):
        for sequence in X:
            node.reset()
            for value in sequence:
                post_state = node.call(value.reshape(1, -1))
                a, b = ip(node, node.internal_state, post_state)
                node.set_param("a", a)
                node.set_param("b", b)


def construct_sequence_model(architecture):
    from src.utils import nodeConstructors

    constructors = dict(nodeConstructors, IPReservoir=SequenceIPReservoir, NVAR=SequenceNVAR)
    nodes = [constructors[node["type"]](**node["params"]) for node in architecture["nodes"]]
    first, second = architecture["edges"][0]
    model = nodes[first] >> nodes[second]
    for first, second in architecture["edges"][1:]:
        model = model & (nodes[first] >> nodes[second])
    return model


def fit_sequences(model, train_x, train_y, warmup, ridge_passes=2):
    """Preserve main's repeated Ridge fit policy with segment-specific warmup."""
    has_offline = any(node.is_trained_offline for node in model.nodes)
    has_online = any(node.is_trained_online for node in model.nodes)
    if has_offline:
        # Fit once before online learning; a final fit updates any offline
        # readout after an intermediate online node has learned.
        model.fit(train_x, train_y, warmup=warmup, reset=True)
    if has_online:
        for sequence, targets in zip(train_x, train_y):
            model.train(sequence, targets, reset=True)
    terminal_offline = any(node.is_trained_offline for node in model.output_nodes)
    if has_offline and terminal_offline:
        for _ in range(ridge_passes - 1):
            model.fit(train_x, train_y, warmup=warmup, reset=True)
    return model


def sequence_scores(model, inputs):
    from src.utils import runModel

    predictions = []
    for sequence in inputs:
        model.reset()
        scores = np.asarray(runModel(model, sequence))
        if scores.shape != (len(sequence), 5) or not np.isfinite(scores).all():
            raise ValueError("Invalid or non-finite class scores.")
        predictions.append(scores)
    return np.stack(predictions)


def predict_segments(model, inputs, warmup=3):
    scores = sequence_scores(model, inputs)[:, warmup:].mean(axis=1)
    return scores.argmax(axis=1), scores


class SegmentMetric:
    """Picklable metric for ESNAS: one decision per complete motion segment."""

    def __init__(self, frames=20, warmup=3, metric="f1_error"):
        self.frames, self.warmup, self.metric = frames, warmup, metric

    def __call__(self, targets, scores):
        from sklearn.metrics import accuracy_score, f1_score

        targets = np.asarray(targets).reshape(-1, self.frames, 5)[:, 0].argmax(axis=1)
        scores = np.asarray(scores).reshape(-1, self.frames, 5)
        predicted = scores[:, self.warmup:].mean(axis=1).argmax(axis=1)
        if self.metric == "f1_error":
            return 1.0 - float(f1_score(targets, predicted, labels=list(range(5)),
                                        average="macro", zero_division=0))
        if self.metric == "accuracy":
            return float(accuracy_score(targets, predicted))
        raise ValueError(f"Unknown metric: {self.metric}")
