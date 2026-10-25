"""The supplied paper SVM and fold-selection procedure, unchanged."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC


def paper_pipeline():
    return Pipeline([("scaler", MaxAbsScaler()), ("pca", PCA()), ("clf", SVC(probability=True))])


def fit_paper_svm(inputs, labels, seed=1234123123, n_jobs=1, folds=10):
    """Select the maximum-scoring fold's fitted estimator; do not refit it."""
    if min(np.bincount(labels, minlength=5)) < folds:
        raise ValueError("Every activity needs at least one segment per CV fold.")
    np.random.seed(seed)
    cv = cross_validate(paper_pipeline(), inputs, labels,
                        cv=StratifiedKFold(n_splits=folds), scoring="balanced_accuracy",
                        return_estimator=True, n_jobs=n_jobs, error_score="raise")
    selected = int(cv["test_score"].argmax(axis=0))
    report = {
        "cv_balanced_accuracy": cv["test_score"].tolist(),
        "cv_fit_seconds": cv["fit_time"].tolist(),
        "selected_fold": selected, "selected_validation_score": float(cv["test_score"][selected]),
        "selection": "Best inner-fold balanced accuracy, no final refit (same as supplied code).",
    }
    return cv["estimator"][selected], report


def classification_metrics(targets, predicted):
    return {"accuracy": float(accuracy_score(targets, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(targets, predicted)),
            "macro_f1": float(f1_score(targets, predicted, labels=list(range(5)),
                                       average="macro", zero_division=0))}
