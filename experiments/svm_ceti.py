"""Paper SVM for CeTI activities, with configurable frames per segment."""

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ceti import DEFAULT_DATA_DIR, DEFAULT_SPLIT
from ceti.data import load_svm_inputs, load_svm_test
from ceti.svm import classification_metrics, fit_paper_svm, paper_pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", "--sampling-rate", type=int, default=100,
                        help="Interpolated frames per complete motion, not Hz (paper: 100).")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=1234123123)
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel CV folds; default serial like supplied code.")
    parser.add_argument("--training-subset", choices=("full", "small"), default="full")
    parser.add_argument("--train-per-class", type=int, default=40, help="Only used with --training-subset small.")
    parser.add_argument("--cv-folds", type=int, default=10)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Two folds, four training segments/class; verifies wiring only.")
    args = parser.parse_args(argv)
    if args.smoke_test:
        args.training_subset, args.train_per_class, args.cv_folds = "small", 4, 2
    if args.frames < 2 or args.cv_folds < 2 or args.train_per_class < 1 or args.n_jobs < 1:
        parser.error("Need frames/folds >= 2 and positive subset size/worker count.")
    output = args.output_dir or ROOT / "results/svm_ceti" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(pipeline=str(paper_pipeline()), python=sys.version,
                  protocol="Supplied SVM pipeline and best-fold selection; explicit shared subject split.")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    started = time.perf_counter()
    train_x, train_y, metadata = load_svm_inputs(args.data_dir, args.split_manifest, args.frames,
                                               args.seed, args.training_subset, args.train_per_class)
    (output / "data.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"SVM inputs: {train_x.shape}; {args.frames} frames/segment; "
          f"{len(metadata['test_records'])} held-out test segments.", flush=True)
    if args.prepare_only:
        print(f"Prepared data manifest: {output}")
        return 0
    model, report = fit_paper_svm(train_x, train_y, args.seed, args.n_jobs, args.cv_folds)
    report["fit_and_cv_seconds"] = time.perf_counter() - started
    # Load the complete shared holdout only after CV has selected the estimator.
    test_x, test_y = load_svm_test(args.data_dir, metadata)
    start = time.perf_counter()
    predicted = model.predict(test_x)
    report.update(test_seconds=time.perf_counter() - start,
                  test_metrics=classification_metrics(test_y, predicted),
                  test_segments=len(test_y), train_segments=len(train_y),
                  pca_components=int(model["pca"].n_components_),
                  qualification="Shared explicit holdout, not the recovered paper test IDs. "
                    "20-frame runs change only temporal representation; full mode keeps the paper's 40-subject pool.")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject", "segment", "target", "prediction"])
        for record, target, prediction in zip(metadata["test_records"], test_y, predicted):
            writer.writerow([record["subject"], record["path"], metadata["class_names"][int(target)],
                             metadata["class_names"][int(prediction)]])
    if args.save_model:
        import joblib
        joblib.dump(model, output / "best_fold.joblib")
    print(f"Selected fold {report['selected_fold'] + 1}: validation balanced accuracy "
          f"{report['selected_validation_score']:.4f}; test {report['test_metrics']}")
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
