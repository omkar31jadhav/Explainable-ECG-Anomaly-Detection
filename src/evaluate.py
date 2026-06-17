"""Evaluate a trained ECG CNN and save classification artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .config import (
    DEFAULT_LABEL_ENCODER_PATH,
    DEFAULT_MODEL_PATH,
    RESULTS_DIR,
    DataConfig,
    SplitConfig,
)
from .dataset import build_beat_dataset, split_dataset
from .model import load_trained_model
from .preprocessing import load_label_encoder
from .utils import (
    plot_confusion_matrix,
    save_confusion_matrix_csv,
    save_json,
    save_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained ECG CNN.")
    parser.add_argument("--data-dir", type=Path, default=DataConfig.data_dir)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--encoder-path", type=Path, default=DEFAULT_LABEL_ENCODER_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--samples-before", type=int, default=100)
    parser.add_argument("--samples-after", type=int, default=100)
    parser.add_argument("--label-scheme", choices=["aami", "symbol"], default="aami")
    parser.add_argument("--exclude-unknown", action="store_true")
    parser.add_argument("--validation-size", type=float, default=SplitConfig.validation_size)
    parser.add_argument("--test-size", type=float, default=SplitConfig.test_size)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def evaluate_pipeline(args: argparse.Namespace | None = None) -> dict:
    args = args or parse_args()
    label_encoder = load_label_encoder(args.encoder_path)
    model = load_trained_model(args.model_path)

    data_config = DataConfig(
        data_dir=args.data_dir,
        signal_channel=args.channel,
        samples_before=args.samples_before,
        samples_after=args.samples_after,
        label_scheme=args.label_scheme,
        include_unknown=not args.exclude_unknown,
    )
    split_config = SplitConfig(
        validation_size=args.validation_size,
        test_size=args.test_size,
    )

    dataset = build_beat_dataset(
        data_config=data_config,
        max_records=args.max_records,
        label_encoder=label_encoder,
        encoder_path=None,
    )
    splits = split_dataset(
        dataset.X,
        dataset.y,
        metadata=dataset.metadata(),
        split_config=split_config,
    )

    probabilities = model.predict(splits.X_test, batch_size=args.batch_size, verbose=1)
    return evaluate_predictions(
        y_true=splits.y_test,
        probabilities=probabilities,
        classes=label_encoder.classes_,
        results_dir=args.results_dir,
    )


def evaluate_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    results_dir: str | Path = RESULTS_DIR,
) -> dict:
    """Compute and save test metrics from predicted probabilities."""

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    y_pred = np.asarray(probabilities).argmax(axis=1)
    label_ids = np.arange(len(classes))
    accuracy = accuracy_score(y_true, y_pred)
    macro = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_ids,
        average="macro",
        zero_division=0,
    )
    weighted = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_ids,
        average="weighted",
        zero_division=0,
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=[str(value) for value in classes],
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=label_ids)

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "classes": [str(value) for value in classes],
    }
    save_json(metrics, results_dir / "evaluation_metrics.json")
    save_text(report, results_dir / "classification_report.txt")
    save_confusion_matrix_csv(matrix, classes, results_dir / "confusion_matrix.csv")
    plot_confusion_matrix(matrix, classes, results_dir / "confusion_matrix.png")

    return {
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": matrix,
        "y_pred": y_pred,
    }


if __name__ == "__main__":
    evaluate_pipeline()

