"""Shared utility functions for training and evaluation artifacts."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from .config import MODELS_DIR, RESULTS_DIR


def ensure_project_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # type: ignore

        tf.random.set_seed(seed)
    except ImportError:
        pass


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(data), indent=2), encoding="utf-8")


def save_text(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_history(history: Any, results_dir: str | Path = RESULTS_DIR) -> dict:
    """Save Keras training history to JSON and CSV."""

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    history_dict = getattr(history, "history", history)
    save_json(history_dict, results_dir / "training_history.json")

    keys = list(history_dict.keys())
    row_count = max((len(history_dict[key]) for key in keys), default=0)
    with (results_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", *keys])
        for index in range(row_count):
            writer.writerow(
                [
                    index + 1,
                    *[
                        history_dict[key][index] if index < len(history_dict[key]) else ""
                        for key in keys
                    ],
                ]
            )
    return history_dict


def plot_training_history(history: Any, results_dir: str | Path = RESULTS_DIR) -> None:
    """Save loss and accuracy curves."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    history_dict = getattr(history, "history", history)

    if "loss" in history_dict:
        plt.figure(figsize=(8, 5))
        plt.plot(history_dict["loss"], label="train_loss")
        if "val_loss" in history_dict:
            plt.plot(history_dict["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / "loss_curve.png", dpi=150)
        plt.close()

    if "accuracy" in history_dict:
        plt.figure(figsize=(8, 5))
        plt.plot(history_dict["accuracy"], label="train_accuracy")
        if "val_accuracy" in history_dict:
            plt.plot(history_dict["val_accuracy"], label="val_accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / "accuracy_curve.png", dpi=150)
        plt.close()


def save_confusion_matrix_csv(
    matrix: np.ndarray,
    classes: list[str] | np.ndarray,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    classes = [str(value) for value in classes]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label", *classes])
        for label, row in zip(classes, matrix):
            writer.writerow([label, *[int(value) for value in row]])


def plot_confusion_matrix(
    matrix: np.ndarray,
    classes: list[str] | np.ndarray,
    path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    classes = [str(value) for value in classes]

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2.0 if matrix.size and matrix.max() > 0 else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                int(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value
