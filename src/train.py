"""Train the MIT-BIH 1D CNN heartbeat classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from .config import (
    DEFAULT_LABEL_ENCODER_PATH,
    DEFAULT_MODEL_PATH,
    RESULTS_DIR,
    DataConfig,
    ModelConfig,
    SplitConfig,
    TrainingConfig,
)
from .dataset import build_beat_dataset, dataset_summary, save_dataset, split_dataset
from .model import build_cnn_model
from .utils import ensure_project_dirs, plot_training_history, save_history, save_json, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ECG CNN on MIT-BIH beats.")
    parser.add_argument("--data-dir", type=Path, default=DataConfig.data_dir)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--encoder-path", type=Path, default=DEFAULT_LABEL_ENCODER_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--samples-before", type=int, default=100)
    parser.add_argument("--samples-after", type=int, default=100)
    parser.add_argument("--label-scheme", choices=["aami", "symbol"], default="aami")
    parser.add_argument("--exclude-unknown", action="store_true")
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--validation-size", type=float, default=SplitConfig.validation_size)
    parser.add_argument("--test-size", type=float, default=SplitConfig.test_size)
    parser.add_argument("--learning-rate", type=float, default=ModelConfig.learning_rate)
    parser.add_argument("--dropout-rate", type=float, default=ModelConfig.dropout_rate)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--cache-dataset", action="store_true")
    return parser.parse_args()


def train_pipeline(args: argparse.Namespace | None = None) -> dict:
    args = args or parse_args()
    ensure_project_dirs()

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
    model_config = ModelConfig(
        input_length=data_config.window_size,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
    )
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_class_weights=not args.no_class_weights,
    )

    set_global_seed(training_config.random_state)

    dataset = build_beat_dataset(
        data_config=data_config,
        max_records=args.max_records,
        encoder_path=args.encoder_path,
    )
    summary = dataset_summary(dataset)
    save_json(summary, Path(args.results_dir) / "dataset_summary.json")
    if args.cache_dataset:
        save_dataset(dataset, Path(args.results_dir) / "mitbih_beat_dataset.npz")

    splits = split_dataset(
        dataset.X,
        dataset.y,
        metadata=dataset.metadata(),
        split_config=split_config,
    )

    model = build_cnn_model(
        input_shape=dataset.X.shape[1:],
        num_classes=len(dataset.classes),
        model_config=model_config,
    )
    callbacks = _build_callbacks(args.model_path, training_config)
    class_weights = _compute_class_weights(splits.y_train) if training_config.use_class_weights else None

    history = model.fit(
        splits.X_train,
        splits.y_train,
        validation_data=(splits.X_val, splits.y_val),
        epochs=training_config.epochs,
        batch_size=training_config.batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_path)
    history_dict = save_history(history, args.results_dir)
    plot_training_history(history_dict, args.results_dir)

    training_metadata = {
        "model_path": str(args.model_path),
        "encoder_path": str(args.encoder_path),
        "classes": [str(value) for value in dataset.classes],
        "train_examples": int(len(splits.y_train)),
        "validation_examples": int(len(splits.y_val)),
        "test_examples": int(len(splits.y_test)),
        "data_config": data_config.__dict__,
        "split_config": split_config.__dict__,
        "model_config": model_config.__dict__,
        "training_config": training_config.__dict__,
    }
    save_json(training_metadata, Path(args.results_dir) / "training_metadata.json")
    return {
        "model": model,
        "history": history_dict,
        "dataset": dataset,
        "splits": splits,
        "metadata": training_metadata,
    }


def _build_callbacks(model_path: Path, training_config: TrainingConfig) -> list:
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    model_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=training_config.early_stopping_patience,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=training_config.reduce_lr_factor,
            patience=training_config.reduce_lr_patience,
            min_lr=training_config.min_learning_rate,
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]


def _compute_class_weights(y_train: np.ndarray) -> dict[int, float]:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(class_id): float(weight) for class_id, weight in zip(classes, weights)}


if __name__ == "__main__":
    train_pipeline()

