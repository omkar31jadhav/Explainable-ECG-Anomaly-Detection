"""Dataset construction and stratified splitting for MIT-BIH heartbeats."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import (
    DEFAULT_DATASET_CACHE_PATH,
    DEFAULT_LABEL_ENCODER_PATH,
    DataConfig,
    SplitConfig,
)
from .data_loader import MITBIHDataLoader, RecordInfo
from .preprocessing import (
    encode_labels,
    extract_heartbeat_windows,
    map_symbol_to_label,
    prepare_beat_features,
    preprocess_signal_for_dataset,
)


@dataclass(frozen=True)
class BeatDataset:
    """In-memory heartbeat dataset."""

    X: np.ndarray
    y: np.ndarray
    labels: np.ndarray
    symbols: np.ndarray
    record_names: np.ndarray
    r_peaks: np.ndarray
    label_encoder: LabelEncoder
    skipped_beats: int

    @property
    def classes(self) -> np.ndarray:
        return self.label_encoder.classes_

    def metadata(self) -> dict[str, np.ndarray]:
        return {
            "labels": self.labels,
            "symbols": self.symbols,
            "record_names": self.record_names,
            "r_peaks": self.r_peaks,
        }


@dataclass(frozen=True)
class DatasetSplit:
    """Train/validation/test arrays plus aligned metadata."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train_metadata: dict[str, np.ndarray]
    val_metadata: dict[str, np.ndarray]
    test_metadata: dict[str, np.ndarray]


def build_beat_dataset(
    data_config: DataConfig = DataConfig(),
    loader: MITBIHDataLoader | None = None,
    records: Sequence[str | RecordInfo] | None = None,
    max_records: int | None = None,
    label_encoder: LabelEncoder | None = None,
    encoder_path: str | Path | None = DEFAULT_LABEL_ENCODER_PATH,
) -> BeatDataset:
    """Load records, extract heartbeat windows, preprocess, and encode labels."""

    loader = loader or MITBIHDataLoader.from_config(data_config)
    discovered = list(records) if records is not None else loader.discover_records()
    if max_records is not None:
        discovered = discovered[:max_records]
    if not discovered:
        raise FileNotFoundError(
            f"No complete MIT-BIH records found in {Path(data_config.data_dir).resolve()}."
        )

    beat_arrays: list[np.ndarray] = []
    label_values: list[str] = []
    symbol_values: list[str] = []
    record_values: list[str] = []
    r_peak_values: list[int] = []
    skipped_total = 0

    for record_info in discovered:
        record = loader.load_record(record_info)
        annotation = loader.load_annotation(record_info)
        signal = preprocess_signal_for_dataset(record.signal, data_config)

        valid_peaks: list[int] = []
        valid_symbols: list[str] = []
        valid_labels: list[str] = []
        for sample, symbol in zip(annotation.samples, annotation.symbols):
            label = map_symbol_to_label(
                symbol,
                label_scheme=data_config.label_scheme,
                include_unknown=data_config.include_unknown,
            )
            if label is None:
                continue
            valid_peaks.append(int(sample))
            valid_symbols.append(symbol)
            valid_labels.append(label)

        extraction = extract_heartbeat_windows(
            signal=signal,
            r_peaks=valid_peaks,
            labels=valid_labels,
            symbols=valid_symbols,
            samples_before=data_config.samples_before,
            samples_after=data_config.samples_after,
            channel=data_config.signal_channel,
        )
        skipped_total += extraction.skipped_count
        if extraction.beats.size == 0:
            continue

        beat_arrays.append(extraction.beats)
        label_values.extend(extraction.labels)
        symbol_values.extend(extraction.symbols)
        record_values.extend([record.name] * len(extraction.labels))
        r_peak_values.extend(extraction.r_peaks.tolist())

    if not beat_arrays:
        raise ValueError("No heartbeat windows were extracted from the selected records.")

    raw_beats = np.vstack(beat_arrays)
    X = prepare_beat_features(
        raw_beats,
        standardize=data_config.standardize_beats,
        add_channel_axis=True,
    )
    y, fitted_encoder = encode_labels(
        label_values,
        encoder=label_encoder,
        save_path=encoder_path,
    )

    return BeatDataset(
        X=X,
        y=y,
        labels=np.asarray(label_values),
        symbols=np.asarray(symbol_values),
        record_names=np.asarray(record_values),
        r_peaks=np.asarray(r_peak_values, dtype=np.int64),
        label_encoder=fitted_encoder,
        skipped_beats=skipped_total,
    )


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    metadata: dict[str, np.ndarray] | None = None,
    split_config: SplitConfig = SplitConfig(),
) -> DatasetSplit:
    """Create stratified train/validation/test splits when class counts allow."""

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of examples.")
    if split_config.validation_size < 0 or split_config.test_size < 0:
        raise ValueError("validation_size and test_size must be non-negative.")
    holdout_size = split_config.validation_size + split_config.test_size
    if not 0 < holdout_size < 1:
        raise ValueError("validation_size + test_size must be between 0 and 1.")

    metadata = metadata or {}
    indices = np.arange(len(y))
    stratify = _stratify_labels(y, holdout_size)
    train_idx, holdout_idx = train_test_split(
        indices,
        test_size=holdout_size,
        random_state=split_config.random_state,
        stratify=stratify,
    )

    relative_test_size = split_config.test_size / holdout_size
    holdout_y = y[holdout_idx]
    holdout_stratify = _stratify_labels(holdout_y, relative_test_size)
    val_idx_rel, test_idx_rel = train_test_split(
        np.arange(len(holdout_idx)),
        test_size=relative_test_size,
        random_state=split_config.random_state,
        stratify=holdout_stratify,
    )
    val_idx = holdout_idx[val_idx_rel]
    test_idx = holdout_idx[test_idx_rel]

    return DatasetSplit(
        X_train=X[train_idx],
        X_val=X[val_idx],
        X_test=X[test_idx],
        y_train=y[train_idx],
        y_val=y[val_idx],
        y_test=y[test_idx],
        train_metadata=_slice_metadata(metadata, train_idx),
        val_metadata=_slice_metadata(metadata, val_idx),
        test_metadata=_slice_metadata(metadata, test_idx),
    )


def save_dataset(dataset: BeatDataset, path: str | Path = DEFAULT_DATASET_CACHE_PATH) -> None:
    """Persist a compressed beat dataset cache."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=dataset.X,
        y=dataset.y,
        labels=dataset.labels,
        symbols=dataset.symbols,
        record_names=dataset.record_names,
        r_peaks=dataset.r_peaks,
        classes=dataset.classes,
        skipped_beats=np.asarray([dataset.skipped_beats], dtype=np.int64),
    )


def dataset_summary(dataset: BeatDataset) -> dict:
    """Return high-level dataset statistics."""

    class_counts = {
        str(label): int(count)
        for label, count in zip(*np.unique(dataset.labels, return_counts=True))
    }
    symbol_counts = {
        str(symbol): int(count)
        for symbol, count in zip(*np.unique(dataset.symbols, return_counts=True))
    }
    record_counts = {
        str(record): int(count)
        for record, count in zip(*np.unique(dataset.record_names, return_counts=True))
    }
    return {
        "num_beats": int(len(dataset.y)),
        "window_shape": list(dataset.X.shape[1:]),
        "num_classes": int(len(dataset.classes)),
        "classes": [str(value) for value in dataset.classes],
        "class_distribution": class_counts,
        "symbol_distribution": symbol_counts,
        "record_distribution": record_counts,
        "skipped_boundary_beats": int(dataset.skipped_beats),
    }


def _slice_metadata(metadata: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.asarray(values)[indices] for name, values in metadata.items()}


def _stratify_labels(y: np.ndarray, test_size: float | int) -> np.ndarray | None:
    labels = np.asarray(y)
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) <= 1 or counts.min() < 2:
        return None

    n_samples = len(labels)
    n_test = ceil(n_samples * test_size) if isinstance(test_size, float) else int(test_size)
    n_train = n_samples - n_test
    if n_test < len(classes) or n_train < len(classes):
        return None
    return labels

