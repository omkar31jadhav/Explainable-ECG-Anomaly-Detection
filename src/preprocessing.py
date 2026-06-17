"""Signal preprocessing, beat extraction, and label encoding utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

from .config import (
    AAMI_CLASS_MAP,
    NON_BEAT_SYMBOLS,
    DataConfig,
    get_label_mapping,
)


@dataclass(frozen=True)
class BeatExtractionResult:
    """Windowed heartbeat extraction output."""

    beats: np.ndarray
    labels: list[str]
    symbols: list[str]
    r_peaks: np.ndarray
    skipped_count: int


def handle_missing_values(signal: np.ndarray | Sequence[float]) -> np.ndarray:
    """Replace NaN/inf values with robust per-channel medians."""

    arr = np.asarray(signal, dtype=np.float32).copy()
    if arr.size == 0:
        return arr

    arr[~np.isfinite(arr)] = np.nan
    if not np.isnan(arr).any():
        return arr

    if arr.ndim == 1:
        fill_value = np.nanmedian(arr)
        if not np.isfinite(fill_value):
            fill_value = 0.0
        arr[np.isnan(arr)] = fill_value
        return arr

    flat = arr.reshape(-1, arr.shape[-1])
    medians = np.nanmedian(flat, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    nan_rows, nan_cols = np.where(np.isnan(flat))
    flat[nan_rows, nan_cols] = medians[nan_cols]
    return flat.reshape(arr.shape)


def normalize_signal(
    signal: np.ndarray | Sequence[float],
    method: str = "zscore",
    axis: int | tuple[int, ...] | None = 0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Normalize a signal with z-score, min-max, robust, or no scaling."""

    arr = handle_missing_values(signal)
    method = method.lower().strip()

    if method in {"none", "identity", "raw"}:
        return arr.astype(np.float32)

    if method in {"zscore", "standard", "standardize"}:
        mean = np.mean(arr, axis=axis, keepdims=True)
        std = np.std(arr, axis=axis, keepdims=True)
        return ((arr - mean) / np.maximum(std, eps)).astype(np.float32)

    if method in {"minmax", "min-max"}:
        min_value = np.min(arr, axis=axis, keepdims=True)
        max_value = np.max(arr, axis=axis, keepdims=True)
        return ((arr - min_value) / np.maximum(max_value - min_value, eps)).astype(
            np.float32
        )

    if method == "robust":
        median = np.median(arr, axis=axis, keepdims=True)
        q75 = np.percentile(arr, 75, axis=axis, keepdims=True)
        q25 = np.percentile(arr, 25, axis=axis, keepdims=True)
        return ((arr - median) / np.maximum(q75 - q25, eps)).astype(np.float32)

    raise ValueError(
        f"Unsupported normalization method '{method}'. "
        "Use 'zscore', 'minmax', 'robust', or 'none'."
    )


def standardize_beats(beats: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Apply per-beat z-score standardization along the time axis."""

    arr = handle_missing_values(beats)
    if arr.size == 0:
        return arr.astype(np.float32)
    if arr.ndim == 1:
        mean = np.mean(arr, keepdims=True)
        std = np.std(arr, keepdims=True)
        return ((arr - mean) / np.maximum(std, eps)).astype(np.float32)
    mean = np.mean(arr, axis=1, keepdims=True)
    std = np.std(arr, axis=1, keepdims=True)
    return ((arr - mean) / np.maximum(std, eps)).astype(np.float32)


def select_channel(signal: np.ndarray, channel: int = 0) -> np.ndarray:
    """Select one ECG channel from a WFDB p_signal array."""

    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D ECG signal, got shape {arr.shape}.")
    if channel < 0 or channel >= arr.shape[1]:
        raise ValueError(
            f"Signal channel {channel} is out of range for signal shape {arr.shape}."
        )
    return arr[:, channel]


def map_symbol_to_label(
    symbol: str,
    label_scheme: str = "aami",
    include_unknown: bool = True,
) -> str | None:
    """Map a WFDB annotation symbol to a model class label."""

    mapping = get_label_mapping(label_scheme)
    if symbol in mapping:
        label = mapping[symbol]
        if label == "Unknown" and not include_unknown:
            return None
        return label
    if symbol in NON_BEAT_SYMBOLS:
        return None
    return "Unknown" if include_unknown and label_scheme.lower() == "aami" else None


def extract_heartbeat_windows(
    signal: np.ndarray | Sequence[float],
    r_peaks: Sequence[int] | np.ndarray,
    labels: Sequence[str] | None = None,
    symbols: Sequence[str] | None = None,
    samples_before: int = 100,
    samples_after: int = 100,
    channel: int = 0,
) -> BeatExtractionResult:
    """Extract fixed-length heartbeat windows centered on annotation samples."""

    if samples_before < 0 or samples_after <= 0:
        raise ValueError("samples_before must be >= 0 and samples_after must be > 0.")

    channel_signal = select_channel(np.asarray(signal, dtype=np.float32), channel=channel)
    window_size = samples_before + samples_after
    peaks = np.asarray(r_peaks, dtype=np.int64)
    labels = list(labels) if labels is not None else [None] * len(peaks)
    symbols = list(symbols) if symbols is not None else [None] * len(peaks)

    if len(labels) != len(peaks) or len(symbols) != len(peaks):
        raise ValueError("r_peaks, labels, and symbols must have the same length.")

    beats: list[np.ndarray] = []
    kept_labels: list[str] = []
    kept_symbols: list[str] = []
    kept_peaks: list[int] = []
    skipped = 0

    for peak, label, symbol in zip(peaks, labels, symbols):
        start = int(peak) - samples_before
        end = int(peak) + samples_after
        if start < 0 or end > len(channel_signal):
            skipped += 1
            continue
        beat = channel_signal[start:end]
        if beat.shape[0] != window_size:
            skipped += 1
            continue
        beats.append(beat.astype(np.float32))
        if label is not None:
            kept_labels.append(str(label))
        if symbol is not None:
            kept_symbols.append(str(symbol))
        kept_peaks.append(int(peak))

    if beats:
        beat_array = np.vstack([beat[np.newaxis, :] for beat in beats]).astype(np.float32)
    else:
        beat_array = np.empty((0, window_size), dtype=np.float32)

    return BeatExtractionResult(
        beats=beat_array,
        labels=kept_labels,
        symbols=kept_symbols,
        r_peaks=np.asarray(kept_peaks, dtype=np.int64),
        skipped_count=skipped,
    )


def prepare_beat_features(
    beats: np.ndarray,
    standardize: bool = True,
    add_channel_axis: bool = True,
) -> np.ndarray:
    """Prepare extracted beats for Conv1D input."""

    arr = np.asarray(beats, dtype=np.float32)
    if standardize:
        arr = standardize_beats(arr)
    else:
        arr = handle_missing_values(arr)
    if add_channel_axis and arr.ndim == 2:
        arr = arr[..., np.newaxis]
    return arr.astype(np.float32)


def encode_labels(
    labels: Sequence[str],
    encoder: LabelEncoder | None = None,
    save_path: str | Path | None = None,
) -> tuple[np.ndarray, LabelEncoder]:
    """Fit or apply a sklearn LabelEncoder and optionally persist it."""

    if not labels:
        raise ValueError("Cannot encode an empty label sequence.")
    if encoder is None:
        encoder = LabelEncoder()
        y = encoder.fit_transform(list(labels))
    else:
        y = encoder.transform(list(labels))
    if save_path is not None:
        save_label_encoder(encoder, save_path)
    return np.asarray(y, dtype=np.int64), encoder


def save_label_encoder(encoder: LabelEncoder, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, path)


def load_label_encoder(path: str | Path) -> LabelEncoder:
    return joblib.load(Path(path))


def preprocess_signal_for_dataset(signal: np.ndarray, config: DataConfig) -> np.ndarray:
    """Clean and normalize a full ECG record before beat extraction."""

    arr = handle_missing_values(signal)
    return normalize_signal(arr, method=config.normalization, axis=0)

