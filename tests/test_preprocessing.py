from __future__ import annotations

import numpy as np

from src.preprocessing import (
    encode_labels,
    extract_heartbeat_windows,
    handle_missing_values,
    map_symbol_to_label,
    normalize_signal,
    prepare_beat_features,
)


def test_extract_heartbeat_windows_skips_boundaries():
    signal = np.arange(500, dtype=np.float32)
    result = extract_heartbeat_windows(
        signal=signal,
        r_peaks=[20, 150, 480],
        labels=["Normal", "PVC", "Normal"],
        symbols=["N", "V", "N"],
        samples_before=50,
        samples_after=50,
    )

    assert result.beats.shape == (1, 100)
    assert result.labels == ["PVC"]
    assert result.symbols == ["V"]
    assert result.r_peaks.tolist() == [150]
    assert result.skipped_count == 2


def test_missing_values_and_normalization_are_finite():
    signal = np.array([1.0, np.nan, np.inf, 4.0], dtype=np.float32)

    filled = handle_missing_values(signal)
    normalized = normalize_signal(signal, method="zscore", axis=None)

    assert np.isfinite(filled).all()
    assert np.isfinite(normalized).all()
    assert abs(float(normalized.mean())) < 1e-6


def test_prepare_beat_features_adds_channel_axis():
    beats = np.array([[1, 2, 3, 4], [4, 5, 6, 7]], dtype=np.float32)

    features = prepare_beat_features(beats)

    assert features.shape == (2, 4, 1)
    assert np.allclose(features.mean(axis=1), 0.0, atol=1e-6)


def test_label_mapping_and_encoding():
    labels = [
        map_symbol_to_label("N"),
        map_symbol_to_label("V"),
        map_symbol_to_label("+"),
    ]
    labels = [label for label in labels if label is not None]

    y, encoder = encode_labels(labels)

    assert labels == ["Normal", "PVC"]
    assert y.shape == (2,)
    assert set(encoder.classes_) == {"Normal", "PVC"}

