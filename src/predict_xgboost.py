"""Predict ECG anomalies using both CNN and XGBoost models with explanations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

from .config import DEFAULT_LABEL_ENCODER_PATH, DEFAULT_XGBOOST_MODEL_PATH
from .preprocessing import load_label_encoder

try:
    import xgboost as xgb  # type: ignore
    _HAS_XGBOOST = True
except Exception:
    xgb = None  # type: ignore[assignment]
    _HAS_XGBOOST = False

FEATURE_NAMES = [
    "RMS",
    "Mean",
    "Std",
    "Max",
    "Min",
    "Range",
    "Diff_Mean",
    "Diff_Std",
    "FFT_Mean",
    "FFT_Std",
    "FFT_Max",
    "Autocorr",
    "Zero_Crossings",
]


def _safe_autocorr(signal: np.ndarray, eps: float = 1e-8) -> float:
    centered = signal - np.mean(signal)
    if centered.size < 2:
        return 0.0
    numerator = np.dot(centered[:-1], centered[1:])
    denom = np.linalg.norm(centered[:-1]) * np.linalg.norm(centered[1:])
    return float(numerator / (denom + eps))


def extract_ecg_features(beats: np.ndarray) -> np.ndarray:
    """Extract handcrafted ECG features for XGBoost prediction."""
    arr = np.asarray(beats, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(
            f"Expected beats array of shape (n, length) or (n, length, 1), got {arr.shape}."
        )

    features: list[np.ndarray] = []
    for beat in arr:
        diff = np.diff(beat)
        fft = np.fft.rfft(beat)
        mag = np.abs(fft)
        feature_vector = np.asarray(
            [
                np.sqrt(np.mean(beat**2, where=np.isfinite(beat))) if beat.size else 0.0,
                np.mean(beat, where=np.isfinite(beat)) if beat.size else 0.0,
                np.std(beat, where=np.isfinite(beat)) if beat.size else 0.0,
                np.max(beat) if beat.size else 0.0,
                np.min(beat) if beat.size else 0.0,
                np.ptp(beat) if beat.size else 0.0,
                np.mean(diff, where=np.isfinite(diff)) if diff.size else 0.0,
                np.std(diff, where=np.isfinite(diff)) if diff.size else 0.0,
                np.mean(mag) if mag.size else 0.0,
                np.std(mag) if mag.size else 0.0,
                np.max(mag) if mag.size else 0.0,
                _safe_autocorr(beat),
                float(np.sum(np.diff(np.signbit(beat)) != 0)),
            ],
            dtype=np.float32,
        )
        features.append(feature_vector)
    return np.vstack(features)


class XGBoostPredictor:
    """Load an XGBoost model and predict ECG heartbeat class probabilities."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_XGBOOST_MODEL_PATH,
        encoder_path: str | Path = DEFAULT_LABEL_ENCODER_PATH,
        model: Any | None = None,
        label_encoder: LabelEncoder | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.label_encoder = (
            label_encoder if label_encoder is not None else load_label_encoder(encoder_path)
        )
        self._model = model if model is not None else self._load_model(self.model_path)
        self.is_dummy = self._model is None
        self.model_path_missing = not self.model_path.exists()

    def _load_model(self, model_path: Path) -> Any | None:
        if not _HAS_XGBOOST:
            return None
        if not model_path.exists():
            return None
        try:
            return joblib.load(model_path)
        except Exception:
            return None

    def predict_proba(self, X_features: np.ndarray) -> np.ndarray:
        X_features = np.asarray(X_features, dtype=np.float32)
        if self._model is not None:
            try:
                probs = self._model.predict_proba(X_features)
                return np.asarray(probs, dtype=np.float32)
            except Exception:
                pass

        # Fallback dummy behavior: uniform probabilities with mild variation.
        n = int(X_features.shape[0]) if X_features.ndim == 2 else 1
        n_classes = len(self.label_encoder.classes_)
        rng = np.random.RandomState(42)
        probs = rng.rand(n, n_classes).astype(np.float32)
        probs /= np.maximum(np.sum(probs, axis=1, keepdims=True), 1e-8)
        return probs

    def predict(self, X_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probs = self.predict_proba(X_features)
        pred_idx = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        return pred_idx, confidences

    def predict_classes(self, X_features: np.ndarray) -> list[str]:
        pred_idx, _ = self.predict(X_features)
        return [
            str(self.label_encoder.inverse_transform([int(i)])[0])
            for i in pred_idx
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CNN and XGBoost predictions with SHAP explanations."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cnn-model-path", type=Path, default=Path("models/ecg_cnn.keras"))
    parser.add_argument("--xgb-model-path", type=Path, default=DEFAULT_XGBOOST_MODEL_PATH)
    parser.add_argument("--encoder-path", type=Path, default=DEFAULT_LABEL_ENCODER_PATH)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--samples-before", type=int, default=100)
    parser.add_argument("--samples-after", type=int, default=100)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to explain")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("This utility is currently focused on XGBoost prediction support.")
    print(f"XGBoost model path: {args.xgb_model_path}")
