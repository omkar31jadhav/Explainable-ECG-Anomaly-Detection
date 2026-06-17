"""Reusable prediction API for XAI and dashboard integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import DEFAULT_LABEL_ENCODER_PATH, DEFAULT_MODEL_PATH, DataConfig
from .model import load_trained_model
from .preprocessing import load_label_encoder, prepare_beat_features


class ECGPredictor:
    """Load model artifacts and predict one heartbeat window."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        encoder_path: str | Path = DEFAULT_LABEL_ENCODER_PATH,
        model: Any | None = None,
        label_encoder: Any | None = None,
        data_config: DataConfig = DataConfig(),
    ) -> None:
        self.data_config = data_config
        self.model = model if model is not None else load_trained_model(model_path)
        self.label_encoder = (
            label_encoder if label_encoder is not None else load_label_encoder(encoder_path)
        )

    def predict(self, heartbeat_signal: np.ndarray | list[float]) -> dict:
        """Predict class and probabilities for one heartbeat signal."""

        features = self._prepare_single_heartbeat(heartbeat_signal)
        try:
            probabilities = self.model.predict(features, verbose=0)
        except TypeError:
            probabilities = self.model.predict(features)
        probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)

        if probabilities.size != len(self.label_encoder.classes_):
            raise ValueError(
                "Model probability output size does not match the label encoder classes."
            )

        predicted_index = int(np.argmax(probabilities))
        if hasattr(self.label_encoder, "inverse_transform"):
            predicted_class = str(
                self.label_encoder.inverse_transform([predicted_index])[0]
            )
        else:
            predicted_class = str(self.label_encoder.classes_[predicted_index])

        return {
            "predicted_class": predicted_class,
            "confidence": float(probabilities[predicted_index]),
            "class_probabilities": {
                str(label): float(probability)
                for label, probability in zip(self.label_encoder.classes_, probabilities)
            },
        }

    def _prepare_single_heartbeat(self, heartbeat_signal: np.ndarray | list[float]) -> np.ndarray:
        arr = np.asarray(heartbeat_signal, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[-1] == 1:
            arr = arr[:, 0]
        if arr.ndim != 1:
            raise ValueError(
                f"Expected a 1D heartbeat window, got shape {arr.shape}."
            )
        if arr.shape[0] != self.data_config.window_size:
            raise ValueError(
                f"Expected heartbeat length {self.data_config.window_size}, "
                f"got {arr.shape[0]}."
            )
        return prepare_beat_features(
            arr[np.newaxis, :],
            standardize=self.data_config.standardize_beats,
            add_channel_axis=True,
        )


def predict_ecg(
    heartbeat_signal: np.ndarray | list[float],
    model_path: str | Path = DEFAULT_MODEL_PATH,
    encoder_path: str | Path = DEFAULT_LABEL_ENCODER_PATH,
) -> dict:
    """Convenience prediction function for downstream modules."""

    predictor = ECGPredictor(model_path=model_path, encoder_path=encoder_path)
    return predictor.predict(heartbeat_signal)

