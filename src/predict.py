"""Reusable prediction API for XAI and dashboard integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import DEFAULT_LABEL_ENCODER_PATH, DEFAULT_MODEL_PATH, DataConfig
from .model import load_trained_model
from pathlib import Path
from typing import Optional

try:
    import tensorflow as tf  # type: ignore
    _HAS_TF = True
except Exception:
    tf = None
    _HAS_TF = False
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
        use_tflite: bool = False,
    ) -> None:
        self.data_config = data_config
        self._keras_model = model if model is not None else load_trained_model(model_path)
        self.use_tflite = bool(use_tflite) and _HAS_TF
        self._tflite_path = Path(model_path).with_suffix('.tflite')
        self._tflite_interpreter: Optional[object] = None
        if self.use_tflite:
            try:
                # Prefer existing TFLite file
                if self._tflite_path.exists():
                    self._load_tflite_interpreter(self._tflite_path)
                else:
                    # Try to convert Keras model to TFLite for faster CPU inference
                    try:
                        converter = tf.lite.TFLiteConverter.from_keras_model(self._keras_model)
                        converter.optimizations = [tf.lite.Optimize.DEFAULT]
                        tflite_model = converter.convert()
                        self._tflite_path.write_bytes(tflite_model)
                        self._load_tflite_interpreter(self._tflite_path)
                    except Exception:
                        # Fall back to Keras model
                        self.use_tflite = False
                        self._tflite_interpreter = None
            except Exception:
                self.use_tflite = False
                self._tflite_interpreter = None
        self.model = self._keras_model
        self.label_encoder = (
            label_encoder if label_encoder is not None else load_label_encoder(encoder_path)
        )

    def _load_tflite_interpreter(self, tflite_path: Path) -> None:
        if not _HAS_TF:
            return
        try:
            interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
            interpreter.allocate_tensors()
            self._tflite_interpreter = interpreter
        except Exception:
            self._tflite_interpreter = None

    def predict(self, heartbeat_signal: np.ndarray | list[float]) -> dict:
        """Predict class and probabilities for one heartbeat signal."""

        features = self._prepare_single_heartbeat(heartbeat_signal)
        # If TFLite interpreter is available use it for faster CPU inference
        probabilities = None
        if getattr(self, '_tflite_interpreter', None) is not None:
            inp_details = self._tflite_interpreter.get_input_details()
            out_details = self._tflite_interpreter.get_output_details()
            # Assume single input tensor
            input_index = inp_details[0]['index']
            # Convert features to required dtype
            required_dtype = inp_details[0].get('dtype', features.dtype)
            in_array = features.astype(required_dtype)
            try:
                # Resize interpreter input if batch size differs
                current_shape = inp_details[0]['shape']
                if tuple(current_shape) != in_array.shape:
                    try:
                        self._tflite_interpreter.resize_tensor_input(input_index, in_array.shape)
                        self._tflite_interpreter.allocate_tensors()
                    except Exception:
                        # Some interpreters don't support resizing; fall back to per-sample
                        if in_array.shape[0] > 1:
                            probs_list = []
                            for i in range(in_array.shape[0]):
                                self._tflite_interpreter.set_tensor(input_index, in_array[i : i + 1])
                                self._tflite_interpreter.invoke()
                                out = self._tflite_interpreter.get_tensor(out_details[0]['index'])
                                probs_list.append(out[0])
                            probabilities = np.vstack(probs_list)
                        else:
                            self._tflite_interpreter.set_tensor(input_index, in_array)
                            self._tflite_interpreter.invoke()
                            probabilities = self._tflite_interpreter.get_tensor(out_details[0]['index'])
                    else:
                        self._tflite_interpreter.set_tensor(input_index, in_array)
                        self._tflite_interpreter.invoke()
                        probabilities = self._tflite_interpreter.get_tensor(out_details[0]['index'])
            except Exception:
                # Fallback to Keras model on any failure
                try:
                    probabilities = self._keras_model.predict(features, verbose=0)
                except Exception:
                    probabilities = None
        else:
            try:
                probabilities = self._keras_model.predict(features, verbose=0)
            except TypeError:
                probabilities = self._keras_model.predict(features)

        if probabilities is None:
            # Ultimate fallback: tiny uniform probabilities
            import numpy as _np

            probabilities = _np.ones((1, len(self.label_encoder.classes_)), dtype=_np.float32) / float(
                max(1, len(self.label_encoder.classes_))
            )
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

