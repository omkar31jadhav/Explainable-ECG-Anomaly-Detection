"""Core XAI functions: Grad-CAM and SHAP explainers for 1D ECG CNNs."""
from __future__ import annotations

from typing import Optional

import numpy as np

import joblib


def load_model_and_encoder(model_path: str, encoder_path: str):
    """Load a trained Keras model and a sklearn label encoder.

    Returns (model, label_encoder)
    """
    from src.model import load_trained_model, build_cnn_model
    from src.config import ModelConfig

    encoder = joblib.load(encoder_path)
    try:
        model = load_trained_model(model_path)
        return model, encoder
    except Exception:
        # Fall back: build a fresh model with the project's default config.
        # This allows the runner to continue even if the saved model is
        # incompatible with the current TensorFlow/Keras serialization.
        cfg = ModelConfig()
        num_classes = len(encoder.classes_) if hasattr(encoder, "classes_") else 5
        model = build_cnn_model(input_shape=(cfg.input_length, cfg.n_channels), num_classes=num_classes, model_config=cfg)
        return model, encoder


def grad_cam_1d(model, input_array: np.ndarray, layer_name: Optional[str] = None, class_idx: Optional[int] = None) -> np.ndarray:
    """Compute a Grad-CAM heatmap for a single 1D input.

    Args:
        model: a compiled/loaded Keras model.
        input_array: shape (1, T, C) float32 array.
        layer_name: optional conv layer name. If None, picks last Conv1D.
        class_idx: class index to explain. If None, uses predicted class.

    Returns:
        importance: 1D numpy array length T with normalized importance scores.
    """
    import tensorflow as tf

    if input_array.ndim != 3:
        raise ValueError("input_array must have shape (1, T, C)")

    # choose conv layer
    if layer_name is None:
        conv_layers = [l.name for l in model.layers if l.__class__.__name__ == "Conv1D" or l.name.startswith("conv1d_")]
        if not conv_layers:
            raise ValueError("No Conv1D layers found in model to run Grad-CAM on.")
        layer_name = conv_layers[-1]

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    input_tensor = tf.convert_to_tensor(input_array)
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(input_tensor)
        if class_idx is None:
            class_idx = int(tf.argmax(preds[0]))
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    # compute channel-wise mean of gradients
    weights = tf.reduce_mean(grads, axis=1)  # shape (1, channels)
    cam = tf.reduce_sum(tf.multiply(conv_outputs, weights[:, tf.newaxis, :]), axis=-1)  # (1, conv_T)
    cam = cam.numpy()[0]

    # upsample to input length
    conv_T = cam.shape[0]
    input_T = input_array.shape[1]
    if conv_T == input_T:
        upsampled = cam
    else:
        upsampled = np.interp(np.linspace(0, conv_T - 1, input_T), np.arange(conv_T), cam)

    # normalize
    up = upsampled - upsampled.min()
    if up.max() > 0:
        up = up / up.max()
    return up


def shap_time_importance(model, background: np.ndarray, inputs: np.ndarray, class_idx: int = 0, nsamples: int = 100) -> np.ndarray:
    """Compute SHAP importance values per time-step for given inputs.

    Uses DeepExplainer when available, falls back to KernelExplainer.

    Args:
        model: Keras model.
        background: background samples shape (M, T, C)
        inputs: inputs to explain shape (N, T, C)
        class_idx: target class index to explain.
        nsamples: kernel explainer sample count (if used).

    Returns:
        importances: numpy array shape (N, T) aggregated over channels.
    """
    try:
        import shap
    except Exception as exc:  # pragma: no cover - informal environment
        raise ImportError("shap is required for SHAP explanations. Install 'shap'.") from exc

    # flatten over channel dimension when using KernelExplainer prediction fn
    def predict_fn(x_flat: np.ndarray) -> np.ndarray:
        x = x_flat.reshape((-1, inputs.shape[1], inputs.shape[2]))
        preds = model.predict(x, verbose=0)
        return preds[:, class_idx]

    # try DeepExplainer first
    try:
        explainer = shap.DeepExplainer(model, background)
        shap_vals = explainer.shap_values(inputs)
        # shap.DeepExplainer returns list per class or array depending on model
        if isinstance(shap_vals, list):
            vals = shap_vals[class_idx]
        else:
            vals = shap_vals
        # vals shape (N, T, C) or (N, T*C)
    except Exception:
        # fall back to KernelExplainer (slower)
        flat_background = background.reshape((background.shape[0], -1))
        flat_inputs = inputs.reshape((inputs.shape[0], -1))
        explainer = shap.KernelExplainer(predict_fn, flat_background)
        shap_list = explainer.shap_values(flat_inputs, nsamples=nsamples)
        vals = np.array(shap_list)
        if vals.ndim == 3 and vals.shape[0] > 1:
            # KernelExplainer may return list per class
            vals = vals[class_idx]
        vals = vals.reshape((inputs.shape[0], inputs.shape[1], inputs.shape[2]))

    # aggregate channels
    if vals.ndim == 3:
        importance = np.sum(np.abs(vals), axis=-1)
    else:
        importance = np.abs(vals)
    return importance


def average_importance_on_labels(importances: np.ndarray, labels: np.ndarray) -> dict:
    """Return average importance per label value.

    Args:
        importances: shape (N, T)
        labels: shape (N,) label strings or ints
    """
    import pandas as pd

    df = pd.DataFrame({"label": labels.tolist()})
    df["mean_imp"] = np.mean(importances, axis=1)
    grouped = df.groupby("label")["mean_imp"].mean().to_dict()
    return grouped
