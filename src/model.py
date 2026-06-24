"""TensorFlow/Keras model definition for ECG heartbeat classification."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .config import DEFAULT_MODEL_PATH, ModelConfig


def _import_tensorflow():
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for model creation/training. "
            "Install dependencies with `pip install -r requirements.txt` in a "
            "TensorFlow-compatible Python environment."
        ) from exc
    return tf


def build_cnn_model(
    input_shape: tuple[int, int] | None = None,
    num_classes: int = 5,
    model_config: ModelConfig = ModelConfig(),
):
    """Build and compile a compact 1D CNN classifier.

    The convolutional trunk retains temporal resolution for Grad-CAM while the
    dual global-pooling head avoids the large parameter count of ``Flatten``.
    """

    resolved_input_shape = input_shape or (
        model_config.input_length,
        model_config.n_channels,
    )
    _validate_model_config(resolved_input_shape, num_classes, model_config)

    tf = _import_tensorflow()
    kernel_initializer = _kernel_initializer_for(model_config.activation)

    inputs = tf.keras.Input(shape=resolved_input_shape, name="ecg_heartbeat")
    x = inputs
    for layer_index, filters in enumerate(model_config.conv_filters, start=1):
        x = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=model_config.kernel_size,
            padding="same",
            use_bias=not model_config.batch_normalization,
            kernel_initializer=kernel_initializer,
            name=f"conv1d_{layer_index}",
        )(x)
        if model_config.batch_normalization:
            x = tf.keras.layers.BatchNormalization(name=f"batch_norm_{layer_index}")(x)
        x = tf.keras.layers.Activation(
            model_config.activation,
            name=f"activation_{layer_index}",
        )(x)
        x = tf.keras.layers.MaxPooling1D(
            pool_size=model_config.pool_size,
            padding="same",
            name=f"max_pool_{layer_index}",
        )(x)
        if model_config.spatial_dropout_rate:
            x = tf.keras.layers.SpatialDropout1D(
                model_config.spatial_dropout_rate,
                name=f"spatial_dropout_{layer_index}",
            )(x)

    average_features = tf.keras.layers.GlobalAveragePooling1D(
        name="global_average_pool",
    )(x)
    max_features = tf.keras.layers.GlobalMaxPooling1D(name="global_max_pool")(x)
    x = tf.keras.layers.Concatenate(name="global_feature_pool")(
        [average_features, max_features]
    )
    x = tf.keras.layers.Dense(
        model_config.dense_units,
        use_bias=not model_config.batch_normalization,
        kernel_initializer=kernel_initializer,
        name="dense_features",
    )(x)
    if model_config.batch_normalization:
        x = tf.keras.layers.BatchNormalization(name="dense_batch_norm")(x)
    x = tf.keras.layers.Activation(
        model_config.activation,
        name="dense_activation",
    )(x)
    x = tf.keras.layers.Dropout(model_config.dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="class_probabilities",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="mitbih_compact_1d_cnn",
    )
    optimizer_kwargs = {
        "learning_rate": model_config.learning_rate,
        "clipnorm": model_config.gradient_clip_norm,
    }
    if model_config.weight_decay:
        optimizer = tf.keras.optimizers.AdamW(
            weight_decay=model_config.weight_decay,
            **optimizer_kwargs,
        )
    else:
        optimizer = tf.keras.optimizers.Adam(**optimizer_kwargs)

    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    if num_classes > 2:
        metrics.append(
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=min(3, num_classes),
                name="top_3_accuracy",
            )
        )
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=metrics,
    )
    return model


def load_trained_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    *,
    compile: bool = True,
):
    """Load a saved Keras model."""

    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    tf = _import_tensorflow()
    return tf.keras.models.load_model(model_path, compile=compile)


def _validate_model_config(
    input_shape: Sequence[int],
    num_classes: int,
    model_config: ModelConfig,
) -> None:
    """Fail early with actionable errors for invalid architecture settings."""

    if len(input_shape) != 2 or any(
        not isinstance(size, int) or isinstance(size, bool) or size <= 0
        for size in input_shape
    ):
        raise ValueError(
            "input_shape must contain two positive integers: (timesteps, channels)."
        )
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2 for classification.")
    if not model_config.conv_filters or any(
        not isinstance(filters, int)
        or isinstance(filters, bool)
        or filters <= 0
        for filters in model_config.conv_filters
    ):
        raise ValueError("conv_filters must contain positive integers.")
    if model_config.kernel_size <= 0:
        raise ValueError("kernel_size must be positive.")
    if model_config.pool_size <= 0:
        raise ValueError("pool_size must be positive.")
    if model_config.dense_units <= 0:
        raise ValueError("dense_units must be positive.")
    if not 0.0 <= model_config.dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in the interval [0, 1).")
    if not 0.0 <= model_config.spatial_dropout_rate < 1.0:
        raise ValueError("spatial_dropout_rate must be in the interval [0, 1).")
    if model_config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if model_config.weight_decay < 0.0:
        raise ValueError("weight_decay cannot be negative.")
    if model_config.gradient_clip_norm <= 0.0:
        raise ValueError("gradient_clip_norm must be positive.")
    if not isinstance(model_config.activation, str) or not model_config.activation.strip():
        raise ValueError("activation must be a non-empty string.")


def _kernel_initializer_for(activation: str) -> str:
    """Choose an initializer suited to the configured activation."""

    normalized = activation.lower().strip()
    if normalized in {"relu", "leaky_relu", "elu", "selu", "gelu"}:
        return "he_normal"
    return "glorot_uniform"
