"""TensorFlow/Keras model definition for ECG heartbeat classification."""

from __future__ import annotations

from pathlib import Path

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
    """Build and compile a configurable 1D CNN classifier."""

    if num_classes < 2:
        raise ValueError("num_classes must be at least 2 for classification.")

    tf = _import_tensorflow()
    input_shape = input_shape or (model_config.input_length, model_config.n_channels)

    inputs = tf.keras.Input(shape=input_shape, name="ecg_heartbeat")
    x = inputs
    for layer_index, filters in enumerate(model_config.conv_filters, start=1):
        x = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=model_config.kernel_size,
            padding="same",
            activation=model_config.activation,
            name=f"conv1d_{layer_index}",
        )(x)
        if model_config.batch_normalization:
            x = tf.keras.layers.BatchNormalization(name=f"batch_norm_{layer_index}")(x)
        x = tf.keras.layers.MaxPooling1D(
            pool_size=model_config.pool_size,
            name=f"max_pool_{layer_index}",
        )(x)

    x = tf.keras.layers.Dropout(model_config.dropout_rate, name="dropout")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(
        model_config.dense_units,
        activation=model_config.activation,
        name="dense_features",
    )(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="class_probabilities",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mitbih_1d_cnn")
    optimizer = tf.keras.optimizers.Adam(learning_rate=model_config.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_trained_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    """Load a saved Keras model."""

    tf = _import_tensorflow()
    return tf.keras.models.load_model(Path(model_path))

