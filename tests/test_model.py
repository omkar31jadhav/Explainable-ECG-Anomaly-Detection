from __future__ import annotations

from dataclasses import replace

import pytest

from src.config import ModelConfig
from src.model import build_cnn_model


def test_build_cnn_model_creates_compiled_model_when_tensorflow_available():
    pytest.importorskip("tensorflow")

    model = build_cnn_model(input_shape=(200, 1), num_classes=5)

    assert model.input_shape == (None, 200, 1)
    assert model.output_shape == (None, 5)
    assert model.loss == "sparse_categorical_crossentropy"
    assert model.get_layer("global_average_pool") is not None
    assert model.get_layer("global_max_pool") is not None
    assert not any(layer.__class__.__name__ == "Flatten" for layer in model.layers)
    assert model.count_params() < 50_000


@pytest.mark.parametrize(
    ("input_shape", "num_classes", "config", "message"),
    [
        ((0, 1), 5, ModelConfig(), "input_shape"),
        ((200, 1), 1, ModelConfig(), "num_classes"),
        ((200, 1), 5, replace(ModelConfig(), conv_filters=()), "conv_filters"),
        ((200, 1), 5, replace(ModelConfig(), dropout_rate=1.0), "dropout_rate"),
        (
            (200, 1),
            5,
            replace(ModelConfig(), spatial_dropout_rate=-0.1),
            "spatial_dropout_rate",
        ),
        ((200, 1), 5, replace(ModelConfig(), learning_rate=0.0), "learning_rate"),
    ],
)
def test_build_cnn_model_rejects_invalid_configuration(
    input_shape,
    num_classes,
    config,
    message,
):
    with pytest.raises(ValueError, match=message):
        build_cnn_model(
            input_shape=input_shape,
            num_classes=num_classes,
            model_config=config,
        )
