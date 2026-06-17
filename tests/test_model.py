from __future__ import annotations

import pytest

from src.model import build_cnn_model


def test_build_cnn_model_creates_compiled_model_when_tensorflow_available():
    pytest.importorskip("tensorflow")

    model = build_cnn_model(input_shape=(200, 1), num_classes=5)

    assert model.input_shape == (None, 200, 1)
    assert model.output_shape == (None, 5)
    assert model.loss == "sparse_categorical_crossentropy"

