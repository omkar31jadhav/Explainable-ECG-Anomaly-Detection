from __future__ import annotations

import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.config import DataConfig
from src.predict import ECGPredictor


class FakeModel:
    def predict(self, X, verbose=0):
        assert X.shape == (1, 200, 1)
        return np.array([[0.02, 0.96, 0.02]], dtype=np.float32)


def test_prediction_pipeline_returns_expected_api_shape():
    encoder = LabelEncoder()
    encoder.fit(["Fusion", "PVC", "SVEB"])
    predictor = ECGPredictor(
        model=FakeModel(),
        label_encoder=encoder,
        data_config=DataConfig(samples_before=100, samples_after=100),
    )

    result = predictor.predict(np.sin(np.linspace(0, 2 * np.pi, 200)))

    assert result["predicted_class"] == "PVC"
    assert result["confidence"] > 0.9
    assert set(result["class_probabilities"]) == {"Fusion", "PVC", "SVEB"}

