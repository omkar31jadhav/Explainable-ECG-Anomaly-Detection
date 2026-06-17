from __future__ import annotations

import numpy as np

from src.config import SplitConfig
from src.dataset import split_dataset


def test_split_dataset_preserves_array_alignment():
    X = np.arange(60, dtype=np.float32).reshape(12, 5, 1)
    y = np.array([0, 1] * 6)
    metadata = {"record_names": np.asarray([f"r{i}" for i in range(12)])}

    split = split_dataset(
        X,
        y,
        metadata=metadata,
        split_config=SplitConfig(validation_size=0.25, test_size=0.25, random_state=7),
    )

    assert len(split.X_train) == 6
    assert len(split.X_val) == 3
    assert len(split.X_test) == 3
    assert len(split.train_metadata["record_names"]) == len(split.y_train)

