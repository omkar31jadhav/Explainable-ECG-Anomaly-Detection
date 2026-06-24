# Explainable ECG Arrhythmia Detection using Deep Learning

## Overview

This repository implements a complete pipeline for ECG heartbeat classification using the PhysioNet MIT-BIH Arrhythmia Database. It focuses on data loading, beat extraction, preprocessing, CNN model training, evaluation, and a small prediction API. Model explainability and dashboard components are handled in companion repositories.

Key features:

- ECG loading with WFDB
- Beat-centered segmentation and normalization
- A lightweight 1D CNN for heartbeat classification
- Training utilities with callbacks and model persistence
- Evaluation and result artifacts (plots, metrics, confusion matrix)

---

## Quick demo visuals

The primary training and evaluation artifacts are embedded below so GitHub shows the overview immediately:

**Accuracy Curve:**

![Accuracy Curve](results/accuracy_curve.png)

**Loss Curve:**

![Loss Curve](results/loss_curve.png)

**Confusion Matrix:**

![Confusion Matrix](results/confusion_matrix.png)

---

## Dataset

Dataset: MIT-BIH Arrhythmia Database (PhysioNet)

Files required: `.dat`, `.hea`, `.atr` per record. Place the dataset under the `data/` folder.

Recommended layout:

```
data/
    100.dat 100.hea 100.atr
    101.dat 101.hea 101.atr
    ...
```

---

## Model & Architecture

Input: 200-sample heartbeat windows centered on the R-peak (100 samples before, 100 after).

The model is a compact 1D CNN with repeated Conv1D → BatchNorm → Activation → Pool blocks, light spatial dropout, and global pooling before the dense classifier. The trained model artifact is saved as `models/ecg_cnn.keras`.

---

## Training

Train with:

```bash
python -m src.train --data-dir data --epochs 50 --batch-size 128
```

Quick smoke test:

```bash
python -m src.train --data-dir data --max-records 3 --epochs 2
```

Produced artifacts (examples):

- `models/ecg_cnn.keras`
- `models/label_encoder.joblib`
- `results/training_history.csv`, `results/training_history.json`
- `results/accuracy_curve.png`, `results/loss_curve.png`

---

## Evaluation

Run evaluation after training:

```bash
python -m src.evaluate --data-dir data
```

Outputs:

- `results/evaluation_metrics.json`
- `results/classification_report.txt`
- `results/confusion_matrix.csv`
- `results/confusion_matrix.png`

---

## Prediction API

Quick example:

```python
import numpy as np
from src.predict import ECGPredictor

heartbeat = np.load("example_heartbeat.npy")  # shape (200,)
predictor = ECGPredictor()
result = predictor.predict(heartbeat)
print(result)
```

Returned structure includes predicted class, confidence, and class probabilities.

---

## Tests

Run unit tests:

```bash
python -m pytest -v
```

---

## Where to look next

- Code: `src/` (data loading, model, training, evaluation)
- Saved model: `models/ecg_cnn.keras`
- Results: `results/` (plots, metrics)

If you'd like, I can also:

- Add a model architecture diagram to `docs/` and embed it here
- Add badges (build / tests / dataset) to the top of this `README.md`
- Create a short `docs/README.md` with image thumbnails and links

---

Team: Machine Learning, Explainability, and Dashboard contributors

License: See project or organization policy
