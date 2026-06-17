"""Simple runner to generate explanations (SHAP + Grad-CAM) for test beats.

Produces PNGs in results/xai and a small CSV summary.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from pyache.xai import load_model_and_encoder, grad_cam_1d, shap_time_importance, average_importance_on_labels
from pyache.visualize import plot_signal_with_importance
from src.dataset import build_beat_dataset, split_dataset
from src.config import RESULTS_DIR, DEFAULT_MODEL_PATH, DEFAULT_LABEL_ENCODER_PATH, DataConfig, SplitConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--encoder", type=Path, default=DEFAULT_LABEL_ENCODER_PATH)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR / "xai")
    p.add_argument("--n-examples", type=int, default=20)
    p.add_argument("--random-seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    data_config = DataConfig()
    dataset = build_beat_dataset(data_config=data_config)
    splits = split_dataset(dataset.X, dataset.y, metadata=dataset.metadata())

    model, encoder = load_model_and_encoder(str(args.model), str(args.encoder))

    rng = np.random.default_rng(args.random_seed)
    idxs = rng.choice(len(splits.X_test), size=min(args.n_examples, len(splits.X_test)), replace=False)

    records = []
    # build small background for SHAP from train set
    bg = splits.X_train[:50] if splits.X_train.shape[0] >= 10 else splits.X_train

    for i, idx in enumerate(idxs):
        x = splits.X_test[idx: idx + 1]
        y = splits.y_test[idx]
        label_name = str(encoder.inverse_transform([int(y)])[0])

        # Grad-CAM
        try:
            cam = grad_cam_1d(model, x)
        except Exception as exc:
            cam = np.zeros((x.shape[1],), dtype=float)

        # SHAP
        try:
            shap_imp = shap_time_importance(model, bg[:20], x, class_idx=int(y), nsamples=50)[0]
        except Exception:
            shap_imp = np.zeros((x.shape[1],), dtype=float)

        # save images
        base = args.results_dir / f"example_{i:03d}_label_{label_name}"
        plot_signal_with_importance(x[0], cam, base.with_suffix(".gradcam.png"), title=f"Grad-CAM: {label_name}")
        plot_signal_with_importance(x[0], shap_imp, base.with_suffix(".shap.png"), title=f"SHAP: {label_name}")

        records.append({"example": int(i), "test_index": int(idx), "label": label_name, "gradcam_mean": float(np.mean(cam)), "shap_mean": float(np.mean(shap_imp))})

    df = pd.DataFrame.from_records(records)
    df.to_csv(args.results_dir / "summary.csv", index=False)
    print("Saved explanations to:", args.results_dir)


if __name__ == "__main__":
    main()
