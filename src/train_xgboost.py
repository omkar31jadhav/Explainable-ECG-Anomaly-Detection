"""Train an XGBoost heartbeat classifier on MIT-BIH features."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from .config import DEFAULT_LABEL_ENCODER_PATH, DEFAULT_XGBOOST_MODEL_PATH, DataConfig
from .dataset import build_beat_dataset
from .predict_xgboost import extract_ecg_features


def train_xgboost_pipeline(
    data_dir: Path | str = DataConfig().data_dir,
    encoder_path: Path | str = DEFAULT_LABEL_ENCODER_PATH,
    model_path: Path | str = DEFAULT_XGBOOST_MODEL_PATH,
    max_records: int | None = None,
    test_size: float = 0.15,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    early_stopping_rounds: int = 10,
) -> dict[str, object]:
    data_config = DataConfig(data_dir=Path(data_dir))
    dataset = build_beat_dataset(
        data_config=data_config,
        max_records=max_records,
        encoder_path=encoder_path,
    )

    X_features = extract_ecg_features(dataset.X)
    if X_features.shape[0] != dataset.y.shape[0]:
        raise ValueError(
            f"Feature extraction returned {X_features.shape[0]} rows but labels have {dataset.y.shape[0]} rows."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X_features,
        dataset.y,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.y,
    )

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(dataset.classes),
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        use_label_encoder=False,
        eval_metric="mlogloss",
        n_jobs=1,
        random_state=random_state,
    )

    fit_kwargs: dict[str, object] = {
        "eval_set": [(X_test, y_test)],
        "verbose": False,
    }

    if early_stopping_rounds is not None and early_stopping_rounds > 0:
        print(
            "Warning: XGBoost version does not support early_stopping_rounds with sklearn wrapper; "
            "training will run for the full n_estimators instead."
        )

    model.fit(
        X_train,
        y_train,
        **fit_kwargs,
    )

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    report = classification_report(
        y_test,
        y_pred,
        target_names=[str(label) for label in dataset.classes],
        zero_division=0,
    )

    return {
        "model_path": str(model_path),
        "encoder_path": str(encoder_path),
        "accuracy": accuracy,
        "classification_report": report,
        "classes": [str(label) for label in dataset.classes],
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_features.shape[1]),
        "n_estimators": n_estimators,
        "max_depth": max_depth,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost ECG heartbeat classifier and save the model artifact."
    )
    parser.add_argument("--data-dir", type=Path, default=DataConfig().data_dir)
    parser.add_argument("--encoder-path", type=Path, default=DEFAULT_LABEL_ENCODER_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_XGBOOST_MODEL_PATH)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--early-stopping-rounds", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = train_xgboost_pipeline(
        data_dir=args.data_dir,
        encoder_path=args.encoder_path,
        model_path=args.model_path,
        max_records=args.max_records,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stopping_rounds,
        random_state=args.random_state,
    )
    print("Saved XGBoost model to", results["model_path"])
    print(f"Accuracy on test split: {results['accuracy']:.4f}")
    print("Classes:", results["classes"])
    print(results["classification_report"])
