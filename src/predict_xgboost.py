"""Predict ECG anomalies using both CNN and XGBoost models with explanations."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import shap
from sklearn.metrics import accuracy_score

from .config import DEFAULT_LABEL_ENCODER_PATH, DataConfig, SplitConfig
from .dataset import build_beat_dataset, split_dataset
from .model import load_trained_model
from .preprocessing import load_label_encoder
from .train_xgboost import extract_ecg_features, DEFAULT_XGBOOST_MODEL_PATH, DEFAULT_XGBOOST_EXPLAINER_PATH
from .config import DEFAULT_MODEL_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CNN and XGBoost predictions with SHAP explanations.")
    parser.add_argument("--data-dir", type=Path, default=DataConfig.data_dir)
    parser.add_argument("--cnn-model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--xgb-model-path", type=Path, default=DEFAULT_XGBOOST_MODEL_PATH)
    parser.add_argument("--xgb-explainer-path", type=Path, default=DEFAULT_XGBOOST_EXPLAINER_PATH)
    parser.add_argument("--encoder-path", type=Path, default=DEFAULT_LABEL_ENCODER_PATH)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--samples-before", type=int, default=100)
    parser.add_argument("--samples-after", type=int, default=100)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to explain")
    return parser.parse_args()


def predict_and_explain(args: argparse.Namespace | None = None) -> None:
    """Compare CNN and XGBoost predictions with SHAP explanations."""
    args = args or parse_args()
    
    print("📂 Loading label encoder...")
    label_encoder = load_label_encoder(args.encoder_path)
    
    # Load dataset
    print("📊 Loading dataset...")
    data_config = DataConfig(
        data_dir=args.data_dir,
        signal_channel=args.channel,
        samples_before=args.samples_before,
        samples_after=args.samples_after,
    )
    split_config = SplitConfig()
    
    dataset = build_beat_dataset(
        data_config=data_config,
        max_records=args.max_records,
        encoder_path=args.encoder_path,
    )
    
    splits = split_dataset(
        dataset.X, dataset.y, metadata=dataset.metadata(), split_config=split_config
    )
    
    # Extract features for XGBoost
    X_features = extract_ecg_features(dataset.X)
    splits_xgb = split_dataset(
        X_features, dataset.y, metadata=dataset.metadata(), split_config=split_config
    )
    
    # Load CNN model
    print("🧠 Loading CNN model...")
    try:
        cnn_model = load_trained_model(args.cnn_model_path)
        cnn_available = True
    except Exception as e:
        print(f"⚠️  CNN model not available (TensorFlow error): {e}")
        cnn_model = None
        cnn_available = False
    
    # Load XGBoost model and explainer
    print("🌳 Loading XGBoost model...")
    if not args.xgb_model_path.exists():
        print(f"❌ XGBoost model not found. Train it first with: python -m src.train_xgboost")
        return
    
    xgb_model = joblib.load(args.xgb_model_path)
    
    # Try to load explainer, but it's optional
    explainer = None
    if args.xgb_explainer_path.exists():
        try:
            explainer = joblib.load(args.xgb_explainer_path)
        except Exception as e:
            print(f"⚠️  SHAP explainer could not be loaded: {type(e).__name__}")
    
    # Evaluate both models
    print("\n" + "="*70)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*70)
    
    xgb_pred = xgb_model.predict(splits_xgb.X_test)
    xgb_acc = accuracy_score(splits_xgb.y_test, xgb_pred)
    
    print(f"\n📊 XGBoost Model Accuracy: {xgb_acc:.4f}")
    
    if cnn_available:
        cnn_pred = cnn_model.predict(splits.X_test, verbose=0).argmax(axis=1)
        cnn_acc = accuracy_score(splits.y_test, cnn_pred)
        print(f"📊 CNN Model Accuracy:     {cnn_acc:.4f}")
        print(f"🏆 Ensemble Accuracy:     {(cnn_acc + xgb_acc) / 2:.4f} (avg)")
    else:
        print("⚠️  CNN model not available for comparison")
        cnn_pred = None
    
    # Show sample predictions with explanations
    print("\n" + "="*70)
    print(f"SAMPLE PREDICTIONS & EXPLANATIONS (First {args.num_samples} test samples)")
    print("="*70)
    
    feature_names = [
        "RMS", "Mean", "Std", "Max", "Min", "Range",
        "Diff_Mean", "Diff_Std", "FFT_Mean", "FFT_Std", "FFT_Max",
        "Autocorr", "Zero_Crossings"
    ]
    
    # Generate SHAP values if explainer is available
    shap_values = None
    if explainer:
        shap_values = explainer.shap_values(splits_xgb.X_test[:args.num_samples])
    else:
        print("ℹ️  SHAP explanations not available (use: python -m src.shap_visualize to generate)")
    
    for i in range(min(args.num_samples, len(splits.X_test))):
        true_label = label_encoder.inverse_transform([splits.y_test[i]])[0]
        xgb_pred_label = label_encoder.inverse_transform([xgb_pred[i]])[0]
        
        print(f"\n{'─'*70}")
        print(f"Sample #{i+1}")
        print(f"{'─'*70}")
        print(f"True Label:      {true_label}")
        if cnn_available and cnn_pred is not None:
            cnn_pred_label = label_encoder.inverse_transform([cnn_pred[i]])[0]
            print(f"CNN Prediction:  {cnn_pred_label} {'✓' if cnn_pred[i] == splits.y_test[i] else '✗'}")
        print(f"XGBoost Pred:    {xgb_pred_label} {'✓' if xgb_pred[i] == splits_xgb.y_test[i] else '✗'}")
        
        # Get SHAP explanation (for predicted class)
        pred_class = xgb_pred[i]
        if isinstance(shap_values, list):
            shap_val = shap_values[pred_class][i]
        else:
            shap_val = shap_values[i]
        
        # Top 3 influential features
        feature_importance = np.argsort(np.abs(shap_val))[-3:][::-1]
        print(f"\nTop Features Contributing to '{xgb_pred_label}' prediction:")
        for rank, feat_idx in enumerate(feature_importance, 1):
            impact = "increases" if shap_val[feat_idx] > 0 else "decreases"
            print(f"  {rank}. {feature_names[feat_idx]}: {shap_val[feat_idx]:.4f} ({impact})")
    
    print("\n" + "="*70)
    print("✅ Prediction comparison complete!")
    print("="*70)


if __name__ == "__main__":
    predict_and_explain()
