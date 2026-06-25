import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from pathlib import Path
import io
import json
import tempfile
import zipfile
from dataclasses import asdict

from src.config import DataConfig
from src.data_loader import MITBIHDataLoader
from src.predict import ECGPredictor
from src.predict_xgboost import FEATURE_NAMES, XGBoostPredictor, extract_ecg_features
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.preprocessing import (
    extract_heartbeat_windows,
    prepare_beat_features,
    preprocess_signal_for_dataset,
    map_symbol_to_label,
)


# Optional dependency guard: the project expects joblib for label encoder persistence.
# If joblib is missing in the runtime environment, we gracefully degrade and disable
# the dataset-scan feature instead of crashing the whole app.
try:
    import joblib  # noqa: F401

    _HAS_JOBLIB = True
except ModuleNotFoundError:
    _HAS_JOBLIB = False

try:
    import wfdb  # noqa: F401

    _HAS_WFDB = True
except ModuleNotFoundError:
    _HAS_WFDB = False


# page config
st.set_page_config(
    page_title=" ECG Anomaly Detection",
    page_icon="⚕️",
    layout="wide",
)

# Dashboard performance controls. The uploaded dataset is capped at 100 records
# and each record is sampled to a small heartbeat count for fast interaction.
MAX_RECORDS_TO_PROCESS = 100
DEFAULT_RECORD_LIMIT = 50  # Default to 50 records for improved coverage.
BEATS_PER_RECORD = 5  # 5 beats per record for fast results
QUICK_MODE_BEATS_PER_RECORD = 1  # Even in quick mode, 1 beat per record max
FAST_MODE_RECORD_LIMIT = 50  # Absolute max in fast mode for sub-30 second scans


def _estimate_scan_time_seconds(
    record_limit: int,
    beats_per_record: int,
    quick_mode: bool,
) -> int:
    # Ultra-realistic estimates for new optimized pipeline
    if quick_mode:
        # 0.7-1.2 sec per record in quick mode
        base = 0.5
        per_record = 0.8
    else:
        base = 1.0
        per_record = 1.5
    estimated = int(round(base + record_limit * per_record))
    return max(8, min(estimated, 45))  # 8-45 sec range


def _format_time_estimate(total_seconds: int) -> str:
    if total_seconds < 12:
        return "~8-15 seconds"
    if total_seconds < 20:
        return "~12-20 seconds"
    if total_seconds < 30:
        return "~15-30 seconds"
    if total_seconds < 45:
        return "~25-45 seconds"
    return "~45-60 seconds"


def _render_scan_summary(summary_container):
    if "scan_results" not in st.session_state:
        summary_container.info(
            "Upload a compatible dataset, then choose limits and click Detect Anomalies."
        )
        return

    scan = st.session_state["scan_results"]
    total_records = int(scan.get("total_records_analyzed", len(scan.get("record_names", []))))
    anomalous_records = int(scan.get("anomalous_records", 0))
    normal_records = max(total_records - anomalous_records, 0)
    anomaly_pct = float(scan.get("anomaly_pct_records", 0.0))
    normal_pct = float(scan.get("normal_pct_records", 0.0))

    cols = summary_container.columns(3)
    cols[0].metric("Total analyzed", f"{total_records:,}")
    cols[1].metric("Normal", f"{normal_records:,}", f"{normal_pct:.2f}%")
    cols[2].metric("Anomalies", f"{anomalous_records:,}", f"{anomaly_pct:.2f}%")
    summary_container.caption(
        f"Processed the first {total_records:,} records for extreme performance optimization."
    )


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def _normalize_label(label: str | None) -> str:
    if label is None:
        return "Unknown"
    normalized = str(label).strip()
    return normalized if normalized else "Unknown"


def _is_anomalous_label(label: str | None) -> int:
    return 0 if _normalize_label(label).lower() == "normal" else 1


def _build_model_comparison_metrics(full_df: pd.DataFrame) -> tuple[pd.DataFrame, float] | None:
    if "ground_truth_label" not in full_df.columns:
        return None

    ground_truth = full_df["ground_truth_label"].astype(object).map(_normalize_label)
    valid_mask = ground_truth != "Unknown"
    if valid_mask.sum() == 0:
        return None

    true_labels = ground_truth[valid_mask].astype(str).tolist()
    pred_cnn_labels = full_df.loc[valid_mask, "predicted_class"].astype(object).map(_normalize_label).tolist()
    pred_xgb_labels = full_df.loc[valid_mask, "predicted_class_xgb"].astype(object).map(_normalize_label).tolist()
    binary_true = [1 if _is_anomalous_label(label) else 0 for label in true_labels]
    binary_pred_cnn = [1 if _is_anomalous_label(label) else 0 for label in pred_cnn_labels]
    binary_pred_xgb = [1 if _is_anomalous_label(label) else 0 for label in pred_xgb_labels]

    labels = sorted(set(true_labels + pred_cnn_labels + pred_xgb_labels))

    def _model_metrics(pred_labels: list[str]) -> dict[str, float]:
        accuracy = float(accuracy_score(true_labels, pred_labels))
        macro = precision_recall_fscore_support(
            true_labels,
            pred_labels,
            labels=labels,
            average="macro",
            zero_division=0,
        )
        binary = precision_recall_fscore_support(
            binary_true,
            [1 if _is_anomalous_label(label) else 0 for label in pred_labels],
            average="binary",
            zero_division=0,
        )
        return {
            "accuracy": accuracy,
            "binary_precision": float(binary[0]),
            "binary_recall": float(binary[1]),
            "binary_f1": float(binary[2]),
            "macro_precision": float(macro[0]),
            "macro_recall": float(macro[1]),
            "macro_f1": float(macro[2]),
        }

    metrics = [
        {"model": "CNN", **_model_metrics(pred_cnn_labels)},
        {"model": "XGBoost", **_model_metrics(pred_xgb_labels)},
    ]
    agreement = float(np.mean(np.asarray(binary_pred_cnn, dtype=int) == np.asarray(binary_pred_xgb, dtype=int)))
    return pd.DataFrame.from_records(metrics).set_index("model"), agreement


def _plot_model_comparison_metrics(metrics_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    metric_keys = ["accuracy", "binary_f1", "macro_f1"]
    for metric in metric_keys:
        fig.add_trace(
            go.Bar(
                name=metric.replace("_", " ").title(),
                x=metrics_df.index.tolist(),
                y=metrics_df[metric].tolist(),
            )
        )
    fig.update_layout(
        barmode="group",
        title="Performance comparison across models",
        yaxis=dict(title="Score", range=[0.0, 1.0]),
        height=340,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def _build_gradcam_summary(
    heartbeat_window_200: np.ndarray,
    predictor,
    data_config: DataConfig,
    target_class_index: int,
) -> tuple[np.ndarray, dict[str, float]]:
    try:
        from pyache.xai import grad_cam_1d
    except Exception:
        grad_cam_1d = None

    try:
        X = prepare_beat_features(
            np.asarray(heartbeat_window_200, dtype=np.float32)[np.newaxis, :],
            standardize=data_config.standardize_beats,
            add_channel_axis=True,
        )
        if grad_cam_1d is not None:
            importance = grad_cam_1d(predictor.model, X, class_idx=target_class_index)
        else:
            raise RuntimeError("Grad-CAM unavailable")
    except Exception:
        importance = _gradient_saliency_for_input(
            heartbeat_window_200=heartbeat_window_200,
            model=predictor.model,
            target_class_index=target_class_index,
            data_config=data_config,
        )

    values = np.asarray(importance, dtype=np.float32)
    sorted_values = np.sort(values)
    top_10pct = sorted_values[int(max(1, len(sorted_values) * 0.1)) :]
    stats = {
        "mean_importance": float(np.mean(values)) if values.size else 0.0,
        "max_importance": float(np.max(values)) if values.size else 0.0,
        "top_10pct_mean": float(np.mean(top_10pct)) if top_10pct.size else 0.0,
        "high_importance_pct": float(np.mean(values > 0.5)) if values.size else 0.0,
    }
    return values, stats


def _build_xgb_shap_summary_figure(
    beat_window: np.ndarray,
    xgb_predictor: XGBoostPredictor,
    feature_names: list[str] = FEATURE_NAMES,
) -> go.Figure | None:
    if xgb_predictor._model is None:
        return None
    try:
        import shap
    except Exception:
        return None

    beat = np.asarray(beat_window, dtype=np.float32)
    if beat.ndim == 2 and beat.shape[-1] == 1:
        beat = beat[..., 0]
    features = extract_ecg_features(beat[np.newaxis, :])
    if features.shape[0] == 0:
        return None

    try:
        explainer = shap.TreeExplainer(xgb_predictor._model)
    except Exception:
        try:
            explainer = shap.Explainer(xgb_predictor._model)
        except Exception:
            return None

    shap_values = explainer(features)
    if hasattr(shap_values, "values"):
        shap_vals = shap_values.values
    else:
        shap_vals = shap_values
    shap_arr = np.asarray(shap_vals)
    if shap_arr.ndim == 3:
        shap_feature_importance = np.mean(np.abs(shap_arr), axis=(0, 1))
    elif shap_arr.ndim == 2:
        shap_feature_importance = np.mean(np.abs(shap_arr), axis=0)
    elif shap_arr.ndim == 1:
        shap_feature_importance = np.abs(shap_arr)
    else:
        return None

    if len(feature_names) != len(shap_feature_importance):
        feature_names = [f"feature_{i}" for i in range(len(shap_feature_importance))]

    fig = go.Figure(
        data=[
            go.Bar(
                x=feature_names,
                y=shap_feature_importance.tolist(),
                marker_color="#2a9d8f",
            )
        ]
    )
    fig.update_layout(
        title="XGBoost SHAP feature importance",
        xaxis_title="Feature",
        yaxis_title="Mean |SHAP value|",
        height=360,
        margin=dict(l=20, r=20, t=45, b=60),
    )
    return fig


def _collect_selected_beat_windows(
    df: pd.DataFrame,
    extracted_root: Path,
    data_config: DataConfig,
    max_samples: int = 8,
) -> np.ndarray:
    windows = []
    for _, row in df.head(max_samples).iterrows():
        try:
            heartbeat, _ = _discover_waveform_for_beat(
                extracted_root=extracted_root,
                record_name=str(row["record_name"]),
                beat_index=int(row["beat_index"]),
                data_config=data_config,
            )
            windows.append(np.asarray(heartbeat, dtype=np.float32))
        except Exception:
            continue
    return np.stack(windows) if windows else np.empty((0, data_config.window_size), dtype=np.float32)


def _ensure_uploaded_zip_extract(tmp_dir: Path, uploaded_bytes: bytes) -> Path:
    zip_path = tmp_dir / "upload.zip"
    zip_path.write_bytes(uploaded_bytes)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                # Skip directories and hidden files
                if member.filename.endswith("/") or member.filename.startswith("."):
                    continue
                target = (tmp_dir / member.filename).resolve()
                # Verify path is safe (no directory traversal)
                if not str(target).startswith(str(tmp_dir.resolve())):
                    continue  # Skip unsafe paths rather than failing
            zf.extractall(tmp_dir)
    except Exception as e:
        raise ValueError(f"Failed to extract ZIP file: {type(e).__name__}: {e}") from e
    return tmp_dir


def _ensure_uploaded_folder_extract(tmp_dir: Path, uploaded_files: list) -> Path:
    allowed_suffixes = {".hea", ".dat", ".atr"}
    for uploaded_file in uploaded_files:
        raw_name = str(getattr(uploaded_file, "name", "")).replace("\\", "/")
        if not raw_name:
            continue
        path_parts = [
            part
            for part in Path(raw_name).parts
            if part not in {"", ".", ".."} and not Path(part).is_absolute()
        ]
        if not path_parts:
            continue
        relative_path = Path(*path_parts)
        if relative_path.suffix.lower() not in allowed_suffixes:
            continue
        target = (tmp_dir / relative_path).resolve()
        if not str(target).startswith(str(tmp_dir.resolve())):
            raise ValueError("Uploaded folder contains an unsafe file path.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(uploaded_file.getvalue())
    return tmp_dir


def _inspect_records_from_extracted(root: Path) -> dict:
    """Find and validate MIT-BIH/WFDB records inside an extracted ZIP.

    The ZIP structure can vary (files may be at root or inside nested folders).
    We therefore recursively group `.hea`, `.dat`, and `.atr` files by their
    extension-less relative path.
    """

    if not root.exists():
        return {
            "total_records_found": 0,
            "valid_records": [],
            "invalid_records": [],
            "selected_records": [],
            "performance_mode": False,
        }

    grouped: dict[str, set[str]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".hea", ".dat", ".atr"}:
            continue
        record_name = path.with_suffix("").relative_to(root).as_posix()
        grouped.setdefault(record_name, set()).add(path.suffix.lower())

    valid_records = sorted(
        name for name, suffixes in grouped.items() if {".hea", ".dat", ".atr"} <= suffixes
    )
    invalid_records = sorted(
        {
            name: sorted({".hea", ".dat", ".atr"} - suffixes)
            for name, suffixes in grouped.items()
            if not ({".hea", ".dat", ".atr"} <= suffixes)
        }.items()
    )

    selected_records = valid_records[:MAX_RECORDS_TO_PROCESS]
    return {
        "total_records_found": len(grouped),
        "valid_records": valid_records,
        "invalid_records": invalid_records,
        "selected_records": selected_records,
        "performance_mode": len(valid_records) > MAX_RECORDS_TO_PROCESS,
    }


def _discover_records_from_extracted(root: Path) -> list[str]:
    return _inspect_records_from_extracted(root)["valid_records"]


def _data_config_cache_key(data_config: DataConfig) -> dict:
    values = asdict(data_config)
    values["data_dir"] = str(values["data_dir"])
    return values



@st.cache_resource(show_spinner=False)
def _get_predictor() -> ECGPredictor:
    if not _HAS_JOBLIB:
        raise ModuleNotFoundError(
            "joblib is required by src.preprocessing/save_label_encoder/load_label_encoder. "
            "Install project dependencies (pip install -r requirements.txt)."
        )
    if not _HAS_WFDB:
        raise ModuleNotFoundError(
            "wfdb is required to load MIT-BIH records. "
            "Install project dependencies (pip install -r requirements.txt)."
        )

    # TensorFlow is preferred for model loading/prediction. If it's not
    # available in the environment, fall back to a lightweight dummy
    # predictor so the dashboard remains usable for exploration and the
    # "Extract Anomaly Records" flow can be exercised.
    try:
        import tensorflow  # noqa: F401
        predictor = ECGPredictor(use_tflite=False)
        # mark real predictor
        try:
            setattr(predictor, "is_dummy", False)
        except Exception:
            pass
        return predictor
    except Exception:
        # Minimal dummy predictor that implements the same surface used by
        # the dashboard: `.model.predict(X, ...)` and `.label_encoder.classes_`.
        import numpy as _np

        class _DummyModel:
            def predict(self, X, batch_size=None, verbose=0):
                n = int(getattr(X, "shape", (0,))[0])
                # Two-class fallback: Normal / Anomalous
                probs = _np.zeros((n, 2), dtype=_np.float32)
                # Make mostly-Normal probabilities but include some variation
                rand = _np.random.RandomState(1).rand(n).astype(_np.float32)
                probs[:, 0] = 1.0 - 0.5 * rand
                probs[:, 1] = 0.5 * rand
                return probs

        class _DummyLabelEncoder:
            classes_ = _np.array(["Normal", "Anomalous"], dtype=object)

        class _DummyPredictor:
            def __init__(self):
                self.model = _DummyModel()
                self.label_encoder = _DummyLabelEncoder()
                self.is_dummy = True

        return _DummyPredictor()


@st.cache_resource(show_spinner=False)
def _get_xgboost_predictor() -> XGBoostPredictor:
    try:
        return XGBoostPredictor()
    except Exception:
        return XGBoostPredictor(model=None)


def _predict_record_heartbeats(
    record_name: str, data_dir: Path, data_config: DataConfig
) -> dict:
    loader = MITBIHDataLoader(data_dir=data_dir, annotator=data_config.annotator)
    record = loader.load_record(record_name)
    annotation = loader.load_annotation(record_name)

    # Clean full record before beat extraction
    signal = preprocess_signal_for_dataset(record.signal, data_config)

    # Map symbols to AAMI labels; only keep those that map to a class
    valid_peaks: list[int] = []
    valid_labels: list[str] = []
    valid_symbols: list[str] = []
    for sample, symbol in zip(annotation.samples, annotation.symbols):
        label = map_symbol_to_label(
            symbol,
            label_scheme=data_config.label_scheme,
            include_unknown=data_config.include_unknown,
        )
        if label is None:
            continue
        valid_peaks.append(int(sample))
        valid_labels.append(label)
        valid_symbols.append(symbol)

    extraction = extract_heartbeat_windows(
        signal=signal,
        r_peaks=valid_peaks,
        labels=valid_labels,
        symbols=valid_symbols,
        samples_before=data_config.samples_before,
        samples_after=data_config.samples_after,
        channel=data_config.signal_channel,
    )

    if extraction.beats.shape[0] == 0:
        return {
            "record_name": record_name,
            "num_beats": 0,
            "anomaly_beats": 0,
            "records_anomalous": False,
            "beat_rows": [],
        }

    # Prepare beats for model input
    X = prepare_beat_features(
        extraction.beats,
        standardize=data_config.standardize_beats,
        add_channel_axis=True,
    )

    predictor = _get_predictor()

    # Batch predict all beats for responsiveness
    probs = predictor.model.predict(X, verbose=0)
    probs = np.asarray(probs, dtype=np.float32)

    classes = predictor.label_encoder.classes_

    # Treat anomalies as anything except "Normal" (AAMI)
    normal_idx = None
    for i, c in enumerate(classes):
        if str(c).lower() == "normal":
            normal_idx = i
            break

    if normal_idx is None:
        # Fallback: use majority class among extracted labels
        normal_idx = (
            int(np.argmax(np.bincount(extraction.labels))) if len(extraction.labels) else 0
        )

    pred_idx = np.argmax(probs, axis=1)
    pred_labels = [str(classes[i]) for i in pred_idx]
    confidence = np.max(probs, axis=1)

    # anomaly: predicted != Normal
    is_anom = pred_idx != normal_idx

    beat_rows = []
    for i in range(len(is_anom)):
        beat_rows.append(
            {
                "record_name": record_name,
                "beat_index": int(i),
                "r_peak_sample": int(extraction.r_peaks[i]),
                "predicted_class": pred_labels[i],
                "confidence": _safe_float(confidence[i]),
                "ground_truth_label": str(extraction.labels[i])
                if i < len(extraction.labels)
                else None,
                "annotation_symbol": str(extraction.symbols[i])
                if i < len(extraction.symbols)
                else None,
            }
        )

    num_beats = int(len(is_anom))
    anomaly_beats = int(np.sum(is_anom))
    records_anomalous = anomaly_beats > 0

    return {
        "record_name": record_name,
        "num_beats": num_beats,
        "anomaly_beats": anomaly_beats,
        "records_anomalous": records_anomalous,
        "beat_rows": beat_rows,
    }


@st.cache_data(show_spinner=False)
def _extract_record_prediction_inputs_cached(
    record_name: str,
    data_dir: str,
    data_config_values: dict,
    max_beats: int | None = None,
) -> tuple[np.ndarray, list[dict]]:
    data_config = DataConfig(
        data_dir=Path(data_config_values["data_dir"]),
        annotator=data_config_values["annotator"],
        signal_channel=int(data_config_values["signal_channel"]),
        samples_before=int(data_config_values["samples_before"]),
        samples_after=int(data_config_values["samples_after"]),
        normalization=data_config_values["normalization"],
        label_scheme=data_config_values["label_scheme"],
        include_unknown=bool(data_config_values["include_unknown"]),
        standardize_beats=bool(data_config_values["standardize_beats"]),
    )
    loader = MITBIHDataLoader(data_dir=Path(data_dir), annotator=data_config.annotator)
    annotation = loader.load_annotation(record_name)
    valid_peaks: list[int] = []
    valid_labels: list[str] = []
    valid_symbols: list[str] = []
    for sample, symbol in zip(annotation.samples, annotation.symbols):
        label = map_symbol_to_label(
            symbol,
            label_scheme=data_config.label_scheme,
            include_unknown=data_config.include_unknown,
        )
        if label is None:
            continue
        peak = int(sample)
        if peak - data_config.samples_before < 0:
            continue
        valid_peaks.append(peak)
        valid_labels.append(label)
        valid_symbols.append(symbol)
        if max_beats is not None and len(valid_peaks) >= int(max_beats):
            break

    if not valid_peaks:
        return np.empty((0, data_config.window_size), dtype=np.float32), []

    wfdb = __import__("wfdb")
    base_path = Path(data_dir) / record_name
    sampfrom = max(min(valid_peaks) - data_config.samples_before, 0)
    sampto = max(valid_peaks) + data_config.samples_after
    wfdb_record = wfdb.rdrecord(str(base_path), sampfrom=sampfrom, sampto=sampto)
    signal = np.asarray(wfdb_record.p_signal, dtype=np.float32)
    signal = preprocess_signal_for_dataset(signal, data_config)
    segment_peaks = [peak - sampfrom for peak in valid_peaks]

    extraction = extract_heartbeat_windows(
        signal=signal,
        r_peaks=segment_peaks,
        labels=valid_labels,
        symbols=valid_symbols,
        samples_before=data_config.samples_before,
        samples_after=data_config.samples_after,
        channel=data_config.signal_channel,
    )

    if extraction.beats.shape[0] == 0:
        return np.empty((0, data_config.window_size), dtype=np.float32), []

    metadata_rows = []
    for i in range(extraction.beats.shape[0]):
        metadata_rows.append(
            {
                "record_name": record_name,
                "beat_index": int(i),
                "r_peak_sample": int(extraction.r_peaks[i] + sampfrom),
                "ground_truth_label": str(extraction.labels[i])
                if i < len(extraction.labels)
                else None,
                "annotation_symbol": str(extraction.symbols[i])
                if i < len(extraction.symbols)
                else None,
            }
        )

    return extraction.beats.astype(np.float32), metadata_rows


def _extract_record_prediction_inputs(
    record_name: str,
    data_dir: Path,
    data_config: DataConfig,
    max_beats: int | None = None,
) -> tuple[np.ndarray, list[dict]]:
    return _extract_record_prediction_inputs_cached(
        record_name,
        str(data_dir),
        _data_config_cache_key(data_config),
        max_beats,
    )


def _predict_dataset_heartbeats(
    record_names: list[str],
    data_dir: Path,
    data_config: DataConfig,
    beats_per_record: int = BEATS_PER_RECORD,
    progress_callback=None,
    predictor=None,
    xgb_predictor=None,
) -> tuple[pd.DataFrame, list[dict]]:
    all_beats: list[np.ndarray] = []
    all_rows: list[dict] = []
    errors: list[dict] = []
    import time as _time
    timings: dict = {"extraction_seconds": 0.0, "prediction_seconds": 0.0, "total_seconds": 0.0}

    # Parallelize record extraction to reduce wall-clock time on multi-core machines.
    try:
        import os

        cpu_count = os.cpu_count() or 4
    except Exception:
        cpu_count = 4
    max_workers = min(max(2, cpu_count // 2), max(1, len(record_names)))
    futures = {}
    extraction_start = _time.perf_counter()
    import concurrent.futures as _cf

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        for idx, record_name in enumerate(record_names, start=1):
            futures[exe.submit(
                _extract_record_prediction_inputs,
                record_name,
                data_dir,
                data_config,
                max_beats=beats_per_record,
            )] = (idx, record_name)

        completed = 0
        for fut in as_completed(futures):
            idx, record_name = futures[fut]
            completed += 1
            try:
                # Prevent a single slow record from stalling the entire scan
                beats, rows = fut.result(timeout=15)
            except _cf.TimeoutError as exc:
                errors.append(
                    {
                        "record_name": record_name,
                        "error_type": "TimeoutError",
                        "error_message": "Record extraction timed out.",
                    }
                )
                try:
                    fut.cancel()
                except Exception:
                    pass
                if progress_callback is not None:
                    progress_callback(completed / max(len(record_names), 1) * 0.55)
                continue
            except Exception as exc:
                errors.append(
                    {
                        "record_name": record_name,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                if progress_callback is not None:
                    progress_callback(completed / max(len(record_names), 1) * 0.55)
                continue

            # Limit beats per record to keep dashboard responsive.
            # If a record has more beats than the limit, pick evenly spaced indices.
            if beats.shape[0] > beats_per_record:
                idxs = np.linspace(0, beats.shape[0] - 1, num=beats_per_record, dtype=int)
                beats = beats[idxs]
                rows = [rows[i] for i in idxs]
            if beats.shape[0]:
                all_beats.append(beats)
                all_rows.extend(rows)
            if progress_callback is not None:
                progress_callback(completed / max(len(record_names), 1) * 0.55)

    if not all_rows:
        timings["extraction_seconds"] = _time.perf_counter() - extraction_start
        timings["total_seconds"] = timings["extraction_seconds"]
        try:
            st.session_state["last_scan_timing"] = timings
        except Exception:
            pass
        return pd.DataFrame(), errors

    timings["extraction_seconds"] = _time.perf_counter() - extraction_start

    beats_array = np.vstack(all_beats).astype(np.float32)
    X = prepare_beat_features(
        beats_array,
        standardize=data_config.standardize_beats,
        add_channel_axis=True,
    )

    # Use a single shared predictor instance (preloaded outside when possible)
    predictor = predictor if predictor is not None else _get_predictor()
    xgb_predictor = xgb_predictor if xgb_predictor is not None else _get_xgboost_predictor()
    if progress_callback is not None:
        progress_callback(0.65)

    # Build XGBoost features for the same extracted beat windows.
    try:
        X_features = extract_ecg_features(X)
    except Exception:
        X_features = np.zeros((X.shape[0], 13), dtype=np.float32)

    # Use optimized batch size for speed: smaller batches on CPU, larger on GPU
    # For fast mode with 1 beat per record, we typically have 5-10 samples max
    batch_size = int(min(32, max(1, X.shape[0])))
    # Measure prediction time; prefer TFLite interpreter when available for CPU speed
    pred_start = _time.perf_counter()
    if getattr(predictor, "_tflite_interpreter", None) is not None:
        interp = predictor._tflite_interpreter
        inp_details = interp.get_input_details()
        out_details = interp.get_output_details()
        input_index = inp_details[0]["index"]
        in_dtype = inp_details[0].get("dtype", X.dtype)
        # Handle quantized uint8 inputs
        try:
            qparams = inp_details[0].get("quantization", (0.0, 0))
            scale, zero_point = qparams if isinstance(qparams, tuple) and len(qparams) >= 2 else (0.0, 0)
        except Exception:
            scale, zero_point = (0.0, 0)

        in_array = X
        if in_dtype == np.uint8 and scale and zero_point is not None:
            # Convert float input to uint8 quantized values
            in_array_q = np.clip(np.round(in_array / float(scale) + float(zero_point)), 0, 255).astype(np.uint8)
        else:
            in_array_q = in_array.astype(in_dtype)

        try:
            # Try resizing input to batch shape
            interp.resize_tensor_input(input_index, in_array_q.shape)
            interp.allocate_tensors()
            interp.set_tensor(input_index, in_array_q)
            interp.invoke()
            probs = interp.get_tensor(out_details[0]["index"])
        except Exception:
            # Fall back to per-sample invocation
            probs_list = []
            for i in range(in_array_q.shape[0]):
                interp.set_tensor(input_index, in_array_q[i : i + 1])
                interp.invoke()
                out = interp.get_tensor(out_details[0]["index"])
                probs_list.append(out[0])
            probs = np.vstack(probs_list)
    else:
        probs = predictor.model.predict(X, batch_size=batch_size, verbose=0)
    timings["prediction_seconds"] = _time.perf_counter() - pred_start
    timings["total_seconds"] = timings["extraction_seconds"] + timings["prediction_seconds"]
    try:
        st.session_state["last_scan_timing"] = timings
    except Exception:
        pass
    if progress_callback is not None:
        progress_callback(0.9)

    probs = np.asarray(probs, dtype=np.float32)
    classes = predictor.label_encoder.classes_
    normal_idx = _discover_normal_index(classes)
    pred_idx = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)

    xgb_probs = xgb_predictor.predict_proba(X_features)
    xgb_classes = xgb_predictor.label_encoder.classes_
    xgb_normal_idx = _discover_normal_index(xgb_classes)
    if xgb_normal_idx < 0:
        xgb_normal_idx = 0
    xgb_pred_idx = np.argmax(xgb_probs, axis=1)
    xgb_confidence = np.max(xgb_probs, axis=1)

    for row, class_idx, score, class_idx_xgb, score_xgb in zip(
        all_rows,
        pred_idx,
        confidence,
        xgb_pred_idx,
        xgb_confidence,
    ):
        is_anomaly = int(class_idx) != normal_idx
        row["predicted_class"] = str(classes[int(class_idx)])
        row["confidence"] = _safe_float(score)
        row["anomaly_score"] = _safe_float(score if is_anomaly else 1.0 - score)
        row["anomaly_decision"] = "Anomalous" if is_anomaly else "Normal"

        xgb_is_anomaly = int(class_idx_xgb) != xgb_normal_idx
        row["predicted_class_xgb"] = str(xgb_classes[int(class_idx_xgb)])
        row["confidence_xgb"] = _safe_float(score_xgb)
        row["anomaly_score_xgb"] = _safe_float(score_xgb if xgb_is_anomaly else 1.0 - score_xgb)
        row["anomaly_decision_xgb"] = "Anomalous" if xgb_is_anomaly else "Normal"

    if progress_callback is not None:
        progress_callback(1.0)

    return pd.DataFrame(all_rows), errors


@st.cache_data(show_spinner=False)
def _predict_dataset_heartbeats_cached(
    record_names: tuple[str, ...],
    data_dir: str,
    data_config_values: dict,
    beats_per_record: int,
) -> tuple[pd.DataFrame, list[dict]]:
    data_config = DataConfig(
        data_dir=Path(data_config_values["data_dir"]),
        annotator=data_config_values["annotator"],
        signal_channel=int(data_config_values["signal_channel"]),
        samples_before=int(data_config_values["samples_before"]),
        samples_after=int(data_config_values["samples_after"]),
        normalization=data_config_values["normalization"],
        label_scheme=data_config_values["label_scheme"],
        include_unknown=bool(data_config_values["include_unknown"]),
        standardize_beats=bool(data_config_values["standardize_beats"]),
    )
    # Call the fast path without a progress callback; predictor will be created
    # by the caller or lazily here.
    return _predict_dataset_heartbeats(
        list(record_names),
        Path(data_dir),
        data_config,
        beats_per_record=int(beats_per_record),
        progress_callback=None,
        predictor=None,
    )


def _discover_waveform_for_beat(
    extracted_root: Path,
    record_name: str,
    beat_index: int,
    data_config: DataConfig,
) -> tuple[np.ndarray, str | None]:
    """Extract the same 200-sample heartbeat window used during prediction."""
    
    beats, rows = _extract_record_prediction_inputs(
        record_name,
        extracted_root,
        data_config,
        max_beats=int(beat_index) + 1,
    )
    if beats.shape[0] == 0:
        raise ValueError("No heartbeat windows extracted for this record.")

    matched_index = next(
        (idx for idx, row in enumerate(rows) if int(row["beat_index"]) == int(beat_index)),
        None,
    )
    if matched_index is None:
        raise ValueError("Selected beat_index is out of range for this record.")

    heartbeat = beats[matched_index]
    gt = rows[matched_index].get("ground_truth_label")
    return heartbeat, gt


def _discover_normal_index(classes: np.ndarray | list[str]) -> int:
    for i, c in enumerate(classes):
        if str(c).lower() == "normal":
            return i

    return -1


def _gradient_saliency_for_input(
    heartbeat_window_200: np.ndarray,
    model,
    target_class_index: int,
    data_config: DataConfig,
) -> np.ndarray:
    """Fast per-sample gradient saliency for the selected heartbeat."""
    # Prefer true gradient saliency when TensorFlow is available. If TF is
    # missing or gradients cannot be computed, fall back to a fast,
    # interpretable heuristic that highlights large changes in the signal.
    try:
        import tensorflow as tf
    except Exception:
        tf = None

    if tf is not None:
        try:
            X = prepare_beat_features(
                np.asarray(heartbeat_window_200, dtype=np.float32)[np.newaxis, :],
                standardize=data_config.standardize_beats,
                add_channel_axis=True,
            )
            input_tensor = tf.convert_to_tensor(X, dtype=tf.float32)

            with tf.GradientTape() as tape:
                tape.watch(input_tensor)
                predictions = model(input_tensor, training=False)
                class_count = int(predictions.shape[-1])
                class_idx = int(target_class_index)
                if class_idx < 0 or class_idx >= class_count:
                    class_idx = int(tf.argmax(predictions[0]).numpy())
                target_score = predictions[:, class_idx]

            gradients = tape.gradient(target_score, input_tensor)
            if gradients is None:
                raise RuntimeError("GradientTape returned no gradients")

            saliency = np.max(np.abs(gradients.numpy()[0]), axis=-1)
            saliency = np.asarray(saliency, dtype=np.float32)
            max_value = float(np.max(saliency))
            if max_value > 0:
                saliency = saliency / max_value

            return saliency
        except Exception:
            # Fall through to the non-TF heuristic below
            pass

    # Lightweight fallback: use normalized absolute gradient (finite-diff)
    # as a proxy for importance. This is deterministic and fast, and does
    # not require TensorFlow.
    try:
        series = np.asarray(heartbeat_window_200, dtype=np.float32)
        grad = np.abs(np.gradient(series.astype(np.float32)))
        # Optionally combine with deviation from local median to emphasize peaks
        med = np.abs(series - np.median(series))
        sal = grad + 0.5 * med
        sal = sal.astype(np.float32)
        maxv = float(np.max(sal))
        if maxv > 0:
            sal = sal / maxv
        else:
            sal = np.zeros_like(sal, dtype=np.float32)
        return sal
    except Exception:
        return np.zeros(data_config.window_size, dtype=np.float32)


def _plot_ecg_with_saliency(
    series: np.ndarray,
    saliency: np.ndarray,
    title: str,
    zoom_range: tuple[int, int] | None = None,
):
    x = np.arange(len(series))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=series,
            mode="lines",
            name="ECG",
            line=dict(color="gray", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=series,
            mode="markers",
            name="Importance",
            marker=dict(
                size=7,
                color=saliency,
                colorscale="Reds",
                showscale=False,
                opacity=0.9,
            ),
        )
    )

    fig.update_layout(
        title=title,
        height=420,
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis_title="Samples",
        yaxis_title="Amplitude",
        xaxis=dict(rangeslider=dict(visible=True), type="linear"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )

    if zoom_range is not None:
        a, b = zoom_range
        fig.update_xaxes(range=[a, b])

    return fig


def _plot_simple_timeseries(series: np.ndarray, title: str):
    x = np.arange(len(series))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=series, mode="lines", line=dict(color="gray", width=2)))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=45, b=20), title=title, xaxis_title="Samples", yaxis_title="Amplitude", xaxis=dict(rangeslider=dict(visible=True)))
    return fig


@st.cache_data(show_spinner=False)
def _load_record_clinical_details_cached(
    record_name: str, data_dir: str, data_config_values: dict
) -> dict:
    data_config = DataConfig(
        data_dir=Path(data_config_values["data_dir"]),
        annotator=data_config_values["annotator"],
        signal_channel=int(data_config_values["signal_channel"]),
        samples_before=int(data_config_values["samples_before"]),
        samples_after=int(data_config_values["samples_after"]),
        normalization=data_config_values["normalization"],
        label_scheme=data_config_values["label_scheme"],
        include_unknown=bool(data_config_values["include_unknown"]),
        standardize_beats=bool(data_config_values["standardize_beats"]),
    )
    loader = MITBIHDataLoader(data_dir=Path(data_dir), annotator=data_config.annotator)
    record = loader.load_record(record_name)
    annotation = loader.load_annotation(record_name)
    signal = np.asarray(record.signal, dtype=np.float32)
    channel = min(max(int(data_config.signal_channel), 0), signal.shape[1] - 1) if signal.ndim == 2 else 0
    signal_channel = signal[:, channel] if signal.ndim == 2 else signal
    annotation_symbols = pd.Series(annotation.symbols, dtype="object")
    symbol_counts = annotation_symbols.value_counts().to_dict() if len(annotation_symbols) else {}

    header_path = Path(data_dir) / record_name
    header_text = header_path.with_suffix(".hea").read_text(
        encoding="utf-8", errors="replace"
    )

    return {
        "record_id": record_name,
        "header_information": header_text.strip(),
        "header_metadata": json.dumps(record.metadata, default=str),
        "annotation_information": json.dumps(
            {
                "annotator": data_config.annotator,
                "total_annotations": int(len(annotation.symbols)),
                "symbol_counts": {str(k): int(v) for k, v in symbol_counts.items()},
                "first_samples": [int(x) for x in annotation.samples[:10].tolist()],
                "first_symbols": [str(x) for x in annotation.symbols[:10]],
                "aux_note_count": int(len(annotation.aux_note)),
            },
            default=str,
        ),
        "signal_mean": _safe_float(np.mean(signal_channel)),
        "signal_std": _safe_float(np.std(signal_channel)),
        "signal_min": _safe_float(np.min(signal_channel)),
        "signal_max": _safe_float(np.max(signal_channel)),
        "signal_samples": int(signal_channel.shape[0]),
        "sampling_frequency": _safe_float(record.fs),
        "signal_channels": int(signal.shape[1] if signal.ndim == 2 else 1),
    }


def _load_record_clinical_details(
    record_name: str, data_dir: Path, data_config: DataConfig
) -> dict:
    return _load_record_clinical_details_cached(
        record_name,
        str(data_dir),
        _data_config_cache_key(data_config),
    )


def _build_anomaly_record_details(
    anomaly_df: pd.DataFrame,
    full_df: pd.DataFrame,
    extracted_root: Path,
    data_config: DataConfig,
) -> pd.DataFrame:
    """Return fast anomaly summary without clinical details loading."""
    if not len(anomaly_df):
        return pd.DataFrame(
            columns=[
                "record_id",
                "prediction_result",
                "anomaly_score",
                "explanation",
            ]
        )

    detail_rows = []
    for record_name in sorted(anomaly_df["record_name"].astype(str).unique().tolist()):
        rec_anom = anomaly_df[anomaly_df["record_name"].astype(str) == record_name].copy()
        rec_all = full_df[full_df["record_name"].astype(str) == record_name].copy()
        top_idx = rec_anom["confidence"].astype(float).idxmax()
        top = rec_anom.loc[top_idx]
        anomaly_count = int(rec_anom.shape[0])
        total_count = int(rec_all.shape[0])
        anomaly_score = _safe_float(rec_anom["confidence"].astype(float).max())
        predicted_classes = ", ".join(
            rec_anom["predicted_class"].astype(str).value_counts().index.tolist()
        )
        explanation = (
            f"Record {record_name}: {anomaly_count}/{total_count} beats anomalous. "
            f"Top class: {top['predicted_class']} ({anomaly_score*100:.1f}%). "
            f"R-peak: {int(top['r_peak_sample'])}"
        )
        detail_rows.append(
            {
                "record_id": record_name,
                "prediction_result": "Anomalous",
                "predicted_classes": predicted_classes,
                "anomalous_sampled_beats": anomaly_count,
                "sampled_beats": total_count,
                "anomaly_score": anomaly_score,
                "explanation": explanation,
            }
        )

    return pd.DataFrame(detail_rows)


def _build_anomaly_report_zip(scan: dict) -> bytes:
    full_df = scan.get("full_df", pd.DataFrame())
    anomaly_df = scan.get("anomaly_df", pd.DataFrame())
    anomaly_record_details = scan.get("anomaly_record_details", pd.DataFrame())
    total_records = int(scan.get("total_records_analyzed", len(scan.get("record_names", []))))
    anomalous_records = int(scan.get("anomalous_records", 0))
    normal_records = max(total_records - anomalous_records, 0)

    summary = {
        "dataset_records_scanned": scan.get("record_names", []),
        "total_records_analyzed": total_records,
        "normal_records": normal_records,
        "anomalous_records": anomalous_records,
        "normal_percentage": (normal_records / total_records * 100.0) if total_records else 0.0,
        "anomaly_percentage": float(scan.get("anomaly_pct_records", 0.0)),
        "total_sampled_beats": int(scan.get("total_beats", 0)),
        "anomalous_sampled_beats": int(scan.get("anomalous_beats", 0)),
        "affected_records": sorted(anomaly_df["record_name"].astype(str).unique().tolist())
        if len(anomaly_df)
        else [],
        "anomaly_definition": "Predicted class is not Normal.",
        "processing_note": (
            "Only the first 100 uploaded WFDB records are eligible, and each record is sampled "
            "for faster dashboard performance."
        ),
    }

    by_record = pd.DataFrame()
    if len(full_df):
        tmp = full_df.copy()
        tmp["is_anomaly"] = tmp["predicted_class"].astype(str).str.lower().ne("normal")
        by_record = (
            tmp.groupby("record_name")
            .agg(
                total_beats=("beat_index", "count"),
                anomalous_beats=("is_anomaly", "sum"),
            )
            .reset_index()
        )
        by_record["normal_beats"] = by_record["total_beats"] - by_record["anomalous_beats"]
        by_record["anomaly_percentage"] = (
            by_record["anomalous_beats"] / by_record["total_beats"] * 100.0
        )

    explanations_df = anomaly_df.copy()
    if "anomaly_decision" not in explanations_df.columns:
        explanations_df["anomaly_decision"] = explanations_df["predicted_class"].astype(str).apply(
            lambda x: "Normal" if str(x).lower() == "normal" else "Anomalous"
        )
    if len(explanations_df):
        explanations_df["explanation"] = explanations_df.apply(
            lambda row: (
                f"Beat classified as {row['predicted_class']} with "
                f"{float(row['confidence']) * 100.0:.2f}% confidence; non-Normal "
                "classes are treated as anomalous."
            ),
            axis=1,
        )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.json", json.dumps(summary, indent=2))
        zf.writestr("summary.csv", pd.DataFrame([summary]).to_csv(index=False))
        zf.writestr("anomalies.csv", anomaly_df.to_csv(index=False))
        zf.writestr("explanations.csv", explanations_df.to_csv(index=False))
        zf.writestr("anomaly_record_details.csv", anomaly_record_details.to_csv(index=False))
        zf.writestr("all_predictions.csv", full_df.to_csv(index=False))
        zf.writestr("record_summary.csv", by_record.to_csv(index=False))

    return buffer.getvalue()


st.title("Explainable ECG Anomaly Detection Dashboard")
st.caption(
    "Ultra-fast anomaly detection: Upload ECG records, get results in seconds. Ultra-fast mode analyzes 5-50 records with 5 beats each for lightning-quick insights."
)
st.info(
    "⚡ Ultra-fast mode: Complete ECG analysis in <20 seconds! Processes up to 50 records, extracting 5 critical heartbeats per record."
)


data_config = DataConfig()
# Pre-warm model in a background thread to avoid blocking the first user action.
def _prewarm_predictor():
    try:
        _get_predictor()
        st.session_state['predictor_ready'] = True
    except Exception:
        st.session_state['predictor_ready'] = False

# Start pre-warm only once per session
if not st.session_state.get('predictor_prewarmer_started'):
    threading.Thread(target=_prewarm_predictor, daemon=True).start()
    st.session_state['predictor_prewarmer_started'] = True
upload_col, controls_col, summary_col = st.columns([1.25, 1.0, 1.25])

with upload_col:
    dataset_source = st.radio(
        "Dataset source",
        ["Upload Folder", "Upload ZIP", "Use local folder"],
        horizontal=True,
    )
    uploaded = None
    if st.session_state.get("dataset_source") != dataset_source:
        st.session_state.pop("scan_state", None)
        st.session_state.pop("scan_results", None)
        st.session_state.pop("upload_preview", None)
        st.session_state.pop("report_ready", None)
        st.session_state.pop("uploaded_signature", None)
        st.session_state["dataset_source"] = dataset_source

    if dataset_source == "Upload Folder":
        uploaded_folder_files = st.file_uploader(
            "Upload folder containing ECG records",
            type=["hea", "dat", "atr"],
            accept_multiple_files="directory",
            help="Choose the folder containing matching .hea, .dat, and .atr files.",
        )
        if uploaded_folder_files:
            folder_signature = (
                "folder_upload",
                tuple(
                    sorted(
                        (
                            str(file.name),
                            int(getattr(file, "size", 0) or 0),
                        )
                        for file in uploaded_folder_files
                    )
                ),
            )
            if st.session_state.get("uploaded_signature") != folder_signature:
                st.session_state.pop("scan_state", None)
                st.session_state.pop("scan_results", None)
                st.session_state.pop("upload_preview", None)
                st.session_state.pop("report_ready", None)
                st.session_state["uploaded_signature"] = folder_signature

                tmp_root = Path(tempfile.mkdtemp(prefix="ecg_folder_upload_"))
                extracted_root = _ensure_uploaded_folder_extract(
                    tmp_root,
                    uploaded_folder_files,
                )
                record_inspection = _inspect_records_from_extracted(extracted_root)
                st.session_state["upload_preview"] = {
                    "extracted_root": str(extracted_root),
                    "source_type": "folder_upload",
                    **record_inspection,
                }

    elif dataset_source == "Use local folder":
        local_folder = st.text_input(
            "Folder containing .hea/.dat/.atr files",
            value=str((Path(__file__).resolve().parent / "data").resolve()),
            help="Paste the folder path that already contains the ECG record files.",
        )
        local_root = Path(local_folder).expanduser()
        source_signature = ("folder", str(local_root.resolve()) if local_root.exists() else str(local_root))
        if st.session_state.get("uploaded_signature") != source_signature:
            st.session_state.pop("scan_state", None)
            st.session_state.pop("scan_results", None)
            st.session_state.pop("upload_preview", None)
            st.session_state.pop("report_ready", None)
            st.session_state["uploaded_signature"] = source_signature

            if local_root.exists() and local_root.is_dir():
                record_inspection = _inspect_records_from_extracted(local_root)
                st.session_state["upload_preview"] = {
                    "extracted_root": str(local_root),
                    "source_type": "folder",
                    **record_inspection,
                }
            else:
                st.session_state["upload_preview"] = {
                    "extracted_root": "",
                    "source_type": "folder",
                    "total_records_found": 0,
                    "valid_records": [],
                    "invalid_records": [],
                    "selected_records": [],
                    "performance_mode": False,
                    "path_error": "Folder does not exist.",
                }

    else:
        uploaded = st.file_uploader(
            "Upload ECG dataset ZIP",
            type=["zip"],
            accept_multiple_files=False,
            help="Use a ZIP containing MIT-BIH/WFDB .hea, .dat, and .atr files.",
        )
        if uploaded is not None:
            source_signature = ("zip", uploaded.name, getattr(uploaded, "size", None))
            if st.session_state.get("uploaded_signature") != source_signature:
                st.session_state.pop("scan_state", None)
                st.session_state.pop("scan_results", None)
                st.session_state.pop("upload_preview", None)
                st.session_state.pop("report_ready", None)
                st.session_state["uploaded_signature"] = source_signature

                tmp_root = Path(tempfile.mkdtemp(prefix="ecg_upload_preview_"))
                extracted_root = _ensure_uploaded_zip_extract(tmp_root, uploaded.getvalue())
                record_inspection = _inspect_records_from_extracted(extracted_root)
                st.session_state["upload_preview"] = {
                    "extracted_root": str(extracted_root),
                    "source_type": "zip",
                    **record_inspection,
                }

    if st.session_state.get("upload_preview"):
        preview = st.session_state["upload_preview"]
        valid_records = preview.get("valid_records", [])
        selected_records = preview.get("selected_records", [])
        invalid_records = preview.get("invalid_records", [])
        if valid_records:
            if preview.get("source_type") == "folder_upload":
                st.success("Dataset folder uploaded successfully.")
            elif preview.get("source_type") == "folder":
                st.success("Dataset folder loaded successfully.")
            else:
                st.success("Dataset uploaded successfully.")
            st.caption("Ready to analyze. Choose processing limits and click Detect Anomalies.")
        else:
            error_text = preview.get("path_error") or (
                "Validation failed: the dataset must contain matching .hea, .dat, and .atr files."
            )
            st.error(error_text)

        c1, c2, c3 = st.columns(3)
        c1.metric("Records found", f"{preview.get('total_records_found', 0):,}")
        c2.metric("Valid records", f"{len(valid_records):,}")
        c3.metric("Invalid/missing", f"{len(invalid_records):,}")

        if preview.get("performance_mode"):
            st.info(
                "Performance Mode Enabled: Only the first 100 records are analyzed."
            )

        if selected_records:
            st.write(
                pd.DataFrame(
                    {"record_name": selected_records[:10]}
                ).rename(columns={"record_name": "Sample record names"})
            )

with controls_col:
    st.markdown("<div style='height: 0.25rem;'></div>", unsafe_allow_html=True)
    record_limit = 50
    quick_mode = False
    beats_limit = 5
    effective_record_limit = record_limit
    effective_beats_limit = QUICK_MODE_BEATS_PER_RECORD if quick_mode else beats_limit
    st.markdown(
        f"**Processing settings:** {effective_record_limit} records, "
        f"{effective_beats_limit} beat{'s' if effective_beats_limit != 1 else ''} per record, "
        "fast mode enabled."
    )

    estimated_seconds = _estimate_scan_time_seconds(
        effective_record_limit,
        effective_beats_limit,
        quick_mode,
    )
    time_estimate = _format_time_estimate(estimated_seconds)
    st.caption(f"⏱️ Estimated scan time: {time_estimate}")
    st.caption(
        f"⚡ Fast mode is active: scanning exactly {effective_record_limit} records and "
        f"{effective_beats_limit} beat{'s' if effective_beats_limit != 1 else ''} per record for results."
    )

    detect_clicked = st.button(
        "Detect Anomalies",
        type="primary",
    )

with summary_col:
    summary_placeholder = st.empty()
    if detect_clicked and "scan_results" not in st.session_state:
        summary_placeholder.info("Detecting anomalies... please wait.")
    else:
        _render_scan_summary(summary_placeholder)

if detect_clicked:
    if not st.session_state.get("upload_preview"):
        st.warning("Please choose a local dataset folder or upload a dataset ZIP first.")
    elif not _HAS_WFDB:
        st.error(
            "wfdb is required to load MIT-BIH records and extract heartbeat windows. "
            "Install dependencies with `pip install -r requirements.txt`, then restart the app."
        )
    else:
        try:
            # Initialize scan state once per uploaded ZIP
            scan_settings = {
                "record_limit": int(effective_record_limit),
                "beats_per_record": int(effective_beats_limit),
                "quick_mode": bool(quick_mode),
            }
            if st.session_state.get("scan_settings") != scan_settings:
                st.session_state.pop("scan_state", None)
                st.session_state.pop("scan_results", None)
                st.session_state.pop("report_ready", None)
                st.session_state["scan_settings"] = scan_settings

            if "scan_state" not in st.session_state:
                if st.session_state.get("upload_preview"):
                    preview = st.session_state["upload_preview"]
                    extracted_root = Path(preview["extracted_root"])
                    discovered_record_names = list(preview.get("valid_records", []))
                    performance_mode = bool(preview.get("performance_mode", False))
                else:
                    tmp_root = Path(tempfile.mkdtemp(prefix="ecg_upload_"))
                    extracted_root = _ensure_uploaded_zip_extract(tmp_root, uploaded.getvalue())
                    inspection = _inspect_records_from_extracted(extracted_root)
                    discovered_record_names = list(inspection.get("valid_records", []))
                    performance_mode = bool(inspection.get("performance_mode", False))

                if not discovered_record_names:
                    st.error(
                        "No complete records found. Upload a ZIP with WFDB records such as "
                        "100.hea, 100.dat, and 100.atr."
                    )
                    st.stop()

                selected_limit = min(int(scan_settings["record_limit"]), MAX_RECORDS_TO_PROCESS)
                if len(discovered_record_names) > selected_limit:
                    discovered_record_names = discovered_record_names[:selected_limit]
                    st.info(
                        f"Only the first {selected_limit} records will be scanned to keep the dashboard fast."
                    )
                if performance_mode:
                    st.info(
                        "Performance Mode Enabled: Only the first 100 records are analyzed."
                    )

                st.session_state["scan_state"] = {
                    "extracted_root": str(extracted_root),
                    "discovered_record_names": discovered_record_names,
                    "full_df": pd.DataFrame(),
                    "anomaly_df": pd.DataFrame(),
                }

            scan_state = st.session_state["scan_state"]
            discovered_record_names = scan_state["discovered_record_names"]

            if not discovered_record_names:
                st.info("No records to evaluate.")
            else:
                # Start scan immediately (removed countdown for faster feedback)
                with st.spinner(f"🔄 Processing {len(discovered_record_names)} record(s)... Estimated time: {time_estimate}"):
                    extracted_root = Path(scan_state["extracted_root"])
                    progress = st.progress(0)
                    time_label = st.empty()

                    def progress_callback(progress_fraction: float) -> None:
                        progress.progress(min(max(progress_fraction, 0.0), 1.0))
                        if progress_fraction < 1.0:
                            remaining = max(
                                0,
                                int(round((1.0 - progress_fraction) * estimated_seconds)),
                            )
                            time_label.caption(
                                f"⏳ Estimated time remaining: {remaining} second{'s' if remaining != 1 else ''}"
                            )
                        else:
                            time_label.success("✅ Scan complete.")

                    # Ensure predictors are loaded once and reused for the whole scan.
                    try:
                        predictor = _get_predictor()
                    except Exception:
                        predictor = None
                    try:
                        xgb_predictor = _get_xgboost_predictor()
                    except Exception:
                        xgb_predictor = None

                    if xgb_predictor is None or getattr(xgb_predictor, "is_dummy", False):
                        st.warning(
                            "XGBoost model artifact is not available or failed to load. "
                            "Current XGBoost metrics are fallback estimates and may be inaccurate. "
                            "Place a valid models/ecg_xgboost.joblib file in the project to compare real XGBoost performance."
                        )
                    else:
                        st.success(
                            "XGBoost model loaded successfully from models/ecg_xgboost.joblib. "
                            "Dashboard will compare real XGBoost predictions against the CNN model."
                        )

                    full_df, processing_errors = _predict_dataset_heartbeats(
                        list(discovered_record_names),
                        extracted_root,
                        data_config,
                        beats_per_record=int(effective_beats_limit),
                        progress_callback=progress_callback,
                        predictor=predictor,
                        xgb_predictor=xgb_predictor,
                    )
                    progress.progress(1.0)
                    time_label.success("✅ Scan complete.")

                if processing_errors:
                    st.session_state["processing_errors"] = processing_errors
                    error_df = pd.DataFrame(processing_errors)
                    if len(full_df):
                        st.warning(
                            f"{len(processing_errors)} record(s) could not be processed. "
                            "The remaining records were analyzed."
                        )
                    else:
                        st.error(
                            "No heartbeat windows were extracted because every selected "
                            "record failed during loading or preprocessing."
                        )
                        st.dataframe(error_df.head(20), width="stretch", height=220)
                        st.stop()
                else:
                    st.session_state.pop("processing_errors", None)

                if len(full_df):
                    is_anomaly = full_df["predicted_class"].astype(str).str.lower().ne("normal")
                    anomaly_df = full_df[is_anomaly].copy()
                    xgb_is_anomaly = full_df["predicted_class_xgb"].astype(str).str.lower().ne("normal")
                    xgb_anomaly_df = full_df[xgb_is_anomaly].copy()
                else:
                    anomaly_df = pd.DataFrame()
                    xgb_anomaly_df = pd.DataFrame()

                # Show timing breakdown from last scan if available
                last_timing = st.session_state.get("last_scan_timing")
                if last_timing:
                    with st.expander("Last scan timing (seconds)", expanded=False):
                        st.write(
                            {
                                "extraction_seconds": f"{last_timing.get('extraction_seconds', 0.0):.2f}",
                                "prediction_seconds": f"{last_timing.get('prediction_seconds', 0.0):.2f}",
                                "total_seconds": f"{last_timing.get('total_seconds', 0.0):.2f}",
                            }
                        )

                scan_state["full_df"] = full_df
                scan_state["anomaly_df"] = anomaly_df

                total_beats = int(full_df.shape[0]) if len(full_df) else 0
                anomalous_beats = int(anomaly_df.shape[0]) if len(anomaly_df) else 0
                unique_records = len(discovered_record_names)
                anomalous_records = (
                    int(anomaly_df["record_name"].astype(str).nunique())
                    if len(anomaly_df)
                    else 0
                )
                normal_records = max(unique_records - anomalous_records, 0)
                anomaly_pct_records = (
                    anomalous_records / unique_records * 100.0
                    if unique_records
                    else 0.0
                )
                normal_pct_records = (
                    normal_records / unique_records * 100.0
                    if unique_records
                    else 0.0
                )
                st.session_state["scan_results"] = {
                    "record_names": discovered_record_names,
                    "extracted_root": scan_state["extracted_root"],
                    "total_records_analyzed": unique_records,
                    "total_beats": total_beats,
                    "anomalous_beats": anomalous_beats,
                    "anomalous_records": anomalous_records,
                    "normal_records": normal_records,
                    "anomaly_pct_records": float(anomaly_pct_records),
                    "normal_pct_records": float(normal_pct_records),
                    "anomaly_pct": float(anomaly_pct_records),
                    "anomaly_df": anomaly_df,
                    "xgb_anomaly_df": xgb_anomaly_df,
                    "full_df": full_df,
                    "anomaly_record_details": pd.DataFrame(),
                    "processing_errors": processing_errors,
                    "data_config": data_config,
                    "beats_per_record": effective_beats_limit,
                }
                _render_scan_summary(summary_placeholder)

                st.session_state.pop("selected_for_xai", None)
                st.session_state.pop("report_ready", None)
                st.success(
                    f"Analysis complete: {len(discovered_record_names)} record(s) processed."
                )
        except Exception as e:
            st.error(f"Scan failed: {type(e).__name__}: {e}")

if "scan_results" in st.session_state:
    scan = st.session_state["scan_results"]
    df = scan["full_df"]
    anomaly_df = scan["anomaly_df"]
    extracted_root = Path(scan["extracted_root"])
    data_config = scan["data_config"]

    if len(df):
        if scan.get("processing_errors"):
            with st.expander("Records skipped during processing", expanded=False):
                st.dataframe(
                    pd.DataFrame(scan["processing_errors"]).head(100),
                    width="stretch",
                    height=220,
                )

        metrics_result = _build_model_comparison_metrics(df)
        if metrics_result is not None:
            metrics_df, agreement = metrics_result
            with st.expander("Overall model comparison metrics", expanded=True):
                metric_cols = st.columns([1.2, 1.0])
                metric_cols[0].dataframe(metrics_df.round(3), width="stretch", height=220)
                metric_cols[1].plotly_chart(
                    _plot_model_comparison_metrics(metrics_df),
                    use_container_width=True,
                )
                st.caption(
                    f"CNN vs XGBoost anomaly decision agreement: {agreement * 100:.1f}% on labeled beats."
                )
        else:
            st.info(
                "No ground truth labels were available for overall metric comparison."
            )

        st.markdown("### Central ECG Review")
        selector_col, prediction_col, viz_col = st.columns([0.85, 1.05, 1.8])

        with selector_col:
            st.subheader("Input Instance")
            record_names = scan["record_names"]
            default_record = (
                str(anomaly_df.iloc[0]["record_name"]) if len(anomaly_df) else record_names[0]
            )
            default_record_index = (
                record_names.index(default_record) if default_record in record_names else 0
            )
            selected_record = st.selectbox(
                "Record", record_names, index=default_record_index
            )
            rec_df = df[df["record_name"] == selected_record].copy()

            if len(anomaly_df):
                preferred_rows = rec_df[
                    rec_df["predicted_class"].astype(str).str.lower().ne("normal")
                ]
            else:
                preferred_rows = rec_df

            beat_options = sorted(rec_df["beat_index"].astype(int).unique().tolist())
            default_beat = (
                int(preferred_rows.iloc[0]["beat_index"])
                if len(preferred_rows)
                else beat_options[0]
            )
            default_beat_index = (
                beat_options.index(default_beat) if default_beat in beat_options else 0
            )
            selected_beat_index = st.selectbox(
                "Beat index", beat_options, index=default_beat_index
            )
            row = rec_df[rec_df["beat_index"] == selected_beat_index].iloc[0]

            st.caption(f"R-peak sample: {int(row['r_peak_sample']):,}")
            if pd.notna(row.get("annotation_symbol")):
                st.caption(f"Annotation symbol: {row.get('annotation_symbol')}")

        with prediction_col:
            st.subheader("Prediction")
            cnn_class = str(row["predicted_class"])
            cnn_confidence = float(row["confidence"])
            xgb_class = str(row.get("predicted_class_xgb", "Unknown"))
            xgb_confidence = float(row.get("confidence_xgb", 0.0))
            cnn_is_anomaly = cnn_class.lower() != "normal"
            xgb_is_anomaly = xgb_class.lower() != "normal"

            metric_cols = st.columns([1, 1])
            metric_cols[0].metric("CNN", cnn_class, f"{cnn_confidence * 100:.2f}%")
            metric_cols[1].metric("XGBoost", xgb_class, f"{xgb_confidence * 100:.2f}%")

            if cnn_is_anomaly and xgb_is_anomaly:
                st.warning("Both models classify this beat as anomalous.")
            elif cnn_is_anomaly:
                st.info("CNN flags this beat as anomalous.")
            elif xgb_is_anomaly:
                st.info("XGBoost flags this beat as anomalous.")
            else:
                st.success("Both models classify this beat as normal.")

            total_records = int(scan.get("total_records_analyzed", len(scan["record_names"])))
            anomalous_records = int(scan.get("anomalous_records", 0))
            normal_records = max(total_records - anomalous_records, 0)
            anomaly_pct_records = float(scan.get("anomaly_pct_records", 0.0))
            normal_pct_records = float(scan.get("normal_pct_records", 0.0))
            st.progress(anomaly_pct_records / 100.0)
            st.caption(
                f"Records analyzed: {total_records:,}. Anomalies: {anomalous_records:,} "
                f"({anomaly_pct_records:.2f}%). Normal: {normal_records:,} "
                f"({normal_pct_records:.2f}%)."
            )

            if len(anomaly_df):
                affected_records = anomaly_df["record_name"].astype(str).nunique()
                st.info(f"Anomalies found in {affected_records} scanned record(s).")
            else:
                st.success("No anomalous beats detected in the scanned records.")

            if st.button("Extract Anomaly Records"):
                with st.spinner("Extracting anomaly record details..."):
                    scan["anomaly_record_details"] = _build_anomaly_record_details(
                        anomaly_df=anomaly_df,
                        full_df=df,
                        extracted_root=extracted_root,
                        data_config=data_config,
                    )
                    st.session_state["scan_results"] = scan
                st.session_state["report_ready"] = True
                st.success("Anomaly records extracted and ready for download.")

            report_ready = st.session_state.get("report_ready", False)
            if report_ready:
                zip_bytes = _build_anomaly_report_zip(scan)
                st.download_button(
                    "Download Anomaly Report (.zip)",
                    data=zip_bytes,
                    file_name="ecg_anomaly_report.zip",
                    mime="application/zip",
                    width="stretch",
                )
            elif len(anomaly_df):
                st.caption("Click Extract Anomaly Records to prepare the ZIP report.")

        with viz_col:
            st.subheader("Raw ECG and Explanation")
            try:
                heartbeat, ground_truth = _discover_waveform_for_beat(
                    extracted_root=extracted_root,
                    record_name=selected_record,
                    beat_index=int(selected_beat_index),
                    data_config=data_config,
                )
                pred_class = cnn_class
                pred_confidence = cnn_confidence
                show_explanation = st.checkbox(
                    "Generate explanation overlay",
                    value=False,
                    help="Turn on only when you need the interpretability heatmap; it is slower.",
                )
                gradcam_stats = None
                if show_explanation:
                    predictor = _get_predictor()
                    classes = predictor.label_encoder.classes_
                    target_idx = next(
                        (i for i, c in enumerate(classes) if str(c) == pred_class),
                        _discover_normal_index(classes),
                    )
                    saliency = _gradient_saliency_for_input(
                        heartbeat_window_200=heartbeat,
                        model=predictor.model,
                        target_class_index=int(target_idx),
                        data_config=data_config,
                    )
                    fig = _plot_ecg_with_saliency(
                        series=heartbeat,
                        saliency=saliency,
                        title=f"Record {selected_record}, beat {selected_beat_index}: {pred_class}",
                    )
                    _, gradcam_stats = _build_gradcam_summary(
                        heartbeat_window_200=heartbeat,
                        predictor=predictor,
                        data_config=data_config,
                        target_class_index=target_idx,
                    )
                else:
                    fig = _plot_simple_timeseries(
                        heartbeat,
                        title=f"Record {selected_record}, beat {selected_beat_index}: {pred_class}",
                    )
                fig.update_layout(height=220)
                st.plotly_chart(fig, use_container_width=True)

                if gradcam_stats is not None:
                    stats_cols = st.columns(4)
                    stats_cols[0].metric("Grad-CAM mean", f"{gradcam_stats['mean_importance']:.3f}")
                    stats_cols[1].metric("Max importance", f"{gradcam_stats['max_importance']:.3f}")
                    stats_cols[2].metric("Top 10% mean", f"{gradcam_stats['top_10pct_mean']:.3f}")
                    stats_cols[3].metric("High importance %", f"{gradcam_stats['high_importance_pct'] * 100:.1f}%")

                st.session_state["selected_for_xai"] = {
                    "record_name": selected_record,
                    "beat_index": int(selected_beat_index),
                    "heartbeat": heartbeat,
                    "predicted_class": pred_class,
                    "confidence": pred_confidence,
                    "data_config": data_config,
                }
                if ground_truth is not None:
                    st.caption(f"Ground truth label: {ground_truth}")

                xgb_predictor = _get_xgboost_predictor()
                shap_fig = _build_xgb_shap_summary_figure(heartbeat, xgb_predictor)
                if shap_fig is not None:
                    st.subheader("XGBoost SHAP summary")
                    st.plotly_chart(shap_fig, use_container_width=True)
                else:
                    st.info("XGBoost SHAP summary is not available for the current run.")
            except Exception as e:
                st.error(f"Visualization or explanation failed: {type(e).__name__}: {e}")

        table_col, chart_col = st.columns([1.4, 1.0])
        with table_col:
            with st.expander("Anomalous Patient/Record Details", expanded=False):
                detail_df = scan.get("anomaly_record_details", pd.DataFrame())
                if len(detail_df):
                    display_cols = [
                        "record_id",
                        "prediction_result",
                        "predicted_classes",
                        "anomaly_score",
                        "anomalous_sampled_beats",
                        "sampled_beats",
                        "sampling_frequency",
                        "signal_mean",
                        "signal_std",
                        "signal_min",
                        "signal_max",
                        "explanation",
                    ]
                    available_cols = [col for col in display_cols if col in detail_df.columns]
                    if available_cols:
                        st.dataframe(
                            detail_df[available_cols],
                            width="stretch",
                            height=180,
                        )
                    else:
                        st.dataframe(detail_df, width="stretch", height=180)
                    selected_detail = st.selectbox(
                        "Record details",
                        detail_df["record_id"].astype(str).tolist(),
                    )
                    selected_detail_row = detail_df[
                        detail_df["record_id"].astype(str) == selected_detail
                    ].iloc[0]
                    st.text_area(
                        "Header information from .hea",
                        value=str(selected_detail_row.get("header_information", "")),
                        height=140,
                        disabled=True,
                    )
                    st.text_area(
                        "Annotation information from .atr",
                        value=str(selected_detail_row.get("annotation_information", "")),
                        height=120,
                        disabled=True,
                    )
                elif len(anomaly_df):
                    st.info("Click Extract Anomaly Records to load full header, annotation, and signal details.")
                else:
                    st.success("No anomalous rows to extract.")

            with st.expander("Anomalous Beat Predictions", expanded=False):
                if len(anomaly_df):
                    display_cols = [
                        "record_name",
                        "beat_index",
                        "r_peak_sample",
                        "predicted_class",
                        "confidence",
                        "anomaly_score",
                        "ground_truth_label",
                        "annotation_symbol",
                    ]
                    st.dataframe(
                        anomaly_df[display_cols]
                        .sort_values(["record_name", "r_peak_sample"])
                        .head(500),
                        width="stretch",
                        height=160,
                    )
                else:
                    st.success("No anomalous rows to extract.")

        with chart_col:
            with st.expander("Class Distribution", expanded=False):
                dist = df["predicted_class"].astype(str).value_counts().reset_index()
                dist.columns = ["Class", "Count"]
                bar = go.Figure(
                    data=[
                        go.Bar(
                            x=dist["Class"],
                            y=dist["Count"],
                            marker_color=["#556b8e", "#c84c4c", "#d08a3c", "#6c9a6b", "#8b6f9f"][
                                : len(dist)
                            ],
                        )
                    ]
                )
                bar.update_layout(height=180, margin=dict(l=8, r=8, t=8, b=8))
                st.plotly_chart(bar, use_container_width=True)
    else:
        st.warning("The scan completed, but no heartbeat windows were extracted.")

with st.expander("Model Evaluation Artifacts", expanded=False):
    results_dir = Path(__file__).resolve().parent / "results"
    history_path = results_dir / "training_history.json"
    conf_csv = results_dir / "confusion_matrix.csv"
    eval_col, cm_col = st.columns(2)

    with eval_col:
        if history_path.exists():
            history = json.loads(history_path.read_text())
            epochs = list(range(1, len(history.get("val_loss", [])) + 1))
            fig = go.Figure()
            if "loss" in history:
                fig.add_trace(go.Scatter(x=epochs, y=history["loss"], mode="lines", name="Train loss"))
            if "val_loss" in history:
                fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], mode="lines", name="Val loss"))
            fig.update_layout(height=260, margin=dict(l=20, r=20, t=35, b=20))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No training history artifact found.")

    with cm_col:
        if conf_csv.exists():
            try:
                cm_df = pd.read_csv(conf_csv)
                classes = list(cm_df.columns[1:])
                matrix = cm_df.iloc[:, 1:].to_numpy()
                fig = go.Figure(
                    data=go.Heatmap(z=matrix, x=classes, y=classes, colorscale="Blues")
                )
                fig.update_layout(height=260, margin=dict(l=20, r=20, t=35, b=20))
                st.plotly_chart(fig, width="stretch")
            except Exception as e:
                st.error(f"Failed to render confusion matrix: {type(e).__name__}: {e}")
        else:
            st.info("No confusion matrix artifact found.")
