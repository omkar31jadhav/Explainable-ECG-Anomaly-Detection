"""Central configuration for the MIT-BIH ECG classification pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

DEFAULT_MODEL_PATH = MODELS_DIR / "ecg_cnn.keras"
DEFAULT_LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
DEFAULT_DATASET_CACHE_PATH = RESULTS_DIR / "mitbih_beat_dataset.npz"


# Non-beat annotation symbols appear in MIT-BIH rhythm/comment annotations.
NON_BEAT_SYMBOLS = {
    "+",
    "~",
    "|",
    '"',
    "!",
    "[",
    "]",
    "(",
    ")",
    "p",
    "t",
    "u",
    "`",
    "'",
    "^",
    "=",
}


# AAMI EC57-style heartbeat superclasses. This keeps the output compact and
# clinically useful for downstream XAI/dashboard work.
AAMI_CLASS_MAP: Mapping[str, str] = {
    "N": "Normal",
    "L": "Normal",
    "R": "Normal",
    "e": "Normal",
    "j": "Normal",
    "A": "SVEB",
    "a": "SVEB",
    "J": "SVEB",
    "S": "SVEB",
    "V": "PVC",
    "E": "PVC",
    "F": "Fusion",
    "/": "Unknown",
    "f": "Unknown",
    "Q": "Unknown",
}


# Optional symbol-level labels for experiments that should keep bundle-branch
# blocks and other beat types separate.
SYMBOL_CLASS_MAP: Mapping[str, str] = {
    "N": "Normal",
    "L": "LBBB",
    "R": "RBBB",
    "A": "APB",
    "a": "Aberrated_APB",
    "J": "Nodal_APB",
    "S": "SVPB",
    "V": "PVC",
    "F": "Fusion",
    "e": "Atrial_Escape",
    "j": "Nodal_Escape",
    "E": "Ventricular_Escape",
    "/": "Paced",
    "f": "Fusion_Paced_Normal",
    "Q": "Unclassifiable",
}


CLASS_DESCRIPTIONS: Mapping[str, str] = {
    "Normal": "Normal and bundle-branch block beats grouped by AAMI.",
    "SVEB": "Supraventricular ectopic beats.",
    "PVC": "Premature ventricular contraction and ventricular escape beats.",
    "Fusion": "Fusion of ventricular and normal beats.",
    "Unknown": "Paced, fusion paced, or unclassifiable beats.",
}


def get_label_mapping(label_scheme: str = "aami") -> Mapping[str, str]:
    """Return the annotation-symbol mapping for a supported label scheme."""

    normalized = label_scheme.lower().strip()
    if normalized == "aami":
        return AAMI_CLASS_MAP
    if normalized in {"symbol", "symbolic", "beat"}:
        return SYMBOL_CLASS_MAP
    raise ValueError(
        f"Unsupported label scheme '{label_scheme}'. Use 'aami' or 'symbol'."
    )


@dataclass(frozen=True)
class DataConfig:
    """Data loading and preprocessing settings."""

    data_dir: Path = DATA_DIR
    annotator: str = "atr"
    signal_channel: int = 0
    samples_before: int = 100
    samples_after: int = 100
    normalization: str = "zscore"
    label_scheme: str = "aami"
    include_unknown: bool = True
    standardize_beats: bool = True

    @property
    def window_size(self) -> int:
        return self.samples_before + self.samples_after


@dataclass(frozen=True)
class SplitConfig:
    """Train/validation/test split settings."""

    validation_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42


@dataclass(frozen=True)
class ModelConfig:
    """1D CNN hyperparameters."""

    input_length: int = 200
    n_channels: int = 1
    conv_filters: tuple[int, ...] = (32, 64)
    kernel_size: int = 5
    pool_size: int = 2
    dense_units: int = 128
    dropout_rate: float = 0.3
    activation: str = "relu"
    learning_rate: float = 1e-3
    batch_normalization: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    """Training loop settings."""

    epochs: int = 50
    batch_size: int = 128
    early_stopping_patience: int = 8
    reduce_lr_patience: int = 4
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-6
    use_class_weights: bool = True
    random_state: int = 42

