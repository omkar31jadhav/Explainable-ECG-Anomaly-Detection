"""WFDB-based loading utilities for the original MIT-BIH database."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import DATA_DIR, DataConfig, get_label_mapping
from .preprocessing import map_symbol_to_label


def _import_wfdb():
    try:
        import wfdb  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "wfdb is required to read MIT-BIH .dat/.hea/.atr files. "
            "Install the project dependencies with `pip install -r requirements.txt`."
        ) from exc
    return wfdb


@dataclass(frozen=True)
class RecordInfo:
    """Filesystem information for one MIT-BIH record."""

    name: str
    base_path: Path
    dat_path: Path
    hea_path: Path
    atr_path: Path


@dataclass(frozen=True)
class ECGRecord:
    """Loaded ECG signal and metadata."""

    name: str
    signal: np.ndarray
    fs: float
    metadata: dict


@dataclass(frozen=True)
class ECGAnnotation:
    """Loaded beat annotation samples and symbols."""

    record_name: str
    samples: np.ndarray
    symbols: list[str]
    aux_note: list[str]


class MITBIHDataLoader:
    """Discover and load MIT-BIH Arrhythmia Database records with WFDB."""

    def __init__(
        self,
        data_dir: str | Path = DATA_DIR,
        annotator: str = "atr",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.annotator = annotator

    @classmethod
    def from_config(cls, config: DataConfig) -> "MITBIHDataLoader":
        return cls(data_dir=config.data_dir, annotator=config.annotator)

    def discover_records(self, validate: bool = True) -> list[RecordInfo]:
        """Discover MIT-BIH records from RECORDS or direct .hea files."""

        record_names = self._read_records_file()
        if not record_names:
            record_names = sorted(path.stem for path in self.data_dir.glob("*.hea"))

        records: list[RecordInfo] = []
        seen: set[str] = set()
        for name in record_names:
            if not name or name in seen:
                continue
            info = self._build_record_info(name)
            if not validate or self._record_files_exist(info):
                records.append(info)
                seen.add(name)

        return records

    def load_record(self, record: str | RecordInfo) -> ECGRecord:
        """Load ECG signals and metadata for one record."""

        wfdb = _import_wfdb()
        info = self._coerce_record_info(record)
        wfdb_record = wfdb.rdrecord(str(info.base_path))
        signal = np.asarray(wfdb_record.p_signal, dtype=np.float32)
        metadata = {
            "record_name": info.name,
            "fs": float(getattr(wfdb_record, "fs", 0.0)),
            "n_sig": int(getattr(wfdb_record, "n_sig", signal.shape[1] if signal.ndim == 2 else 1)),
            "sig_len": int(getattr(wfdb_record, "sig_len", signal.shape[0])),
            "sig_name": list(getattr(wfdb_record, "sig_name", [])),
            "units": list(getattr(wfdb_record, "units", [])),
            "adc_gain": list(getattr(wfdb_record, "adc_gain", [])),
            "baseline": list(getattr(wfdb_record, "baseline", [])),
            "comments": list(getattr(wfdb_record, "comments", [])),
        }
        return ECGRecord(
            name=info.name,
            signal=signal,
            fs=float(metadata["fs"]),
            metadata=metadata,
        )

    def load_annotation(self, record: str | RecordInfo) -> ECGAnnotation:
        """Load R-peak sample locations and annotation symbols."""

        wfdb = _import_wfdb()
        info = self._coerce_record_info(record)
        annotation = wfdb.rdann(str(info.base_path), self.annotator)
        return ECGAnnotation(
            record_name=info.name,
            samples=np.asarray(annotation.sample, dtype=np.int64),
            symbols=list(annotation.symbol),
            aux_note=list(getattr(annotation, "aux_note", [])),
        )

    def dataset_statistics(
        self,
        records: Iterable[str | RecordInfo] | None = None,
        label_scheme: str = "aami",
        include_unknown: bool = True,
    ) -> dict:
        """Generate record, annotation, and class-distribution statistics."""

        discovered = list(records) if records is not None else self.discover_records()
        symbol_counts: Counter[str] = Counter()
        class_counts: Counter[str] = Counter()
        total_annotations = 0
        mapped_beats = 0

        get_label_mapping(label_scheme)
        for record in discovered:
            annotation = self.load_annotation(record)
            symbol_counts.update(annotation.symbols)
            total_annotations += len(annotation.symbols)
            for symbol in annotation.symbols:
                label = map_symbol_to_label(
                    symbol,
                    label_scheme=label_scheme,
                    include_unknown=include_unknown,
                )
                if label is not None:
                    class_counts[label] += 1
                    mapped_beats += 1

        return {
            "num_records": len(discovered),
            "total_annotations": total_annotations,
            "num_beats": mapped_beats,
            "symbol_distribution": dict(sorted(symbol_counts.items())),
            "class_distribution": dict(sorted(class_counts.items())),
        }

    def _read_records_file(self) -> list[str]:
        records_path = self.data_dir / "RECORDS"
        if not records_path.exists():
            return []
        return [
            line.strip()
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    def _build_record_info(self, name: str) -> RecordInfo:
        base_path = self.data_dir / name
        return RecordInfo(
            name=name,
            base_path=base_path,
            dat_path=base_path.with_suffix(".dat"),
            hea_path=base_path.with_suffix(".hea"),
            atr_path=base_path.with_suffix(f".{self.annotator}"),
        )

    @staticmethod
    def _record_files_exist(info: RecordInfo) -> bool:
        return info.dat_path.exists() and info.hea_path.exists() and info.atr_path.exists()

    def _coerce_record_info(self, record: str | RecordInfo) -> RecordInfo:
        if isinstance(record, RecordInfo):
            return record

        candidate = Path(record)
        if candidate.suffix:
            candidate = candidate.with_suffix("")
        if not candidate.is_absolute():
            candidate = self.data_dir / candidate

        name = candidate.name
        return RecordInfo(
            name=name,
            base_path=candidate,
            dat_path=candidate.with_suffix(".dat"),
            hea_path=candidate.with_suffix(".hea"),
            atr_path=candidate.with_suffix(f".{self.annotator}"),
        )

