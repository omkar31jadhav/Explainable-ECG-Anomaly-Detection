from __future__ import annotations

import numpy as np

from src import data_loader
from src.data_loader import MITBIHDataLoader

FIXTURE_DIR = __import__("pathlib").Path(__file__).resolve().parent / "fixtures" / "mitbih_minimal"


def test_discover_records_from_records_file():
    loader = MITBIHDataLoader(FIXTURE_DIR)
    records = loader.discover_records()

    assert [record.name for record in records] == ["100", "101"]
    assert records[0].base_path == FIXTURE_DIR / "100"


def test_load_record_uses_wfdb(monkeypatch):
    class FakeWFDBRecord:
        p_signal = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        fs = 360
        n_sig = 2
        sig_len = 2
        sig_name = ["MLII", "V5"]
        units = ["mV", "mV"]
        adc_gain = [200, 200]
        baseline = [1024, 1024]
        comments = ["synthetic"]

    class FakeWFDB:
        @staticmethod
        def rdrecord(path):
            assert path.endswith("100")
            return FakeWFDBRecord()

    monkeypatch.setattr(data_loader, "_import_wfdb", lambda: FakeWFDB)

    record = MITBIHDataLoader(FIXTURE_DIR).load_record("100")

    assert record.name == "100"
    assert record.fs == 360
    assert record.signal.shape == (2, 2)
    assert record.metadata["sig_name"] == ["MLII", "V5"]


def test_load_annotation_uses_wfdb(monkeypatch):
    class FakeAnnotation:
        sample = np.array([100, 220, 350])
        symbol = ["N", "V", "+"]
        aux_note = ["", "", "(N"]

    class FakeWFDB:
        @staticmethod
        def rdann(path, extension):
            assert path.endswith("100")
            assert extension == "atr"
            return FakeAnnotation()

    monkeypatch.setattr(data_loader, "_import_wfdb", lambda: FakeWFDB)

    annotation = MITBIHDataLoader(FIXTURE_DIR).load_annotation("100")

    assert annotation.record_name == "100"
    assert annotation.samples.tolist() == [100, 220, 350]
    assert annotation.symbols == ["N", "V", "+"]
