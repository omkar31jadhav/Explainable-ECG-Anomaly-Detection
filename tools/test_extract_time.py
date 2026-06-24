from pathlib import Path
import time
import sys
sys.path.insert(0, str(Path('.').resolve()))
from src.data_loader import MITBIHDataLoader
from src.config import DataConfig
from src.preprocessing import extract_heartbeat_windows


def main():
    data_dir = Path('data')
    loader = MITBIHDataLoader(data_dir=data_dir)
    records = [p.stem for p in sorted(data_dir.glob('*.hea'))][:10]
    cfg = DataConfig(data_dir=data_dir)
    for name in records:
        try:
            t0 = time.perf_counter()
            rec = loader.load_record(name)
            t1 = time.perf_counter()
            ann = loader.load_annotation(name)
            t2 = time.perf_counter()
            res = extract_heartbeat_windows(
                rec.signal,
                ann.samples,
                labels=ann.symbols,
                symbols=ann.symbols,
                samples_before=cfg.samples_before,
                samples_after=cfg.samples_after,
                channel=cfg.signal_channel,
            )
            t3 = time.perf_counter()
            print(f"{name}: load_record={t1-t0:.2f}s, load_annotation={t2-t1:.2f}s, extract={t3-t2:.2f}s, beats={res.beats.shape[0]}")
        except Exception as e:
            print(f"{name}: ERROR {type(e).__name__}: {e}")


if __name__ == '__main__':
    main()
