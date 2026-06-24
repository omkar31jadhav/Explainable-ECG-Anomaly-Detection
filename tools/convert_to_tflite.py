"""Convert the trained Keras model to a quantized TFLite model using a
representative dataset drawn from the MIT-BIH files in `data/`.

Run this from the project root using the project's Python env.
"""
from pathlib import Path
import numpy as np
import tensorflow as tf

from src.config import DataConfig
from src.data_loader import MITBIHDataLoader
from src.preprocessing import extract_heartbeat_windows, prepare_beat_features


def collect_representative_samples(data_dir: str, max_samples: int = 256):
    cfg = DataConfig(data_dir=data_dir)
    loader = MITBIHDataLoader(data_dir=data_dir)
    samples = []
    for hea in sorted(Path(data_dir).glob('*.hea')):
        name = hea.stem
        try:
            rec = loader.load_record(name)
            ann = loader.load_annotation(name)
            res = extract_heartbeat_windows(
                signal=rec.signal,
                r_peaks=ann.samples,
                labels=ann.symbols,
                symbols=ann.symbols,
                samples_before=cfg.samples_before,
                samples_after=cfg.samples_after,
                channel=cfg.signal_channel,
            )
        except Exception:
            continue
        if res is None or res.beats.shape[0] == 0:
            continue
        for b in res.beats:
            samples.append(b.astype(np.float32))
            if len(samples) >= max_samples:
                break
        if len(samples) >= max_samples:
            break

    if not samples:
        raise RuntimeError('No representative samples collected from data/.')

    arr = np.stack(samples)
    X = prepare_beat_features(arr, standardize=cfg.standardize_beats, add_channel_axis=True)
    return X


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / 'data'
    model_path = project_root / 'models' / 'ecg_cnn.keras'
    out_path = project_root / 'models' / 'ecg_cnn_quant.tflite'

    print('Collecting representative samples...')
    rep = collect_representative_samples(str(data_dir), max_samples=256)
    print('Representative samples shape:', rep.shape)

    print('Loading Keras model...')
    model = tf.keras.models.load_model(model_path, compile=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def rep_gen():
        for i in range(rep.shape[0]):
            yield [rep[i:i+1]]

    converter.representative_dataset = rep_gen
    # Prefer full integer quantization if supported
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    print('Converting to TFLite (this may take a moment)...')
    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)
    print('Wrote quantized TFLite model to', out_path)


if __name__ == '__main__':
    main()
