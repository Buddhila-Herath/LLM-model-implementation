from typing import Dict, List, Union

import numpy as np
import librosa


def extract_audio_features(audio_path: str) -> Dict[str, Union[List, float]]:
    y, sr = librosa.load(audio_path, sr=None)
    if y.size == 0:
        return {
            "mfcc": [],
            "pitch": [],
            "tempo": 0.0,
            "zcr": 0.0,
            "energy": 0.0,
        }

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T.tolist()
    pitch = _extract_pitch(y, sr)

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = float(np.asarray(tempo).mean())
    except Exception:
        tempo_val = 0.0

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = float(np.mean(zcr))

    energy = librosa.feature.rms(y=y)
    energy_mean = float(np.mean(energy))

    return {
        "mfcc": mfcc,
        "pitch": pitch,
        "tempo": tempo_val,
        "zcr": zcr_mean,
        "energy": energy_mean,
    }


def _extract_pitch(y: np.ndarray, sr: int) -> List[float]:
    try:
        import parselmouth  # type: ignore

        sound = parselmouth.Sound(y, sr)
        pitch = sound.to_pitch()
        values = pitch.selected_array["frequency"]
        return [float(v) for v in values if v > 0]
    except Exception:
        f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
        return [float(v) for v in f0 if v > 0]
