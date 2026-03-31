from typing import Dict, List

import numpy as np
import librosa


def extract_audio_features(audio_path: str) -> Dict[str, List]:
    y, sr = librosa.load(audio_path, sr=None)
    if y.size == 0:
        return {"mfcc": [], "pitch": []}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T.tolist()
    pitch = _extract_pitch(y, sr)
    return {"mfcc": mfcc, "pitch": pitch}


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
