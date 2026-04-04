from typing import Dict

import whisper


_MODEL_CACHE: Dict[str, whisper.Whisper] = {}


def _get_model(model_name: str) -> whisper.Whisper:
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = whisper.load_model(model_name)
    return _MODEL_CACHE[model_name]


def transcribe_audio(audio_path: str, model_name: str = "small") -> str:
    model = _get_model(model_name)
    result = model.transcribe(audio_path)
    return result.get("text", "").strip()
