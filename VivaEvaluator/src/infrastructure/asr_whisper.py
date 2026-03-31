from typing import Optional

import whisper


_MODEL: Optional[whisper.Whisper] = None


def _get_model() -> whisper.Whisper:
    global _MODEL
    if _MODEL is None:
        _MODEL = whisper.load_model("base")
    return _MODEL


def transcribe_audio(audio_path: str) -> str:
    model = _get_model()
    result = model.transcribe(audio_path)
    return result.get("text", "").strip()
