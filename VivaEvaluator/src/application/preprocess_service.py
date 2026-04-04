import json
from typing import Dict

from src.application.audio_features import extract_audio_features
from src.infrastructure.mediapipe_extractors import extract_face_hand_landmarks
from src.infrastructure.asr_whisper import transcribe_audio


def preprocess_sample(
    video_path: str, audio_path: str, whisper_model: str = "small"
) -> Dict:
    face_landmarks, hand_landmarks = extract_face_hand_landmarks(video_path)
    transcript = transcribe_audio(audio_path, model_name=whisper_model)
    audio_features = extract_audio_features(audio_path)

    return {
        "id": "sample",
        "transcript": transcript,
        "audio_features": audio_features,
        "face_landmarks": face_landmarks,
        "hand_landmarks": hand_landmarks,
        "labels": {"confidence": 0, "quality": 0},
    }


def save_features(features: Dict, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(features, f)
