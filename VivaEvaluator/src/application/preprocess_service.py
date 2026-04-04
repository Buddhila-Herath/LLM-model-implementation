import json
from typing import Any, Dict

from src.application.audio_features import extract_audio_features
from src.application.gesture_features import compute_hand_movement
from src.infrastructure.mediapipe_extractors import extract_vision_features
from src.infrastructure.asr_whisper import transcribe_audio


def preprocess_sample(
    video_path: str, audio_path: str, whisper_model: str = "small"
) -> Dict[str, Any]:
    vision = extract_vision_features(video_path)
    face_landmarks = vision["face_landmarks"]
    hand_landmarks = vision["hand_landmarks"]
    pose_features = vision["pose_features"]
    emotion_results = vision["emotion_results"]

    transcript = transcribe_audio(audio_path, model_name=whisper_model)
    audio_features = extract_audio_features(audio_path)

    movement_mean = compute_hand_movement(hand_landmarks)

    final_features = {
        "face": face_landmarks,
        "hands": hand_landmarks,
        "pose": pose_features,
        "emotion": emotion_results,
        "audio": audio_features,
    }

    return {
        "id": "sample",
        "transcript": transcript,
        "audio_features": audio_features,
        "face_landmarks": face_landmarks,
        "hand_landmarks": hand_landmarks,
        "pose_features": pose_features,
        "emotion_results": emotion_results,
        "hand_gesture_stats": {"movement_mean": movement_mean},
        "final_features": final_features,
        "labels": {"confidence": 0, "quality": 0},
    }


def save_features(features: Dict, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(features, f)
