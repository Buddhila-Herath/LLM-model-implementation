from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Labels:
    confidence: int
    quality: int


@dataclass
class FeatureSample:
    sample_id: str
    transcript: str
    audio_features: Dict[str, Any]
    face_landmarks: List
    hand_landmarks: List
    labels: Labels
    pose_features: Optional[List] = None
    emotion_results: Optional[List] = None
    hand_gesture_stats: Optional[Dict[str, float]] = None
    final_features: Optional[Dict[str, Any]] = None
