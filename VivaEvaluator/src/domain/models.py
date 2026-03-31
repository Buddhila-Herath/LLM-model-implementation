from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Labels:
    confidence: int
    quality: int


@dataclass
class FeatureSample:
    sample_id: str
    transcript: str
    audio_features: Dict[str, List]
    face_landmarks: List
    hand_landmarks: List
    labels: Labels
