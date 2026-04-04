from typing import List, Sequence, Union

import numpy as np

Number = Union[float, int]


def compute_hand_movement(hand_landmarks: Sequence[Sequence[Sequence[Number]]]) -> float:
    """Mean Euclidean movement between consecutive hand poses (21×3 landmarks each)."""
    if len(hand_landmarks) < 2:
        return 0.0

    movements: List[float] = []
    for i in range(1, len(hand_landmarks)):
        prev = np.asarray(hand_landmarks[i - 1], dtype=np.float64)
        curr = np.asarray(hand_landmarks[i], dtype=np.float64)
        if prev.shape != curr.shape:
            continue
        diff = np.linalg.norm(curr - prev)
        movements.append(float(diff))

    return float(np.mean(movements)) if movements else 0.0
