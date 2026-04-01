import json
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

NUM_HAND_LANDMARKS = 21
LANDMARK_DIMS = 3
DEFAULT_SEQ_LEN = 30

class HandLSTM(nn.Module):
    """Sequence classifier over hand landmarks using LSTM."""
    def __init__(self, hidden_size: int = 32, num_classes: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(NUM_HAND_LANDMARKS * LANDMARK_DIMS, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq_len, 21, 3]
        batch_size, seq_len, n_lm, n_dim = x.shape
        x = x.view(batch_size, seq_len, n_lm * n_dim)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

def _normalize_hand_frame(frame: List[List[float]]) -> List[List[float]]:
    if not frame:
        return [[0.0, 0.0, 0.0] for _ in range(NUM_HAND_LANDMARKS)]
    fixed = frame[:NUM_HAND_LANDMARKS]
    if len(fixed) < NUM_HAND_LANDMARKS:
        fixed += [[0.0, 0.0, 0.0] for _ in range(NUM_HAND_LANDMARKS - len(fixed))]
    normalized: List[List[float]] = []
    for point in fixed:
        if not isinstance(point, list):
            normalized.append([0.0, 0.0, 0.0])
            continue
        p = point[:LANDMARK_DIMS]
        if len(p) < LANDMARK_DIMS:
            p += [0.0] * (LANDMARK_DIMS - len(p))
        normalized.append([float(p[0]), float(p[1]), float(p[2])])
    return normalized

def _sequence_from_hand_landmarks(hand_landmarks: List, seq_len: int) -> torch.Tensor:
    frames = [
        _normalize_hand_frame(frame)
        for frame in hand_landmarks
        if isinstance(frame, list)
    ]
    if len(frames) >= seq_len:
        frames = frames[:seq_len]
    else:
        padding = [
            [[0.0, 0.0, 0.0] for _ in range(NUM_HAND_LANDMARKS)]
            for _ in range(seq_len - len(frames))
        ]
        frames = frames + padding
    return torch.tensor(frames, dtype=torch.float32)

def load_hand_samples(features_path: str, seq_len: int = DEFAULT_SEQ_LEN) -> Tuple[torch.Tensor, torch.Tensor]:
    with open(features_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data if isinstance(data, list) else [data]
    xs: List[torch.Tensor] = []
    ys: List[int] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        hand_landmarks = record.get("hand_landmarks", [])
        if not hand_landmarks:
            continue
        seq = _sequence_from_hand_landmarks(hand_landmarks, seq_len)
        label = int(record.get("labels", {}).get("confidence", 0))
        ys.append(0 if label <= 0 else 1)
        xs.append(seq)
    if not xs:
        raise ValueError("No valid hand landmark samples found")
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)

def synthetic_hand_dataset(samples: int = 120, seq_len: int = DEFAULT_SEQ_LEN) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(samples, seq_len, NUM_HAND_LANDMARKS, LANDMARK_DIMS)
    y = torch.randint(0, 2, (samples,), dtype=torch.long)
    return x, y

def train_hand_model(features_path: str, epochs: int = 3) -> Tuple[HandLSTM, float, str]:
    source = "features"
    try:
        x, y = load_hand_samples(features_path)
        if x.shape[0] < 8 or len(torch.unique(y)) < 2:
            raise ValueError("Insufficient label diversity for training")
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        x, y = synthetic_hand_dataset()
        source = "synthetic"
    loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True)
    model = HandLSTM()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    last_loss = 0.0
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
    return model, last_loss, source

def infer_hand_probability(model_path: str, features_path: str, seq_len: int = DEFAULT_SEQ_LEN) -> Tuple[float, str]:
    model = HandLSTM()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    source = "features"
    try:
        x, _ = load_hand_samples(features_path, seq_len=seq_len)
        sample = x[:1]
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        sample = torch.randn(1, seq_len, NUM_HAND_LANDMARKS, LANDMARK_DIMS)
        source = "synthetic"
    with torch.no_grad():
        logits = model(sample)
        probability = torch.softmax(logits, dim=1)[0, 1].item()
    return probability, source