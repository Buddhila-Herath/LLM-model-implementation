import json
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


NUM_FACE_LANDMARKS = 468
LANDMARK_DIMS = 3
DEFAULT_SEQ_LEN = 30


class FaceCNNLSTM(nn.Module):
    """Sequence classifier over face landmarks using CNN frame encoder + LSTM."""

    def __init__(self, hidden_size: int = 64, num_classes: int = 2) -> None:
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv1d(in_channels=LANDMARK_DIMS, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.temporal = nn.LSTM(input_size=64, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq_len, 468, 3]
        batch_size, seq_len, _, _ = x.shape
        x = x.view(batch_size * seq_len, NUM_FACE_LANDMARKS, LANDMARK_DIMS)
        x = x.permute(0, 2, 1)  # [batch*seq, 3, 468]
        frame_embeddings = self.frame_encoder(x).squeeze(-1)  # [batch*seq, 64]
        frame_embeddings = frame_embeddings.view(batch_size, seq_len, -1)

        _, (h_n, _) = self.temporal(frame_embeddings)
        return self.classifier(h_n[-1])


def _normalize_frame(frame: List[List[float]]) -> List[List[float]]:
    if not frame:
        return [[0.0, 0.0, 0.0] for _ in range(NUM_FACE_LANDMARKS)]

    fixed = frame[:NUM_FACE_LANDMARKS]
    if len(fixed) < NUM_FACE_LANDMARKS:
        fixed += [[0.0, 0.0, 0.0] for _ in range(NUM_FACE_LANDMARKS - len(fixed))]

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


def _sequence_from_landmarks(face_landmarks: List, seq_len: int) -> torch.Tensor:
    frames = [
        _normalize_frame(frame)
        for frame in face_landmarks
        if isinstance(frame, list)
    ]

    if len(frames) >= seq_len:
        frames = frames[:seq_len]
    else:
        padding = [
            [[0.0, 0.0, 0.0] for _ in range(NUM_FACE_LANDMARKS)]
            for _ in range(seq_len - len(frames))
        ]
        frames = frames + padding

    return torch.tensor(frames, dtype=torch.float32)


def load_face_samples(features_path: str, seq_len: int = DEFAULT_SEQ_LEN) -> Tuple[torch.Tensor, torch.Tensor]:
    with open(features_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data if isinstance(data, list) else [data]

    xs: List[torch.Tensor] = []
    ys: List[int] = []

    for record in records:
        if not isinstance(record, dict):
            continue
        face_landmarks = record.get("face_landmarks", [])
        if not face_landmarks:
            continue

        seq = _sequence_from_landmarks(face_landmarks, seq_len)
        label = int(record.get("labels", {}).get("confidence", 0))
        ys.append(0 if label <= 0 else 1)
        xs.append(seq)

    if not xs:
        raise ValueError("No valid face landmark samples found")

    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)


def synthetic_face_dataset(samples: int = 120, seq_len: int = DEFAULT_SEQ_LEN) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(samples, seq_len, NUM_FACE_LANDMARKS, LANDMARK_DIMS)
    y = torch.randint(0, 2, (samples,), dtype=torch.long)
    return x, y


def train_face_model(features_path: str, epochs: int = 3) -> Tuple[FaceCNNLSTM, float, str]:
    source = "features"
    try:
        x, y = load_face_samples(features_path)
        if x.shape[0] < 8 or len(torch.unique(y)) < 2:
            raise ValueError("Insufficient label diversity for training")
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        x, y = synthetic_face_dataset()
        source = "synthetic"

    loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True)

    model = FaceCNNLSTM()
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


def infer_face_probability(model_path: str, features_path: str, seq_len: int = DEFAULT_SEQ_LEN) -> Tuple[float, str]:
    model = FaceCNNLSTM()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    source = "features"
    try:
        x, _ = load_face_samples(features_path, seq_len=seq_len)
        sample = x[:1]
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        sample = torch.randn(1, seq_len, NUM_FACE_LANDMARKS, LANDMARK_DIMS)
        source = "synthetic"

    with torch.no_grad():
        logits = model(sample)
        probability = torch.softmax(logits, dim=1)[0, 1].item()

    return probability, source
