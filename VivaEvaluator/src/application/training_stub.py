from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMAudio(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(4, 16, batch_first=True)
        self.fc = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


def train_dummy_model() -> Tuple[LSTMAudio, float]:
    x = torch.randn(100, 10, 4)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = LSTMAudio()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    last_loss = 0.0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())

    return model, last_loss
