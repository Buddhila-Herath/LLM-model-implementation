from typing import Tuple

import torch
import torch.nn as nn


class LSTMAudio(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(4, 16, batch_first=True)
        self.fc = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


def infer_probability(model_path: str) -> Tuple[float, torch.Tensor]:
    model = LSTMAudio()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    input_tensor = torch.randn(1, 10, 4)
    with torch.no_grad():
        output = model(input_tensor)
    prob = torch.softmax(output, dim=1)[0, 1].item()
    return prob, output
