from src.application.paths import models_dir
from src.application.training_stub import train_dummy_model
import torch


def main() -> None:
    model, loss = train_dummy_model()
    output_path = models_dir() / "audio_model.pth"
    torch.save(model.state_dict(), output_path)
    print(f"Trained dummy model. Loss: {loss:.4f}")
    print(f"Saved model to {output_path}")


if __name__ == "__main__":
    main()
